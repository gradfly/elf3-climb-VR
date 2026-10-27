/**
 * motion.js —— 网页端「运动仿真」回放
 *
 * 背景：浏览器里的 MuJoCo WASM 没有策略网络（没有 onnxruntime），所以"运动仿真"没法
 * 在网页里现算。做法是先在本地用 `web/tools/bake_motion.py` 把仿真跑一遍，
 * 把每个控制步（50 Hz）的 qpos 录成 float32 二进制；网页端只负责按时间插值回放。
 *
 * 数据格式（见 web/assets/motion_*.json 的 meta）：
 *   qpos = [x, y, z, qw, qx, qy, qz, 关节角 × 29]   —— 与 elf3.xml 的自由基座 + 29 关节一致
 *
 * 采样：位置线性插值，四元数用最短路径 slerp（直接对四元数做线性插值会让旋转矩阵
 * 出现缩放，MuJoCo 的 mj_forward 不会帮你归一化，所以这里必须 slerp）。
 */

export class MotionPlayer {
  /**
   * @param {object}     meta    motion_*.json
   * @param {Float32Array} frames (nframes, nq) 行优先
   */
  constructor(meta, frames) {
    this.meta = meta;
    this.frames = frames;
    this.nq = meta.nq | 0;
    this.nframes = meta.nframes | 0;
    this.fps = meta.fps || 50;
    this.duration = this.nframes / this.fps;
    this.label = meta.label || meta.name;
    this.kind = meta.kind;

    this.time = 0;          // 当前播放时间（秒）
    this.speed = 1;
    this.loop = true;
    this.paused = false;

    this._out = new Float64Array(this.nq);
  }

  /** 从 assets/ 加载 motion_<name>.json + 同名 .bin */
  static async load(assetsBase, name) {
    const metaRes = await fetch(`${assetsBase}motion_${name}.json`);
    if (!metaRes.ok) throw new Error(`HTTP ${metaRes.status} — motion_${name}.json`);
    const meta = await metaRes.json();

    // 带版本号请求二进制：数据重新录制后 URL 会变，避免任何中间层/浏览器拿旧缓存
    const ver = meta.version || meta.generated || meta.nframes || '';
    const binUrl = `${assetsBase}${meta.bin}${ver ? `?v=${encodeURIComponent(ver)}` : ''}`;
    const binRes = await fetch(binUrl);
    if (!binRes.ok) throw new Error(`HTTP ${binRes.status} — ${meta.bin}`);
    const buf = await binRes.arrayBuffer();
    const frames = new Float32Array(buf);
    const need = meta.nframes * meta.nq;
    if (frames.length !== need) {
      throw new Error(`${meta.bin} 长度不符：期望 ${need} 个 float32，实际 ${frames.length}`);
    }
    return new MotionPlayer(meta, frames);
  }

  /** 播放进度 0..1 */
  get progress() {
    return this.duration > 0 ? (this.time % this.duration) / this.duration : 0;
  }

  seek(t) {
    const d = this.duration;
    this.time = d > 0 ? ((t % d) + d) % d : 0;
    return this;
  }

  restart() {
    this.time = 0;
    return this;
  }

  /** 前进 dt 秒（按 speed 缩放）；到末尾按 loop 决定回卷或停在末帧 */
  advance(dt) {
    if (this.paused) return this;
    let t = this.time + dt * this.speed;
    if (!this.loop && t >= this.duration) {
      t = this.duration;
      this.paused = true;
    }
    this.seek(t);
    return this;
  }

  /**
   * 采样当前时刻的 qpos，写进 out（可直接传 MuJoCo 的 data.qpos 视图）。
   * @returns {Float64Array} out
   */
  writeQpos(out) {
    const { frames, nq, nframes, fps } = this;
    const dst = out || this._out;

    let x = this.time * fps;
    if (this.loop) {
      x %= nframes;
      if (x < 0) x += nframes;
    } else if (x > nframes - 1) {
      x = nframes - 1;
    }
    let i0 = Math.floor(x);
    let a = x - i0;
    let i1 = i0 + 1;
    if (i1 >= nframes) {
      if (this.loop) i1 = 0;
      else { i1 = nframes - 1; a = 0; }
    }

    const o0 = i0 * nq;
    const o1 = i1 * nq;
    for (let k = 0; k < nq; k++) {
      const v0 = frames[o0 + k];
      dst[k] = v0 + (frames[o1 + k] - v0) * a;
    }
    this._slerpQuat(dst, o0, o1, a);
    return dst;
  }

  /** 自由基座四元数（qpos[3..6]，wxyz）做最短路径 slerp + 归一化 */
  _slerpQuat(dst, o0, o1, a) {
    const A = this.frames;
    let w0 = A[o0 + 3], x0 = A[o0 + 4], y0 = A[o0 + 5], z0 = A[o0 + 6];
    let w1 = A[o1 + 3], x1 = A[o1 + 4], y1 = A[o1 + 5], z1 = A[o1 + 6];

    let dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1;
    if (dot < 0) { w1 = -w1; x1 = -x1; y1 = -y1; z1 = -z1; dot = -dot; }

    let s0;
    let s1;
    if (dot > 0.9995) {
      s0 = 1 - a;
      s1 = a;
    } else {
      const th = Math.acos(dot > 1 ? 1 : dot);
      const st = Math.sin(th);
      s0 = Math.sin((1 - a) * th) / st;
      s1 = Math.sin(a * th) / st;
    }

    let w = s0 * w0 + s1 * w1;
    let x = s0 * x0 + s1 * x1;
    let y = s0 * y0 + s1 * y1;
    let z = s0 * z0 + s1 * z1;
    const n = Math.hypot(w, x, y, z) || 1;
    dst[3] = w / n;
    dst[4] = x / n;
    dst[5] = y / n;
    dst[6] = z / n;
  }

  /** 供 HUD 显示 */
  describe() {
    return `${this.label} · ${this.kind === 'ref' ? '参考动作' : '策略仿真'}`
      + ` · ${this.time.toFixed(1)}/${this.duration.toFixed(1)} s`;
  }
}
