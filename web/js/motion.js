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
 * 采样：位置/关节用均匀 Catmull-Rom（C1 连续，见 writeQpos），基座四元数在相邻帧
 * 之间做最短路径插值（直接对四元数做线性插值会让旋转矩阵出现缩放，MuJoCo 的
 * mj_forward 不会帮你归一化，所以每次都要归一化）。`?interp=linear` 可退回旧的线性插值。
 *
 * meta.loop === false 时播完不会跳回第一帧，而是冻结在末帧 —— 单段「下蹲 / 起立」
 * 单段参考动作（下蹲 / 起立）用这个语义（和真机部署的 freeze_at_end 一致）；循环版不设该字段。
 */

/**
 * 插值模式。`?interp=linear` 保留旧的逐帧线性插值：
 * 真机上万一觉得 CR 让动作「发软」，或者怀疑曲线过冲把脚压进地面，可以一键回退 + A/B。
 */
const INTERP_MODE = (() => {
  try {
    return new URLSearchParams(location.search).get('interp') === 'linear' ? 'linear' : 'cr';
  } catch { return 'cr'; }
})();

/** 进程内已解码的回放数据：key = `${assetsBase}|${name}` */
const MOTION_CACHE = new Map();   // key -> { meta, frames, version }
/** 同名的 in-flight 请求：下拉连点 / 预取与点选撞车时只发一次 */
const MOTION_PENDING = new Map(); // key -> Promise<MotionPlayer>

/**
 * 预取顺序：默认片打头（点一下就能看），1.76 MB 的 fight_ref 压轴，
 * 免得大文件把小文件挤在后面。
 */
export const MOTION_PREFETCH = [
  'fight_policy', 'crawl', 'parade_seq', 'climb_policy', 'crab', 'squat_cycle',
  'policy_crab', 'goose_walk', 'longjump_hi', 'squat_down', 'squat_up',
  'goose_inplace', 'salute', 'fight_ref',
];

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
    this.loop = meta.loop !== false;   // 单段蹲起：播完冻结在末帧
    this.paused = false;
    this.interp = INTERP_MODE;         // 'cr' | 'linear'

    this._out = new Float64Array(this.nq);
  }

  /**
   * 从 assets/ 加载 motion_<name>.json + 同名 .bin。
   *
   * 同一份数据在会话内只下载一次：`MOTION_CACHE` 存已解码的帧，`MOTION_PENDING`
   * 让并发请求共享同一个 in-flight。`revalidate` 只重取 2 KB 的 json 比对 version，
   * 版本没变就直接复用已有的 Float32Array（重烘数据后刷新页面即可）。
   */
  static async load(assetsBase, name, { revalidate = false } = {}) {
    const key = `${assetsBase}|${name}`;
    const hit = MOTION_CACHE.get(key);
    if (hit && !revalidate) return new MotionPlayer(hit.meta, hit.frames);

    let p = MOTION_PENDING.get(key);
    if (!p) {
      p = MotionPlayer._fetch(assetsBase, name, key, hit)
        .finally(() => MOTION_PENDING.delete(key));
      MOTION_PENDING.set(key, p);
    }
    return p;
  }

  static async _fetch(assetsBase, name, key, prev) {
    const metaRes = await fetch(`${assetsBase}motion_${name}.json`);
    if (!metaRes.ok) throw new Error(`HTTP ${metaRes.status} — motion_${name}.json`);
    const meta = await metaRes.json();

    // 带版本号请求二进制：数据重新录制后 URL 会变，避免任何中间层/浏览器拿旧缓存
    const ver = meta.version || meta.generated || meta.nframes || '';

    // 版本没变（revalidate 路径）就不重下 bin
    let frames = (prev && prev.version === ver) ? prev.frames : null;
    if (!frames) {
      const binUrl = `${assetsBase}${meta.bin}${ver ? `?v=${encodeURIComponent(ver)}` : ''}`;
      const binRes = await fetch(binUrl);
      if (!binRes.ok) throw new Error(`HTTP ${binRes.status} — ${meta.bin}`);
      frames = new Float32Array(await binRes.arrayBuffer());
      const need = meta.nframes * meta.nq;
      if (frames.length !== need) {
        throw new Error(`${meta.bin} 长度不符：期望 ${need} 个 float32，实际 ${frames.length}`);
      }
    }
    MOTION_CACHE.set(key, { meta, frames, version: ver });
    return new MotionPlayer(meta, frames);
  }

  /**
   * 并发预取（默认 6 路，与 manifest 的 fetchConcurrency 一致）。
   * 任何一段失败都只记进返回值，不抛——预取是锦上添花，不能影响主流程。
   */
  static async prefetch(assetsBase, names = MOTION_PREFETCH, { concurrency = 6, onProgress } = {}) {
    const queue = names.slice();
    let done = 0;
    let failed = 0;
    const worker = async () => {
      for (;;) {
        const nm = queue.shift();
        if (!nm) return;
        try {
          await MotionPlayer.load(assetsBase, nm);
        } catch {
          failed++;
        }
        done++;
        onProgress?.(done, names.length, nm);
      }
    };
    const n = Math.max(1, Math.min(concurrency, queue.length));
    await Promise.all(new Array(n).fill(0).map(worker));
    return { done, failed, total: names.length };
  }

  /** 调试用：清空进程内缓存（重烘数据后不想刷新页面时） */
  static clearCache() {
    MOTION_CACHE.clear();
    MOTION_PENDING.clear();
  }

  static get cacheSize() {
    return MOTION_CACHE.size;
  }

  /** 播放进度 0..1（冻结末帧的回放停在 1，不能取模回 0） */
  get progress() {
    if (this.duration <= 0) return 0;
    const t = this.loop ? this.time % this.duration
      : Math.min(this.duration, Math.max(0, this.time));
    return t / this.duration;
  }

  seek(t) {
    const d = this.duration;
    if (!this.loop) {
      this.time = d > 0 ? Math.min(d, Math.max(0, t)) : 0;
      return this;
    }
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
      // 直接赋 time，别走 seek()：冻结模式下 time == duration 是合法终态
      this.time = this.duration;
      this.paused = true;
      return this;
    }
    this.seek(t);
    return this;
  }

  /**
   * 采样当前时刻的 qpos，写进 out（可直接传 MuJoCo 的 data.qpos 视图）。
   *
   * 默认走均匀 Catmull-Rom（τ=0.5）。线性插值的速度在每 20 ms 边界上是分段常数，
   * 而 72/90 Hz 与 50 Hz 不整除，折点会不断落在不同的显示帧上 → 节拍性的「顿一下」。
   * CR 让一阶导连续，同一份数据看起来明显更顺。
   *
   * 两个必须的护栏：
   *   1) a == 0 时逐位拷贝关键帧 —— 自检里有 `|x-3.152|<1e-3`、`|z-zEnd|<0.01` 这类
   *      端点断言，必须精确命中；
   *   2) 每个通道把结果钳在本段两端点之间 —— 循环接缝处（有几段数据标了 loop 但首尾
   *      差 2~10 m）不钳制会把「瞬移」摊成 5~6 帧每帧 10 cm 的滑行，比线性更糟。
   *      实测只有 0.4%~6% 的采样点会被钳到，99% 以上仍是 C1 连续。
   *
   * @returns {Float64Array} out
   */
  writeQpos(out) {
    const { frames, nq, nframes, fps, loop } = this;
    const dst = out || this._out;
    const last = nframes - 1;

    let x = this.time * fps;
    if (loop) {
      x %= nframes;
      if (x < 0) x += nframes;
    } else if (x < 0) x = 0;
    else if (x > last) x = last;

    let i1 = Math.floor(x);
    let a = x - Math.floor(x);
    if (i1 > last) { i1 = last; a = 0; }
    if (!loop && i1 >= last) { i1 = last; a = 0; }

    if (a <= 0) return this._copyFrame(dst, i1);

    const o1 = i1 * nq;
    // 循环取模；非循环两端钳位
    let i2 = i1 + 1;
    if (loop) { if (i2 >= nframes) i2 = 0; } else if (i2 > last) i2 = last;
    const o2 = i2 * nq;

    if (this.interp === 'linear') {
      for (let k = 0; k < nq; k++) {
        const v1 = frames[o1 + k];
        dst[k] = v1 + (frames[o2 + k] - v1) * a;
      }
      this._slerpQuat(dst, o1, o2, a);
      return dst;
    }

    let i0 = i1 - 1;
    if (loop) { if (i0 < 0) i0 += nframes; } else if (i0 < 0) i0 = 0;
    let i3 = i1 + 2;
    if (loop) { if (i3 >= nframes) i3 -= nframes; } else if (i3 > last) i3 = last;
    const o0 = i0 * nq;
    const o3 = i3 * nq;

    const a2 = a * a;
    const a3 = a2 * a;
    const h00 = 2 * a3 - 3 * a2 + 1;
    const h10 = a3 - 2 * a2 + a;
    const h01 = -2 * a3 + 3 * a2;
    const h11 = a3 - a2;

    for (let k = 0; k < nq; k++) {
      if (k >= 3 && k <= 6) continue;            // 四元数单独处理
      const v1 = frames[o1 + k];
      const v2 = frames[o2 + k];
      const v = h00 * v1 + h10 * 0.5 * (v2 - frames[o0 + k])
        + h01 * v2 + h11 * 0.5 * (frames[o3 + k] - v1);
      // 钳在本段两端点之间
      dst[k] = v1 < v2
        ? (v < v1 ? v1 : (v > v2 ? v2 : v))
        : (v < v2 ? v2 : (v > v1 ? v1 : v));
    }
    this._quatCR(dst, o0, o1, o2, o3, h00, h10, h01, h11);
    return dst;
  }

  /** a==0 的端点：逐位拷贝整帧（四元数仍归一化，mj_kinematics 不会替你归一化） */
  _copyFrame(dst, i) {
    const { frames, nq } = this;
    const o = i * nq;
    for (let k = 0; k < nq; k++) dst[k] = frames[o + k];
    if (nq >= 7) {
      const n = Math.hypot(dst[3], dst[4], dst[5], dst[6]) || 1;
      dst[3] /= n; dst[4] /= n; dst[5] /= n; dst[6] /= n;
    }
    return dst;
  }

  /**
   * 基座四元数（qpos[3..6]，wxyz）的三次插值。
   *
   * 不用 squad：那要算四元数的 log/exp（4~6 次四元数乘 + 三角函数），代码量和出错
   * 概率都高；50 Hz 下帧间角速度很小，「符号对齐 → 分量三次 → 归一化」与真 squad
   * 的差异 <1e-3 rad。符号对齐是必须的：q 与 -q 是同一姿态，不对齐会绕 360° 远路。
   */
  _quatCR(dst, o0, o1, o2, o3, h00, h10, h01, h11) {
    const A = this.frames;
    const w1 = A[o1 + 3], x1 = A[o1 + 4], y1 = A[o1 + 5], z1 = A[o1 + 6];
    const s0 = (A[o0 + 3] * w1 + A[o0 + 4] * x1 + A[o0 + 5] * y1 + A[o0 + 6] * z1) < 0 ? -1 : 1;
    const s2 = (A[o2 + 3] * w1 + A[o2 + 4] * x1 + A[o2 + 5] * y1 + A[o2 + 6] * z1) < 0 ? -1 : 1;
    const s3 = (A[o3 + 3] * w1 + A[o3 + 4] * x1 + A[o3 + 5] * y1 + A[o3 + 6] * z1) < 0 ? -1 : 1;
    const w0 = A[o0 + 3] * s0, x0 = A[o0 + 4] * s0, y0 = A[o0 + 5] * s0, z0 = A[o0 + 6] * s0;
    const w2 = A[o2 + 3] * s2, x2 = A[o2 + 4] * s2, y2 = A[o2 + 5] * s2, z2 = A[o2 + 6] * s2;
    const w3 = A[o3 + 3] * s3, x3 = A[o3 + 4] * s3, y3 = A[o3 + 5] * s3, z3 = A[o3 + 6] * s3;

    let w = h00 * w1 + h10 * 0.5 * (w2 - w0) + h01 * w2 + h11 * 0.5 * (w3 - w1);
    let x = h00 * x1 + h10 * 0.5 * (x2 - x0) + h01 * x2 + h11 * 0.5 * (x3 - x1);
    let y = h00 * y1 + h10 * 0.5 * (y2 - y0) + h01 * y2 + h11 * 0.5 * (y3 - y1);
    let z = h00 * z1 + h10 * 0.5 * (z2 - z0) + h01 * z2 + h11 * 0.5 * (z3 - z1);

    // 钳到两端点之间（接缝处防绕远路），再归一化
    w = w1 < w2 ? (w < w1 ? w1 : (w > w2 ? w2 : w)) : (w < w2 ? w2 : (w > w1 ? w1 : w));
    x = x1 < x2 ? (x < x1 ? x1 : (x > x2 ? x2 : x)) : (x < x2 ? x2 : (x > x1 ? x1 : x));
    y = y1 < y2 ? (y < y1 ? y1 : (y > y2 ? y2 : y)) : (y < y2 ? y2 : (y > y1 ? y1 : y));
    z = z1 < z2 ? (z < z1 ? z1 : (z > z2 ? z2 : z)) : (z < z2 ? z2 : (z > z1 ? z1 : z));

    const n = Math.hypot(w, x, y, z) || 1;
    dst[3] = w / n;
    dst[4] = x / n;
    dst[5] = y / n;
    dst[6] = z / n;
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
    // 面板里统一叫「参考动作」：参考动作 / 参考轨迹（蹲起）/ 脚本步态（螃蟹步）/
    // 脚本动作（正步）四类数据都是本机烘焙好的轨迹，只有 onnx 闭环录制的才叫策略仿真。
    const kindText = (this.kind === 'ref' || this.kind === 'squat'
      || this.kind === 'crab' || this.kind === 'parade' || this.kind === 'jump'
      || this.kind === 'crawl') ? '参考动作' : '策略仿真';
    return `${this.label} · ${kindText}`
      + ` · ${this.time.toFixed(1)}/${this.duration.toFixed(1)} s`;
  }
}
