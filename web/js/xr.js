/**
 * xr.js —— WebXR 入口按钮：沉浸式 VR + 半透明 VR（透视）
 *
 * 两个按钮的差别只在 XRSession 的 mode 与「背景怎么合成」：
 *
 *   进入 VR      immersive-vr  →  environmentBlendMode = 'opaque'
 *                虚拟世界把现实环境完全遮住，就是常规的 VR 观感。
 *
 *   半透明 VR    immersive-ar  →  environmentBlendMode = 'alpha-blend'
 *                three r170 的 WebGLBackground 在 alpha-blend 下会把清屏 alpha 置 0，
 *                我们再把 scene.background / scene.fog 摘掉、把地面网格调成半透明，
 *                于是「虚拟物体没遮住的地方」直接透出头显摄像头的真实画面
 *                （Quest 的 Passthrough / 透视），机器人像站在你自己的房间里。
 *                退出会话时全部还原，不影响原来的 VR / 桌面观感。
 *
 * 所以这个"半透明"不是把 canvas 调成半透明那么简单：
 *  - 渲染器必须带 alpha 通道（WebGLRenderer 构造时 alpha:true），XR 帧缓冲才有 alpha；
 *  - 背景不能是天空盒贴图（那会画满整个视野并写入 alpha=1），必须置 null；
 *  - 雾会把远处物体拉向雾色，透视下也不合适，一并关掉；
 *  - 地面网格若是不透明的，就仍是"虚拟地板盖住真实地板"，所以淡化到 0.3。
 *
 * 说明：桌面浏览器 / 不支持透视的头显上，两个按钮都会显示为不可用 ——
 * 这是设备能力限制（WebXR 的透视需要头显的彩色摄像头），不是页面问题。
 */

/** 两个入口模式的定义（顺序 = 按钮显示顺序） */
const MODES = [
  {
    id: 'vr',
    sessionMode: 'immersive-vr',
    label: '进入 VR',
    exitLabel: '退出 VR',
    unsupported: 'VR 不可用',
    hint: '沉浸式 VR：虚拟世界完全遮住现实环境',
    through: false,
    features: ['local-floor', 'bounded-floor'],
  },
  {
    id: 'ar',
    sessionMode: 'immersive-ar',
    label: '半透明 VR',
    exitLabel: '退出半透明',
    unsupported: '透视不可用',
    hint: '半透明 VR（透视）：背景透明，透过眼镜能看到真实环境',
    through: true,
    features: ['local-floor', 'bounded-floor', 'hand-tracking'],
  },
];

export class XRControls {
  /**
   * @param {object} o
   * @param {any}    o.renderer  WebGLRenderer（alpha:true，已开启 xr）
   * @param {any}    o.scene     three.js Scene
   * @param {any}    o.slot      按钮挂载点（<div id="vr-slot">）
   * @param {Function} [o.onEnter]        (id, info) => void  id: 'vr' | 'ar'
   * @param {Function} [o.onExit]         (id) => void
   * @param {Function} [o.onSeeThrough]   (on) => void  半透明合成开关（主程序用它淡化地面）
   * @param {Function} [o.toast]
   * @param {Function} [o.log]
   */
  constructor({ renderer, scene, slot, onEnter, onExit, onSeeThrough, toast, log } = {}) {
    this.renderer = renderer;
    this.scene = scene;
    this.slot = slot;
    this.onEnter = onEnter;
    this.onExit = onExit;
    this.onSeeThrough = onSeeThrough;
    this.toast = toast || (() => {});
    this.log = log || (() => {});

    this.modes = MODES.map((m) => m.id);
    this.entries = [];        // [{ def, btn }]
    this.supportedMap = {};   // id -> bool
    this.session = null;      // 当前 XRSession
    this.entry = null;        // 当前会话对应的 entry
    this.through = false;     // 是否处于"透视/半透明"合成
    this._saved = {};         // 进入透视前的背景/雾/清屏 alpha
  }

  // ------------------------------------------------------------------ 初始化
  async init() {
    if (!('xr' in navigator)) {
      this._fallbackLink();
      return this;
    }
    for (const def of MODES) this.entries.push(this._makeButton(def));
    for (const e of this.entries) {
      this.supportedMap[e.def.id] = await this._probe(e.def.sessionMode);
      this._paint();
    }
    this.log(`· WebXR 入口：VR ${this.supportedMap.vr ? '支持' : '不支持'}`
      + ` / 半透明透视 ${this.supportedMap.ar ? '支持' : '不支持'}`);
    return this;
  }

  async _probe(mode) {
    try {
      return (await navigator.xr.isSessionSupported(mode)) === true;
    } catch (e) {
      this.log(`· isSessionSupported('${mode}') 异常：${e?.message ?? e}`);
      return false;
    }
  }

  _makeButton(def) {
    const btn = document.createElement('button');
    btn.id = `xr-${def.id}`;
    btn.type = 'button';
    btn.className = `xr-btn xr-${def.id}`;
    btn.textContent = def.label;
    btn.title = def.hint;
    btn.disabled = true;
    btn.addEventListener('click', () => { this.toggle(def.id); });
    const entry = { def, btn };
    this.slot.appendChild(btn);
    return entry;
  }

  /** 完全没有 WebXR（或非安全上下文）时的提示链接 */
  _fallbackLink() {
    const a = document.createElement('a');
    if (window.isSecureContext === false) {
      a.href = location.href.replace(/^http:/, 'https:');
      a.textContent = 'WebXR 需要 HTTPS';
    } else {
      a.href = 'https://immersiveweb.dev/';
      a.target = '_blank';
      a.rel = 'noreferrer';
      a.textContent = '此浏览器不支持 WebXR';
    }
    this.slot.appendChild(a);
  }

  _paint() {
    for (const e of this.entries) {
      const sup = !!this.supportedMap[e.def.id];
      const active = !!this.session && this.entry === e;
      e.btn.disabled = !sup || (!!this.session && !active);
      e.btn.textContent = active ? e.def.exitLabel : (sup ? e.def.label : e.def.unsupported);
      e.btn.classList.toggle('active', active);
      e.btn.title = sup
        ? e.def.hint
        : `${e.def.hint}｜当前设备 / 浏览器不支持，请用带彩色摄像头的 WebXR 头显（如 Quest）打开`;
    }
    return this;
  }

  // ------------------------------------------------------------------ 状态
  /** 当前会话模式：null | 'vr' | 'ar' */
  get active() { return this.entry ? this.entry.def.id : null; }
  /** 渲染器上报的混合模式：opaque / alpha-blend / additive */
  get blendMode() { return this.session ? this.session.environmentBlendMode : null; }
  get supported() { return { ...this.supportedMap }; }

  capabilities() {
    return {
      hasXR: 'xr' in navigator,
      vr: !!this.supportedMap.vr,
      ar: !!this.supportedMap.ar,
      active: this.active,
      through: this.through,
      blendMode: this.blendMode,
    };
  }

  // ------------------------------------------------------------------ 进入 / 退出
  async toggle(id) {
    const entry = this.entries.find((e) => e.def.id === id);
    if (!entry || entry.btn.disabled) return null;

    if (this.session) {                       // 已在会话里 → 点它即退出
      if (this.entry !== entry) return null;
      try { await this.session.end(); } catch (e) { this.log(`· 结束 XR 会话失败：${e?.message ?? e}`); }
      return null;
    }

    if (!this.supportedMap[id]) {
      this.toast(`${entry.def.hint} —— 当前设备 / 浏览器不支持`, 4200);
      return null;
    }
    return this.enter(entry);
  }

  /** @param {{def: object, btn: HTMLElement}} entry */
  async enter(entry) {
    const def = entry.def;

    let session;
    try {
      // 注意：这里必须保持"用户手势"上下文 —— 前面不能有 await
      session = await navigator.xr.requestSession(def.sessionMode, {
        optionalFeatures: def.features,
      });
    } catch (err) {
      this.log(`· requestSession('${def.sessionMode}') 失败：${err?.name ?? ''} ${err?.message ?? err}`);
      this.toast(`无法进入「${def.label}」：${err?.message || err?.name || err}`, 4600);
      return null;
    }

    // 参考空间：优先 local-floor（用户脚踩 y=0，正好与 MuJoCo 地面对齐）。
    // 少数设备（手机 ARCore 之类）只有 local，先自己探一下再交给 three，
    // 免得 three 内部 requestReferenceSpace 抛错后会话变黑。
    let refType = 'local-floor';
    try {
      await session.requestReferenceSpace(refType);
    } catch {
      refType = 'local';
    }
    this.renderer.xr.setReferenceSpaceType(refType);

    this.session = session;
    this.entry = entry;
    session.addEventListener('end', () => this._onSessionEnd(), { once: true });

    try {
      await this.renderer.xr.setSession(session);
    } catch (err) {
      this.session = null;
      this.entry = null;
      try { session.end(); } catch { /* ignore */ }
      this.log(`· renderer.xr.setSession 失败：${err?.message ?? err}`);
      this.toast('XR 会话初始化失败，详见浏览器控制台', 4200);
      this._paint();
      return null;
    }

    // 半透明模式：摘掉背景和雾 → 没被虚拟物体遮住的地方露出真实环境。
    // 保险起见按设备上报的混合模式决定：只有 alpha-blend / additive 才真的透明，
    // 万一某设备把 AR 会话上报成 opaque，保持不透明渲染，避免用户看到一片黑。
    const blend = session.environmentBlendMode;
    this.setSeeThrough(def.through && (blend === 'alpha-blend' || blend === 'additive'));
    if (def.through && !this.through) {
      this.log(`· 注意：设备把 ${def.sessionMode} 上报为 environmentBlendMode=${blend}，保持不透明渲染`);
    }
    this._paint();

    this.log(`· 进入 ${def.sessionMode}：参考空间 ${refType}，environmentBlendMode=${blend}`
      + `，清屏 alpha=${this.renderer.getClearAlpha().toFixed(2)}`);
    this.onEnter?.(def.id, { through: this.through, blend, refType });
    return session;
  }

  _onSessionEnd() {
    const id = this.entry ? this.entry.def.id : null;
    this.session = null;
    this.entry = null;
    this.setSeeThrough(false);
    this._paint();
    // three 自己的 'end' 监听会在这之后才把 isPresenting 置回 false，
    // 放到微任务里保证主程序恢复视角时已经不处于 XR 状态。
    Promise.resolve().then(() => this.onExit?.(id));
  }

  // ------------------------------------------------------------------ 半透明合成
  /**
   * 开关「透视 / 半透明」合成（幂等，可直接调用，不依赖真头显 → 便于自检）。
   *   on = true ：背景置空 + 关雾 + 清屏 alpha=0 → 头显里能看到真实环境
   *   on = false：原样还原（背景贴图、雾、清屏 alpha 都是进入前的值）
   * @param {boolean} on
   */
  setSeeThrough(on) {
    on = !!on;
    if (on === this.through) return this;
    const s = this.scene;

    if (on) {
      this._saved = {
        background: s.background,
        fog: s.fog,
        clearAlpha: this.renderer.getClearAlpha(),
      };
      s.background = null;   // 天空盒会写满整个视野且 alpha=1 → 必须摘掉
      s.fog = null;          // 雾让远处物体偏雾色，透视下不需要
      this.renderer.setClearAlpha(0);
    } else if (Object.keys(this._saved).length) {
      const sv = this._saved;
      s.background = sv.background;
      s.fog = sv.fog;
      this.renderer.setClearAlpha(sv.clearAlpha ?? 1);
      this._saved = {};
    }

    this.through = on;
    this.onSeeThrough?.(on);
    return this;
  }

  /** 主动退出当前会话（等价于点「退出」按钮） */
  async exit() {
    if (!this.session) return;
    try { await this.session.end(); } catch { /* ignore */ }
  }
}
