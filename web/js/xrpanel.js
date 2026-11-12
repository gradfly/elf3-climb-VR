/**
 * xrpanel.js —— 头显内可操作的设置面板
 *
 * 为什么需要这一层
 * ---------------
 * WebXR 沉浸会话里浏览器只合成 WebGL 画布，**DOM 不会进到虚拟画面**（Quest 浏览器
 * 也不支持 dom-overlay）。所以「在头显里点复选框」只有一个办法：把面板画进场景——
 * 用 2D canvas 画出界面 → CanvasTexture 贴到一个平面上 → 用手柄射线打到平面上，
 * 把命中点换算回画布像素，再改成对应的控件。
 *
 * 不去复制业务逻辑
 * ---------------
 * 页面上的状态真值全在 DOM 控件里（`opt-*` 的 checked / value，按钮的 click），
 * main.js 的 setupUI() 只监听这些事件。于是这里做两件事就够了：
 *
 *   读：LAYOUT 白名单列出头显里要显示的控件（最小两栏集合），label / option /
 *       checked / value / 按钮文案一律实时读 #panel 的 DOM，桌面那份完整面板不删控件。
 *   写：点一下 → 改 checked / value → dispatchEvent(new Event('change'))，
 *       按钮则直接 .click()。切换场景、地形、回放机位、toast 全部照旧生效。
 *
 * 每 1/UI.pollHz 秒采一次 DOM 签名，变了才重画 → 桌面端改开关，头显里跟着变。
 * 悬停射线按 UI.hoverHz 跑，重绘按 UI.rebuildMs 合并；头锁每帧只改 group 变换，
 * 不触发重绘。
 *
 * 版式与摆放
 * ----------
 * 宽扁条（1024×~290）贴到 0.80 m 宽的平面上，默认整块 50% 半透明（标题栏可切
 * 100%）。头锁每帧摆在眼睛前方 UI.dist、上方 UI.raiseY，并朝眼睛下倾；「归位」
 * 在「跟随头部」与「固定在世界」之间切换。回放下拉 11 项放不进短画布，打开时
 * rebuild() 临时撑高画布并按顶边锚定，关闭后缩回。
 *
 * 交互
 * ----
 *   扳机 / 侧键 = 点击（手势追踪没有扳机时也走 squeeze）
 *   左摇杆按下  = 收起 / 展开
 *   标题栏四个按钮：50%/100% 透明度 / 归位 / 退出（结束 XR 会话）/ 收起
 *
 * 无头自检
 * --------
 * 非 XR 时用桌面 camera 代替头显摆位，于是 show()/tap()/exportPNG() 全部可在
 * verify.mjs 里跑，真头显只用来微调 UI.worldW / dist / raiseY / font 这几个常量。
 */

const FONT = '"Noto Sans SC", "Source Han Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif';

/** 版式与摆放参数（要按头显观感微调时只改这里） */
const UI = {
  pxW: 1024,        // 画布宽（固定），高度按内容算
  pad: 22,
  gap: 44,          // 两栏间距
  colW: 468,        // (1024 - 2*22 - 44) / 2
  rowH: 52,         // 行距
  chipH: 44,        // 按钮/单选条高度
  headH: 54,        // 分组标题行高
  blockGap: 10,     // 分组之间的空隙
  titleH: 66,       // 标题栏
  statusH: 0,       // 底部状态条已删除（不再占位）
  font: 31,
  headFont: 24,
  worldW: 0.80,     // 平面宽度（米）→ 文字张角 ≈ 1.2°，头显里够读
  dist: 1.5,       // 摆放距离（水平，头锁后每帧重算）
  raiseY: 0.5,     // 面板中心比眼睛高多少 → 默认置于视野上方（真机微调）
  tabW: 0.30,       // 收起后的小签宽度
  pollHz: 3,        // DOM 状态轮询（桌面改开关 → 头显面板跟上的延迟 = 1/pollHz）
  hoverHz: 30,      // 悬停射线检测频率（每帧做两次 Raycaster 没必要）
  rebuildMs: 120,   // 重绘合并窗口：一次重绘 = 全画布 2D 重画 + 最多 3.2 MB 纹理上传 + mip 链
  tapCooldown: 240, // 点击防抖（ms）
  flashMs: 200,
  statusMs: 1800,
  showStatus: false, // 精简面板不显示任何反馈文字
};

const C = {
  bg: '#10161f',
  line: 'rgba(120,160,200,0.30)',
  lineSoft: 'rgba(120,160,200,0.14)',
  fg: '#dce6f2',
  dim: '#8fa2b8',
  accent: '#54d1ff',
  ink: '#06202e',
  chip: 'rgba(255,255,255,0.06)',
  on: 'rgba(84,209,255,0.16)',
  onLine: 'rgba(84,209,255,0.55)',
  hover: 'rgba(84,209,255,0.10)',
  hoverLine: 'rgba(84,209,255,0.45)',
};

/**
 * 头显面板的最小控件集：显式两栏白名单，精确对应设计稿。
 * 只列 DOM id / data-view / 分组标题；label、option、checked、value 一律实时读
 * #panel 的 DOM，桌面那份面板保持完整，文案也不会出现两份不同步。
 */
const LAYOUT = [
  { title: '场景', col: 0, widgets: [
    { kind: 'radio', sel: '.seg' },
    { kind: 'select', id: 'opt-motion' },
    { kind: 'btnrow', ids: ['btn-motion-restart', 'btn-motion-pause'] },
  ] },
  { title: '视角', col: 1, widgets: [
    { kind: 'btnrow', views: ['front', 'back', 'side', 'squat', 'grid', 'wide'] },
    { kind: 'check', id: 'opt-follow' },
  ] },
];

export class XRPanel {
  /**
   * @param {object} o
   * @param {any} o.THREE
   * @param {any} o.renderer  WebGLRenderer（已开启 xr）
   * @param {any} o.scene     three.js Scene（面板挂在 scene，不挂 stage）
   * @param {any} o.camera    桌面相机（非 XR 时当作"头"）
   * @param {any} o.vrNav     VRNav（取 controllers 做射线）
   * @param {HTMLElement} o.panelEl  #panel
   * @param {any} [o.xr]      XRControls（标题栏「退出」用）
   */
  constructor({ THREE, renderer, scene, camera, vrNav, panelEl, xr } = {}) {
    this.THREE = THREE;
    this.renderer = renderer;
    this.scene = scene;
    this.camera = camera;
    this.vrNav = vrNav;
    this.panelEl = panelEl;
    this.xr = xr;

    this.visible = false;
    this.collapsed = false;
    this._groups = [];
    this._hits = [];
    this._drawList = [];
    this._overlay = null;
    this._hoverKey = null;
    this._flashKey = null;
    this._flashUntil = 0;
    this._statusText = '';
    this._statusUntil = 0;
    this._sig = '';
    this._hoverAcc = 0;        // 悬停射线节流
    this._rebuildTimer = null; // 重绘合并定时器
    this._rebuildCount = 0;    // 自检用：累计重绘次数
    this._poll = 0;
    this._lastTap = 0;
    this._stickLatch = false;
    this._placeDelay = 0;
    this._presenting = false;
    this._opacity = 0.5;       // 整个贴图面板默认 50% 半透明
    this._headLocked = true;   // 默认每帧跟随头部，固定置于视野上方

    this._vP = new THREE.Vector3();
    this._vF = new THREE.Vector3();
    this._vC = new THREE.Vector3();
    this._raycaster = new THREE.Raycaster();
    this._raycaster.far = 3.2;
    this._ray = new THREE.Ray();
  }

  // ================================================================ 初始化
  init() {
    const { THREE, scene } = this;

    this._group = new THREE.Group();
    this._group.name = 'xr-panel';
    this._group.visible = false;
    scene.add(this._group);

    // —— 主面板 ——
    this._canvas = document.createElement('canvas');
    this._canvas.width = UI.pxW;
    this._canvas.height = 800;
    this._ctx = this._canvas.getContext('2d');
    this._tex = null;                 // 由 _makeTexture() 建（画布尺寸变了必须重建，见下）
    this._anisotropy = 1;
    // 各向异性只给 2：面板挂在 1.15 m、宽 0.80 m，张角约 30°，屏幕投影对画布是
    // 1.57:1 的**缩小**采样，8× 各向异性在这么小的倾斜下没有可见收益，却要每片元
    // 采 8 次（1024×782 × 双目）。
    try { this._anisotropy = Math.min(2, this.renderer.capabilities.getMaxAnisotropy()); } catch { /* ignore */ }

    this._mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: null, toneMapped: false, side: THREE.FrontSide, fog: false,
        transparent: true, opacity: this._opacity, depthWrite: false,
      }),
    );
    this._makeTexture();              // 建纹理并挂到材质上
    this._mesh.name = 'xr-panel-quad';
    this._group.add(this._mesh);

    // —— 收起后的小签 ——
    this._tabCanvas = document.createElement('canvas');
    this._tabCanvas.width = 384;
    this._tabCanvas.height = 160;
    this._tabCtx = this._tabCanvas.getContext('2d');
    this._tabTex = new THREE.CanvasTexture(this._tabCanvas);
    this._tabTex.colorSpace = THREE.SRGBColorSpace;
    this._tabMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        map: this._tabTex, toneMapped: false, side: THREE.FrontSide, fog: false,
        transparent: true, opacity: this._opacity, depthWrite: false,
      }),
    );
    this._tabMesh.name = 'xr-panel-tab';
    this._tabMesh.visible = false;
    this._group.add(this._tabMesh);
    this._drawTab();

    // —— 手柄：扳机点击、侧键兼作点击 ——
    for (const ctrl of (this.vrNav?.controllers || [])) {
      ctrl.addEventListener('selectstart', () => this._activate(ctrl));
      ctrl.addEventListener('squeezestart', () => this._activate(ctrl));
    }

    this.rebuild();
    return this;
  }

  /**
   * 建（或重建）主面板纹理。
   *
   * ⚠️ 画布尺寸一变**必须换一张新的 CanvasTexture**，不能只靠 `needsUpdate = true`：
   * three r170 走 `texStorage2D()` 分配的是**不可变**的纹理存储，尺寸锁死在第一次上传时。
   * 下拉覆盖层把画布从 1024×290 撑到 1024×782 后，再往旧存储里传 782 高的数据，
   * GL 直接报 INVALID_VALUE，纹理里还是那张 290 高的旧图 → 平面上看到的是「整块被拉伸、
   * 下拉列表完全看不见」。so 修法：dispose 旧纹理（连同 GL 对象一起释放）
   * + new CanvasTexture + 换掉材质的 map，让 three 重新走一遍 initTexture。
   */
  _makeTexture() {
    const { THREE } = this;
    if (this._tex) {
      this._tex.dispose();
      this._tex = null;
    }
    const tex = new THREE.CanvasTexture(this._canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.generateMipmaps = true;
    tex.minFilter = THREE.LinearMipmapLinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.anisotropy = this._anisotropy || 1;
    this._tex = tex;
    if (this._mesh && this._mesh.material) {
      this._mesh.material.map = tex;
      this._mesh.material.needsUpdate = true;
    }
    return tex;
  }

  // ================================================================ 显隐
  /** 显示：在用户当前视线上方重新摆位（半透明 VR / 沉浸 VR 都走这里） */
  show() {
    const was = this.visible;
    this.visible = true;
    this.collapsed = false;
    this._headLocked = true;   // 每次进会话默认头锁置顶；透明度保留会话内的选择
    this._group.visible = true;
    // 先不显示，等 _place() 摆好再打开，避免头显里头两帧闪现在世界原点
    this._mesh.visible = false;
    this._tabMesh.visible = false;
    this._placeDelay = this.renderer.xr.isPresenting ? 2 : 0;
    if (!this._placeDelay) this._place();
    if (!was) this.rebuild();
    return this;
  }

  hide() {
    this.visible = false;
    this.collapsed = false;
    this._overlay = null;
    this._hoverKey = null;
    this._flashKey = null;
    this._statusText = '';
    clearTimeout(this._rebuildTimer);
    this._rebuildTimer = null;
    if (this._group) this._group.visible = false;
    return this;
  }

  toggle() { return this.visible ? this.hide() : this.show(); }

  /** 收起成小签（位置留在原处，点了会在当前视线前重新展开） */
  collapse() {
    if (!this.visible) return this;
    this.collapsed = true;
    this._overlay = null;
    this._statusText = '';       // 小签上没有状态条，别留到展开后
    this._mesh.visible = false;
    this._tabMesh.visible = true;
    return this;
  }

  expand() {
    if (!this.visible) return this.show();
    this.collapsed = false;
    this._overlay = null;
    this._mesh.visible = false;
    this._tabMesh.visible = false;
    this._placeDelay = this.renderer.xr.isPresenting ? 1 : 0;
    if (!this._placeDelay) this._place();
    this.rebuild();
    return this;
  }

  // ================================================================ 每帧
  update(dt) {
    if (!this.visible || !this._group) return;

    // 会话建立的那一刻渲染器可能还没把 isPresenting 置真（onEnter 的时序不保证），
    // 于是这里补一次：一变成 XR 就按头显视线重新摆位，别留在桌面相机前方。
    const presenting = this.renderer.xr.isPresenting;
    if (presenting !== this._presenting) {
      this._presenting = presenting;
      if (presenting) {
        this._placeDelay = 2;
        this._mesh.visible = false;
      }
    }

    if (this._placeDelay > 0) {
      this._placeDelay--;
      if (this._placeDelay === 0) this._place();
    } else if (this._headLocked) {
      // 头锁：每帧按头部姿态重算，折叠态也跑（小签始终留在视野上方）。
      // 必须排在 _pollStick / _pollHover 之前：_place() 内 updateMatrixWorld(true)
      // 刷新子网格矩阵，30 Hz 的悬停射线才不会用到上一帧位姿。
      // 只改 group 变换，绝不调 rebuild() —— 否则 _rebuildCount 会暴涨。
      this._place();
    }

    // 摇杆按下 = 收起 / 展开（给"走远了够不到小签"留一条退路）
    this._pollStick();

    // 悬停射线按 hoverHz 跑，而不是每帧 —— 一体机上每次 Raycaster + 命中判定
    // 都是 XR 帧预算里的固定开销
    this._hoverAcc = (this._hoverAcc || 0) + dt;
    if (this._hoverAcc >= 1 / UI.hoverHz) {
      this._hoverAcc = 0;
      this._pollHover();
    }

    this._poll += dt;
    if (this._poll >= 1 / UI.pollHz) {
      this._poll = 0;
      const sig = this._signature();
      if (sig !== this._sig) this._scheduleRebuild(UI.rebuildMs);
    }

    const now = performance.now();
    if (this._flashKey && now > this._flashUntil) { this._flashKey = null; this.rebuild(); }
    if (this._statusText && now > this._statusUntil) { this._statusText = ''; this.rebuild(); }
  }

  /**
   * 合并窗口内的重绘请求。
   *
   * rebuild() 是从 XR 的 rAF 回调里同步执行的（frame() → xrPanel.update()）：
   * 一次 = 若干次 querySelectorAll + 全画布 1024×~800 的 fillText + 3.2 MB 纹理上传
   * + mipmap 生成。手柄扫过面板时悬停项每秒变十几次，直接同步重绘会把主线程拖成长任务，
   * 头显里表现为"手柄都迟钝"。这里把窗口内的多次请求压成一次。
   */
  _scheduleRebuild(ms = UI.rebuildMs) {
    if (this._rebuildTimer) return;
    if (ms <= 0) { this._rebuildTimer = null; this.rebuild(); return; }
    this._rebuildTimer = setTimeout(() => {
      this._rebuildTimer = null;
      if (this.visible) this.rebuild();
    }, ms);
    return this;
  }

  /** 轮询左右摇杆按下 → 收起/展开（VRNav 只用 axes 和 A/B/X/Y，不冲突） */
  _pollStick() {
    const session = this.renderer.xr.getSession ? this.renderer.xr.getSession() : null;
    if (!session) return;
    let pressed = false;
    for (const src of session.inputSources) {
      const b = src.gamepad && src.gamepad.buttons && src.gamepad.buttons[3];
      if (b && b.pressed) pressed = true;
    }
    if (pressed && !this._stickLatch) {
      this._stickLatch = true;
      if (this.collapsed) this.expand(); else this.collapse();
    } else if (!pressed) {
      this._stickLatch = false;
    }
  }

  /** 每个控制器各算一次命中，悬停项变了才重画 */
  _pollHover() {
    const presenting = this.renderer.xr.isPresenting;
    let hover = null;
    let changed = false;
    for (const ctrl of (this.vrNav?.controllers || [])) {
      const w = presenting ? this._pick(ctrl) : null;
      const prev = ctrl.userData.__xrHover || null;
      const key = w ? w.key : null;
      const prevKey = prev ? prev.key : null;
      if (key !== prevKey) changed = true;
      ctrl.userData.__xrHover = w;
      if (key) hover = key;
    }
    if (hover !== this._hoverKey) changed = true;
    this._hoverKey = hover;
    if (changed) this._scheduleRebuild(UI.rebuildMs);
  }

  // ================================================================ 摆位
  /** 头显（或桌面相机）的世界位置与朝向 */
  _head(P, F) {
    const cam = this.renderer.xr.isPresenting ? this.renderer.xr.getCamera() : this.camera;
    cam.getWorldPosition(P);
    cam.getWorldDirection(F);
  }

  /** 头锁：摆在眼睛前方 UI.dist 米、比眼睛高 UI.raiseY，平面朝眼睛下倾 */
  _place() {
    const P = this._vP, F = this._vF, c = this._vC;
    this._head(P, F);
    F.y = 0;
    if (F.lengthSq() < 1e-8) F.set(0, 0, -1);
    F.normalize();
    c.copy(P).addScaledVector(F, UI.dist);   // 水平距离恒 UI.dist
    c.y = P.y + UI.raiseY;                   // 视野上方

    this._group.position.copy(c);
    // 目标点用眼睛真实高度：three 的非相机 lookAt 让 +Z 指向目标，
    // PlaneGeometry 正面即 +Z，面板因高于眼睛而朝眼下倾。
    this._group.lookAt(P.x, P.y, P.z);
    this._group.updateMatrixWorld(true);
    if (!this.collapsed) this._mesh.visible = true;
    return this;
  }

  // ================================================================ 命中
  /** 手柄射线 → 平面 → 画布像素 → 控件 */
  _pick(ctrl) {
    const target = this.collapsed ? this._tabMesh : this._mesh;
    if (!target || !target.visible) return null;

    this._ray.origin.setFromMatrixPosition(ctrl.matrixWorld);
    this._ray.direction.set(0, 0, -1).transformDirection(ctrl.matrixWorld);
    this._raycaster.ray.copy(this._ray);

    const hits = this._raycaster.intersectObject(target, false);
    if (!hits.length || !hits[0].uv) return null;

    if (this.collapsed) return { key: 'panel:expand', label: '展开设置' };

    const px = hits[0].uv.x * this._canvas.width;
    const py = (1 - hits[0].uv.y) * this._canvas.height;
    return this._hitAt(px, py);
  }

  _hitAt(px, py) {
    for (let i = this._hits.length - 1; i >= 0; i--) {
      const h = this._hits[i];
      const r = h.rect;
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return h;
    }
    return null;
  }

  _activate(ctrl) {
    const now = performance.now();
    if (now - this._lastTap < UI.tapCooldown) return;
    const w = ctrl.userData.__xrHover || null;
    if (!w) {
      if (this._overlay) { this._overlay = null; this.rebuild(); }
      return;
    }
    this._lastTap = now;
    this._haptic(ctrl);
    this.tap(w.key);
  }

  _haptic(ctrl) {
    try {
      const src = ctrl.userData.inputSource;
      const act = src && src.gamepad && src.gamepad.hapticActuators && src.gamepad.hapticActuators[0];
      if (act && act.pulse) act.pulse(0.35, 30);
    } catch { /* 没有触觉反馈就算了 */ }
  }

  // ================================================================ 动作
  /**
   * 直接触发某个控件（射线点击与自检共用）。
   * key 形如 check:opt-follow / radio:opt-scene:single / select:opt-motion /
   *        option:fight_policy / btn:btn-motion-pause / btn:view:front /
   *        panel:collapse / panel:opacity / panel:recenter
   */
  tap(key) {
    if (!key) return false;
    const parts = String(key).split(':');
    const kind = parts[0];

    if (kind === 'overlay') { this._overlay = null; this.rebuild(); return true; }

    if (kind === 'option') {
      const ov = this._overlay;
      const value = parts.slice(1).join(':');
      const sel = ov && document.getElementById(ov.id);
      if (!sel) { this._overlay = null; this.rebuild(); return false; }
      const label = (ov.items.find((i) => i.value === value) || {}).label || value;
      this._overlay = null;
      if (sel.value !== value) {
        sel.value = value;
        sel.dispatchEvent(new Event('change'));
      }
      this._flash('select:' + sel.id);
      this._status(`${ov.label}：${label}`);
      this.rebuild();
      return true;
    }

    if (kind === 'panel') {
      const act = parts[1];
      if (act === 'opacity') {
        this._opacity = this._opacity < 1 ? 1.0 : 0.5;
        this._mesh.material.opacity = this._opacity;
        this._tabMesh.material.opacity = this._opacity;
        this.rebuild();   // 标题 chip 的 50% / 100% 文案要跟着变
        return true;
      }
      if (act === 'recenter') {
        // 头锁下「归位」= 在「跟随头部」与「固定在世界」之间切换；
        // 切回跟随时立即摆回当前视野上方。
        this._headLocked = !this._headLocked;
        if (this._headLocked) this._place();
        this._status(this._headLocked ? '面板跟随头部' : '面板固定在世界');
        this.rebuild();
        return true;
      }
      if (act === 'exit') { this.xr?.exit?.(); return true; }
      if (act === 'collapse') { this.collapse(); return true; }
      if (act === 'expand') { this.expand(); return true; }
      return false;
    }

    if (kind === 'check') {
      const el = document.getElementById(parts[1]);
      if (!el) return false;
      el.checked = !el.checked;
      el.dispatchEvent(new Event('change'));
      const w = this._find('check:' + parts[1]);
      this._flash('check:' + parts[1]);
      this._status(`${w ? w.label : parts[1]}：${el.checked ? '开' : '关'}`);
      this.rebuild();
      return true;
    }

    if (kind === 'radio') {
      const name = parts[1];
      const value = parts.slice(2).join(':');
      const inputs = [...document.querySelectorAll(`input[name="${name}"]`)];
      const el = inputs.find((i) => i.value === value);
      if (!el) return false;
      if (!el.checked) {
        el.checked = true;
        el.dispatchEvent(new Event('change'));
      }
      const w = this._find('radio:' + name + ':' + value);
      this._flash('radio:' + name + ':' + value);
      this._status(`${w ? w.label : value}`);
      this.rebuild();
      return true;
    }

    if (kind === 'select') {
      const hit = this._find('select:' + parts[1]);
      const spec = hit && (hit.spec || hit);
      if (!spec || !Array.isArray(spec.options)) return false;
      this._openOverlay(spec);
      return true;
    }

    if (kind === 'btn') {
      if (parts[1] === 'view') {
        const view = parts.slice(2).join(':');
        const el = document.querySelector(`#views button[data-view="${view}"]`);
        if (!el) return false;
        el.click();
        this._flash('btn:view:' + view);
        this._status(el.textContent.trim());
      } else {
        const el = document.getElementById(parts[1]);
        if (!el) return false;
        el.click();
        this._flash('btn:' + parts[1]);
        this._status(el.textContent.trim());
      }
      this.rebuild();
      return true;
    }
    return false;
  }

  _find(key) {
    for (const h of this._hits) if (h.key === key) return h;
    return null;
  }

  _flash(key) {
    this._flashKey = key;
    this._flashUntil = performance.now() + UI.flashMs;
  }

  _status(text) {
    if (!UI.showStatus) return;
    this._statusText = text;
    this._statusUntil = performance.now() + UI.statusMs;
  }

  _openOverlay(w) {
    const items = w.options.map((o) => ({
      key: `option:${o.value}`,
      value: o.value,
      label: o.label,
      on: o.value === w.value,
      rect: { x: 0, y: 0, w: 0, h: 0 },
    }));
    this._overlay = { id: w.id, label: w.label, items };
    this.rebuild();
    return this._overlay;
  }

  // ================================================================ DOM → 控件清单
  /**
   * 头显面板只投影 LAYOUT 白名单里的控件（最小集合）。
   * 分组标题、控件标签、下拉选项、选中态全部实时读 #panel 的 DOM，
   * 桌面那份完整面板一个字都不用改。
   */
  _buildGroups() {
    const body = this.panelEl && this.panelEl.querySelector('.body');
    if (!body) return [];
    const h2s = [...body.querySelectorAll('h2')];

    // 先索引所有按钮：id 型（运动 / 物理按钮）与 data-view 型（视角按钮）
    const buttons = new Map();
    for (const b of body.querySelectorAll('.btns button')) {
      buttons.set(b.dataset.view ? `view:${b.dataset.view}` : `id:${b.id}`, b);
    }

    return LAYOUT.map((spec) => {
      const h2 = h2s.find((e) => e.textContent.trim() === spec.title);
      const items = [];
      for (const w of spec.widgets) {
        if (w.kind === 'radio') {
          const el = body.querySelector(w.sel || '.seg');
          const inputs = el ? [...el.querySelectorAll('input[type=radio]')] : [];
          if (!inputs.length) continue;
          items.push({
            kind: 'radio',
            name: inputs[0].name,
            items: inputs.map((inp) => ({
              value: inp.value,
              label: (inp.closest('label')?.querySelector('span')?.textContent || inp.value).trim(),
              on: inp.checked,
            })),
          });
        } else if (w.kind === 'select') {
          const el = document.getElementById(w.id);
          if (!el) continue;
          items.push({
            kind: 'select', id: el.id,
            label: el.closest('.rowsel')?.querySelector('span')?.textContent?.trim() || w.id,
            value: el.value,
            options: [...el.options].map((o) => ({ value: o.value, label: o.textContent.trim() })),
          });
        } else if (w.kind === 'check') {
          const el = document.getElementById(w.id);
          if (!el) continue;
          items.push({
            kind: 'check', id: el.id,
            label: el.closest('.chk')?.querySelector('span')?.textContent?.trim() || w.id,
            checked: el.checked,
          });
        } else if (w.kind === 'btnrow') {
          const keys = w.ids ? w.ids.map((id) => `id:${id}`)
            : w.views.map((v) => `view:${v}`);
          const its = keys.map((k) => buttons.get(k)).filter(Boolean).map((b) => ({
            id: b.id || null,
            view: b.dataset.view || null,
            label: b.textContent.trim(),
          }));
          if (its.length) items.push({ kind: 'btnrow', items: its });
        }
      }
      return { title: h2 ? h2.textContent.trim() : spec.title, col: spec.col, items };
    }).filter((g) => g.items.length);
  }

  // ================================================================ 版式
  _perRow(n) { return n <= 2 ? Math.max(1, n) : 3; }

  _layout() {
    const groups = this._groups;
    const cols = [[], []];
    // 设计稿是固定两栏：左「场景」右「视角」，不做贪心均衡
    for (const g of groups) cols[g.col ?? 0].push(g);

    const x0 = [UI.pad, UI.pad + UI.colW + UI.gap];
    const top = UI.titleH + 4;
    this._hits = [];
    this._drawList = [];
    let bottom = top;

    for (let c = 0; c < 2; c++) {
      let y = top;
      for (const g of cols[c]) {
        this._drawList.push({
          kind: 'h', text: g.title, rect: { x: x0[c], y, w: UI.colW, h: UI.headH },
        });
        y += UI.headH;

        for (const w of g.items) {
          if (w.kind === 'btnrow') {
            const n = w.items.length;
            const per = this._perRow(n);
            const rows = Math.ceil(n / per);
            const cw = (UI.colW - (per - 1) * 8) / per;
            w.rect = { x: x0[c], y, w: UI.colW, h: rows * UI.rowH - 8 };
            for (let i = 0; i < n; i++) {
              const r = Math.floor(i / per), k = i % per;
              const it = w.items[i];
              it.rect = {
                x: x0[c] + k * (cw + 8),
                y: y + r * UI.rowH,
                w: cw,
                h: UI.chipH,
              };
              it.key = it.view ? `btn:view:${it.view}` : `btn:${it.id}`;
              this._hits.push({ key: it.key, label: it.label, kind: 'btn', rect: it.rect });
            }
            this._drawList.push(w);
            y += rows * UI.rowH;
            continue;
          }

          w.rect = { x: x0[c], y, w: UI.colW, h: UI.rowH };
          if (w.kind === 'radio') {
            const n = w.items.length;
            const gap = 10;
            const cw = (UI.colW - (n - 1) * gap) / n;
            for (let i = 0; i < n; i++) {
              const it = w.items[i];
              it.rect = { x: x0[c] + i * (cw + gap), y: y + 4, w: cw, h: UI.chipH };
              it.key = `radio:${w.name}:${it.value}`;
              this._hits.push({ key: it.key, label: it.label, kind: 'radio', rect: it.rect });
            }
          } else if (w.kind === 'select') {
            w.key = `select:${w.id}`;
            // spec 指回控件本体：点它要拿 options / id，命中表里只有矩形
            this._hits.push({ key: w.key, label: w.label, kind: 'select', rect: w.rect, spec: w });
          } else if (w.kind === 'check') {
            w.key = `check:${w.id}`;
            this._hits.push({ key: w.key, label: w.label, kind: 'check', rect: w.rect, spec: w });
          }
          this._drawList.push(w);
          y += UI.rowH;
        }
        y += UI.blockGap;
      }
      bottom = Math.max(bottom, y);
    }

    this._titleHits = this._titleHitRects();
    this._hits.push(...this._titleHits);
    this._contentH = bottom;
    this._canvasH = Math.ceil(bottom + UI.statusH);
    return this;
  }

  /** 标题栏右侧四个按钮的矩形 */
  _titleHitRects() {
    const w = UI.pxW;
    const h = 36;
    const y = (UI.titleH - h) / 2 + 4;
    const defs = [
      { key: 'panel:collapse', label: '收起', w: 92 },
      { key: 'panel:exit', label: this.xr && this.xr.active === 'ar' ? '退出透视' : '退出 VR', w: 124 },
      { key: 'panel:recenter', label: '归位', w: 92 },
      { key: 'panel:opacity', label: this._opacity < 1 ? '50%' : '100%', w: 100 },
    ];
    let x = w - UI.pad;
    const out = [];
    for (const d of defs) {
      x -= d.w;
      out.push({ key: d.key, label: d.label, kind: 'panel', rect: { x, y, w: d.w, h } });
      x -= 10;
    }
    return out;
  }

  /** 下拉列表覆盖层：整块画布盖住，只保留选项矩形可点 */
  _layoutOverlay() {
    const ov = this._overlay;
    const W = this._canvas.width;
    const H = this._canvas.height;
    const top = UI.titleH + 34;
    // 面板现在是宽扁条（≈290 px 高），11 项列表放不下。
    // 这里固定行高，由 rebuild() 按 n 把画布临时撑到 H = top + n*rowH + ... px；
    // 命中坐标按 canvas.height 换算，所以 _pick() 不用改。
    const gap = 8;
    const itemH = UI.rowH;
    const x = 120;
    const w = W - 240;

    ov.items.forEach((it, i) => {
      it.rect = { x, y: top + i * (itemH + gap), w, h: itemH };
    });

    // 关闭用的整屏矩形必须排在**最前面**：_hitAt 从后往前找，
    // 排后面会把所有选项都吞掉（整屏矩形命中一切）。
    // 标题栏（设置 / 50% / 归位 / 退出 VR / 收起）的命中区保留在中间，
    // 这样下拉打开时这 4 个按钮既能看见也能点（否则标题栏被蒙层压住且点不到，
    // 看着像面板"丢了头"）。_hitAt 先查选项、再查标题、最后才落到整屏关闭。
    this._hits = [{ key: 'overlay:close', label: '', kind: 'option', rect: { x: 0, y: 0, w: W, h: H } }]
      .concat(this._titleHits || [])
      .concat(ov.items.map((it) => ({
        key: it.key, label: it.label, kind: 'option', rect: it.rect,
      })));
    return this;
  }

  // ================================================================ 重画
  rebuild() {
    if (!this._ctx) return this;
    this._rebuildCount = (this._rebuildCount || 0) + 1;   // 自检：空闲时不该再涨
    this._groups = this._buildGroups();
    this._layout();

    const W = UI.pxW;
    const baseH = this._canvasH;
    const n = this._overlay ? this._overlay.items.length : 0;
    // 下拉覆盖层打开时临时撑高画布，容纳 11 项回放列表；关闭后自动缩回宽扁条。
    const H = n
      ? Math.max(baseH, UI.titleH + 34 + n * UI.rowH + (n - 1) * 8 + 30)
      : baseH;
    if (this._canvas.width !== W || this._canvas.height !== H) {
      this._canvas.width = W;
      this._canvas.height = H;
      // 画布尺寸变了 → 纹理必须重建（texStorage2D 分配不可变，详见 _makeTexture）。
      // 旧写法只改 canvas 尺寸 + needsUpdate，GL 会报 INVALID_VALUE 且平面上仍是
      // 拉伸的旧图，就是「点回放，面板被拉伸、下拉选项看不见」的根因。
      this._makeTexture();
    }
    // 画布高宽比变了 → 平面尺寸跟着变，文字物理大小保持恒定
    const worldH = UI.worldW * (H / W);
    const baseWorldH = UI.worldW * (baseH / W);
    this._mesh.scale.set(UI.worldW, worldH, 1);
    // 顶边锚定：PlaneGeometry 中心在原点，dy + worldH/2 = baseWorldH/2
    // → dy = (baseWorldH - worldH)/2（负值 = 中心下移，顶边留在原处）。
    this._mesh.position.y = (baseWorldH - worldH) / 2;

    if (this._overlay) this._layoutOverlay();

    this._draw();
    this._tex.needsUpdate = true;
    this._sig = this._signature();
    return this;
  }

  _draw() {
    const ctx = this._ctx;
    const W = this._canvas.width;
    const H = this._canvas.height;

    ctx.clearRect(0, 0, W, H);
    this._round(1, 1, W - 2, H - 2, 18);
    ctx.fillStyle = C.bg;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = C.line;
    ctx.stroke();

    this._drawTitle();
    // 覆盖层撑高后上半部不画底层控件，避免蒙层下露出被压暗的旧内容
    if (!this._overlay) for (const w of this._drawList) this._drawWidget(w);
    if (this._overlay) { this._drawOverlay(); this._drawTitle(); } // 蒙层之上重绘标题栏，避免"丢头"
  }

  _drawTitle() {
    const ctx = this._ctx;
    const W = this._canvas.width;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.font = this._font(30, 700);
    ctx.fillStyle = C.fg;
    ctx.fillText('设置', UI.pad, UI.titleH / 2 + 3);

    for (const h of this._titleHits) {
      const on = this._hoverKey === h.key;
      this._chip(h.rect, {
        hover: on,
        accent: h.key === 'panel:exit',
        active: h.key === 'panel:recenter' && !this._headLocked,
      });
      ctx.font = this._font(23, 600);
      ctx.fillStyle = on ? C.fg : C.dim;
      ctx.textAlign = 'center';
      ctx.fillText(h.label, h.rect.x + h.rect.w / 2, h.rect.y + h.rect.h / 2 + 1);
    }

    ctx.strokeStyle = C.lineSoft;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(UI.pad, UI.titleH);
    ctx.lineTo(W - UI.pad, UI.titleH);
    ctx.stroke();
  }

  _drawWidget(w) {
    const ctx = this._ctx;
    const r = w.rect;
    if (w.kind === 'h') {
      ctx.font = this._font(UI.headFont, 700);
      ctx.fillStyle = C.dim;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(w.text.toUpperCase(), r.x + 2, r.y + r.h / 2 + 4);
      ctx.strokeStyle = C.lineSoft;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(r.x, r.y + r.h - 8);
      ctx.lineTo(r.x + r.w, r.y + r.h - 8);
      ctx.stroke();
      return;
    }

    if (w.kind === 'check') {
      const on = this._hoverKey === w.key;
      const flash = this._flashKey === w.key;
      if (on) {
        this._round(r.x - 6, r.y + 4, r.w + 12, r.h - 8, 10);
        ctx.fillStyle = C.hover;
        ctx.fill();
      }
      const bs = 26;
      const bx = r.x + 6;
      const by = r.y + (r.h - bs) / 2;
      this._round(bx, by, bs, bs, 7);
      ctx.lineWidth = 2;
      ctx.strokeStyle = flash || w.checked ? C.accent : 'rgba(160,190,215,0.5)';
      ctx.fillStyle = flash ? C.on : (w.checked ? C.accent : 'rgba(255,255,255,0.04)');
      ctx.fill();
      ctx.stroke();
      if (w.checked) {
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 3.2;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(bx + 6, by + 13.5);
        ctx.lineTo(bx + 11.5, by + 19);
        ctx.lineTo(bx + 20, by + 7.5);
        ctx.stroke();
        ctx.lineCap = 'butt';
      }
      ctx.font = this._font(UI.font);
      ctx.fillStyle = C.fg;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(this._ellipsis(w.label, r.w - bs - 26), bx + bs + 12, r.y + r.h / 2 + 1);
      return;
    }

    if (w.kind === 'radio') {
      for (const it of w.items) {
        const on = this._hoverKey === it.key || this._flashKey === it.key;
        this._chip(it.rect, { hover: on, active: it.on });
        ctx.font = this._font(UI.font - 2, it.on ? 600 : 500);
        ctx.fillStyle = it.on ? C.fg : C.dim;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this._ellipsis(it.label, it.rect.w - 16),
          it.rect.x + it.rect.w / 2, it.rect.y + it.rect.h / 2 + 1);
      }
      return;
    }

    if (w.kind === 'select') {
      const on = this._hoverKey === w.key || this._flashKey === w.key;
      ctx.font = this._font(UI.font);
      ctx.fillStyle = C.fg;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const labW = Math.min(ctx.measureText(w.label).width, r.w * 0.32);
      ctx.fillText(w.label, r.x + 2, r.y + r.h / 2 + 1);

      const vw = r.w - labW - 20;
      const vr = { x: r.x + r.w - vw, y: r.y + 4, w: vw, h: r.h - 8 };
      this._chip(vr, { hover: on });
      const cur = w.options.find((o) => o.value === w.value);
      ctx.font = this._font(UI.font - 3, 500);
      ctx.fillStyle = C.fg;
      ctx.textAlign = 'left';
      ctx.fillText(this._ellipsis(cur ? cur.label : String(w.value), vw - 40),
        vr.x + 12, vr.y + vr.h / 2 + 1);

      const cx = vr.x + vw - 18;
      const cy = vr.y + vr.h / 2;
      ctx.beginPath();
      ctx.moveTo(cx - 6, cy - 3);
      ctx.lineTo(cx + 6, cy - 3);
      ctx.lineTo(cx, cy + 5);
      ctx.closePath();
      ctx.fillStyle = C.dim;
      ctx.fill();
      return;
    }

    if (w.kind === 'btnrow') {
      for (const it of w.items) {
        const on = this._hoverKey === it.key;
        const flash = this._flashKey === it.key;
        this._chip(it.rect, { hover: on, active: flash });
        ctx.font = this._font(UI.font - 3, 500);
        ctx.fillStyle = on || flash ? C.fg : C.dim;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this._ellipsis(it.label, it.rect.w - 14),
          it.rect.x + it.rect.w / 2, it.rect.y + it.rect.h / 2 + 1);
      }
    }
  }

  _drawOverlay() {
    const ctx = this._ctx;
    const W = this._canvas.width;
    const H = this._canvas.height;
    const ov = this._overlay;

    ctx.fillStyle = 'rgba(6,10,16,0.92)';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = C.line;
    ctx.lineWidth = 2;
    this._round(1, 1, W - 2, H - 2, 18);
    ctx.stroke();

    ctx.font = this._font(26, 600);
    ctx.fillStyle = C.dim;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(ov.label, 120, UI.titleH + 2);

    for (const it of ov.items) {
      const on = this._hoverKey === it.key;
      this._chip(it.rect, { hover: on, active: it.on });
      ctx.font = this._font(UI.font - 3, it.on ? 600 : 500);
      ctx.fillStyle = it.on ? C.fg : C.dim;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(this._ellipsis(it.label, it.rect.w - 24), it.rect.x + 14, it.rect.y + it.rect.h / 2 + 1);
    }
  }

  /** 收起态的小签：三条横线 + 「设置」 */
  _drawTab() {
    const ctx = this._tabCtx;
    const W = this._tabCanvas.width;
    const H = this._tabCanvas.height;
    ctx.clearRect(0, 0, W, H);
    this._round(2, 2, W - 4, H - 4, 26);
    ctx.fillStyle = C.bg;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = C.onLine;
    ctx.stroke();

    ctx.strokeStyle = C.accent;
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(52, 56 + i * 24);
      ctx.lineTo(112, 56 + i * 24);
      ctx.stroke();
    }
    ctx.lineCap = 'butt';

    ctx.font = this._font(52, 600);
    ctx.fillStyle = C.fg;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('设置', 136, H / 2 + 2);

    const worldH = UI.tabW * (H / W);
    this._tabMesh.scale.set(UI.tabW, worldH, 1);
    this._tabTex.needsUpdate = true;
  }

  // ================================================================ 小工具
  _font(px, weight = 500) { return `${weight} ${px}px ${FONT}`; }

  _round(x, y, w, h, r) {
    const ctx = this._ctx;
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  _chip(rect, { hover = false, active = false, accent = false } = {}) {
    const ctx = this._ctx;
    this._round(rect.x, rect.y, rect.w, rect.h, Math.min(12, rect.h / 2));
    ctx.fillStyle = active ? C.on : (hover ? C.hover : C.chip);
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = active ? C.onLine
      : hover ? (accent ? 'rgba(255,180,84,0.6)' : C.hoverLine)
        : (accent ? 'rgba(255,180,84,0.35)' : C.lineSoft);
    ctx.stroke();
  }

  _ellipsis(text, maxW) {
    const ctx = this._ctx;
    const s = String(text == null ? '' : text);
    if (ctx.measureText(s).width <= maxW) return s;
    let out = s;
    while (out.length > 1 && ctx.measureText(out + '…').width > maxW) out = out.slice(0, -1);
    return out + '…';
  }

  /** 面板状态签名：任何一项变了就重画 */
  _signature() {
    const p = this.panelEl;
    if (!p) return '';
    const out = [];
    for (const s of p.querySelectorAll('select')) out.push(s.value);
    for (const c of p.querySelectorAll('input[type=checkbox]')) out.push(c.checked ? 1 : 0);
    for (const r of p.querySelectorAll('input[type=radio]')) out.push(r.checked ? 1 : 0);
    for (const b of p.querySelectorAll('.body button')) out.push(b.textContent.trim());
    return out.join('|');
  }

  // ================================================================ 调试 / 自检
  /** 所有可点控件（扁平，带矩形） */
  widgets() {
    return this._hits
      .filter((h) => h.key !== 'overlay:close')
      .map((h) => ({
        key: h.key, kind: h.kind, label: h.label,
        rect: { x: h.rect.x, y: h.rect.y, w: h.rect.w, h: h.rect.h },
      }));
  }

  state() {
    const counts = { radio: 0, select: 0, check: 0, btn: 0, panel: 0, option: 0 };
    for (const h of this.widgets()) counts[h.kind] = (counts[h.kind] || 0) + 1;
    return {
      visible: this.visible,
      collapsed: this.collapsed,
      overlay: this._overlay ? this._overlay.id : null,
      canvas: { w: this._canvas.width, h: this._canvas.height },
      world: { w: UI.worldW, h: UI.worldW * (this._canvas.height / this._canvas.width) },
      groups: this._groups.map((g) => g.title),
      counts,
      hover: this._hoverKey,
      status: this._statusText,
      sig: this._sig,
      opacity: this._opacity,
      headLocked: this._headLocked,
      baseH: this._canvasH,
    };
  }

  /** 把画好的面板导成 PNG dataURL —— 上机看中文有没有变豆腐块 */
  exportPNG() { return this._canvas.toDataURL('image/png'); }
}
