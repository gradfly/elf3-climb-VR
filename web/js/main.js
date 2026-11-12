/**
 * main.js —— elf3 网页端 MuJoCo 查看器 / VR 沉浸式观看
 *
 * 流程：加载 MuJoCo WASM → 把 XML/STL 灌进虚拟文件系统 → 编译模型 →
 *       用 data.geom_xpos/geom_xmat 驱动 three.js 场景 → 静止站立（默认）或实时物理
 *
 * 三种"看什么"：
 *   1) 静止站立（默认）：mj_forward 一次，机器人保持装配姿态；
 *   2) 运动仿真回放：读 web/assets/motion_*.bin（本机用 tools/bake_motion.py
 *      跑 MuJoCo + policy.onnx 闭环录下来的 qpos 序列），按时间插值写回 data.qpos，
 *      再 mj_kinematics 刷新几何位姿 → 就是"策略跟踪打拳动作"的仿真画面，VR 里也能看；
 *   3) 本地物理（可选）：直接跑 mj_step，无控制器，机器人会倒地（演示物理用）。
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import loadMujoco from '../vendor/mujoco.js';
import { MjScene } from './mjscene.js';
import { VRNav } from './vrnav.js';
import { MotionPlayer, MOTION_PREFETCH } from './motion.js';
import { XRControls } from './xr.js';
import { XRPanel } from './xrpanel.js';

// ---------------------------------------------------------------- 配置
const CFG = {
  assetsBase: new URL('../assets/', import.meta.url).href,
  entryXml: 'elf3.xml',
  fetchConcurrency: 6,
  maxPhysicsStepsPerFrame: 24,
  defaultTimestep: 0.002,
  motionDefault: 'fight_policy',  // ?motion=1 时用的默认回放数据
  followLerp: 0.07,               // 桌面视角跟随的每秒收敛比例（每帧）
  xrFloorOpacity: 0.3,            // 半透明 VR（透视）时地面网格的不透明度
};

/** 矩阵场景的行列与间距（米，MuJoCo 坐标；整体以原点为中心） */
const GRID = { rows: 8, cols: 12, pitchX: 1.08, pitchZ: 1.17 };

// ---------------------------------------------------------------- 启动期画质参数
/**
 * 这两个值只能在启动时定，运行期改不了：
 *   AA    —— 写进 WebGL context attributes。three 在 XR 里把 antialias 透传进
 *            XRWebGLLayer，以及投影层的 `samples: antialias ? 4 : 0`
 *            （vendor/three.module.js:27451），一体机上 4× MSAA 是最贵的一笔带宽。
 *   XR_SS —— XRWebGLLayer / XRProjectionLayer 的 framebufferScaleFactor，必须早于
 *            renderer.xr.setSession 才有意义（vendor:27303 在 presenting 时只警告）。
 *
 * 一体机（PICO / Quest 的浏览器 UA 都带 Android）默认关 MSAA，**帧缓冲不缩放（1.0）**。
 *
 * ⚠️ **不要把 XR_SS 默认调成 0.8**：three 在有 `XRWebGLBinding` 的设备上走投影层分支，
 * 用 `scaleFactor` 创建 XRProjectionLayer 再按 textureWidth/Height 建 render target
 * （vendor/three.module.js:27427-27453）。PICO 4 Ultra 上 0.8 会拿到无效尺寸
 * （或合成器不接受该纹理），**整帧被丢弃 → 进入 VR 全黑**，连机器人和地面都看不到。
 * 已实测：`?xrss=1` 恢复，0.8 全黑。要换性能请显式 `?xrss=0.8`，且必须先在真机确认能出图。
 */
const QS = new URLSearchParams(location.search);
const IS_STANDALONE_XR = /Android/i.test(navigator.userAgent) && 'xr' in navigator;
const AA = QS.has('aa') ? QS.get('aa') !== '0' : !IS_STANDALONE_XR;
const XR_SS = (() => {
  const v = Number(QS.get('xrss'));
  return Number.isFinite(v) && v > 0 ? Math.min(1, Math.max(0.5, v)) : 1;
})();

/**
 * 注视点渲染（fixed foveation）等级：0 = 全程全分辨率，1 = 边缘最狠。
 *
 * ⚠️ **不要默认拉满**。FFR 在头显侧是「按着色率（shading rate）分块」实现的：
 * 边缘区域每 N×N 像素才真正着色一次。细长 / 面积小的几何能不能被采样到，取决于
 * 三角形与采样网格的**亚像素对齐** —— 而左右眼是两个独立的采样网格（还有各自的
 * 透镜/注视点映射），于是同一块几何可能「左眼有、右眼整只没了或只剩一半」。
 * 越是细 / 小 / 远的物体越容易中招（机器人肢体、十几米外的整只机器人）；
 * 地面网格虽然也被降采样，但它连成一片，少几条线看不出来，所以表现为「只有机器人丢」。
 * 这也是为什么现象和场景强相关：正步/蹲起/高跳把机器人放在画面中心（fovea 区，全分辨率），
 * 贴地爬行（低伏）、爬楼梯（远）、螃蟹步（偏边）都把它推到边缘区。
 * 曾经把这里设成 1.0，在 PICO 4 上稳定复现「右眼没有机器人」。默认 0.5；
 * 想省性能可 ?ffr=1，怀疑 FFR 就 ?ffr=0。
 */
const FFR = (() => {
  if (!QS.has('ffr')) return 0.5;
  const v = Number(QS.get('ffr'));
  return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0.5;
})();

/**
 * 矩阵模式逐实例视锥剔除。`?cull=0` 关掉（万一边缘出现闪，用来排除）。
 *
 * GRID_CULL_R = 机器人包围球半径（约 0.9 m）+ 余量。余量要盖住两件事：
 * ① 剔除用的是上一帧的机器人中心（1 帧位姿滞后，≤3 cm）；
 * ② 头显快速转头时，本帧视锥和上一帧最多差 4~5°（10 m 处横向 ~0.8 m）。
 */
const CULL_GRID = QS.get('cull') !== '0';
const GRID_CULL_R = 1.7;

/**
 * 机器人着色模型：`lambert`（默认，哑光塑料，片元成本约 1/2~1/3）或 `standard`
 * （PBR + 环境贴图，金属质感）。见 `mjscene.setRobotMaterial`。
 * `?mat=standard` 或运行期 `__elf.setRobotMaterial('standard')` 切回 PBR。
 */
const ROBOT_MAT = QS.get('mat') === 'standard' ? 'standard' : 'lambert';

// ---------------------------------------------------------------- DOM
const $ = (id) => document.getElementById(id);
const canvas = $('view');
const elLoading = $('loading');
const elBar = $('ld-bar');
const elStep = $('ld-step');
const elPct = $('ld-pct');
const elLog = $('ld-log');
const elHint = $('ld-hint');

let logLines = 0;
/** 只在值真的变了时才写 DOM（textContent 赋值会标脏样式/布局，XR 帧里白花时间） */
function setText(el, s) {
  if (el && el.textContent !== s) el.textContent = s;
}
function setProgress(pct, step) {
  const p = Math.max(0, Math.min(100, pct));
  elBar.style.width = `${p.toFixed(1)}%`;
  elPct.textContent = `${Math.round(p)}%`;
  if (step) elStep.textContent = step;
}
function log(msg) {
  logLines++;
  elLog.textContent += (logLines > 1 ? '\n' : '') + msg;
  elLog.scrollTop = elLog.scrollHeight;
}
function fail(err) {
  elLoading.classList.add('error');
  elStep.textContent = '启动失败';
  elHint.textContent = String(err && err.message ? err.message : err);
  elHint.style.color = '#ff8f8f';
  console.error(err);
}
let toastTimer = null;
function toast(msg, ms = 2200) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), ms);
}

// 把运行时异常也写进加载日志，方便排查（headless 自检也会用到）
window.addEventListener('error', (e) => {
  log(`! 运行时错误: ${e.message} @ ${e.filename ?? ''}:${e.lineno ?? 0}`);
  $('ld-hint').textContent = `运行时错误：${e.message}`;
  $('ld-hint').style.color = '#ff8f8f';
});
window.addEventListener('unhandledrejection', (e) => {
  log(`! 未处理的 Promise 拒绝: ${e.reason?.message || e.reason}`);
});

// ---------------------------------------------------------------- 工具
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
  const len = Number(res.headers.get('content-length')) || 0;
  if (!res.body || !res.body.getReader) {
    const buf = await res.arrayBuffer();
    onBytes?.(buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.byteLength;
    onBytes?.(value.byteLength);
  }
  const out = new Uint8Array(len || got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.byteLength; }
  return out.buffer;
}

async function mapLimit(items, limit, worker) {
  const ret = new Array(items.length);
  let i = 0;
  const runners = new Array(Math.min(limit, items.length)).fill(0).map(async () => {
    for (;;) {
      const idx = i++;
      if (idx >= items.length) return;
      ret[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(runners);
  return ret;
}

function makeSkyTexture() {
  const cv = document.createElement('canvas');
  cv.width = 8; cv.height = 256;
  const ctx = cv.getContext('2d');
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0.0, '#060a11');
  g.addColorStop(0.42, '#14263a');
  g.addColorStop(0.53, '#2c4560');
  g.addColorStop(0.62, '#16232f');
  g.addColorStop(1.0, '#080c12');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 8, 256);
  const tex = new THREE.CanvasTexture(cv);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ---------------------------------------------------------------- 全局状态
let renderer, scene, camera, controls, stage, mjScene, vrNav, xr, xrPanel, mujoco, model, data;
let simulating = false, physicsAcc = 0, timestep = CFG.defaultTimestep;
let fpsAcc = 0, fpsFrames = 0, fps = 0;
let standQpos = null; // 落地对齐后的静止站立姿态（复位用）
let kinematics = null; // 回放用的前向计算（优先 mj_kinematics，比 mj_forward 省掉碰撞检测）
let motion = null;     // 当前回放（MotionPlayer）
let motionName = 'off';
let following = true;  // 视角是否跟随机器人
let sceneMode = 'matrix';     // 场景：'single' 单人 / 'matrix' 8×12 矩阵（默认矩阵）
let terrainWanted = { stairs: true, crab: false };  // 当前回放需要的地形（矩阵模式下暂存，退出时恢复）
let motionHudAcc = 0;
let motionPausedUi = false;   // 暂停按钮当前显示的状态（冻结末帧时自动同步）
let keyLight = null;          // 主方向光（投阴影那盏），VR 降载时关它的 castShadow
let inXR = false;             // 当前是否处于"XR 降载"状态（由 enterXRQuality 置位）
let preXr = null;             // 进 VR 前的 { shadows }，退出时还原阴影（场景保留 VR 里的选择）
let lastWrittenTime = NaN;    // 上次真正写进 data.qpos 的回放时刻（跳帧用）
let needSync = true;          // 一次性重算请求（复位姿态 / 换回放 / 切开关）
let boundsAcc = 0;            // robotBounds 的节流累加器（VR 里降到 15 Hz）
let motionReqId = 0;          // 换回放的请求序号：连点下拉时防「后发先至」
let lastFrameTime = 0;        // 上一个 animation-frame 回调的 timestamp（ms），0 = 需重新锚定
let dtSmooth = 1 / 72;        // 平滑帧间隔。只给「跟随 / 阻尼」用；播放时间必须用原始 dt，
                              // 否则持续掉帧时会被 EMA 永久性拖慢
const FT_N = 120;             // 帧时间环形缓冲长度（≈1.7 s @72 Hz）
const _ft = new Float32Array(FT_N);
let _ftIdx = 0;
let _ftCount = 0;
const _vRobot = new THREE.Vector3();
const _vDelta = new THREE.Vector3();

/**
 * 帧时间统计（p50 / p95 / max / 标准差）。
 *
 * 头显上拿不到 GPU timer（没有 EXT_disjoint_timer_query_webgl2），所以「稳定低帧」
 * 和「长任务抖动」只能靠分布区分：p95/p50 ≈ 1 → 稳定被 GPU 卡住（降几何/像素才有用）；
 * p95/p50 > 2 → 帧里有长任务（GC / 纹理上传 / 着色器编译）。
 */
function frameStats() {
  const n = _ftCount;
  if (!n) return { frames: 0, p50: 0, p95: 0, max: 0, avg: 0, std: 0 };
  const a = _ft.slice(0, n);          // 拷一份再排，别原地打乱环形缓冲
  a.sort();
  let sum = 0;
  for (let i = 0; i < n; i++) sum += a[i];
  const avg = sum / n;
  let vs = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - avg; vs += d * d; }
  const r = (v) => +v.toFixed(2);
  return {
    frames: n,
    p50: r(a[Math.floor(0.5 * (n - 1))]),
    p95: r(a[Math.floor(0.95 * (n - 1))]),
    max: r(a[n - 1]),
    avg: r(avg),
    std: r(Math.sqrt(vs / n)),
  };
}

const VIEWS = {
  front: { pos: [0.0, 1.30, 2.75], target: [0, 0.95, 0] },
  back:  { pos: [0.0, 1.35, -2.75], target: [0, 0.95, 0] },
  side:  { pos: [2.90, 1.35, 0.0], target: [0, 0.95, 0] },
  head:  { pos: [0.95, 1.42, 1.35], target: [0, 1.22, 0] },
  top:   { pos: [0.01, 5.20, 3.10], target: [0, 0.80, 0] },
  // 螃蟹步台阶沿 MuJoCo +Y 上升，剖面在 YZ 平面上，所以虚拟相机要站在 +X 侧
  // （three 的 -Z 方向）才看得见"台阶"而不是一堵斜面墙。下面这组等于
  // "机器人中心 +（0, 1.5, -6.0）"，既适合开视角跟随时做追尾机位，
  // 关掉跟随时也能看到起步段和左侧整条楼梯。
  crab:  { pos: [-1.35, 2.50, -6.00], target: [-1.35, 1.00, 0] },
  // 蹲起：原地不动，用 3/4 机位同时看得见膝角与躯干前倾；蹲下后躯干只有
  // 0.52 m 高，所以视点比正面机位略低，别把目标点还留在 0.95 m。
  squat: { pos: [1.30, 1.05, 1.65], target: [0, 0.70, 0] },
  // 正步 / 敬礼：机器人正面朝 three −Z（见 mjscene 的 yaw = +PI/2），所以
  // 正面 3/4 机位必须站在 −Z 一侧，否则看到的是后背（旧配置就吃到这个坑）。
  parade: { pos: [-1.45, 1.15, -1.75], target: [0, 0.98, 0] },
  // 矩阵场景：8 行 × 12 列（间距 1.08 / 1.17 m）铺开约 13 m × 8 m，机位要退到十几米外
  // 并俯视 10°，否则前排会把后排全挡住；目标点抬到 0.4 m，避免整片机器人挤在画面下半部。
  // 这组位置按「16:9 下整片四角落在 NDC 0.9 以内」反解（列数 16→12 后横向收窄 3.2 m）。
  grid:  { pos: [8.24, 1.40, -7.19], target: [0, 0.40, 0] },
  wide:  { pos: [7.20, 4.60, 10.5], target: [1.6, 0.90, 0] },
  // 高跳：略高的 3/4 机位，跟随看腾空弧线（机器人朝 +X，前跳幅度很小，基本原地起跳）
  jump:  { pos: [2.40, 1.60, 2.40], target: [0, 1.00, 0] },
  // 贴地爬行：站立 1.03 m → 趴姿 0.10 m，还要沿 three -Z 推进约 1.1 m。
  // 站在 +X/-Z 的侧前 3/4 低机位，俯角约 15° 才同时看得见「双手撑地」的
  // 落点和「双脚后伸」的幅度；目标点压到 0.45 m，别用站立回放的 0.95 m
  // （否则趴下后机器人整只掉到画面下缘）。视角跟随会带着相机一起平移。
  crawl: { pos: [1.65, 1.02, -1.10], target: [0, 0.42, -0.15] },
};

function applyView(name) {
  const v = VIEWS[name];
  if (!v) return;
  // VR 沉浸会话里相机就是头显姿态、每帧被覆盖，不能用它改视角。
  // 改为把机器人相对用户摆到对应观察角度（动世界 stage，等价桌面动相机）。
  if (renderer.xr.isPresenting) {
    if (vrNav && vrNav.presentView(name)) {
      // 选了固定机位就关掉自动跟随，否则下一帧又被拉回正前方，按钮看着像没反应。
      const fe = $('opt-follow');
      if (fe && fe.checked) { fe.checked = false; fe.dispatchEvent(new Event('change')); }
    }
    return;
  }
  if (!controls) return;
  camera.position.set(...v.pos);
  controls.target.set(...v.target);
  controls.update();
}

// ---------------------------------------------------------------- 启动
async function boot() {
  setProgress(2, '创建 WebGL 渲染器');

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: AA,
    // alpha:true 是"半透明 VR（透视）"的前提：XR 帧缓冲带 alpha 通道，
    // 清屏 alpha=0 的地方才能透出头显摄像头的真实画面。桌面观感不受影响
    // （场景背景是天空盒贴图，始终不透明）。
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false; // 场景基本静止，阴影贴图按需刷新
  renderer.xr.enabled = true;

  scene = new THREE.Scene();
  scene.background = makeSkyTexture();
  scene.fog = new THREE.Fog(0x16232f, 34, 130);

  camera = new THREE.PerspectiveCamera(58, window.innerWidth / window.innerHeight, 0.05, 300);
  camera.position.set(2.05, 1.52, 3.05);
  scene.add(camera);

  // 环境光照：RoomEnvironment 提供金属件需要的高光反射
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
  scene.environment = envRT.texture;
  scene.environmentIntensity = 0.55;
  pmrem.dispose();

  // world 光照都挂在 stage 里，跟着世界一起平移/旋转
  stage = new THREE.Group();
  stage.name = 'stage';
  scene.add(stage);

  const hemi = new THREE.HemisphereLight(0xbcd8ff, 0x2a3341, 0.75);
  stage.add(hemi);

  const key = new THREE.DirectionalLight(0xffffff, 2.35);
  key.position.set(4.2, 7.6, 5.6);
  key.target.position.set(0, 0.85, 0);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 24;
  key.shadow.camera.left = -3.6;
  key.shadow.camera.right = 3.6;
  key.shadow.camera.top = 3.2;
  key.shadow.camera.bottom = -3.2;
  key.shadow.bias = -0.0012;
  key.shadow.normalBias = 0.02;
  stage.add(key, key.target);
  keyLight = key;          // 阴影开关的真正闸门（见 setShadows）

  const rim = new THREE.DirectionalLight(0x7fb6ff, 0.7);
  rim.position.set(-4.5, 3.2, -5.0);
  stage.add(rim);

  // 计时不用 THREE.Clock：frame(time) 直接用 animation-frame 的 timestamp，
  // 见主循环注释（Clock 会把本帧自身耗时光算进 dt，掉帧时正反馈放大抖动）

  // ---------------- MuJoCo WASM ----------------
  setProgress(5, '加载 MuJoCo WebAssembly 引擎');
  log('· 正在下载 MuJoCo WASM（约 10 MB）…');
  mujoco = await loadMujoco();
  log('· MuJoCo WASM 就绪');
  setProgress(42, '读取资源清单');

  const manifest = await (await fetch(`${CFG.assetsBase}manifest.json`)).json();
  const total = manifest.totalBytes || manifest.files.reduce((a, f) => a + f.size, 0);
  let loaded = 0;
  setProgress(44, `加载模型资源（${manifest.files.length} 个文件 / ${(total / 1048576).toFixed(1)} MB）`);

  const buffers = new Map();
  await mapLimit(manifest.files, CFG.fetchConcurrency, async (f) => {
    const buf = await fetchWithProgress(`${CFG.assetsBase}${f.path}`, (n) => {
      loaded += n;
      setProgress(44 + 38 * (loaded / total), `加载模型资源 ${(loaded / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
    });
    buffers.set(f.path, buf);
  });
  log(`· 资源就绪：${buffers.size} 个文件`);

  // ---------------- 编译模型 ----------------
  setProgress(84, '编译 MuJoCo 模型');
  const vfs = new mujoco.MjVFS();
  for (const f of manifest.files) {
    vfs.addBuffer(f.path, new Uint8Array(buffers.get(f.path)));
  }
  const entryBuf = buffers.get(CFG.entryXml) || buffers.get(manifest.entry);
  const xmlText = new TextDecoder().decode(new Uint8Array(entryBuf));

  const t0 = performance.now();
  model = mujoco.MjModel.from_xml_string(xmlText, vfs);
  data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);
  try { timestep = model.opt?.timestep || CFG.defaultTimestep; } catch { timestep = CFG.defaultTimestep; }
  log(`· 模型编译完成：${model.nbody} 个刚体 / ${model.ngeom} 个几何 / ${model.nq} DOF（${(performance.now() - t0).toFixed(0)} ms）`);

  setProgress(91, '构建三维场景');
  mjScene = new MjScene({ THREE, mujoco, model, data }).build();
  stage.add(mjScene.wrapZUp());
  log(`· 网格三角面 ${mjScene.stats.triangles.toLocaleString()} 个 / 几何体 ${mjScene.stats.geometries} 个`);
  // 材质在 build() 里按 PBR 建好；默认是 Lambert，这里统一对齐到 ROBOT_MAT
  if (mjScene.robotMaterial !== ROBOT_MAT) mjScene.setRobotMaterial(ROBOT_MAT);
  if (ROBOT_MAT === 'standard') log('· 机器人材质：MeshStandard（PBR；?mat= 不传时默认是 Lambert）');

  // 回放时只需要几何位姿：mj_kinematics 不跑碰撞/约束，比 mj_forward 便宜很多
  kinematics = (typeof mujoco.mj_kinematics === 'function') ? mujoco.mj_kinematics : mujoco.mj_forward;
  log(`· 回放前向计算：${kinematics === mujoco.mj_forward ? 'mj_forward（该构建未导出 mj_kinematics）' : 'mj_kinematics'}`);

  // 站立姿态落地对齐：模型初始 free joint 高度让脚底略低于 z=0，
  // 这里按视觉网格最低点整体抬到底面正好贴地，避免脚陷进地板。
  {
    const minZ = mjScene.robotMinZ();
    if (Number.isFinite(minZ) && Math.abs(minZ) > 1e-4) {
      data.qpos[2] -= minZ;
      mujoco.mj_forward(model, data);
      mjScene.sync();
      log(`· 落地对齐：底座${minZ > 0 ? '下降' : '抬高'} ${(Math.abs(minZ) * 100).toFixed(1)} cm（脚底 z=0）`);
    }
    standQpos = Float64Array.from(data.qpos);
  }

  setProgress(97, '初始化交互与 VR');
  setupUI();
  setupControls();
  setupXR();
  onResize();
  window.addEventListener('resize', onResize);
  // 标签页从后台切回来时时间基准要重新锚定，否则会按「离开的时长」积一次
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) lastFrameTime = 0;
  });

  // 场景：默认矩阵（8×12 = 96 份同步播放），URL 可用 ?scene=single 覆盖
  setSceneMode(new URLSearchParams(location.search).get('scene') === 'single' ? 'single' : 'matrix',
    { silent: true });
  log(`· 场景：${sceneMode === 'matrix'
    ? `矩阵 ${GRID.rows}×${GRID.cols}（${GRID.rows * GRID.cols} 份实例）`
    : '单人'}`);

  await applyMotionFromQuery();     // 默认不开回放；URL 带 ?motion=… 时自动进入

  renderer.shadowMap.needsUpdate = true;
  renderer.setAnimationLoop(frame);

  // 调试句柄：控制台里可直接 __elf.applyView('front') / __elf.camera / __elf.mjScene
  window.__elf = {
    THREE, renderer, scene, camera, controls, stage, mjScene, vrNav, xr, xrPanel,
    mujoco, model, data, applyView, setMotion, restoreStandPose,
    setSceneMode, setTerrain, applyMotionView, uiMode: applyUiMode,
    stats: () => mjScene && mjScene.stats,
    // 左右眼现场体检：__elf.xrDiag()（见文件末尾），排查「单眼丢物体」用
    xrDiag,
    get simulating() { return simulating; },
    get following() { return following; },
    get sceneMode() { return sceneMode; },
    get grid() { return mjScene ? mjScene.grid : null; },
    get xrMode() { return xr ? xr.active : null; },
    get seeThrough() { return xr ? xr.through : false; },
    xrCapabilities: () => (xr ? xr.capabilities() : null),
    get kinematicsName() { return kinematics === mujoco.mj_forward ? 'mj_forward' : 'mj_kinematics'; },
    // —— VR 降载 / 性能观测（真机连 chrome://inspect 时用 perf()）——
    xrQuality: {
      enter: enterXRQuality,
      exit: exitXRQuality,
      // 会话里现场改 FFR：__elf.xrQuality.foveation(0) → 立刻生效，不用退出重进
      foveation: (v) => xr.setFoveation(v),
    },
    get xrScale() { return XR_SS; },
    get xrFoveation() { return xr ? xr.foveation : FFR; },
    get aa() { return AA; },
    get inXR() { return inXR; },
    // 机器人着色模型（有损开关）：'standard' | 'lambert'
    setRobotMaterial: (kind) => (mjScene ? mjScene.setRobotMaterial(kind) : null),
    get robotMaterial() { return mjScene ? mjScene.robotMaterial : ROBOT_MAT; },
    get shadows() {
      return {
        enabled: renderer.shadowMap.enabled,
        cast: !!keyLight?.castShadow,
        auto: renderer.shadowMap.autoUpdate,
      };
    },
    perf: () => ({
      fps: +fps.toFixed(1),
      tris: renderer.info.render.triangles,
      calls: renderer.info.render.calls,
      programs: renderer.info.programs.length,
      xrss: XR_SS,
      aa: AA,
      inXR,
      sceneMode,
      shadows: {
        enabled: renderer.shadowMap.enabled,
        cast: !!keyLight?.castShadow,
        auto: renderer.shadowMap.autoUpdate,
      },
      syncIds: mjScene && mjScene._syncIds ? mjScene._syncIds.length : null,
      syncCount: (mjScene && mjScene._syncCount) || 0,
      rebuildCount: (xrPanel && xrPanel._rebuildCount) || 0,
      // 矩阵实例：count 是总份数，drawn 是本帧真正提交的份数（VR 里做了视锥剔除）
      gridDrawn: (mjScene && mjScene.grid && mjScene.grid.drawn) || 0,
      gridTotal: (mjScene && mjScene.grid && mjScene.grid.count) || 0,
      ft: frameStats(),          // 帧时间分布：p95/p50 ≈ 1 → GPU-bound；> 2 → 长任务
      interp: (motion && motion.interp) || null,
      robotMaterial: (mjScene && mjScene.robotMaterial) || ROBOT_MAT,
      motionCache: MotionPlayer.cacheSize,
    }),
    get motion() { return motion; },          // 调试/自检：可直接快进、查 loop 状态
    // 回放数据缓存：自检用来验「预取之后切段不再走网络」
    motionCache: {
      get size() { return MotionPlayer.cacheSize; },
      clear: () => MotionPlayer.clearCache(),
      prefetch: (names) => MotionPlayer.prefetch(CFG.assetsBase, names),
    },
    getMotion: () => (motion ? {
      name: motionName, kind: motion.kind, label: motion.label,
      nframes: motion.nframes, nq: motion.nq, fps: motion.fps,
      duration: +motion.duration.toFixed(3), time: +motion.time.toFixed(3),
      speed: motion.speed, paused: motion.paused, loop: motion.loop,
      stats: motion.meta.stats || null,
    } : { name: motionName, kind: null }),
  };

  setProgress(100, '完成');
  $('hud').classList.remove('hidden');
  $('panel').classList.remove('hidden');
  $('help').classList.remove('hidden');
  setTimeout(() => elLoading.classList.add('hidden'), 260);

  // 首屏就绪之后再预热，不占启动时间
  schedulePrefetch();
}

/**
 * 空闲期预取全部回放数据（14 段 / 3.5 MB）。
 *
 * 现在 `MotionPlayer.load` 是纯按需的：切一段要等 json + bin 两个往返
 * （Wi-Fi 上 0.3~2 s），而 VR 里那 1~2 s 就是纯黑/冻帧。全部预取之后切段是同步的。
 *
 * 三条约束：
 *   1) 排在 `setProgress(100)` 与 `window.__elf` 之后 —— 不能拖慢启动（自检有 80 s 窗口）；
 *   2) 只在没有 XR 会话时启动，别和头显串流抢带宽；
 *   3) 失败完全静默：预取是锦上添花，不能影响主流程，更不能往 console 丢 warn
 *      （自检把 console.warn 也算「控制台有报错」）。
 */
function schedulePrefetch() {
  if (QS.get('noprefetch') === '1') return;
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 800));
  idle(() => {
    if (renderer.xr.isPresenting) return;
    MotionPlayer.prefetch(CFG.assetsBase, MOTION_PREFETCH, {
      onProgress: (done, total) => {
        if (done === total) log(`· 回放数据预取完成：${total} 段`);
      },
    }).catch(() => {});
  }, { timeout: 3000 });
}

// ---------------------------------------------------------------- 运动仿真回放
/** URL 参数：?motion=<off|fight_policy|fight_ref|climb_policy|crab|policy_crab|squat_cycle|squat_down|squat_up|crawl|goose_walk|goose_inplace|parade_seq|salute|longjump_hi|1> &speed=0.5 &follow=0/1 &scene=single|matrix */
async function applyMotionFromQuery() {
  const q = new URLSearchParams(location.search);
  const speed = q.get('speed');
  const follow = q.get('follow');
  if (speed) {
    const el = $('opt-motion-speed');
    if ([...el.options].some((o) => o.value === speed)) el.value = speed;
  }
  if (follow !== null) {
    following = !(follow === '0' || follow === 'false');
    $('opt-follow').checked = following;
    if (vrNav) vrNav.follow = following;
  }
  const m = q.get('motion');
  if (m === null) return null;                    // 不带参数 → 保持原来的静止站立
  const name = (m === '' || m === '1' || m === 'on') ? CFG.motionDefault : m;
  if (name === 'off') return null;
  return setMotion(name);
}

/** 恢复静止站立姿态（落地对齐后的 qpos） */
function restoreStandPose() {
  if (!model || !data) return;
  if (standQpos) data.qpos.set(standQpos);
  mujoco.mj_forward(model, data);
  mjScene.sync();
  renderer.shadowMap.needsUpdate = true;
}

/** 切换两套地形的显隐（见 index.html 的「显示」区） */
function setTerrain({ stairs, crab }) {
  const s = $('opt-stairs');
  const c = $('opt-crab');
  if (s && s.checked !== stairs) { s.checked = stairs; s.dispatchEvent(new Event('change')); }
  if (c && c.checked !== crab) { c.checked = crab; c.dispatchEvent(new Event('change')); }
}

/**
 * 「当前回放需要哪套地形」记在 terrainWanted 里。
 * 矩阵模式只复制机器人、不复制地形，所以这时把地形全藏起来；退出矩阵再按记录恢复。
 */
function setTerrainWanted(t) {
  terrainWanted = t;
  setTerrain(sceneMode === 'matrix' ? { stairs: false, crab: false } : t);
}

/**
 * 阴影总闸。
 *
 * 注意：只翻 renderer.shadowMap.enabled 是**没用的** —— three 的 needsProgramChange
 * 判断里不含 shadowMapEnabled（vendor/three.module.js:30700-30802），材质不会重编译，
 * 着色器里的 USE_SHADOWMAP / PCFSoft 采样照旧。真正的闸门是主光的 castShadow：
 * 它一变，WebGLLights 的 hash.numDirectionalShadows 就变 → state.version 递增
 * （vendor:22008/22015）→ 命中 lightsStateVersion 分支 → 自动换程序。
 * 旧 program 留在 programs 缓存里，所以来回切第二次是 cache hit，不会反复编译。
 */
function setShadows(on, { syncUi = true } = {}) {
  renderer.shadowMap.enabled = on;
  if (keyLight) keyLight.castShadow = on;
  renderer.shadowMap.needsUpdate = on;
  const el = $('opt-shadow');
  if (syncUi && el) el.checked = on;
}

/**
 * 进入 XR 的自动降载：关阴影、压掉阴影每帧重算。
 *
 * **不动场景**：矩阵 8×12（96 份）在 VR 里照常显示，用户在头显里选了什么就是什么。
 * 代价是 1980 万三角/单目、3950 万/双目 —— 一体机会明显掉帧，但这是用户的明确选择
 * （「进 VR 保持矩阵，不降载」）。想流畅就自己在面板里切单人。
 *
 * 退出时只还原阴影，场景保留 VR 里的选择（见 exitXRQuality）。
 */
function enterXRQuality() {
  if (inXR || !mjScene) return;
  inXR = true;
  preXr = { shadows: !!$('opt-shadow')?.checked };
  setShadows(false);
  // syncShadowAutoUpdate() 在「单人 + 有回放」时会把 autoUpdate 打开，这里压回去
  renderer.shadowMap.autoUpdate = false;
  // 关阴影会让「有阴影」那套 program 失效，下一帧要重编译。趁现在（会话刚建立、
  // 合成器还在过渡）同步编好，别让它落到第一帧 XR 渲染里 —— 那就是「进 VR 卡一下」。
  // 只在真的呈现时做：无头自检里 isPresenting 是 false，软渲染下编译要几分钟。
  if (renderer.xr.isPresenting) {
    try { renderer.compile(scene, camera); } catch { /* 预热失败不影响功能 */ }
  }
  // 帧时间基准重新锚定：进会话前后的间隔不可比
  lastFrameTime = 0;
  log(`· XR 降载：阴影 ${preXr.shadows ? '开' : '关'} → 关，场景保持 ${sceneMode}，`
    + `帧缓冲 ${XR_SS}×，注视点 ${FFR}`);
}

function exitXRQuality() {
  if (!inXR) return;
  inXR = false;
  setShadows(!!preXr?.shadows);
  // 场景不还原：VR 里切过的模式带回桌面（用户在头显里做的选择不该被抹掉）
  syncShadowAutoUpdate();
  lastFrameTime = 0;
  preXr = null;
}

/** 什么时候需要每帧重算阴影贴图：跑物理、或单人模式下有回放（XR 里一律不重算） */
function syncShadowAutoUpdate() {
  renderer.shadowMap.autoUpdate = !inXR && (simulating || (!!motion && sceneMode === 'single'));
  renderer.shadowMap.needsUpdate = true;
}

/** 当前回放该用哪个机位（只在单人模式下用得上：矩阵有自己的一片机位） */
function applyMotionView() {
  if (motion && motion.kind === 'crab') applyView('crab');
  else if (motion && motion.kind === 'squat') applyView('squat');
  else if (motion && motion.kind === 'parade' && motion.meta.camera) applyView(motion.meta.camera);
  else if (motion && motion.kind === 'jump') applyView('jump');
  else applyView('front');
}

/**
 * 切换场景模式。
 *   single —— 只有一个机器人（就是模型本身，动画照旧）；
 *   matrix —— 把机器人视觉网格实例化成 8 行 × 12 列 = 96 份，同步播放同一段动画。
 *
 * 矩阵用 InstancedMesh 实现（见 MjScene.enableRobotGrid）：几何 / 材质只有一份，
 * 绘制批次仍等于机器人几何数，96 份也不会掉帧；地形与阴影不参与复制。
 *
 * **VR 里同样可用**：不再拦截（用户明确要求「进 VR 保持矩阵」）。代价是双目
 * 3950 万三角/帧，一体机会明显掉帧 —— 需要流畅就自己在面板里点「单人」。
 *
 * @param {'single'|'matrix'} mode
 * @param {object} [o]
 * @param {boolean} [o.silent]   不弹 toast / 不写日志（启动期与退出还原用）
 * @param {boolean} [o.keepView] 不动机位。XR 期间三维画面由头显位姿决定，
 *   applyView() 改的桌面相机每帧都会被 updateUserCamera 覆盖（vendor:27713），
 *   真正需要保留的是 OrbitControls 的内部球坐标（退出 VR 后才用得上）。
 */
function setSceneMode(mode, { silent = false, keepView = false } = {}) {
  if (!mjScene) return;
  const m = mode === 'single' ? 'single' : 'matrix';
  sceneMode = m;
  // XR 里一律不动机位：桌面相机每帧被头显位姿覆盖，applyView 只会污染
  // OrbitControls 的球坐标（退出后桌面会停在一个莫名其妙的角度）。
  const freezeView = keepView || inXR || renderer.xr.isPresenting;

  for (const el of document.querySelectorAll('input[name="opt-scene"]')) {
    el.checked = el.value === m;
    el.closest('label')?.classList.toggle('on', el.checked);
  }

  let info = null;
  if (m === 'matrix') {
    info = mjScene.enableRobotGrid(GRID);
    setTerrain({ stairs: false, crab: false });   // 地形不复制，直接藏起来
    if (!freezeView) applyView('grid');
  } else {
    mjScene.disableRobotGrid();
    setTerrain(terrainWanted);
    if (!freezeView) applyMotionView();
  }

  syncShadowAutoUpdate();
  refreshSceneHud();
  refreshHudMode();

  if (!silent) {
    toast(m === 'matrix'
      ? `矩阵场景：${GRID.rows} 行 × ${GRID.cols} 列 = ${GRID.rows * GRID.cols} 份同步播放`
      : '已切回单人场景');
    log(m === 'matrix'
      ? `· 矩阵场景：${info.count} 份实例 / ${info.meshes.length} 个 InstancedMesh`
      : '· 已切回单人场景');
  }
  return info;
}

/** HUD「场景」栏 + 面板说明 */
function refreshSceneHud() {
  const el = $('hud-scene');
  if (el) {
    el.textContent = sceneMode === 'matrix'
      ? `矩阵 ${GRID.rows}×${GRID.cols}（${GRID.rows * GRID.cols} 份）`
      : '单人';
  }
  const note = $('scene-note');
  if (note) {
    const n = GRID.rows * GRID.cols;
    const tris = Math.round(n * 20.6);     // 每份约 20.6 万三角 → 单目万面/帧（96 份 ≈ 1978 万）
    note.innerHTML = sceneMode === 'matrix'
      ? `矩阵 = 把单人场景复制 <b>${GRID.rows} 行 × ${GRID.cols} 列 = ${n} 份</b>`
        + '（间距 1.08 / 1.17 m），同一段动画同步播放；地形 / 阴影不复制。'
        + `<br>实例化渲染：绘制批次仍是 30 出头（不是 32×${n}），代价是三角形 ×${n}`
        + `（约 ${tris} 万面/帧）—— 独立显卡流畅，一体机 / 集显建议切回单人。`
      : `单人 = 只有一个机器人。切到矩阵会把机器人复制 ${GRID.rows * GRID.cols} 份同步播放（默认）。`;
  }
}

/**
 * 「正步 + 敬礼」连贯序列的说明文案（motion_parade_seq）。
 *
 * 这段数据的 stats 同时带着正步与敬礼两套指标（bake_actions 从
 * meta.parts_meta 合并两段源 meta），走原来 parade 分支会把两套文案各印一遍，
 * 所以单独写一段。分段信息直接读 meta.sequence.parts，帧数改了也不用改文案。
 */
function paradeSequenceNote(motion, s) {
  const seq = motion.meta.sequence || {};
  const parts = (seq.parts || [])
    .map((p) => `${p.frames} 帧${p.action === 'goose_walk' ? '正步'
      : p.action === 'attention_blend' ? '过渡'
      : p.action === 'salute' ? '敬礼' : '定格'}`)
    .join(' + ');
  return '连贯动作：<code>launch/goose_step.py</code> 的正步（上体前倾 5°、双腿全程绷直、'
    + '支撑脚后置不滑）走到 3.15 m 处收步立定，再用 1 s 收臂成敬礼立正，接 '
    + '<code>launch/salute.py</code> 的军礼（抬手 → 保持 → 礼毕）；'
    + '拼接由 <code>scripts/build_parade_sequence.py</code> 完成'
    + '（smootherstep 过渡，<b>只插值关节、基座沿用正步终点</b>，否则 1 s 内会倒着滑回原点；'
    + '敬礼段整段平移 +3.15 m）。'
    + (s ? `<br>${parts} = <b>${s.frames} 帧</b>；接缝最大差 `
      + `${s.seamMaxDiffRad} rad，脚底 ${s.footMinZCm} ~ ${s.footMaxZCm} cm，`
      + `X 向推进 ${s.travelX} m。`
      + `<br>正步段：踢腿 ${s.kickLiftCm} cm、支撑腿后倾 ${s.stanceBackDeg}°、`
      + `大臂前摆 ${s.armFwdDeg}°；敬礼段：右腕右移 ${s.wristRightCm} cm、`
      + `滚 ${s.wristRollDeg}°（掌心 `
      + `(${(s.palmNormal || []).map((v) => v.toFixed(2)).join(', ')})，手背朝前的立掌）。` : '');
}

/**
 * 切换回放数据（'off' = 关闭）。
 * 回放与本地 mj_step 物理互斥：开回放会自动关掉物理开关。
 */
async function setMotion(name, { hideStairs = true } = {}) {
  const req = ++motionReqId;    // 连点下拉时，只有最后一次的结果能落地
  const sel = $('opt-motion');
  // 下拉里只列当前提供的几段；用 URL / __elf 传了列表外的名字（如 squat_down）时
  // 不要硬写 value，否则 <select> 会变成空白项，看着像坏了。
  if (sel && name && [...sel.options].some((o) => o.value === name)) sel.value = name;
  motionName = name;

  if (name === 'off' || !name) {
    motion = null;
    restoreStandPose();
    setTerrainWanted({ stairs: true, crab: false });  // 回到默认：只显示紧凑障碍
    $('hud-motion').textContent = '—';
    refreshHudMode();
    syncShadowAutoUpdate();
    $('motion-note').innerHTML =
      '回放数据由 <code>tools/bake_motion.py</code>（打拳）/ '
      + '<code>tools/bake_climb.py</code>（爬楼梯）/ '
      + '<code>tools/bake_crab.py</code>（螃蟹步横移）在本机跑一遍仿真后录制；'
      + '<code>tools/bake_squat.py</code>（蹲下 / 起立）直接烘焙参考动作。';
    return null;
  }

  if (simulating) {                                // 与本地物理互斥
    const el = $('opt-sim');
    el.checked = false;
    el.dispatchEvent(new Event('change'));
  }

  $('motion-note').textContent = '正在加载回放数据…';
  let player;
  try {
    player = await MotionPlayer.load(CFG.assetsBase, name);
    if (req !== motionReqId) return null;      // 期间又选了别的，这次结果作废
    if (player.nq !== data.qpos.length) {
      throw new Error(`数据 nq=${player.nq} 与模型 nq=${data.qpos.length} 不一致`);
    }
  } catch (e) {
    if (req !== motionReqId) return null;      // 慢请求的失败也别把 UI 重置成"加载失败"
    motion = null;
    motionName = 'off';
    if (sel) sel.value = 'off';
    $('hud-motion').textContent = '—';
    $('motion-note').innerHTML = `加载失败：${e.message}<br>`
      + '先在项目根目录跑 <code>python web/tools/bake_*.py</code> 生成对应回放数据'
      + '（蹲下 / 起立用 <code>bake_squat.py</code>）。';
    toast(`回放数据加载失败：${e.message}`, 4200);
    return null;
  }

  player.loop = player.meta.loop !== false;   // 单段蹲起播完冻结末帧
  player.speed = Number($('opt-motion-speed').value) || 1;
  player.restart();

  motion = player;
  motionPausedUi = false;
  // 不在这里预先写第 0 帧：主循环下一帧就会写，提前写只会让「切片瞬间」的
  // qpos 提前跳变（自检里「切片后 1.6 s 内位移变大」正是拿切片前的姿态当参照）。
  lastWrittenTime = NaN;
  needSync = true;
  $('btn-motion-pause').textContent = '暂停';

  // 两套地形共用同一个编译好的模型：紧凑障碍沿 +X（爬楼梯策略用的），
  // 螃蟹步台阶沿 +Y（参考动作 · 螃蟹步用的）。回放哪一段就只显示那一段 ——
  // 否则另一套地形会出现在画面里，看着像"走错路"，也白白拉低帧率。
  //   打拳 / 参考动作：平地录的，默认两套都藏起来（hideStairs=false 时保留障碍）
  //   爬楼梯策略：显示紧凑障碍
  //   螃蟹步横移：显示 crab_ 台阶
  const isCrab = motion.kind === 'crab';
  const isSquat = motion.kind === 'squat';
  const isParade = motion.kind === 'parade';
  setTerrainWanted({
    stairs: motion.kind === 'climb' || (!isCrab && !hideStairs),
    crab: isCrab,
  });
  // 矩阵模式下所有实例共用同一个机位，不跟着回放换机位（换了也看不清台阶）
  if (sceneMode === 'single') {
    if (isCrab) applyView('crab');      // 换成能看清台阶剖面的侧视机位
    if (isSquat) applyView('squat');    // 原地蹲起用 3/4 机位看膝角与躯干前倾
    if (isParade && motion.meta.camera) applyView(motion.meta.camera);  // 敬礼用近景
    if (motion.kind === 'jump') applyView('jump');                      // 高跳：3/4 机位看弧线
    if (motion.kind === 'crawl') applyView('crawl');                    // 贴地爬行：低机位看撑地/后伸
  }

  syncShadowAutoUpdate();
  refreshHudMode();
  $('hud-motion').textContent = motion.describe();

  const s = motion.meta.stats;
  // 同为 kind='crab' 的两段数据来源不同：脚本烘焙的参考动作 vs 策略闭环。
  // 靠 json 里的 policy 字段区分（bake_crab.py --policy-name 写入）。
  const crabHead = motion.meta.policy
    ? '策略仿真：本机跑 MuJoCo + <code>' + motion.meta.policy + '</code> 闭环录制'
      + '（残差步态 + 地形高度扫描，<b>没有脚本轨迹</b>）。机器人朝 +X，沿 +Y 侧向横移上楼。'
    : '参考动作：<code>crab_climb.py</code> 的四相位准静态控制器解出的关节轨迹'
      + '（闭式 IK + DLS 精修，<b>没有策略网络</b>）。机器人朝 +X，沿 +Y 侧向横移上楼。';
  const squatHead = motion.meta.loop
    ? '循环回放：下蹲 → 蹲姿保持 → 起立 → 站姿保持，首尾同姿态，无缝循环。'
    : '单段回放：播完<b>冻结在末帧</b>（与真机 <code>freeze_at_end</code> 一致），不跳回起点。'
      + '想重看请点「从头播放」。';

  // 连贯序列（正步 + 敬礼）单独一段说明：它的 stats 同时含两套指标，
  // 落到下面 parade 分支会把正步 / 敬礼两段文案各印一遍。
  if (motion.kind === 'parade' && motion.meta.sequence) {
    $('motion-note').innerHTML = paradeSequenceNote(motion, s)
      + `<br>${motion.nframes} 帧 @ ${motion.fps} fps = ${motion.duration.toFixed(1)} s，`
      + '播完<b>冻结在末帧立正</b>，想重看请点「从头播放」。';
    toast(`已载入回放：${motion.label}`);
    log(`· 回放数据 ${name}：${motion.nframes} 帧 / ${motion.duration.toFixed(1)} s`);
    return motion;
  }
  const note = motion.kind === 'ref'
    ? '参考动作：<code>fight.npz</code> 里重定向后的运动，纯运动学装配（无物理）。'
    : motion.kind === 'parade'
      ? '参考动作：<code>launch/goose_step.py</code> 用解析两连杆 IK 让<b>两条腿始终绷直</b>、'
        + '支撑脚锁死不滑且大腿<b>微微后倾</b>；大臂前摆到最高点时<b>小臂水平横在胸前</b>'
        + '（肩内旋把折 90° 的小臂转到横向）；'
        + '<code>launch/salute.py</code> 用「腕 + 肘」双约束 IK 解出军礼手位。'
        + '逐帧生成，纯运动学装配（<b>无策略网络、不跑 mj_step</b>）。'
        + (s ? `<br>脚底最低点 ${s.footMinZCm} ~ ${s.footMaxZCm} cm，`
          + `单帧最大关节跳变 ${s.maxJointJumpRad} rad，X 向推进 ${s.travelX} m。`
          + (s.kickLiftCm ? `踢腿高度 ${s.kickLiftCm} cm @ ${s.baseSpeed} m/s。` : '')
          + (s.wristRightCm !== undefined
            ? `<br>右手腕沿<b>机器人右侧水平外移 ${s.wristRightCm} cm</b>`
              + `（y = ${s.wristY_m} m，高度不变），距头中心 ${s.wristToHeadMm} mm`
              + `（IK 误差 ${s.wristErrMm} mm）；左手自然下垂，`
              + `小臂竖直度 <b>${s.leftArmDown}</b>（1.0 = 完全垂直向下）。`
            : '')
          + (s.wristRollDeg !== undefined && Array.isArray(s.palmNormal)
            ? `<br>右手腕绕前臂轴滚 <b>${s.wristRollDeg}°</b>：掌心法向 `
              + `(${s.palmNormal.map((v) => v.toFixed(2)).join(', ')})、`
              + `四指 (${s.fingerDir.map((v) => v.toFixed(2)).join(', ')})，`
              + '即<b>掌心朝后、手背朝前</b>的立掌（掌缘朝前、四指并拢斜指太阳穴），'
              + '不是手心正对镜头。'
            : '')
          + (s.wristToHeadMm && s.wristRightCm === undefined
            ? `敬礼手到头中心 ${s.wristToHeadMm} mm（IK 误差 ${s.wristErrMm} mm）。` : '')
          + (s.kneeSupportDeg ? `<br>双腿平均膝屈：支撑 <b>${s.kneeSupportDeg}°</b> / `
            + `摆动 <b>${s.kneeSwingDeg}°</b>（腿基本笔直）；基座起伏 ${s.hipRiseCm} cm，`
            + '这是「腿直 + 脚贴地 + 大步幅」三者兼得的必然结果。' : '')
          + (s.stanceBackDeg ? `<br>支撑脚放在质心后方 ${s.stanceBackCm} cm，`
            + `支撑腿大腿随之后倾 <b>${s.stanceBackDeg}°</b>（膝在髋后），`
            + `前摆峰值大臂前摆 ${s.armFwdDeg}°、小臂水平横向 `
            + `(横向占比 ${s.armMedial})。` : '')
          + (s.leanDeg ? `<br>正步上体<b>前倾 ${s.leanDeg}°</b>，按 FK 实测质心前移 `
            + `${s.balanceShiftCm} cm（收敛残差 ${s.comResidualMm} mm）；`
            + '重心投影落在前脚掌上，是行进中的动态平衡姿势。' : '')
          : '')
        + (motion.meta.loop
          ? '<br>循环回放：首尾同相位，播完自动从起点继续。'
          : '<br>单段回放：播完<b>冻结在末帧</b>，想重看请点「从头播放」。')
      : motion.kind === 'crawl'
      ? '参考动作：<code>launch/prone_crawl.py</code> 逐帧 IK：站立 → 蹲下 → '
        + '<b>双手撑地</b> → 双脚向后伸 → 趴姿，再接 <b>4 个循环</b>的俯卧向前爬行；'
        + '由 <code>scripts/build_prone_crawl_sequence.py</code> 拼成一条连续序列'
        + (s ? `<br>过渡 ${s.standupFrames} 帧 + 爬行 ${s.gaitFrames} 帧 = ${s.frames} 帧，`
          + `接缝 f${s.seamFrame}（qpos 差 ${s.seamMaxDiffRad} rad），`
          + `X 向推进 ${s.travelX} m @ ${s.speed} m/s。`
          + '<br><b>手先撑地、脚再后伸</b>，后伸期双手全程钉住当支点；'
          + '下蹲段已做腿部电机抖动平滑。'
          + `<br>已知视觉遗留：撑地瞬间手 mesh 最低 ${s.visualMinZCm} cm`
          + '（拳头陷入地面），源轨迹与 sim 端一致，碰撞代理是腕部球，不影响动作本身。' : '')
        + '<br>单段回放：播完<b>冻结在趴姿</b>，想重看请点「从头播放」。'
        + '<br>建议切「单人」场景观看（矩阵 8×12 的机位太远，看不清贴地细节）。'
      : motion.kind === 'squat'
      ? '参考动作：<code>logs/export/squat_{down,up}.npz</code> 的 29 关节极限蹲动作，'
        + '纯运动学装配（<b>无策略网络、不跑 mj_step</b>，所以不会摔倒）。'
        + (s ? `<br>基座 z ${s.zStand} → ${s.zSquat} m，蹲姿躯干前倾 ${s.pitchSquatDeg}°；`
          + `${s.framesDown} 帧下蹲 + ${s.framesUp} 帧起立 @ ${s.fps} fps。` : '')
        + `<br>${squatHead}`
      : motion.kind === 'crab'
        ? crabHead
          + (s ? `<br>台阶 ${s.steps} 级 · 级高 ${s.riseMin}~${s.riseMax} m · 进深 `
            + `${s.treadMin}~${s.treadMax} m · 最高 ${s.maxTop} m · 横移 ${s.travelY} m。` : '')
        : motion.kind === 'jump'
          ? '参考动作：<code>long_jump_real.py</code> 纯自驱闭环仿真'
            + '（PD + 摆臂 + 蹬腿，<b>零冲量注入</b>）录制的 29 关节 qpos，'
            + '经 <code>bake_longjump.py</code> 按关节名重排成网页 36 维逐帧回放'
            + '（<b>无策略网络、不跑 mj_step</b>，所以不会摔倒）。'
            + (s ? `<br>相对站立跳高 <b>${s.jumpHeightRelCm} cm</b>，`
              + `质心峰值 ${s.peakComZ} m；X 向推进 ${s.travelXm} m，`
              + `腾空 ${s.flightFrames} 帧（${s.flightS} s）。` : '')
            + '<br>单段回放：播完<b>冻结在末帧站姿</b>，想重看请点「从头播放」。'
          : motion.kind === 'climb'
          ? '策略仿真：本机跑 MuJoCo + <code>normal_depth.onnx</code> 深度视觉闭环录制'
            + '（躯干俯视深度相机；地形 = <code>universal_test</code> 紧凑障碍）。'
            + (s ? `<br>最远 x = ${s.maxX} m（终点线 13.2 m），躯干高度 `
              + `${s.rootZMin} ~ ${s.rootZMax} m，${s.fell ? '末段摔倒' : '全程未摔倒'}。` : '')
          : '策略仿真：本机跑 MuJoCo + <code>policy.onnx</code> 闭环录制。'
            + (s ? `跟踪误差：均值 ${s.anchorErrMean} m / 最大 ${s.anchorErrMax} m，`
              + `躯干最大倾角 ${s.tiltMax}°。` : '');
  $('motion-note').innerHTML = note
    + `<br>${motion.nframes} 帧 @ ${motion.fps} fps = ${motion.duration.toFixed(1)} s。`;
  toast(`已载入回放：${motion.label}`);
  log(`· 回放数据 ${name}：${motion.nframes} 帧 / ${motion.duration.toFixed(1)} s`);
  return motion;
}

/**
 * 机器人视觉中心（three 世界坐标），用于视角跟随 / VR 正对。
 * 注意：robotBounds() 给的是 MuJoCo 世界坐标，而 wrapZUp 那两层 Group
 * （rotation.x=-π/2 再 rotation.y=+π/2）**已经**完成了 Z-up→Y-up 的映射，
 * 所以这里直接 localToWorld，千万不要再手动换轴（会把目标点算到地面以下）。
 */
function robotCenterWorld(out) {
  const c = mjScene.robotBounds().getCenter(out);   // MuJoCo 世界坐标
  return mjScene.root.localToWorld(c);              // → three 世界坐标（含 stage 平移/旋转）
}

/** 每帧：让视角跟着机器人（桌面=平移相机，VR=交给 VRNav 舒适跟随） */
function updateMotionCamera(dt, refreshBounds = true) {
  if (refreshBounds) robotCenterWorld(_vRobot);
  if (vrNav) {
    vrNav.robotWorld.copy(_vRobot);
    if (following) vrNav.autoFollow(dt);
    if (renderer.xr.isPresenting) return;
  }
  if (!following) return;
  // 贴地爬行沿 three -Z 推进约 1.1 m，同时躯干从 1.03 m 降到 0.10 m，
  // 包围盒中心变化比其它回放大得多。默认 0.07/帧的跟随在低帧率下会明显滞后
  // （实测 8 fps 时末帧机器人被挤到画面右下并裁切），给这段单独加快到 0.25。
  const lerp = (motion && motion.kind === 'crawl') ? 0.25 : CFG.followLerp;
  _vDelta.copy(_vRobot).sub(controls.target).multiplyScalar(lerp);
  if (_vDelta.lengthSq() > 1e-10) {
    controls.target.add(_vDelta);
    camera.position.add(_vDelta);
    controls.update();
  }
}

// ---------------------------------------------------------------- 交互
function setupControls() {
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.6;
  controls.maxDistance = 40;
  controls.maxPolarAngle = Math.PI * 0.495; // 不许钻到地面以下
  controls.target.set(0.12, 0.78, 0);
  controls.update();
}

function setupXR() {
  xr = new XRControls({
    renderer,
    scene,
    slot: $('vr-slot'),
    framebufferScale: XR_SS,
    foveation: FFR,
    log,
    toast,
    onEnter: (id, info) => {
      controls.enabled = false;
      // 先降载再建面板：xrPanel.show() 会立刻 rebuild()，这样面板第一帧显示的
      // 就是实际的「单人 / 阴影关」，不会先画一帧矩阵状态再跳。
      enterXRQuality();
      const L = renderer.xr.getBaseLayer();
      const fw = L ? (L.framebufferWidth ?? L.textureWidth) : 0;
      const fh = L ? (L.framebufferHeight ?? L.textureHeight) : 0;
      refreshHudMode();
      applyUiMode(id);
      toast((id === 'ar'
        ? `半透明 VR 已开启（${info.blend}）`
        : '已进入 VR：左摇杆移动 · 右摇杆连续转向 · 扳机点击面板')
        + ` · 场景 ${sceneMode === 'matrix' ? `矩阵 ${GRID.rows}×${GRID.cols}` : '单人'}`
        + (XR_SS < 1 ? ` / ${XR_SS}×` : '')
        + (fw && fh ? ` / ${fw}×${fh}` : ''), 5200);
    },
    onExit: () => {
      controls.enabled = true;
      // 场景不还原（保留 VR 里的选择），只按当前模式给一个合适的桌面机位。
      exitXRQuality();
      applyView(sceneMode === 'matrix' ? 'grid' : 'front');
      refreshHudMode();
      applyUiMode(null);
    },
    // 透视模式：地面网格淡化，真实地面透出来；退出时完全还原
    onSeeThrough: (on) => {
      if (mjScene) mjScene.setFloorOpacity(on ? CFG.xrFloorOpacity : 1);
      renderer.shadowMap.needsUpdate = true;
    },
  });
  xr.init();

  vrNav = new VRNav({ THREE, renderer, camera, scene, stage });
  vrNav.init();
  // 手动转向 = 用户主动接管观察角度：关掉「视角跟随机器人」并同步面板复选框，
  // 与 applyView() 里点视角预设的处理一致（否则转大角度会被 autoFollow 慢慢拉回）。
  vrNav.onManualTurn = () => {
    if (!following) return;
    const fe = $('opt-follow');
    if (fe && fe.checked) { fe.checked = false; fe.dispatchEvent(new Event('change')); }
  };

  // 头显内面板：WebXR 会话里浏览器只合成 WebGL 画布、DOM 不会进虚拟画面，
  // 所以另画一块 CanvasTexture 贴图面板（见 xrpanel.js），控件仍由 #panel 的 DOM 驱动。
  xrPanel = new XRPanel({ THREE, renderer, scene, camera, vrNav, panelEl: $('panel'), xr });
  xrPanel.init();
}

/**
 * XR 会话期间的界面形态：2D 面板进精简模式（只去说明段落），
 * 头显里则显示贴图面板。退出会话全部还原。
 * @param {'vr'|'ar'|null} id
 */
function applyUiMode(id) {
  const on = !!id;
  const panel = $('panel');
  if (panel) panel.classList.toggle('compact', on);
  if (!xrPanel) return;
  if (on) xrPanel.show(); else xrPanel.hide();
}

/** HUD「状态」栏：XR 会话 > 回放 > 本地物理 > 静止站立（优先级从高到低） */
function refreshHudMode() {
  const el = $('hud-mode');
  if (!el) return;
  const mode = xr ? xr.active : null;
  if (mode === 'ar') el.textContent = '半透明 VR（透视）';
  else if (mode === 'vr') el.textContent = 'VR 沉浸观看';
  else if (motion) el.textContent = (motion.kind === 'squat' || motion.kind === 'crawl')
    ? '参考动作回放' : '运动仿真回放';
  else if (simulating) el.textContent = '物理运行（无控制器）';
  else el.textContent = '静止站立';
}

function setupUI() {
  // 场景：单人 / 矩阵（默认矩阵）
  for (const el of document.querySelectorAll('input[name="opt-scene"]')) {
    el.addEventListener('change', () => { if (el.checked) setSceneMode(el.value); });
  }
  refreshSceneHud();

  // 运动仿真回放
  $('opt-motion').addEventListener('change', () => { setMotion($('opt-motion').value); });
  $('opt-motion-speed').addEventListener('change', () => {
    if (motion) motion.speed = Number($('opt-motion-speed').value) || 1;
  });
  $('opt-follow').addEventListener('change', () => {
    following = $('opt-follow').checked;
    if (vrNav) vrNav.follow = following;
    toast(following ? '视角会跟着机器人走' : '已关闭视角跟随');
  });
  $('btn-motion-restart').addEventListener('click', () => {
    if (!motion) { toast('先在「回放」里选一段运动数据'); return; }
    motion.restart();
    motion.paused = false;
    motionPausedUi = false;
    $('btn-motion-pause').textContent = '暂停';
    toast('已从头播放');
  });
  $('btn-motion-pause').addEventListener('click', () => {
    if (!motion) { toast('先在「回放」里选一段运动数据'); return; }
    // 单段蹲起播完会冻结末帧；此时「继续」等价于从头播放，否则点了也不动。
    const frozen = !motion.loop && motion.paused
      && motion.time >= motion.duration - 1e-6;
    if (frozen) {
      motion.restart();
      motion.paused = false;
      toast('已从头播放');
    } else {
      motion.paused = !motion.paused;
      toast(motion.paused ? '回放已暂停' : '继续播放');
    }
    motionPausedUi = motion.paused;
    $('btn-motion-pause').textContent = motionPausedUi ? '继续' : '暂停';
  });

  // 显示开关
  const bind = (id, key) => {
    const el = $(id);
    el.addEventListener('change', () => {
      mjScene.setOption(key, el.checked);
      renderer.shadowMap.needsUpdate = true;
    });
  };
  bind('opt-robot', 'showRobot');
  bind('opt-stairs', 'showStairs');
  bind('opt-crab', 'showCrab');
  bind('opt-floor', 'showFloor');
  bind('opt-collision', 'showCollision');

  const shadowEl = $('opt-shadow');
  shadowEl.addEventListener('change', () => {
    setShadows(shadowEl.checked, { syncUi: false });
  });

  // 视角预设
  $('views').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-view]');
    if (btn) { applyView(btn.dataset.view); toast(`视角：${btn.textContent}`); }
  });

  // 物理开关
  const simEl = $('opt-sim');
  simEl.addEventListener('change', () => {
    simulating = simEl.checked;
    physicsAcc = 0;
    if (simulating && motion) setMotion('off');      // 与回放互斥
    syncShadowAutoUpdate();                          // 机器人动起来后阴影要跟着更新
    if (!simulating && standQpos) {
      // 关掉物理时回到静止站立姿态
      data.qpos.set(standQpos);
      mujoco.mj_forward(model, data);
      mjScene.sync();
    }
    refreshHudMode();
    toast(simulating ? '已开启 MuJoCo 物理：机器人会因重力倒地' : '已回到静止站立姿态');
  });

  $('btn-reset-pose').addEventListener('click', () => {
    if (motion) {                                    // 回放中 → 回到视频开头
      motion.restart();
      motion.paused = false;
      $('btn-motion-pause').textContent = '暂停';
      toast('回放已从头开始');
      return;
    }
    try {
      mujoco.mj_resetData(model, data);
      if (standQpos) data.qpos.set(standQpos);
    } catch {
      data.qpos.set(standQpos || model.qpos0);
    }
    mujoco.mj_forward(model, data);
    mjScene.sync();
    renderer.shadowMap.needsUpdate = true;
    toast('已复位到初始站立姿态');
  });

  $('btn-reset-stage').addEventListener('click', () => {
    if (vrNav) vrNav.resetStage();
    toast('世界位置已复位');
  });

  $('panel-fold').addEventListener('click', () => $('panel').classList.toggle('folded'));

  // HUD 信息
  $('hud-model').textContent =
    `${model.nbody} 刚体 · ${model.ngeom} 几何 · ${mjScene.stats.triangles.toLocaleString()} 面`;
}

function onResize() {
  const w = window.innerWidth, h = window.innerHeight;
  // XR 会话里帧缓冲尺寸是 three 按 glProjLayer.textureWidth/Height（或 XRWebGLLayer）
  // 设好的；这里再 setSize 会把它改回窗口尺寸 —— 投影层路径上就是视口错位/黑屏。
  if (renderer.xr.isPresenting) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

/**
 * XR 现场体检：把**左右眼各自的视口 + 机器人包围盒在那一只眼里的 NDC 范围**打出来。
 *
 * 为什么需要它：three 的三目（stereo）绘制列表是**每帧一次、左右眼共用**的 ——
 * `projectObject()` 只用 `cameraXR` 的联合视锥裁一次，之后 `renderScene()` 才按眼
 * 分视口画。所以「只有一只眼丢物体」**不可能是剔除**，只可能是：
 *   (a) 那只眼的视口 / 投影矩阵不对  → 下面两眼的 ndc 范围会明显不一样，或整段落在 ±1 之外；
 *   (b) 几何两眼里都在，丢的是**片元**   → 两行 ndc 几乎一致，那就是降采样 / FFR 把
 *       细小几何的采样点漏掉了（把 foveation 调 0 复测即可确认）。
 *
 * 头显用 chrome://inspect 连上后，在 console 里敲 `__elf.xrDiag()`（会话内、退出前都行）。
 */
function xrDiag() {
  const camXR = renderer.xr.getCamera();
  const L = renderer.xr.getBaseLayer ? renderer.xr.getBaseLayer() : null;
  const box = mjScene ? mjScene.robotBounds() : null;   // MuJoCo 世界坐标
  const pts = [];
  if (box && !box.isEmpty()) {
    for (let c = 0; c < 8; c++) {
      // 8 个角点 → three 世界坐标（wrapZUp 那两层已经做完 Z-up→Y-up）
      pts.push(mjScene.root.localToWorld(new THREE.Vector3(
        c & 1 ? box.max.x : box.min.x,
        c & 2 ? box.max.y : box.min.y,
        c & 4 ? box.max.z : box.min.z,
      )));
    }
  }
  const eyes = (camXR.cameras || []).map((cam, i) => {
    let x0 = Infinity; let x1 = -Infinity; let y0 = Infinity; let y1 = -Infinity;
    let offNear = 0;
    for (const p of pts) {
      const q = p.clone().project(cam);       // → NDC
      if (!(q.z >= -1 && q.z <= 1)) offNear++; // 近平面之后 project() 没有意义，单独计数
      x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x);
      y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
    }
    return {
      eye: i === 0 ? 'L' : (i === 1 ? 'R' : `#${i}`),
      viewport: cam.viewport
        ? `${cam.viewport.x},${cam.viewport.y} ${cam.viewport.width}×${cam.viewport.height}` : '—',
      'ndc x': pts.length ? `${x0.toFixed(2)} .. ${x1.toFixed(2)}` : '—',
      'ndc y': pts.length ? `${y0.toFixed(2)} .. ${y1.toFixed(2)}` : '—',
      inFrustum: pts.length && x1 > -1 && x0 < 1 && y1 > -1 && y0 < 1 ? 'yes' : 'NO',
      offNear,
    };
  });
  const info = {
    presenting: renderer.xr.isPresenting,
    mode: xr ? xr.active : null,
    // 非会话时 getFoveation() 返回 undefined（还没有 baseLayer）→ 回落到现场值 / 启动参数
    foveation: (renderer.xr.getFoveation ? renderer.xr.getFoveation() : undefined)
      ?? (xr ? xr.foveation : FFR),
    framebuffer: L ? `${L.framebufferWidth ?? L.textureWidth}×${L.framebufferHeight ?? L.textureHeight}` : '—',
    xrss: XR_SS,
    aa: AA,
    sceneMode,
    shadows: { enabled: renderer.shadowMap.enabled, cast: !!keyLight?.castShadow },
    calls: renderer.info.render.calls,
    triangles: renderer.info.render.triangles,
  };
  console.log('[xrDiag]', info);
  console.table(eyes);
  return { info, eyes };
}

// ---------------------------------------------------------------- 主循环
/**
 * @param {number} [time] animation-frame 的 timestamp（ms）。WebXR 与桌面 rAF 都会传。
 *
 * 为什么不用 `clock.getDelta()`：它算的是「上一个回调结束 → 现在」，把本帧自身的耗时
 * 也算进去，掉帧时会正反馈放大抖动；而且换基准点（进/出 VR、标签页切回来）会吐一个
 * 巨大的 delta。改用绝对 timestamp 求差，进出会话时重新锚定即可。
 */
function frame(time) {
  const now = (typeof time === 'number' && time > 0) ? time : performance.now();
  let dt = lastFrameTime ? (now - lastFrameTime) / 1000 : 1 / 72;
  lastFrameTime = now;
  if (!(dt > 0)) dt = 1 / 72;                  // 首帧 / 时间戳回退
  else if (dt > 0.1) dt = 0.1;                 // 长卡顿限幅，别一次跳过一整段
  // 平滑值只喂「跟随 / 阻尼」；播放时间必须用原始 dt，否则持续掉帧会被 EMA 拖慢
  dtSmooth += (dt - dtSmooth) * 0.2;
  _ft[_ftIdx] = dt * 1000;
  _ftIdx = (_ftIdx + 1) % FT_N;
  if (_ftCount < FT_N) _ftCount++;

  // 矩阵模式：按头显视锥逐实例剔除。
  // InstancedMesh 的包围球罩住整片网格，three 只能「整体画 / 整体不画」，所以
  // 视野外那几十份必须自己筛掉 —— 这是矩阵模式下唯一够大的无损收益。
  if (CULL_GRID && renderer.xr.isPresenting && mjScene && mjScene.grid) {
    mjScene.cullGrid(renderer.xr.getCamera(), _vRobot, GRID_CULL_R);
  }

  let synced = false;

  if (motion) {
    // 运动仿真回放：按时间插值写回 qpos → 刷新几何位姿 → 同步 three.js
    motion.advance(dt);
    if (motion.paused !== motionPausedUi) {       // 单段冻结末帧时自动改按钮文字
      motionPausedUi = motion.paused;
      setText($('btn-motion-pause'), motionPausedUi ? '继续' : '暂停');
    }
    // 只有「回放时刻真的变了」才重算：暂停 / 播完冻结时整条
    // qpos → mj_kinematics → 矩阵同步 的链子直接跳过。
    // 注意用 lastWrittenTime 比较，不能用 advance() 前后 time 差 ——
    // 自检里有 m.paused=true; m.seek(2.20) 这种「暂停下外部改 time」的用法。
    if (motion.time !== lastWrittenTime || needSync || mjScene._dirty) {
      motion.writeQpos(data.qpos);
      kinematics(model, data);
      mjScene.sync();
      lastWrittenTime = motion.time;
      needSync = false;
      synced = true;
    }
    boundsAcc += dt;
    // robotBounds 要遍历 32 个网格 × 8 个角点，VR 帧预算里没必要每帧算；
    // 桌面保持每帧（视角跟随要顺滑）。
    const due = !inXR || boundsAcc >= 1 / 15;
    if (due) boundsAcc = 0;
    updateMotionCamera(dtSmooth, due);
    motionHudAcc += dt;
    if (motionHudAcc >= 0.2) {
      motionHudAcc = 0;
      // 沉浸态浏览器只合成 WebGL 画布，DOM 根本看不见；写 textContent 只会标脏
      // 样式/布局，白占 XR 帧预算
      if (!renderer.xr.isPresenting) setText($('hud-motion'), motion.describe());
    }
  } else if (simulating && model && data) {
    physicsAcc += dt;
    let steps = 0;
    while (physicsAcc >= timestep && steps < CFG.maxPhysicsStepsPerFrame) {
      mujoco.mj_step(model, data);
      physicsAcc -= timestep;
      steps++;
    }
    if (steps > 0) { mjScene.sync(); synced = true; }
    if (physicsAcc > timestep * CFG.maxPhysicsStepsPerFrame) physicsAcc = 0;
  }

  // 静止站立 / 暂停状态下改显示开关、切场景、复位姿态等一次性变化：补一次同步
  if (!synced && (needSync || mjScene._dirty)) {
    mjScene.sync();
    needSync = false;
  }

  if (vrNav) vrNav.update(dt);
  // 头显内面板：头锁摆位、悬停、DOM 状态轮询。
  // three 的动画循环是「先回调、后 requestAnimationFrame」，任何一帧抛异常都会永久
  // 停掉渲染（头显里就是全黑）。这里熔断：面板坏掉只丢面板，场景照常渲染。
  if (xrPanel) {
    try { xrPanel.update(dt); } catch (e) {
      console.error('xrPanel 更新失败，已禁用头显面板', e);
      xrPanel = null;
    }
  }
  if (controls && controls.enabled && !renderer.xr.isPresenting) controls.update();

  renderer.render(scene, camera);

  // FPS / 渲染统计
  fpsAcc += dt; fpsFrames++;
  if (fpsAcc >= 0.5) {
    fps = fpsFrames / fpsAcc;
    fpsAcc = 0; fpsFrames = 0;
    if (!renderer.xr.isPresenting) {
      const r = renderer.info.render;
      setText($('hud-render'),
        `${fps.toFixed(0)} fps · ${(r.triangles / 1000).toFixed(0)}k 三角 · ${r.calls} 批次`);
    }
  }
}

boot().catch(fail);
