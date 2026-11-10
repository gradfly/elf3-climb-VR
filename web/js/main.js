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
import { MotionPlayer } from './motion.js';
import { XRControls } from './xr.js';

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
const GRID = { rows: 10, cols: 20, pitchX: 0.72, pitchZ: 0.78 };

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
let renderer, scene, camera, controls, stage, mjScene, vrNav, xr, mujoco, model, data;
let clock, simulating = false, physicsAcc = 0, timestep = CFG.defaultTimestep;
let fpsAcc = 0, fpsFrames = 0, fps = 0;
let standQpos = null; // 落地对齐后的静止站立姿态（复位用）
let kinematics = null; // 回放用的前向计算（优先 mj_kinematics，比 mj_forward 省掉碰撞检测）
let motion = null;     // 当前回放（MotionPlayer）
let motionName = 'off';
let following = true;  // 视角是否跟随机器人
let sceneMode = 'matrix';     // 场景：'single' 单人 / 'matrix' 10×20 矩阵（默认矩阵）
let terrainWanted = { stairs: true, crab: false };  // 当前回放需要的地形（矩阵模式下暂存，退出时恢复）
let motionHudAcc = 0;
let motionPausedUi = false;   // 暂停按钮当前显示的状态（冻结末帧时自动同步）
const _vRobot = new THREE.Vector3();
const _vDelta = new THREE.Vector3();

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
  // 矩阵场景：10 行 × 20 列铺开约 14 m × 7 m，机位要退到十几米外并俯视 10°，
  // 否则前排会把后排全挡住；8.7/9.9 这组刚好把整片四角都收进画面（再近就会裁掉
  // 近端两角）。目标点抬到 0.4 m，避免整片机器人挤在画面下半部。
  grid:  { pos: [8.0, 1.40, -7], target: [0, 0.40, 0] },
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
  if (!v || !controls) return;
  camera.position.set(...v.pos);
  controls.target.set(...v.target);
  controls.update();
}

// ---------------------------------------------------------------- 启动
async function boot() {
  setProgress(2, '创建 WebGL 渲染器');

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
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

  const rim = new THREE.DirectionalLight(0x7fb6ff, 0.7);
  rim.position.set(-4.5, 3.2, -5.0);
  stage.add(rim);

  clock = new THREE.Clock();

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

  // 场景：默认矩阵（10×20 = 200 份同步播放），URL 可用 ?scene=single 覆盖
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
    THREE, renderer, scene, camera, controls, stage, mjScene, vrNav, xr,
    mujoco, model, data, applyView, setMotion, restoreStandPose,
    setSceneMode, setTerrain, applyMotionView,
    stats: () => mjScene && mjScene.stats,
    get simulating() { return simulating; },
    get following() { return following; },
    get sceneMode() { return sceneMode; },
    get grid() { return mjScene ? mjScene.grid : null; },
    get xrMode() { return xr ? xr.active : null; },
    get seeThrough() { return xr ? xr.through : false; },
    xrCapabilities: () => (xr ? xr.capabilities() : null),
    get kinematicsName() { return kinematics === mujoco.mj_forward ? 'mj_forward' : 'mj_kinematics'; },
    get motion() { return motion; },          // 调试/自检：可直接快进、查 loop 状态
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

/** 什么时候需要每帧重算阴影贴图：跑物理、或单人模式下有回放 */
function syncShadowAutoUpdate() {
  renderer.shadowMap.autoUpdate = simulating || (!!motion && sceneMode === 'single');
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
 *   matrix —— 把机器人视觉网格实例化成 10 行 × 20 列 = 200 份，同步播放同一段动画。
 *
 * 矩阵用 InstancedMesh 实现（见 MjScene.enableRobotGrid）：几何 / 材质只有一份，
 * 绘制批次仍等于机器人几何数，200 份也不会掉帧；地形与阴影不参与复制。
 */
function setSceneMode(mode, { silent = false } = {}) {
  if (!mjScene) return;
  const m = mode === 'single' ? 'single' : 'matrix';
  sceneMode = m;

  for (const el of document.querySelectorAll('input[name="opt-scene"]')) {
    el.checked = el.value === m;
    el.closest('label')?.classList.toggle('on', el.checked);
  }

  let info = null;
  if (m === 'matrix') {
    info = mjScene.enableRobotGrid(GRID);
    setTerrain({ stairs: false, crab: false });   // 地形不复制，直接藏起来
    applyView('grid');
  } else {
    mjScene.disableRobotGrid();
    setTerrain(terrainWanted);
    applyMotionView();
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
    note.innerHTML = sceneMode === 'matrix'
      ? '矩阵 = 把单人场景复制 <b>10 行 × 20 列 = 200 份</b>，同一段动画同步播放；地形 / 阴影不复制。'
        + '<br>实例化渲染：绘制批次仍是 30 出头（不是 32×200），代价是三角形 ×200'
        + '（约 4100 万面/帧）—— 独立显卡流畅，一体机 / 集显建议切回单人。'
      : '单人 = 只有一个机器人。切到矩阵会把机器人复制 200 份同步播放（默认）。';
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
    if (player.nq !== data.qpos.length) {
      throw new Error(`数据 nq=${player.nq} 与模型 nq=${data.qpos.length} 不一致`);
    }
  } catch (e) {
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

  motion = player;
  motion.loop = player.meta.loop !== false;   // 单段蹲起播完冻结末帧
  motion.speed = Number($('opt-motion-speed').value) || 1;
  motion.restart();
  motionPausedUi = false;
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
        + '<br>建议切「单人」场景观看（矩阵 10×20 的机位太远，看不清贴地细节）。'
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
function updateMotionCamera(dt) {
  const c = robotCenterWorld(_vRobot);
  if (vrNav) {
    vrNav.robotWorld.copy(c);
    if (following) vrNav.autoFollow(dt);
    if (renderer.xr.isPresenting) return;
  }
  if (!following) return;
  // 贴地爬行沿 three -Z 推进约 1.1 m，同时躯干从 1.03 m 降到 0.10 m，
  // 包围盒中心变化比其它回放大得多。默认 0.07/帧的跟随在低帧率下会明显滞后
  // （实测 8 fps 时末帧机器人被挤到画面右下并裁切），给这段单独加快到 0.25。
  const lerp = (motion && motion.kind === 'crawl') ? 0.25 : CFG.followLerp;
  _vDelta.copy(c).sub(controls.target).multiplyScalar(lerp);
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
    log,
    toast,
    onEnter: (id, info) => {
      controls.enabled = false;
      renderer.shadowMap.needsUpdate = true;
      refreshHudMode();
      toast(id === 'ar'
        ? `半透明 VR 已开启：背景透明（${info.blend}），能看到眼镜外的真实环境`
        : '已进入 VR：左摇杆移动 · 右摇杆 30° 转向 · A/X 正对机器人', 4600);
    },
    onExit: () => {
      controls.enabled = true;
      renderer.shadowMap.needsUpdate = true;
      applyView('front');
      refreshHudMode();
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
    renderer.shadowMap.enabled = shadowEl.checked;
    renderer.shadowMap.needsUpdate = true;
    mjScene.applyOptions();
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
  if (!renderer.xr.isPresenting) {
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  renderer.setSize(w, h, false);
}

// ---------------------------------------------------------------- 主循环
function frame() {
  const dt = Math.min(clock.getDelta(), 0.12);

  if (motion) {
    // 运动仿真回放：按时间插值写回 qpos → 刷新几何位姿 → 同步 three.js
    motion.advance(dt);
    if (motion.paused !== motionPausedUi) {       // 单段冻结末帧时自动改按钮文字
      motionPausedUi = motion.paused;
      $('btn-motion-pause').textContent = motionPausedUi ? '继续' : '暂停';
    }
    motion.writeQpos(data.qpos);
    kinematics(model, data);
    mjScene.sync();
    updateMotionCamera(dt);
    motionHudAcc += dt;
    if (motionHudAcc >= 0.2) {
      motionHudAcc = 0;
      $('hud-motion').textContent = motion.describe();
    }
  } else if (simulating && model && data) {
    physicsAcc += dt;
    let steps = 0;
    while (physicsAcc >= timestep && steps < CFG.maxPhysicsStepsPerFrame) {
      mujoco.mj_step(model, data);
      physicsAcc -= timestep;
      steps++;
    }
    if (steps > 0) mjScene.sync();
    if (physicsAcc > timestep * CFG.maxPhysicsStepsPerFrame) physicsAcc = 0;
  }

  if (vrNav) vrNav.update(dt);
  if (controls && controls.enabled && !renderer.xr.isPresenting) controls.update();

  renderer.render(scene, camera);

  // FPS / 渲染统计
  fpsAcc += dt; fpsFrames++;
  if (fpsAcc >= 0.5) {
    fps = fpsFrames / fpsAcc;
    fpsAcc = 0; fpsFrames = 0;
    const r = renderer.info.render;
    $('hud-render').textContent =
      `${fps.toFixed(0)} fps · ${(r.triangles / 1000).toFixed(0)}k 三角 · ${r.calls} 批次`;
  }
}

boot().catch(fail);
