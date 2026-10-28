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
let motionHudAcc = 0;
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
  wide:  { pos: [7.20, 4.60, 10.5], target: [1.6, 0.90, 0] },
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

  await applyMotionFromQuery();     // 默认不开回放；URL 带 ?motion=… 时自动进入

  renderer.shadowMap.needsUpdate = true;
  renderer.setAnimationLoop(frame);

  // 调试句柄：控制台里可直接 __elf.applyView('front') / __elf.camera / __elf.mjScene
  window.__elf = {
    THREE, renderer, scene, camera, controls, stage, mjScene, vrNav, xr,
    mujoco, model, data, applyView, setMotion, restoreStandPose,
    stats: () => mjScene && mjScene.stats,
    get simulating() { return simulating; },
    get following() { return following; },
    get xrMode() { return xr ? xr.active : null; },
    get seeThrough() { return xr ? xr.through : false; },
    xrCapabilities: () => (xr ? xr.capabilities() : null),
    get kinematicsName() { return kinematics === mujoco.mj_forward ? 'mj_forward' : 'mj_kinematics'; },
    getMotion: () => (motion ? {
      name: motionName, kind: motion.kind, label: motion.label,
      nframes: motion.nframes, nq: motion.nq, fps: motion.fps,
      duration: +motion.duration.toFixed(3), time: +motion.time.toFixed(3),
      speed: motion.speed, paused: motion.paused,
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
/** URL 参数：?motion=<off|fight_policy|fight_ref|climb_policy|crab|1> &speed=0.5 &follow=0/1 */
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
 * 切换回放数据（'off' = 关闭）。
 * 回放与本地 mj_step 物理互斥：开回放会自动关掉物理开关。
 */
async function setMotion(name, { hideStairs = true } = {}) {
  const sel = $('opt-motion');
  if (sel && sel.value !== name) sel.value = name;
  motionName = name;

  if (name === 'off' || !name) {
    motion = null;
    restoreStandPose();
    setTerrain({ stairs: true, crab: false });      // 回到默认：只显示紧凑障碍
    $('hud-motion').textContent = '—';
    refreshHudMode();
    renderer.shadowMap.autoUpdate = simulating;
    $('motion-note').innerHTML =
      '回放数据由 <code>tools/bake_motion.py</code>（打拳）/ '
      + '<code>tools/bake_climb.py</code>（爬楼梯）/ '
      + '<code>tools/bake_crab.py</code>（螃蟹步横移）在本机跑一遍仿真后录制。';
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
      + '先在项目根目录跑 <code>python web/tools/bake_motion.py</code> 生成回放数据。';
    toast(`回放数据加载失败：${e.message}`, 4200);
    return null;
  }

  motion = player;
  motion.speed = Number($('opt-motion-speed').value) || 1;
  motion.restart();
  $('btn-motion-pause').textContent = '暂停';

  // 两套地形共用同一个编译好的模型：紧凑障碍沿 +X（爬楼梯策略用的），
  // 螃蟹步台阶沿 +Y（脚本步态用的）。回放哪一段就只显示那一段 ——
  // 否则另一套地形会出现在画面里，看着像"走错路"，也白白拉低帧率。
  //   打拳 / 参考动作：平地录的，默认两套都藏起来（hideStairs=false 时保留障碍）
  //   爬楼梯策略：显示紧凑障碍
  //   螃蟹步横移：显示 crab_ 台阶
  const isCrab = motion.kind === 'crab';
  setTerrain({
    stairs: motion.kind === 'climb' || (!isCrab && !hideStairs),
    crab: isCrab,
  });
  if (isCrab) applyView('crab');      // 换成能看清台阶剖面的侧视机位

  renderer.shadowMap.autoUpdate = true;
  renderer.shadowMap.needsUpdate = true;
  refreshHudMode();
  $('hud-motion').textContent = motion.describe();

  const s = motion.meta.stats;
  const note = motion.kind === 'ref'
    ? '参考动作：<code>fight.npz</code> 里重定向后的运动，纯运动学装配（无物理）。'
    : motion.kind === 'crab'
      ? '脚本步态：<code>crab_climb.py</code> 的四相位准静态控制器解出的关节轨迹'
        + '（闭式 IK + DLS 精修，<b>没有策略网络</b>）。机器人朝 +X，沿 +Y 侧向横移上楼。'
        + (s ? `<br>台阶 ${s.steps} 级 · 级高 ${s.riseMin}~${s.riseMax} m · 进深 `
          + `${s.treadMin}~${s.treadMax} m · 最高 ${s.maxTop} m · 横移 ${s.travelY} m。` : '')
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
  _vDelta.copy(c).sub(controls.target).multiplyScalar(CFG.followLerp);
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
  else if (motion) el.textContent = '运动仿真回放';
  else if (simulating) el.textContent = '物理运行（无控制器）';
  else el.textContent = '静止站立';
}

function setupUI() {
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
    $('btn-motion-pause').textContent = '暂停';
    toast('已从头播放');
  });
  $('btn-motion-pause').addEventListener('click', () => {
    if (!motion) { toast('先在「回放」里选一段运动数据'); return; }
    motion.paused = !motion.paused;
    $('btn-motion-pause').textContent = motion.paused ? '继续' : '暂停';
    toast(motion.paused ? '回放已暂停' : '继续播放');
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
    renderer.shadowMap.autoUpdate = simulating || !!motion; // 机器人动起来后阴影要跟着更新
    renderer.shadowMap.needsUpdate = true;
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
