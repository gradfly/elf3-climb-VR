// 端到端验证：加载 → 场景 → UI → 物理开关 → 复位 → VR 入口
// 用法: node tools/verify.mjs [url]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

const CHROME = process.env.CHROME
  || (fs.existsSync('C:/Program Files/Google/Chrome/Application/chrome.exe')
    ? 'C:/Program Files/Google/Chrome/Application/chrome.exe'
    : 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe');
const url = process.argv[2] || 'http://localhost:8090/';
const PORT = 9400 + Math.floor(Math.random() * 300);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'cdp-verify-'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--no-sandbox', '--no-first-run', '--disable-extensions', '--window-size=1600,1000',
  ...(url.startsWith('https:') ? ['--ignore-certificate-errors'] : []),
  'about:blank',
], { stdio: 'ignore' });

let id = 0;
function rpc(ws, method, params = {}) {
  return new Promise((resolve, reject) => {
    const mid = ++id;
    const onMsg = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      if (m.id === mid) {
        ws.removeEventListener('message', onMsg);
        m.error ? reject(new Error(method + ': ' + JSON.stringify(m.error))) : resolve(m.result);
      }
    };
    ws.addEventListener('message', onMsg);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
}
const evalJs = async (ws, expr) => {
  const r = await rpc(ws, 'Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
  return r.result?.value;
};

const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? '  [OK]  ' : '  [FAIL]'} ${name}${detail ? ' — ' + detail : ''}`);
};

try {
  for (let i = 0; i < 100; i++) {
    try { if ((await fetch(`http://127.0.0.1:${PORT}/json/version`)).ok) break; } catch { /* wait */ }
    await sleep(200);
  }
  const t = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  const errors = [];
  ws.addEventListener('message', (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.method === 'Runtime.exceptionThrown') {
      errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    }
    if (m.method === 'Runtime.consoleAPICalled' && (m.params.type === 'error' || m.params.type === 'warning')) {
      errors.push('console.' + m.params.type + ': ' + m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
    }
    if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') {
      errors.push('log: ' + m.params.entry.text);
    }
  });
  await rpc(ws, 'Runtime.enable');
  await rpc(ws, 'Log.enable');
  await rpc(ws, 'Page.enable');

  const t0 = Date.now();
  await rpc(ws, 'Page.navigate', { url });
  let boot = null;
  for (let i = 0; i < 160; i++) {
    await sleep(500);
    boot = await evalJs(ws, `(()=>{try{return {ok:!!window.__elf, hud:!document.getElementById('hud').classList.contains('hidden'), step:document.getElementById('ld-step').textContent, hint:document.getElementById('ld-hint').textContent, err:document.getElementById('loading').classList.contains('error')}}catch(e){return {fatal:String(e)}}})()`);
    if (boot.ok || boot.err) break;
  }
  console.log(`\n加载用时 ≈ ${((Date.now() - t0) / 1000).toFixed(1)} s\n`);
  check('页面完成初始化（无启动失败）', boot.ok === true && boot.err === false, JSON.stringify(boot));
  if (!boot.ok) throw new Error('初始化失败，后续检查跳过');

  // 1. 模型与场景
  const scene = await evalJs(ws, `(()=>{const s=__elf.mjScene,m=__elf.model;
    return {nbody:m.nbody,ngeom:m.ngeom,nmesh:m.nmesh,nq:m.nq,nu:m.nu,
      robotGeoms:s.robotGeomIds.length,tris:s.stats.triangles,geoms:s.stats.geometries,
      minZ:+s.robotMinZ().toFixed(4),
      box:(()=>{const b=s.robotBounds();return [+b.min.x.toFixed(3),+b.max.x.toFixed(3),+b.min.z.toFixed(3),+b.max.z.toFixed(3)]})(),
      children:s.root.children.length};})()`);
  check('视觉网格全部建好（几何数 = 机器人网格数）',
    scene.robotGeoms === scene.geoms && scene.robotGeoms >= 30,
    `${scene.geoms} 个几何 / ${scene.tris} 三角面`);
  check('脚底对齐地面（minZ ≈ 0）', Math.abs(scene.minZ) < 0.01, `minZ = ${scene.minZ} m`);
  check('机器人高度合理（≈1.5 m）', scene.box[3] > 1.3 && scene.box[3] < 1.8, `z ∈ [${scene.box[2]}, ${scene.box[3]}]`);
  check('three.js 场景对象数与几何数一致', scene.children >= scene.ngeom, `${scene.children} 个对象`);

  // 2. HUD / UI
  const ui = await evalJs(ws, `(()=>{const $=id=>document.getElementById(id);
    return {mode:$('hud-mode').textContent.trim(),model:$('hud-model').textContent.trim(),
      hud:!$('hud').classList.contains('hidden'),panel:!$('panel').classList.contains('hidden'),
      help:!$('help').classList.contains('hidden'),loading:$('loading').classList.contains('hidden'),
      xrBtns:[...document.querySelectorAll('#vr-slot button')].map(b=>({id:b.id,text:b.textContent.trim(),cls:b.className})),
      views:[...document.querySelectorAll('#views button')].map(b=>b.dataset.view),
      opts:[...document.querySelectorAll('#panel input[type=checkbox]')].map(i=>i.id),
      log:$('ld-log').textContent.trim().split('\\n').slice(0,8)};})()`);
  check('HUD 显示静止站立', ui.mode.includes('静止站立'), ui.mode);
  check('HUD 显示模型统计', /\d+ 刚体/.test(ui.model), ui.model);
  check('加载遮罩已隐藏 / 三个面板可见', ui.loading && ui.hud && ui.panel && ui.help);
  check('两个 XR 入口按钮已创建（进入 VR / 半透明 VR）',
    ui.xrBtns.map((b) => b.id).join(',') === 'xr-vr,xr-ar', JSON.stringify(ui.xrBtns));
  check('半透明 VR 按钮带独立样式类（磨砂玻璃观感）',
    (ui.xrBtns.find((b) => b.id === 'xr-ar') || {}).cls === 'xr-btn xr-ar',
    (ui.xrBtns.find((b) => b.id === 'xr-ar') || {}).cls);
  check('9 个视角预设齐全（含矩阵全景）', ui.views.length === 9, ui.views.join(','));
  const wantOpts = ['opt-robot', 'opt-stairs', 'opt-crab', 'opt-floor', 'opt-shadow',
    'opt-collision', 'opt-sim', 'opt-follow'];
  check('显示/物理/跟随开关齐全', wantOpts.every((id) => ui.opts.includes(id)), ui.opts.join(','));
  check('运动仿真控件齐全（回放/倍速/暂停）',
    !!(await evalJs(ws, `!!(document.getElementById('opt-motion')&&document.getElementById('opt-motion-speed')&&document.getElementById('btn-motion-pause')&&document.getElementById('btn-motion-restart'))`)));
  console.log('  加载日志:'); for (const l of ui.log) console.log('    ' + l);

  // 2.5 场景模式（单人 / 矩阵）
  const scn = await evalJs(ws, `(()=>{const s=__elf.mjScene,$=id=>document.getElementById(id);
    const out={mode:__elf.sceneMode,
      checked:[...document.querySelectorAll('input[name="opt-scene"]')].filter(i=>i.checked).map(i=>i.value),
      hud:$('hud-scene').textContent.trim(),
      grid:s.grid?{rows:s.grid.rows,cols:s.grid.cols,count:s.grid.count,
        meshes:s.grid.meshes.length,perInst:s.grid.meshes[0].count}:null,
      robotGeoms:s.robotGeomIds.length,
      robotShownOff:s.objs[s.robotGeomIds[0]].visible,
      stairsOn:s.opts.showStairs, crabOn:s.opts.showCrab,
      cam:[+__elf.camera.position.x.toFixed(2),+__elf.camera.position.y.toFixed(2),
        +__elf.camera.position.z.toFixed(2)],
      camDist:+__elf.camera.position.distanceTo(__elf.controls.target).toFixed(2),
      calls:__elf.renderer.info.render.calls};
    const el=$('opt-robot'); el.checked=false; el.dispatchEvent(new Event('change'));
    out.gridHidden=s.grid.meshes.every(m=>!m.visible);
    el.checked=true; el.dispatchEvent(new Event('change'));
    out.gridVisible=s.grid.meshes.every(m=>m.visible);
    return out;})()`);
  check('默认进矩阵场景（单选按钮 matrix / HUD 显示矩阵）',
    scn.mode === 'matrix' && scn.checked.join() === 'matrix' && scn.hud.includes('矩阵'),
    `mode=${scn.mode} checked=${scn.checked} hud=${scn.hud}`);
  check('矩阵 = 8 行 × 12 列 = 96 份实例（每批 InstancedMesh 画 96 次）',
    !!scn.grid && scn.grid.rows === 8 && scn.grid.cols === 12 && scn.grid.count === 96
      && scn.grid.perInst === 96 && scn.grid.meshes === scn.robotGeoms,
    JSON.stringify(scn.grid));
  check('矩阵模式隐藏地形（地形不参与复制）',
    scn.stairsOn === false && scn.crabOn === false,
    `showStairs=${scn.stairsOn} showCrab=${scn.crabOn}`);
  check('矩阵模式原件不画（只画实例）', scn.robotShownOff === false);
  // 机位只校验「是远景」，具体数值允许调（VIEWS.grid 经常被按观感微调）
  check('矩阵机位是远景（相机离方阵中心 6 m 以上，在 OrbitControls 上限内）',
    scn.camDist > 6 && scn.camDist <= 40,
    `camera=${JSON.stringify(scn.cam)} 距 target ${scn.camDist} m`);
  check('「机器人」开关能同时控制 96 份实例', scn.gridHidden && scn.gridVisible);
  console.log(`  （矩阵模式绘制批次 ${scn.calls} 次 / 单人 30 个几何 —— 实例化没有把 draw call 乘 96）`);

  const scn2 = await evalJs(ws, `(()=>{const s=__elf.mjScene,$=id=>document.getElementById(id);
    __elf.setSceneMode('single');
    return {mode:__elf.sceneMode, gridCleared:!s.grid,
      robotVisible:s.objs[s.robotGeomIds[0]].visible,
      stairsOn:s.opts.showStairs, hud:$('hud-scene').textContent.trim(),
      radio:[...document.querySelectorAll('input[name="opt-scene"]')].filter(i=>i.checked).map(i=>i.value),
      labelOn:[...document.querySelectorAll('#scene-mode label')].filter(l=>l.classList.contains('on')).length,
      children:s.root.children.length};})()`);
  check('切回单人：实例全部销毁、原件重新显示',
    scn2.mode === 'single' && scn2.gridCleared && scn2.robotVisible === true,
    `grid=${scn2.gridCleared} robot=${scn2.robotVisible} children=${scn2.children}`);
  check('切回单人：地形恢复、单选按钮与高亮同步',
    scn2.stairsOn === true && scn2.radio.join() === 'single' && scn2.labelOn === 1,
    `showStairs=${scn2.stairsOn} radio=${scn2.radio} 高亮=${scn2.labelOn} hud=${scn2.hud}`);

  // 3. 显示开关
  const toggles = await evalJs(ws, `(async()=>{const s=__elf.mjScene,out={};
    const set=(id,v)=>{const el=document.getElementById(id);el.checked=v;el.dispatchEvent(new Event('change'));};
    set('opt-robot',false); out.robotHidden=s.objs[s.robotGeomIds[0]].visible===false;
    set('opt-robot',true);  out.robotShown=s.objs[s.robotGeomIds[0]].visible===true;
    set('opt-stairs',false);out.stairsHidden=s.envGeomIds.every(g=>!s.objs[g]||!s.objs[g].visible);
    set('opt-stairs',true);
    set('opt-floor',false); out.floorHidden=s.floorGeomIds.every(g=>!s.objs[g]||!s.objs[g].visible);
    set('opt-floor',true);
    set('opt-collision',true); out.collisionShown=[...Array(s.model.ngeom).keys()].filter(g=>s.model.geom_group[g]===3&&s.objs[g]&&s.objs[g].visible).length;
    set('opt-collision',false);
    set('opt-shadow',false); out.shadowOff=__elf.renderer.shadowMap.enabled===false&&__elf.shadows.cast===false;
    set('opt-shadow',true);  out.shadowOn =__elf.renderer.shadowMap.enabled===true &&__elf.shadows.cast===true;
    return out;})()`);
  check('“机器人”开关生效', toggles.robotHidden && toggles.robotShown);
  check('“楼梯/障碍”开关生效', toggles.stairsHidden);
  check('“地面网格”开关生效', toggles.floorHidden);
  check('“碰撞体”调试层可见', toggles.collisionShown > 0, `${toggles.collisionShown} 个碰撞几何`);
  check('“阴影”开关生效（同时翻主光 castShadow，着色器才真的换程序）',
    toggles.shadowOff && toggles.shadowOn);

  // 4. 视角预设
  const views = await evalJs(ws, `(()=>{const out={};
    for(const b of document.querySelectorAll('#views button')){
      b.click();
      const c=__elf.camera.position,t=__elf.controls.target;
      out[b.dataset.view]=[+c.x.toFixed(2),+c.y.toFixed(2),+c.z.toFixed(2),+t.y.toFixed(2)];
    } return out;})()`);
  const allMoved = Object.values(views).every((v) => v.some((x) => x !== 0));
  check('9 个视角预设均可用', Object.keys(views).length === 9 && allMoved, JSON.stringify(views));

  // 5. 物理开关 + 复位
  const phys = await evalJs(ws, `(async()=>{
    const p=__elf.data.qpos.slice(0,3);
    const stand=[p[0],p[1],p[2]];
    const el=document.getElementById('opt-sim');
    el.checked=true; el.dispatchEvent(new Event('change'));
    await new Promise(r=>setTimeout(r,2500));
    const after=Array.from(__elf.data.qpos.slice(0,3));
    el.checked=false; el.dispatchEvent(new Event('change'));
    await new Promise(r=>setTimeout(r,200));
    const back=Array.from(__elf.data.qpos.slice(0,3));
    return {stand,after,back,mode:document.getElementById('hud-mode').textContent.trim()};})()`);
  check('开启物理后机器人在运动（qpos 变化）',
    Math.abs(phys.after[2] - phys.stand[2]) > 1e-3 || Math.abs(phys.after[0] - phys.stand[0]) > 1e-3,
    `z: ${phys.stand[2].toFixed(3)} → ${phys.after[2].toFixed(3)}`);
  check('关闭物理后回到静止站立姿态',
    Math.abs(phys.back[2] - phys.stand[2]) < 1e-6, `z 回到 ${phys.back[2].toFixed(3)}，HUD=${phys.mode}`);

  const reset = await evalJs(ws, `(()=>{document.getElementById('btn-reset-pose').click();
    const a=Array.from(__elf.data.qpos.slice(0,3));
    document.getElementById('btn-reset-stage').click();
    return {qpos:a.map(v=>+v.toFixed(3)),stagePos:[__elf.stage.position.x,__elf.stage.position.z].map(v=>+v.toFixed(3)),
      toast:document.getElementById('toast').textContent};})()`);
  check('“复位姿态”按钮可用', reset.qpos.every(Number.isFinite), JSON.stringify(reset.qpos));

  // 5.5 运动仿真回放
  const mo = await evalJs(ws, `(async()=>{
    const before = { nq: __elf.data.qpos.length, stairs: __elf.mjScene.opts.showStairs,
                     mode: document.getElementById('hud-mode').textContent.trim() };
    const r = await __elf.setMotion('fight_policy');
    const s0 = __elf.getMotion();
    const q0 = Array.from(__elf.data.qpos);
    await new Promise(x=>setTimeout(x,1200));
    const q1 = Array.from(__elf.data.qpos);
    return { loaded: !!r, s0, before, kinematics: __elf.kinematicsName,
      moved: q0.some((v,i)=>Math.abs(v-q1[i])>1e-4),
      movedBy: +Math.max(...q0.map((v,i)=>Math.abs(v-q1[i]))).toFixed(4),
      stairsAfter: __elf.mjScene.opts.showStairs,
      simChecked: document.getElementById('opt-sim').checked,
      mode: document.getElementById('hud-mode').textContent.trim(),
      hud: document.getElementById('hud-motion').textContent.trim() };})()`);
  check('回放数据加载成功（帧数/时长合理）',
    mo.loaded && mo.s0.nframes > 1000 && mo.s0.nq === mo.before.nq,
    `${mo.s0.nframes} 帧 × ${mo.s0.nq} @ ${mo.s0.fps} fps = ${mo.s0.duration} s`);
  check('回放驱动 MuJoCo qpos 变化（机器人真的在动）', mo.moved, `最大变化 ${mo.movedBy}`);
  check('回放时 HUD 显示运动仿真', mo.mode.includes('运动仿真'), `${mo.mode} · ${mo.hud}`);
  check('回放时自动隐藏楼梯（录制在平地上进行）',
    mo.stairsAfter === false, `showStairs: ${mo.before.stairs} → ${mo.stairsAfter}`);
  check('回放与本地物理互斥（自动关掉 mj_step）', mo.simChecked === false);
  console.log(`  （回放前向计算用: ${mo.kinematics}｜跟踪统计: ${JSON.stringify(mo.s0.stats)}）`);

  const mo2 = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    const t0 = __elf.getMotion().time;
    $('btn-motion-pause').click();
    const p0 = Array.from(__elf.data.qpos);
    await new Promise(x=>setTimeout(x,600));
    const p1 = Array.from(__elf.data.qpos);
    const paused = p0.every((v,i)=>Math.abs(v-p1[i])<1e-9);
    $('btn-motion-restart').click();
    const tAfterRestart = __elf.getMotion().time;
    $('btn-motion-pause').click();
    const q0 = Array.from(__elf.data.qpos);
    await new Promise(x=>setTimeout(x,600));
    const moved2 = q0.some((v,i)=>Math.abs(v-__elf.data.qpos[i])>1e-4);
    __elf.setMotion('off');
    await new Promise(x=>setTimeout(x,300));
    return { t0, paused, tAfterRestart, moved2,
      off: { name: __elf.getMotion().name, mode: $('hud-mode').textContent.trim(),
             hud: $('hud-motion').textContent.trim(),
             minZ: +__elf.mjScene.robotMinZ().toFixed(4),
             z: +__elf.data.qpos[2].toFixed(3) } };})()`);
  check('暂停按钮生效（回放停住）', mo2.paused && mo2.t0 > 0, `暂停时 t=${mo2.t0}s`);
  check('从头播放按钮生效', mo2.tAfterRestart < 0.1, `t → ${mo2.tAfterRestart}`);
  check('恢复播放后继续推进', mo2.moved2);
  check('切回“关闭”回到静止站立（脚底仍对齐 z=0）',
    mo2.off.name === 'off' && mo2.off.mode.includes('静止站立') && Math.abs(mo2.off.minZ) < 0.01,
    JSON.stringify(mo2.off));

  // 5.6 「楼梯 / 障碍」里的地形 + 爬楼梯回放
  const terrain = await evalJs(ws, `(()=>{const mc=__elf.mujoco,m=__elf.model;
    const names=[...__elf.mjScene.envGeomIds].map(g=>mc.mj_id2name(m,mc.mjtObj.mjOBJ_GEOM.value,g)||'');
    return {count:names.length, samples:names.slice(0,3),
      low:names.filter(n=>n.includes('low_stair')).length,
      high:names.filter(n=>n.includes('high_stair')).length,
      trench:names.filter(n=>n.includes('trench')).length,
      oldStairs:names.filter(n=>/^step/.test(n)).length};})()`);
  check('“楼梯 / 障碍”里是 universal_test 紧凑地形（低楼梯 / 高楼梯 / 壕沟齐全）',
    terrain.low === 5 && terrain.high === 3 && terrain.trench === 3 && terrain.oldStairs === 0,
    `${terrain.count} 个地形几何：低楼梯 ${terrain.low} / 高楼梯 ${terrain.high} / 壕沟 ${terrain.trench}`);

  const climb = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    __elf.setMotion('off');
    const r = await __elf.setMotion('climb_policy');
    const s0 = __elf.getMotion();
    const q0 = Array.from(__elf.data.qpos);
    await new Promise(x=>setTimeout(x,1600));
    const q1 = Array.from(__elf.data.qpos);
    return { loaded: !!r, kind: s0.kind, nframes: s0.nframes, nq: s0.nq, stats: s0.stats,
      stairsOn: __elf.mjScene.opts.showStairs, stairsChecked: $('opt-stairs').checked,
      moved: q0.some((v,i)=>Math.abs(v-q1[i])>1e-4),
      x0: +q0[0].toFixed(2), x1: +q1[0].toFixed(2),
      mode: $('hud-mode').textContent.trim(),
      hud: $('hud-motion').textContent.trim(),
      note: $('motion-note').textContent.trim() };})()`);
  check('爬楼梯回放加载成功（kind=climb，帧数/时长合理）',
    climb.loaded && climb.kind === 'climb' && climb.nframes > 500,
    `${climb.nframes} 帧 × ${climb.nq}｜${climb.hud}`);
  check('爬楼梯回放保留地形（不自动隐藏“楼梯 / 障碍”）',
    climb.stairsOn === true && climb.stairsChecked === true,
    `showStairs=${climb.stairsOn} / 复选框=${climb.stairsChecked}`);
  check('爬楼梯回放真的在前进（x 增大）',
    climb.moved && climb.x1 > climb.x0 + 0.02, `x ${climb.x0} → ${climb.x1}`);
  check('爬楼梯回放 HUD 正常显示', climb.mode.includes('运动仿真'), `${climb.mode} · ${climb.hud}`);
  console.log(`  （爬楼梯回放统计: ${JSON.stringify(climb.stats && {
    maxX: climb.stats.maxX, fell: climb.stats.fell, rootZMax: climb.stats.rootZMax })}）`);
  console.log(`  （说明文案: ${climb.note.replace(/\s+/g, ' ').slice(0, 110)}…）`);

  const climb2 = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    __elf.setMotion('off');
    const el=$('opt-stairs'); el.checked=false; el.dispatchEvent(new Event('change'));
    const beforeHide = __elf.mjScene.opts.showStairs;
    await __elf.setMotion('climb_policy');
    const afterPick = __elf.mjScene.opts.showStairs;
    await __elf.setMotion('off');
    return { beforeHide, afterPick, checked:$('opt-stairs').checked };})()`);
  check('选“策略仿真 · 爬楼梯”会自动打开“楼梯 / 障碍”',
    climb2.beforeHide === false && climb2.afterPick === true, JSON.stringify(climb2));

  // 5.6 螃蟹步横移回放（参考动作 · 脚本烘焙，另一套地形）
  const crab = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    __elf.setMotion('off');
    const r = await __elf.setMotion('crab');
    const s0 = __elf.getMotion();
    const q0 = Array.from(__elf.data.qpos);
    await new Promise(x=>setTimeout(x,1600));
    const q1 = Array.from(__elf.data.qpos);
    const out = { loaded: !!r, kind: s0.kind, nframes: s0.nframes, stats: s0.stats,
      crabOn: __elf.mjScene.opts.showCrab, stairsOn: __elf.mjScene.opts.showStairs,
      nCrab: __elf.mjScene.crabGeomIds.length,
      movedY: +(Math.abs(q1[1] - q0[1])).toFixed(3),
      mode: $('hud-mode').textContent.trim(),
      hud: $('hud-motion').textContent.trim(),
      camZ: +__elf.camera.position.z.toFixed(2) };
    __elf.setMotion('off');
    return out;})()`);
  check('螃蟹步回放加载成功（kind=crab，帧数合理）',
    crab.loaded && crab.kind === 'crab' && crab.nframes > 500, `${crab.nframes} 帧｜${crab.hud}`);
  check('螃蟹步回放显示 crab_ 台阶、隐藏紧凑障碍',
    crab.crabOn === true && crab.stairsOn === false && crab.nCrab > 0,
    `showCrab=${crab.crabOn} showStairs=${crab.stairsOn} / ${crab.nCrab} 个台阶 geom`);
  check('螃蟹步回放真的在横移（y 增大）', crab.movedY > 0.02, `Δy=${crab.movedY} m`);
  check('螃蟹步回放自动切到“螃蟹步”机位（相机在 -Z 侧）',
    crab.camZ < -3, `camera.z=${crab.camZ}`);
  check('螃蟹步回放 HUD 显示“参考动作”',
    crab.mode.includes('运动仿真') && crab.hud.includes('参考动作'), `${crab.mode} · ${crab.hud}`);
  console.log(`  （螃蟹步回放统计: ${JSON.stringify(crab.stats && {
    steps: crab.stats.steps, maxTop: crab.stats.maxTop, travelY: crab.stats.travelY })}）`);

  // 5.7 蹲下 / 起立参考动作（纯运动学，无策略网络）
  const squat = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    __elf.setMotion('off');
    const r = await __elf.setMotion('squat_cycle');
    const s0 = __elf.getMotion();
    const m = __elf.motion;
    const first = Array.from(m.frames.slice(0, m.nq));
    const last = Array.from(m.frames.slice((m.nframes-1)*m.nq, m.nframes*m.nq));
    const seam = Math.max(...first.map((v,k)=>Math.abs(v-last[k])));
    let zMin = Infinity, zMax = -Infinity;
    for (let i=0;i<m.nframes;i++){const v=m.frames[i*m.nq+2];zMin=Math.min(zMin,v);zMax=Math.max(zMax,v);}
    const out = { loaded:!!r, kind:s0.kind, nframes:s0.nframes, loop:s0.loop,
      stats:s0.stats, seam:+seam.toFixed(6), zMin:+zMin.toFixed(3), zMax:+zMax.toFixed(3),
      crabOn:__elf.mjScene.opts.showCrab, stairsOn:__elf.mjScene.opts.showStairs,
      camX:+__elf.camera.position.x.toFixed(2), camZ:+__elf.camera.position.z.toFixed(2),
      mode:$('hud-mode').textContent.trim(), hud:$('hud-motion').textContent.trim(),
      opts:[...$('opt-motion').options].map(o=>o.value) };
    __elf.setMotion('off');
    return out;})()`);
  check('蹲起循环在下拉里，单段下蹲 / 起立已从下拉移除',
    squat.opts.includes('squat_cycle')
      && !squat.opts.includes('squat_down') && !squat.opts.includes('squat_up'),
    squat.opts.filter(v=>v.startsWith('squat')).join(', '));
  check('下拉里不再有「原地正步 / 敬礼」两个已下线选项',
    !squat.opts.includes('goose_inplace') && !squat.opts.includes('salute'),
    squat.opts.join(', '));
  check('蹲下起立循环加载成功（kind=squat，loop=true）',
    squat.loaded && squat.kind === 'squat' && squat.loop === true && squat.nframes > 300,
    `${squat.nframes} 帧｜${squat.hud}`);
  check('循环首尾帧完全同姿态（不会跳一下）', squat.seam < 1e-6,
    `max|Δ|=${squat.seam}`);
  check('循环里基座真的从站姿降到蹲姿再升回来',
    squat.zMax - squat.zMin > 0.4, `z ${squat.zMin} ~ ${squat.zMax} m`);
  check('蹲起回放隐藏两套台阶并切到 3/4 机位',
    squat.crabOn === false && squat.stairsOn === false
      && squat.camX > 1 && squat.camZ > 1,
    `stairs=${squat.stairsOn} crab=${squat.crabOn} / cam=(${squat.camX}, ${squat.camZ})`);
  check('蹲起回放 HUD 显示「参考动作」',
    squat.mode.includes('参考动作') && squat.hud.includes('参考动作'),
    `${squat.mode} · ${squat.hud}`);

  const frozen = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    const r = await __elf.setMotion('squat_down');
    const m = __elf.motion;
    m.time = m.duration - 0.01;
    await new Promise(x=>setTimeout(x,200));
    const out = { loaded:!!r, loop:m.loop, paused:m.paused,
      time:+m.time.toFixed(3), duration:+m.duration.toFixed(3), progress:+m.progress.toFixed(3),
      z:+__elf.data.qpos[2].toFixed(3), zEnd:m.frames[(m.nframes-1)*m.nq+2],
      btn:$('btn-motion-pause').textContent.trim() };
    $('btn-motion-restart').click();
    out.afterRestart = { paused:m.paused, time:+m.time.toFixed(3), btn:$('btn-motion-pause').textContent.trim() };
    __elf.setMotion('off');
    return out;})()`);
  check('单段下蹲播完冻结在末帧（loop=false，progress=1）',
    frozen.loaded && frozen.loop === false && frozen.paused === true
      && Math.abs(frozen.time - frozen.duration) < 1e-6 && frozen.progress === 1,
    `t=${frozen.time}/${frozen.duration} s · progress=${frozen.progress}`);
  check('冻结时姿态是蹲姿、按钮变「继续」',
    Math.abs(frozen.z - frozen.zEnd) < 0.01 && frozen.btn === '继续',
    `z=${frozen.z} m（末帧 ${(+frozen.zEnd).toFixed(3)}）· ${frozen.btn}`);
  check('点「从头播放」后解除冻结',
    frozen.afterRestart.paused === false && frozen.afterRestart.time === 0
      && frozen.afterRestart.btn === '暂停', JSON.stringify(frozen.afterRestart));

  // 5.6 贴地爬行：动作 1（站立→蹲下→撑地→后伸→趴姿）+ 动作 2（爬行）合并序列
  const crawlSeq = await evalJs(ws, `(async()=>{
    const r = await __elf.setMotion('crawl');
    if (!r) return { loaded:false };
    const m = __elf.motion;
    const out = {
      loaded:true, kind:m.kind, nframes:m.nframes, nq:m.nq, loop:m.loop,
      freeze:m.meta.freezeAtEnd, camera:m.meta.camera,
      cam:__elf.camera.position.toArray().map(v=>+v.toFixed(2)),
      stats:m.meta.stats,
      z0:+m.frames[2].toFixed(3),
      zEnd:+m.frames[(m.nframes-1)*m.nq+2].toFixed(3),
      opts:[...document.getElementById('opt-motion').options].map(o=>o.value),
    };
    m.paused=true; m.seek(2.20);                 // u=0.44 双手撑地
    await new Promise(res=>setTimeout(res,350));
    out.zPlant=+__elf.data.qpos[2].toFixed(3);
    __elf.setMotion('off');
    return out;})()`);
  check('贴地爬行合并序列在下拉里且加载成功',
    crawlSeq.loaded && crawlSeq.opts.includes('crawl')
      && crawlSeq.kind === 'crawl' && crawlSeq.nq === 36,
    `opts 含 crawl=${crawlSeq.opts?.includes('crawl')}｜kind=${crawlSeq.kind}`);
  check('合并序列 649 帧、播完冻结（不跳回站立）',
    crawlSeq.nframes === 649 && crawlSeq.loop === false && crawlSeq.freeze === true,
    `${crawlSeq.nframes} 帧｜loop=${crawlSeq.loop}`);
  check('序列真的从站姿走到趴姿（z 1.03 → 0.10）',
    Math.abs(crawlSeq.z0 - 1.034) < 0.01 && crawlSeq.zEnd < 0.15,
    `z0=${crawlSeq.z0} zEnd=${crawlSeq.zEnd} zPlant=${crawlSeq.zPlant}`);
  check('分段信息完整（250 帧过渡 + 399 帧爬行，接缝 f250）',
    crawlSeq.stats?.standupFrames === 250 && crawlSeq.stats?.gaitFrames === 399
      && crawlSeq.stats?.seamFrame === 250,
    JSON.stringify({standup:crawlSeq.stats?.standupFrames,
      gait:crawlSeq.stats?.gaitFrames, seam:crawlSeq.stats?.seamFrame}));
  check('贴地爬行自动切到低机位（相机在 +X / -Z 侧）',
    crawlSeq.camera === 'crawl' && crawlSeq.cam[0] > 1 && crawlSeq.cam[2] < 0,
    `camera=${crawlSeq.camera} pos=${crawlSeq.cam}`);

  // 5.8 正步 + 敬礼 连贯序列（scripts/build_parade_sequence.py 拼出）
  const paradeSeq = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    __elf.setMotion('off');
    const r = await __elf.setMotion('parade_seq');
    if (!r) return { loaded:false };
    const m = __elf.motion, nq = m.nq;
    const xs = [];
    for (let k=0;k<m.nframes;k++) xs.push(m.frames[k*nq]);
    const p = (m.meta.sequence||{}).parts||[];
    const part = (n)=>p.find(q=>q.action===n)||{};
    let maxBack = 0;
    for (let k=1;k<xs.length;k++) maxBack = Math.min(maxBack, xs[k]-xs[k-1]);
    const sal = xs.slice(part('salute').start_frame, part('salute').end_frame+1);
    const out = { loaded:true, kind:m.kind, nframes:m.nframes, nq, loop:m.loop,
      freeze:m.meta.freezeAtEnd, camera:m.meta.camera,
      x0:+xs[0].toFixed(3), xEnd:+xs[xs.length-1].toFixed(3),
      xGooseEnd:+xs[part('goose_walk').end_frame].toFixed(3),
      saluteSpan:+(Math.max(...sal)-Math.min(...sal)).toFixed(6),
      maxBack:+maxBack.toFixed(5),
      frames:{goose:part('goose_walk').frames, blend:part('attention_blend').frames,
              salute:part('salute').frames, hold:part('hold_attention').frames},
      note:$('motion-note').textContent,
      opts:[...$('opt-motion').options].map(o=>o.value) };
    __elf.setMotion('off');
    return out;})()`);
  check('正步 + 敬礼（连贯）在下拉里且加载成功（kind=parade，641 帧）',
    paradeSeq.loaded && paradeSeq.opts.includes('parade_seq')
      && paradeSeq.kind === 'parade' && paradeSeq.nq === 36
      && paradeSeq.nframes === 641 && paradeSeq.camera === 'parade',
    `nframes=${paradeSeq.nframes}｜camera=${paradeSeq.camera}｜kind=${paradeSeq.kind}`);
  check('净位移链：正步走到 3.152 m，敬礼段原地不动',
    Math.abs(paradeSeq.xGooseEnd - 3.152) < 1e-3
      && Math.abs(paradeSeq.xEnd - 3.152) < 1e-3 && paradeSeq.saluteSpan < 1e-6,
    `x: 0 → ${paradeSeq.xGooseEnd}（敬礼段跨度 ${paradeSeq.saluteSpan} m）`);
  check('过渡段基座没有回退（不会倒着滑回原点）',
    paradeSeq.maxBack >= -1e-6, `单帧最小 Δx = ${paradeSeq.maxBack} m`);
  check('分段完整：341 正步 + 50 过渡 + 225 敬礼 + 25 定格',
    paradeSeq.frames.goose === 341 && paradeSeq.frames.blend === 50
      && paradeSeq.frames.salute === 225 && paradeSeq.frames.hold === 25,
    JSON.stringify(paradeSeq.frames));
  check('面板说明走「连贯动作」专属文案（不串成单段正步 / 敬礼）',
    paradeSeq.note.includes('连贯动作') && paradeSeq.note.includes('build_parade_sequence'),
    (paradeSeq.note || '').slice(0, 46) + '…');

  const paradeFrozen = await evalJs(ws, `(async()=>{const $=id=>document.getElementById(id);
    await __elf.setMotion('parade_seq');
    const m=__elf.motion;
    m.time = m.duration - 0.01;
    await new Promise(x=>setTimeout(x,250));
    const out={ loop:m.loop, paused:m.paused, time:+m.time.toFixed(3),
      duration:+m.duration.toFixed(3), btn:$('btn-motion-pause').textContent.trim(),
      x:+__elf.data.qpos[0].toFixed(3) };
    __elf.setMotion('off');
    return out;})()`);
  check('连贯序列播完冻结在末帧立正（loop=false，按钮变「继续」）',
    paradeFrozen.loop === false && paradeFrozen.paused === true
      && Math.abs(paradeFrozen.time - paradeFrozen.duration) < 1e-6
      && paradeFrozen.btn === '继续' && Math.abs(paradeFrozen.x - 3.152) < 1e-3,
    `t=${paradeFrozen.time}/${paradeFrozen.duration} s · x=${paradeFrozen.x} m · ${paradeFrozen.btn}`);

  // 6. 渲染循环确实在出帧
  const frames = await evalJs(ws, `(async()=>{const r=__elf.renderer;const a=r.info.render.frame;
    await new Promise(x=>setTimeout(x,800));return r.info.render.frame-a;})()`);
  check('渲染循环持续出帧', frames > 0, `800 ms 内 ${frames} 帧`);

  // 7. XR 能力 + 半透明 VR（透视）合成
  const xr = await evalJs(ws, `(async()=>{
    await new Promise(r=>setTimeout(r,800));   // 等 isSessionSupported 探测回来
    const btns=[...document.querySelectorAll('#vr-slot button')].map(b=>({id:b.id,text:b.textContent.trim(),disabled:b.disabled}));
    return {
      xrEnabled:__elf.renderer.xr.enabled,
      alpha: __elf.renderer.getContext().getContextAttributes().alpha,
      caps: __elf.xrCapabilities(),
      btns,
      refSpace: __elf.vrNav && __elf.vrNav.refSpaceType || null,
      vrNavReady: !!__elf.vrNav && !!__elf.vrNav.renderer,
    };})()`);
  check('WebXR 已在渲染器上启用', xr.xrEnabled === true);
  check('VR 漫游模块已就绪', xr.vrNavReady === true, `参考空间: ${xr.refSpace}`);
  check('WebGL 带 alpha 通道（XR 帧缓冲透明的前提）', xr.alpha === true);
  check('两个 XR 按钮的可用状态与设备能力一致',
    xr.btns.length === 2 && xr.btns.every((b) => b.disabled === !xr.caps[b.id === 'xr-vr' ? 'vr' : 'ar']),
    JSON.stringify(xr.btns) + ' / caps=' + JSON.stringify(xr.caps));
  console.log(`  （无头浏览器能力探测: ${JSON.stringify(xr.caps)} — 头显里 vr/ar 才是 true）`);

  // 7.5 半透明合成开关：不需要真头显也能验证（直接调 setSeeThrough）
  const through = await evalJs(ws, `(()=>{
    const r=__elf.renderer, s=__elf.scene, x=__elf.xr;
    const F=__elf.mjScene.objs[__elf.mjScene.floorGeomIds[0]].material;
    const b={bg:s.background, fog:s.fog, alpha:r.getClearAlpha(), floor:F.opacity};
    x.setSeeThrough(true);
    const on={bgNull:s.background===null, fogNull:s.fog===null, alpha:r.getClearAlpha(),
              floor:+F.opacity.toFixed(3), transparent:F.transparent,
              flag:x.through, getter:__elf.seeThrough};
    x.setSeeThrough(false);
    const off={bgBack:s.background===b.bg, fogBack:s.fog===b.fog, alpha:r.getClearAlpha(),
               floor:+F.opacity.toFixed(3), transparent:F.transparent, flag:x.through,
               hud:document.getElementById('hud-mode').textContent.trim()};
    return {before:{bg:!!b.bg, fog:!!b.fog, alpha:b.alpha, floor:b.floor}, on, off};})()`);
  check('半透明 VR：开启后摘掉背景贴图与雾（不再遮住现实）',
    through.on.bgNull && through.on.fogNull, JSON.stringify(through.before));
  check('半透明 VR：清屏 alpha 归零（XR 帧缓冲因此透明）',
    through.on.alpha === 0, `alpha ${through.before.alpha} → ${through.on.alpha}`);
  check('半透明 VR：地面网格淡化到 0.3（露出真实地面）',
    Math.abs(through.on.floor - 0.3) < 1e-6 && through.on.transparent === true,
    `opacity ${through.before.floor} → ${through.on.floor}`);
  check('半透明 VR：状态标记与调试句柄同步',
    through.on.flag === true && through.on.getter === true);
  check('半透明 VR：关闭后背景 / 雾 / alpha / 地面全部还原',
    through.off.bgBack && through.off.fogBack && through.off.alpha === through.before.alpha
      && Math.abs(through.off.floor - 1) < 1e-6 && through.off.transparent === false,
    JSON.stringify(through.off));

  // 7.6 头显内面板（CanvasTexture 贴图）：DOM 在 XR 里不会被合成，面板画成场景里的平面。
  //     无头环境把「头显姿态」换成桌面 camera，所以摆位 / 绘制 / 命中 / 派发都能在这里跑。
  const xp = await evalJs(ws, `(async()=>{
    __elf.uiMode('vr');                       // 等价于进入会话：2D 精简 + 显示贴图面板
    await new Promise(r=>setTimeout(r,320));
    const p=__elf.xrPanel;
    const st=p.state();
    const quad=__elf.scene.getObjectByName('xr-panel-quad');
    const tab=__elf.scene.getObjectByName('xr-panel-tab');
    const grp=__elf.scene.getObjectByName('xr-panel');
    const note=document.getElementById('motion-note');
    const P=new __elf.THREE.Vector3(), v=new __elf.THREE.Vector3();
    __elf.camera.getWorldPosition(P);
    grp.getWorldDirection(v);                 // 非相机对象的 +Z 轴
    const want=P.clone().sub(grp.position).normalize();
    return {
      st,
      quadVisible: !!quad && quad.visible === true,
      inScene: !!grp && grp.parent === __elf.scene,
      panelPos: grp? grp.position.toArray().map(v=>+v.toFixed(3)) : null,
      camPos: __elf.camera.position.toArray().map(v=>+v.toFixed(3)),
      dy: +(grp.position.y - P.y).toFixed(3),
      normal: +v.dot(want).toFixed(3),
      mat: {
        q: [quad.material.transparent, quad.material.opacity, quad.material.depthWrite],
        t: [tab.material.transparent, tab.material.opacity, tab.material.depthWrite],
      },
      compact: document.getElementById('panel').classList.contains('compact'),
      noteDisplay: getComputedStyle(note).display,
      keys: p.widgets().map(w=>w.key),
    };})()`);
  check('头显面板已建在 scene 里并显示（不挂 stage，摇杆漫游不会拖走它）',
    xp.quadVisible && xp.inScene, `位置 ${JSON.stringify(xp.panelPos)} / 相机 ${JSON.stringify(xp.camPos)}`);
  check('面板摆在相机正前方 1.5 m（无头时用桌面 camera 当代替）',
    !!xp.panelPos && Math.abs(Math.hypot(xp.panelPos[0] - xp.camPos[0],
      xp.panelPos[2] - xp.camPos[2]) - 1.5) < 0.02,
    `水平距离 ${xp.panelPos && Math.hypot(xp.panelPos[0] - xp.camPos[0], xp.panelPos[2] - xp.camPos[2]).toFixed(2)} m`);
  check('面板中心高于眼睛 0.5 m（默认置于视野上方）',
    xp.dy > 0 && Math.abs(xp.dy - 0.5) < 0.05, `dy=${xp.dy} m`);
  check('平面 +Z 法向指向眼睛（朝眼下倾，约 10.8°）',
    xp.normal > 0.99, `dot=${xp.normal}`);
  check('主面板与收起小签默认都是 50% 半透明、关闭深度写入',
    JSON.stringify(xp.mat.q) === JSON.stringify([true, 0.5, false])
      && JSON.stringify(xp.mat.t) === JSON.stringify([true, 0.5, false]),
    `quad=${xp.mat.q} tab=${xp.mat.t}`);
  check('state() 暴露默认透明度与头锁状态',
    xp.st.opacity === 0.5 && xp.st.headLocked === true,
    `opacity=${xp.st.opacity} headLocked=${xp.st.headLocked}`);
  const WANT_KEYS = [
    'radio:opt-scene:single', 'radio:opt-scene:matrix', 'select:opt-motion',
    'btn:btn-motion-restart', 'btn:btn-motion-pause',
    'btn:view:front', 'btn:view:back', 'btn:view:side',
    'btn:view:squat', 'btn:view:grid', 'btn:view:wide',
    'check:opt-follow',
    'panel:collapse', 'panel:exit', 'panel:recenter', 'panel:opacity',
  ].sort();
  check('面板精简成固定两栏（场景 / 视角）',
    xp.st.groups.join(',') === '场景,视角', xp.st.groups.join(','));
  check('最小控件集 = 2 单选 + 1 下拉 + 1 复选 + 8 按钮 + 4 标题按钮，说明文字与多余控件全删',
    xp.st.counts.radio === 2 && xp.st.counts.select === 1 && xp.st.counts.check === 1
      && xp.st.counts.btn === 8 && xp.st.counts.panel === 4
      && xp.keys.length === 16
      && JSON.stringify(xp.keys.slice().sort()) === JSON.stringify(WANT_KEYS)
      && !xp.keys.some((k) => /opt-motion-speed|btn-reset-stage|view:(head|top|crab)|note/.test(k)),
    `${JSON.stringify(xp.st.counts)} keys=${xp.keys.length}`);
  check('宽扁画布长宽比来自内容（文字物理大小恒定，不裁切）',
    xp.st.canvas.w === 1024 && xp.st.canvas.h > 200 && xp.st.canvas.h < 420
      && Math.abs(xp.st.world.h - 0.80 * xp.st.canvas.h / xp.st.canvas.w) < 1e-6,
    `画布 ${xp.st.canvas.w}×${xp.st.canvas.h} → 平面 0.80×${xp.st.world.h.toFixed(2)} m`);
  check('进会话时 2D 面板进精简模式：说明段落隐藏、其余照旧',
    xp.compact === true && xp.noteDisplay === 'none',
    `compact=${xp.compact} motion-note display=${xp.noteDisplay}`);

  const xpTap = await evalJs(ws, `(async()=>{
    const p=__elf.xrPanel, $=id=>document.getElementById(id), out={};
    const quad=()=>__elf.scene.getObjectByName('xr-panel-quad');
    const tab =()=>__elf.scene.getObjectByName('xr-panel-tab');
    const label=()=>p.widgets().find(w=>w.key==='panel:opacity')?.label;
    const f0=$('opt-follow').checked;
    p.tap('check:opt-follow');
    out.check={ before:f0, after:$('opt-follow').checked, follow:__elf.vrNav.follow };
    p.tap('check:opt-follow');
    out.checkBack=$('opt-follow').checked;

    p.tap('radio:opt-scene:single');
    await new Promise(r=>setTimeout(r,220));
    out.radio={ mode:__elf.sceneMode,
      on:[...document.querySelectorAll('#scene-mode label')].filter(l=>l.classList.contains('on')).length };
    p.tap('radio:opt-scene:matrix');
    await new Promise(r=>setTimeout(r,220));
    out.radioBack=__elf.sceneMode;

    // 回放下拉 11 项：短画布撑高 + 顶边锚定，选完再缩回。
    // 先把 DOM 值改成 fight_policy 但**不派发 change**（避免触发 .bin 异步加载：
    // 7.6 时场景仍是 matrix 96 份实例，软渲染下主线程极慢，加载会拖几分钟），
    // 再选 option:off —— 既验证「选项写回 DOM + 派发 change」，又不会拉起下载。
    $('opt-motion').value='fight_policy';
    p.rebuild();
    out.beforeSel=$('opt-motion').value;
    const texBefore=p._tex;
    p.tap('select:opt-motion');
    out.overlayOpen = p.state().overlay;
    out.overlayOpts = p.widgets().filter(w=>w.kind==='option').length;
    const g=p.state(), q=quad();
    out.grown={ h:g.canvas.h, baseH:g.baseH, dy:q.position.y, scaleY:q.scale.y };
    // 纹理随画布尺寸重建：texStorage2D 不可变存储锁死首帧尺寸，画布从 290 撑到 782
    // 必须换新 CanvasTexture，否则 GL 报 INVALID_VALUE、平面仍是拉伸的旧图。
    out.texRebuilt = p._tex && p._tex !== texBefore && p._tex.image.height === g.canvas.h
      && q.material.map === p._tex;
    p.tap('option:off');
    out.select={ value:$('opt-motion').value, overlayClosed:p.state().overlay };
    const s=p.state();
    out.shrunk={ h:s.canvas.h, dy:quad().position.y };

    // 面板按钮 → DOM 点击派发：挂一次性监听，不依赖任何回放数据。
    let clicks=0;
    const btn=$('btn-motion-restart'), on=()=>{ clicks++; };
    btn.addEventListener('click', on);
    p.tap('btn:btn-motion-restart');
    btn.removeEventListener('click', on);
    out.restart={ clicks, label:$('btn-motion-pause').textContent.trim() };

    // 视角按钮：先记下正面机位，挪到矩阵机位，再点面板上的「正面」
    __elf.applyView('front');
    const expected=__elf.camera.position.toArray().map(v=>+v.toFixed(3));
    __elf.applyView('grid');
    const gridPos=__elf.camera.position.toArray().map(v=>+v.toFixed(3));
    p.tap('btn:view:front');
    out.view={ expected, gridPos, frontPos:__elf.camera.position.toArray().map(v=>+v.toFixed(3)) };

    // 透明度一键切换 50% ↔ 100%（两个材质 + chip 文案一起变）
    p.tap('panel:opacity');
    out.opacity1={ st:p.state().opacity, q:quad().material.opacity, t:tab().material.opacity, label:label() };
    p.tap('panel:opacity');
    out.opacity2={ st:p.state().opacity, q:quad().material.opacity, t:tab().material.opacity, label:label() };
    return out;})()`);
  check('面板复选框 → DOM 与视角跟随同步',
    xpTap.check.before !== xpTap.check.after && xpTap.check.follow === xpTap.check.after
      && xpTap.checkBack === xpTap.check.before, JSON.stringify(xpTap.check));
  check('面板单选 → 场景模式切换（含 2D 高亮）',
    xpTap.radio.mode === 'single' && xpTap.radio.on === 1 && xpTap.radioBack === 'matrix',
    JSON.stringify(xpTap.radio));
  check('面板下拉 → 覆盖层列出 11 段回放并可选中（写回 DOM + 派发 change）',
    xpTap.overlayOpen === 'opt-motion' && xpTap.overlayOpts === 11
      && xpTap.beforeSel === 'fight_policy' && xpTap.select.value === 'off'
      && xpTap.select.overlayClosed === null,
    `覆盖层=${xpTap.overlayOpen} 选项=${xpTap.overlayOpts} ${xpTap.beforeSel} → ${xpTap.select.value}`);
  check('覆盖层撑高画布且顶边锚定，关闭后自动缩回宽扁条',
    xpTap.grown.h >= 782 && xpTap.grown.baseH > 200 && xpTap.grown.baseH < 420
      && xpTap.grown.dy < 0
      && Math.abs(xpTap.grown.dy - (0.80 * (xpTap.grown.baseH - xpTap.grown.h) / 1024) / 2) < 1e-6
      && Math.abs(xpTap.grown.scaleY - 0.80 * xpTap.grown.h / 1024) < 1e-6
      && xpTap.shrunk.h === xpTap.grown.baseH && Math.abs(xpTap.shrunk.dy) < 1e-6,
    `H=${xpTap.grown.h} baseH=${xpTap.grown.baseH} dy=${xpTap.grown.dy.toFixed(3)} → ${xpTap.shrunk.h}`);
  check('撑高画布时纹理随之重建（新 CanvasTexture 贴到材质上，修「下拉被拉伸看不见」）',
    xpTap.texRebuilt === true, `texRebuilt=${xpTap.texRebuilt}`);
  check('面板视角按钮 → 相机切回正面机位',
    xpTap.view.frontPos.every((v, i) => Math.abs(v - xpTap.view.expected[i]) < 0.05)
      && xpTap.view.gridPos.some((v, i) => Math.abs(v - xpTap.view.expected[i]) > 0.5),
    `grid=${xpTap.view.gridPos} → ${xpTap.view.frontPos}`);
  check('面板「从头播放」→ 代点 DOM 按钮（零业务逻辑复制）',
    xpTap.restart.clicks === 1,
    `DOM click 次数=${xpTap.restart.clicks} 按钮文案=${xpTap.restart.label}`);
  check('标题栏一键切换整个面板 50% ↔ 100%',
    xpTap.opacity1.st === 1 && xpTap.opacity1.q === 1 && xpTap.opacity1.t === 1 && xpTap.opacity1.label === '100%'
      && xpTap.opacity2.st === 0.5 && xpTap.opacity2.q === 0.5 && xpTap.opacity2.t === 0.5
      && xpTap.opacity2.label === '50%',
    `第一次=${JSON.stringify(xpTap.opacity1)} 第二次=${JSON.stringify(xpTap.opacity2)}`);

  const xpLock = await evalJs(ws, `(async()=>{
    const p=__elf.xrPanel, grp=__elf.scene.getObjectByName('xr-panel');
    const P=new __elf.THREE.Vector3();
    const frame=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
    p.tap('panel:recenter');                        // 头锁 → 世界固定
    const off={ locked:p.state().headLocked, pos:grp.position.toArray().map(v=>+v.toFixed(3)) };
    __elf.camera.position.set(3,1.6,3); __elf.camera.updateMatrixWorld(true);
    await frame(); await new Promise(r=>setTimeout(r,150));
    off.posAfter=grp.position.toArray().map(v=>+v.toFixed(3));
    p.tap('panel:recenter');                        // 世界固定 → 头锁，立即摆回视野上方
    await frame();
    __elf.camera.getWorldPosition(P);
    const g=grp.position;
    const on={ locked:p.state().headLocked,
      dist:+Math.hypot(g.x-P.x,g.z-P.z).toFixed(3), dy:+(g.y-P.y).toFixed(3) };
    __elf.applyView('front');                       // 还原相机，别影响后续小节
    return {off,on};})()`);
  check('「归位」切到世界固定：移动相机后面板留在原处',
    xpLock.off.locked === false
      && xpLock.off.pos.every((v, i) => Math.abs(v - xpLock.off.posAfter[i]) < 1e-3),
    `${xpLock.off.pos} → ${xpLock.off.posAfter}`);
  check('「归位」切回头锁：面板立即回到视线上方 1.5 m / +0.5 m',
    xpLock.on.locked === true && Math.abs(xpLock.on.dist - 1.5) < 0.05
      && Math.abs(xpLock.on.dy - 0.5) < 0.05,
    `dist=${xpLock.on.dist} dy=${xpLock.on.dy}`);

  const png = await evalJs(ws, `__elf.xrPanel.exportPNG()`);
  let pngPath = null;
  if (typeof png === 'string' && png.startsWith('data:image/png')) {
    const buf = Buffer.from(png.split(',')[1], 'base64');
    fs.mkdirSync('logs', { recursive: true });
    pngPath = path.resolve('logs/xr_panel_debug.png');
    fs.writeFileSync(pngPath, buf);
  }
  check('面板导出 PNG（可人眼确认中文没变豆腐块）',
    !!pngPath, pngPath ? `${pngPath}（${(fs.statSync(pngPath).size / 1024).toFixed(0)} KB）` : '导出失败');

  const xpCollapse = await evalJs(ws, `(async()=>{
    const p=__elf.xrPanel, q=()=>__elf.scene.getObjectByName('xr-panel-quad').visible,
      t=()=>__elf.scene.getObjectByName('xr-panel-tab').visible;
    p.tap('panel:collapse'); await new Promise(r=>setTimeout(r,150));
    const a={collapsed:p.state().collapsed, mesh:q(), tab:t()};
    p.tap('panel:expand');   await new Promise(r=>setTimeout(r,150));
    const b={collapsed:p.state().collapsed, mesh:q(), tab:t()};
    return {a,b};})()`);
  check('面板「收起 / 展开」在贴图与小签之间切换',
    xpCollapse.a.collapsed && !xpCollapse.a.mesh && xpCollapse.a.tab
      && !xpCollapse.b.collapsed && xpCollapse.b.mesh && !xpCollapse.b.tab,
    JSON.stringify(xpCollapse));

  const xpOff = await evalJs(ws, `(async()=>{
    __elf.uiMode(null);
    await new Promise(r=>setTimeout(r,200));
    const grp=__elf.scene.getObjectByName('xr-panel');
    return { visible:__elf.xrPanel.state().visible, groupVisible:grp.visible,
      compact:document.getElementById('panel').classList.contains('compact'),
      noteDisplay:getComputedStyle(document.getElementById('motion-note')).display };})()`);
  check('退出会话后全部还原（面板隐藏、2D 说明段落回来）',
    xpOff.visible === false && xpOff.groupVisible === false
      && xpOff.compact === false && xpOff.noteDisplay !== 'none',
    `panel=${xpOff.visible} compact=${xpOff.compact} note=${xpOff.noteDisplay}`);

  // 7.7 VR 降载（进会话关阴影 + 帧缓冲缩放 + 每帧 CPU 瘦身；**场景不降载**）
  //     不需要真头显：__elf.xrQuality.enter()/exit() 就是 onEnter/onExit 调的那两个函数。
  const xq = await evalJs(ws, `(async()=>{
    const r=__elf.renderer, s=__elf.mjScene;
    __elf.setSceneMode('matrix');            // 先进矩阵，验证 VR 里会原样保留
    const before={ mode:__elf.sceneMode, grid:!!s.grid, count:s.grid?s.grid.count:0,
      shadow:r.shadowMap.enabled, cast:__elf.shadows.cast };
    __elf.xrQuality.enter();
    await new Promise(k=>requestAnimationFrame(()=>requestAnimationFrame(k)));
    const on={ mode:__elf.sceneMode, grid:!!s.grid, count:s.grid?s.grid.count:0,
      meshes:s.grid?s.grid.meshes.length:0,
      shadow:r.shadowMap.enabled, cast:__elf.shadows.cast, auto:r.shadowMap.autoUpdate,
      robotVisible:s.objs[s.robotGeomIds[0]].visible,
      instVisible:s.grid? s.grid.meshes.filter(m=>m.visible).length : 0,
      syncIds:s._syncIds?s._syncIds.length:null, ngeom:s.model.ngeom };
    // VR 里切单人再切回矩阵：两次都必须真的生效（不再被拦）
    __elf.setSceneMode('single');
    const vSingle={ mode:__elf.sceneMode, grid:!!s.grid };
    __elf.setSceneMode('matrix');
    const vMatrix={ mode:__elf.sceneMode, grid:!!s.grid, count:s.grid?s.grid.count:0 };
    __elf.xrQuality.exit();
    await new Promise(k=>requestAnimationFrame(()=>requestAnimationFrame(k)));
    const off={ mode:__elf.sceneMode, grid:!!s.grid, count:s.grid?s.grid.count:0,
      shadow:r.shadowMap.enabled, cast:__elf.shadows.cast };
    return { before, on, off, vSingle, vMatrix, aa:__elf.aa, xrss:__elf.xrScale,
      inXR:__elf.inXR };})()`);
  check('进 VR 保持矩阵场景（96 份实例原样留在场景里，不再自动降载）',
    xq.on.mode === 'matrix' && xq.on.grid === true && xq.on.count === 96
      && xq.on.meshes === 32 && xq.on.instVisible === 32
      && xq.on.robotVisible === false,
    `mode=${xq.on.mode} grid=${xq.on.grid} 实例=${xq.on.count}/${xq.on.meshes}个Mesh `
      + `可见=${xq.on.instVisible} 原件可见=${xq.on.robotVisible}`);
  check('进 VR 自动关阴影（连主光 castShadow 一起关，阴影贴图也不再每帧重算）',
    xq.on.shadow === false && xq.on.cast === false && xq.on.auto === false,
    `enabled=${xq.on.shadow} cast=${xq.on.cast} autoUpdate=${xq.on.auto}`);
  check('VR 里场景可自由切换（单人 ↔ 矩阵都不被拦）',
    xq.vSingle.mode === 'single' && xq.vSingle.grid === false
      && xq.vMatrix.mode === 'matrix' && xq.vMatrix.grid === true
      && xq.vMatrix.count === 96,
    `入VR=${xq.on.mode} → 单人=${xq.vSingle.mode} → 矩阵=${xq.vMatrix.mode}(${xq.vMatrix.count})`);
  check('退出 VR 保留 VR 里的场景选择，只还原阴影',
    xq.off.mode === 'matrix' && xq.off.grid === true && xq.off.count === 96
      && xq.off.shadow === xq.before.shadow && xq.off.cast === xq.before.cast
      && xq.inXR === false,
    `mode=${xq.off.mode} grid=${xq.off.grid} shadow=${xq.off.shadow} cast=${xq.off.cast}`);
  check('同步清单已瘦身（只写可见 geom，不再遍历全部 94 个）',
    typeof xq.on.syncIds === 'number' && xq.on.syncIds < xq.on.ngeom,
    `${xq.on.syncIds} / ${xq.on.ngeom} 个 geom`);
  check('启动期画质参数可查（桌面仍开抗锯齿，xrss 在 0.5~1 内）',
    xq.aa === true && xq.xrss >= 0.5 && xq.xrss <= 1,
    `antialias=${xq.aa} xrss=${xq.xrss}`);

  const idleSync = await evalJs(ws, `(async()=>{
    __elf.setMotion('off');
    await new Promise(k=>setTimeout(k,250));
    const a=__elf.mjScene._syncCount;
    await new Promise(k=>setTimeout(k,400));
    return { a, b:__elf.mjScene._syncCount };})()`);
  check('静止 / 暂停时不再做无谓的 qpos→kinematics→sync 链',
    idleSync.b - idleSync.a <= 1, `${idleSync.a} → ${idleSync.b} 次 sync / 400 ms`);

  const idlePanel = await evalJs(ws, `(async()=>{
    __elf.uiMode('vr');
    await new Promise(k=>setTimeout(k,300));
    const a=__elf.xrPanel._rebuildCount;
    await new Promise(k=>setTimeout(k,800));
    const b=__elf.xrPanel._rebuildCount;
    __elf.uiMode(null);
    return { a, b };})()`);
  check('头显面板空闲时不再反复重绘（悬停重绘已去抖，一次重绘=3.2 MB 纹理上传）',
    idlePanel.b - idlePanel.a <= 1, `${idlePanel.a} → ${idlePanel.b} 次重绘 / 800 ms`);

  // 7.9 右摇杆连续转向：无头也能验（只查 VRNav 配置，不碰真实 XR 会话）
  const turnCfg = await evalJs(ws, `(()=>{
    const n=__elf.vrNav;
    return { speed:n.turnSpeed, dead:n.turnDeadzone,
      hasSnap:'snapTurn' in n, hasCooldown:'turnCooldown' in n,
      onManualTurn:typeof n.onManualTurn };
  })()`);
  check('右摇杆为连续转向（90°/s + 死区，不再有 snap / 冷却）',
    Math.abs(turnCfg.speed - Math.PI / 2) < 1e-6 && turnCfg.dead > 0
      && !turnCfg.hasSnap && !turnCfg.hasCooldown,
    `speed=${turnCfg.speed}, dead=${turnCfg.dead}`);
  check('手动转向会关掉自动跟随（onManualTurn 已接上）',
    turnCfg.onManualTurn === 'function');

  // 7.10 预加载 / 插值曲线 / 矩阵实例剔除
  const cacheTest = await evalJs(ws, `(async()=>{
    const s0=__elf.motionCache.size;
    await __elf.motionCache.prefetch(['crawl','parade_seq']);
    const s1=__elf.motionCache.size;
    const t0=performance.now();
    await __elf.setMotion('crawl');
    const ms=performance.now()-t0;
    const out={ s0, s1, ms:+ms.toFixed(1), interp:__elf.perf().interp,
      ft:__elf.perf().ft.frames, name:__elf.getMotion().name };
    __elf.setMotion('off');
    return out;})()`);
  check('回放数据可预取并命中内存缓存（切段不再走网络）',
    cacheTest.s1 >= 2 && cacheTest.s1 >= cacheTest.s0 && cacheTest.name === 'crawl',
    `缓存 ${cacheTest.s0} → ${cacheTest.s1} 段，切段耗时 ${cacheTest.ms} ms`);
  check('默认插值曲线是 Catmull-Rom（?interp=linear 可回退）',
    cacheTest.interp === 'cr', `interp=${cacheTest.interp}`);
  check('帧时间探针在跑（perf().ft）',
    cacheTest.ft > 0, `${cacheTest.ft} 帧样本`);

  const interpTest = await evalJs(ws, `(async()=>{
    await __elf.setMotion('squat_cycle');
    const m=__elf.motion; m.paused=true;
    const nq=m.nq, f=m.frames, nf=m.nframes, fps=m.fps;
    const out=new Float64Array(nq);
    let samples=0, bad=0, worst=0;
    // 每 7 帧取一个非整数时刻（a≈0.37），逐通道查是否落在相邻两帧之间。
    // 四元数通道（3..6）要归一化，归一化会改变分量幅值，不参与这条判据。
    for(let i=0;i<nf-2;i+=7){
      m.seek((i+0.37)/fps);
      const xf=m.time*fps, ii=Math.floor(xf), a=xf-ii;
      if(ii>=nf-1||a<1e-6) continue;
      m.writeQpos(out);
      const o1=ii*nq, o2=(ii+1)*nq;
      for(let k=0;k<nq;k++){
        if(k>=3&&k<=6) continue;
        const lo=Math.min(f[o1+k],f[o2+k]), hi=Math.max(f[o1+k],f[o2+k]);
        const ov = out[k]>hi ? out[k]-hi : (out[k]<lo ? lo-out[k] : 0);
        if(ov>1e-6) bad++;
        if(ov>worst) worst=ov;
      }
      samples++;
    }
    const interp=m.interp;
    __elf.setMotion('off');
    return {samples, bad, worst:+worst.toFixed(6), interp};})()`);
  check('Catmull-Rom 不越出相邻两帧（接缝钳制生效，脚不会压进地面）',
    interpTest.interp === 'cr' && interpTest.samples > 30 && interpTest.bad === 0,
    `${interpTest.samples} 个采样 / 越界 ${interpTest.bad} 次（最大 ${interpTest.worst}）`);

  const cullTest = await evalJs(ws, `(()=>{
    const s=__elf.mjScene, T=__elf.THREE;
    if(!s.grid) return {skip:true};
    const c=new T.Vector3();
    s.robotBounds().getCenter(c);
    s.root.localToWorld(c);
    const cam=new T.PerspectiveCamera(48, 1, 0.1, 200);
    cam.position.set(c.x, c.y+0.9, c.z+2.6);   // 站在机器人正前方 2.6 m
    cam.lookAt(c.x, c.y, c.z);
    cam.updateMatrixWorld(true);
    const drawn=s.cullGrid(cam, c, 1.7);
    s.sync();                                  // 让 im.count 落地
    const applied=s.grid.meshes[0].count;
    const total=s.grid.count;
    s.grid.mask.fill(1);                       // 还原，别把桌面场景留在剔除状态
    s.sync();
    return {drawn, applied, total, after:s.grid.meshes[0].count};})()`);
  check('矩阵实例按视锥剔除（视野外的不提交，桌面/无头保持全量）',
    !cullTest.skip && cullTest.drawn > 0 && cullTest.drawn < cullTest.total
      && cullTest.applied === cullTest.drawn && cullTest.after === cullTest.total,
    `可见 ${cullTest.drawn}/${cullTest.total}，提交 ${cullTest.applied}，还原后 ${cullTest.after}`);

  // 7.11 机器人着色模型可切换（有损开关，用来评估「为帧率值不值得牺牲金属感」）
  const matTest = await evalJs(ws, `(()=>{
    const s=__elf.mjScene, ids=s.robotGeomIds;
    const kind=()=>ids.map(i=>s.objs[i].material.type);
    const uniq=(a)=>[...new Set(a)];
    const before=uniq(kind());
    __elf.setRobotMaterial('standard');
    const std=uniq(kind());
    __elf.setRobotMaterial('lambert');
    const lam=uniq(kind());
    const lamGrid=s.grid?uniq(s.grid.meshes.map(m=>m.material.type)):[];
    const lamFlag=__elf.perf().robotMaterial;
    __elf.setRobotMaterial('lambert');
    const back=uniq(kind());
    const backGrid=s.grid?uniq(s.grid.meshes.map(m=>m.material.type)):[];
    return {before, std, lam, lamGrid, lamFlag, back, backGrid,
      flag:__elf.perf().robotMaterial};})()`);
  check('机器人材质可切 MeshStandard 并还原（矩阵实例材质同步）',
    matTest.before.join() === 'MeshLambertMaterial'
      && matTest.std.join() === 'MeshStandardMaterial'
      && matTest.lam.join() === 'MeshLambertMaterial'
      && matTest.lamGrid.join() === 'MeshLambertMaterial'
      && matTest.back.join() === 'MeshLambertMaterial'
      && matTest.backGrid.join() === 'MeshLambertMaterial',
    `默认 ${matTest.before} → ${matTest.std}（实例 ${matTest.backGrid}）→ ${matTest.back}`);
  check('切换后 perf().robotMaterial 与实际材质一致',
    matTest.lamFlag === 'lambert' && matTest.flag === 'lambert',
    `${matTest.lamFlag} → ${matTest.flag}`);

  // 8. URL 参数路径（一键脚本 start_vr_motion.bat / .sh 走的就是这条）
  const q = `${url}${url.includes('?') ? '&' : '?'}motion=fight_policy&speed=0.5&follow=0`;
  await rpc(ws, 'Page.navigate', { url: q });
  let auto = null;
  for (let i = 0; i < 200; i++) {
    await sleep(500);
    auto = await evalJs(ws, `(()=>{try{return{ok:!!window.__elf,m:window.__elf.getMotion(),
      mode:document.getElementById('hud-mode').textContent.trim(),
      speed:document.getElementById('opt-motion-speed').value,
      follow:document.getElementById('opt-follow').checked,
      vrFollow:__elf.vrNav?__elf.vrNav.follow:null}}catch(e){return{err:String(e)}}})()`);
    if (auto && auto.ok && auto.m && auto.m.kind) break;
  }
  check('URL 参数 ?motion=… 自动进入运动仿真回放',
    !!(auto && auto.m && auto.m.kind === 'policy'), JSON.stringify(auto && auto.m));
  check('URL 参数 ?speed / ?follow 生效',
    !!auto && auto.speed === '0.5' && auto.follow === false && auto.vrFollow === false,
    `speed=${auto && auto.speed} follow=${auto && auto.follow}`);
  check('带参数进入时 HUD 直接显示运动仿真',
    !!auto && auto.mode.includes('运动仿真'), auto && auto.mode);

  check('控制台无报错', errors.length === 0, errors.slice(0, 4).join(' | ') || '干净');

  const pass = results.filter((r) => r.ok).length;
  console.log(`\n===== 通过 ${pass}/${results.length} =====`);
  if (pass !== results.length) {
    console.log('失败项：');
    for (const r of results.filter((x) => !x.ok)) console.log('  - ' + r.name + ' :: ' + r.detail);
  }
  process.exitCode = pass === results.length ? 0 : 1;
} catch (e) {
  console.error('验证脚本异常:', e.message);
  process.exitCode = 2;
} finally {
  chrome.kill();
}
