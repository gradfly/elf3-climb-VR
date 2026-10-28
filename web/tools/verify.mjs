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
  check('30 个视觉网格全部建好', scene.robotGeoms === 30 && scene.geoms === 30, `${scene.geoms} 个几何 / ${scene.tris} 三角面`);
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
  check('7 个视角预设齐全', ui.views.length === 7, ui.views.join(','));
  const wantOpts = ['opt-robot', 'opt-stairs', 'opt-crab', 'opt-floor', 'opt-shadow',
    'opt-collision', 'opt-sim', 'opt-follow'];
  check('显示/物理/跟随开关齐全', wantOpts.every((id) => ui.opts.includes(id)), ui.opts.join(','));
  check('运动仿真控件齐全（回放/倍速/暂停）',
    !!(await evalJs(ws, `!!(document.getElementById('opt-motion')&&document.getElementById('opt-motion-speed')&&document.getElementById('btn-motion-pause')&&document.getElementById('btn-motion-restart'))`)));
  console.log('  加载日志:'); for (const l of ui.log) console.log('    ' + l);

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
    set('opt-shadow',false); out.shadowOff=__elf.renderer.shadowMap.enabled===false;
    set('opt-shadow',true);  out.shadowOn=__elf.renderer.shadowMap.enabled===true;
    return out;})()`);
  check('“机器人”开关生效', toggles.robotHidden && toggles.robotShown);
  check('“楼梯/障碍”开关生效', toggles.stairsHidden);
  check('“地面网格”开关生效', toggles.floorHidden);
  check('“碰撞体”调试层可见', toggles.collisionShown > 0, `${toggles.collisionShown} 个碰撞几何`);
  check('“阴影”开关生效', toggles.shadowOff && toggles.shadowOn);

  // 4. 视角预设
  const views = await evalJs(ws, `(()=>{const out={};
    for(const b of document.querySelectorAll('#views button')){
      b.click();
      const c=__elf.camera.position,t=__elf.controls.target;
      out[b.dataset.view]=[+c.x.toFixed(2),+c.y.toFixed(2),+c.z.toFixed(2),+t.y.toFixed(2)];
    } return out;})()`);
  const allMoved = Object.values(views).every((v) => v.some((x) => x !== 0));
  check('7 个视角预设均可用', Object.keys(views).length === 7 && allMoved, JSON.stringify(views));

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

  // 5.6 螃蟹步横移回放（脚本步态，另一套地形）
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
  check('螃蟹步回放 HUD 显示“脚本步态”',
    crab.mode.includes('运动仿真') && crab.hud.includes('脚本步态'), `${crab.mode} · ${crab.hud}`);
  console.log(`  （螃蟹步回放统计: ${JSON.stringify(crab.stats && {
    steps: crab.stats.steps, maxTop: crab.stats.maxTop, travelY: crab.stats.travelY })}）`);

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
