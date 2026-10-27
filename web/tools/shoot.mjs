// 用 Chrome DevTools Protocol 精确截图（等待页面就绪后再抓帧）
// 用法: node tools/shoot.mjs <url> <out.png> [waitMs] [--js=代码]
import fs from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import os from 'node:os';

const CHROME = process.env.CHROME
  || (fs.existsSync('C:/Program Files/Google/Chrome/Application/chrome.exe')
    ? 'C:/Program Files/Google/Chrome/Application/chrome.exe'
    : 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe');

const url = process.argv[2] || 'http://localhost:8090/';
const out = process.argv[3] || 'shot.png';
const waitMs = Number(process.argv[4] || 1500);
const jsArg = process.argv.find((a) => a.startsWith('--js='));
const extraJs = jsArg ? jsArg.slice(5) : null;

const PORT = 9333 + Math.floor(Math.random() * 400);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'cdp-'));

const chrome = spawn(CHROME, [
  '--headless=new',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  '--disable-gpu', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--no-sandbox', '--no-first-run', '--disable-extensions',
  '--window-size=1600,1000',
  ...(url.startsWith('https:') ? ['--ignore-certificate-errors'] : []),
  'about:blank',
], { stdio: ['ignore', 'pipe', 'pipe'] });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function waitEndpoint() {
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (r.ok) return;
    } catch { /* 还没起来 */ }
    await sleep(200);
  }
  throw new Error('Chrome 远程调试端口未就绪');
}

let id = 0;
function rpc(ws, method, params = {}, sessionId) {
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
    ws.send(JSON.stringify({ id: mid, method, params, sessionId }));
  });
}

try {
  await waitEndpoint();
  const t = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: 'PUT' })).json();
  const ws = new WebSocket(t.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  const logs = [];
  ws.addEventListener('message', (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch { return; }
    if (m.method === 'Runtime.consoleAPICalled') {
      logs.push(m.params.type + ': ' + m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
    }
    if (m.method === 'Runtime.exceptionThrown') {
      logs.push('EXCEPTION: ' + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text));
    }
  });

  await rpc(ws, 'Page.enable');
  await rpc(ws, 'Runtime.enable');
  await rpc(ws, 'Page.navigate', { url });

  // 等待应用就绪：加载遮罩被隐藏 + 渲染循环已启动
  let ready = false;
  for (let i = 0; i < 150; i++) {
    await sleep(500);
    const r = await rpc(ws, 'Runtime.evaluate', {
      expression: `(()=>{const l=document.getElementById('loading');return JSON.stringify({hidden:!!l&&l.classList.contains('hidden'),hud:!document.getElementById('hud').classList.contains('hidden'),step:(document.getElementById('ld-step')||{}).textContent||'',pct:(document.getElementById('ld-pct')||{}).textContent||''})})()`,
      returnByValue: true,
    });
    const s = (() => {
      try { return JSON.parse(r.result?.value); } catch { return { hidden: false, hud: false, step: '(页面尚未就绪)', pct: '' }; }
    })();
    if (s.hud) { ready = true; console.log('就绪:', JSON.stringify(s)); break; }
    if (i % 6 === 5) console.log('  等待中…', s.pct, s.step);
  }
  if (!ready) console.log('! 未检测到就绪状态');

  await sleep(waitMs);
  if (process.env.PROBE) {
    const r = await rpc(ws, 'Runtime.evaluate', { expression: process.env.PROBE, returnByValue: true });
    console.log('探针:', JSON.stringify(r.result?.value ?? r.result?.description ?? r.result));
  }
  if (extraJs) {
    const r = await rpc(ws, 'Runtime.evaluate', { expression: extraJs, returnByValue: true, awaitPromise: true });
    console.log('注入结果:', JSON.stringify(r.result?.value ?? r.result?.description ?? r.result));
    await sleep(1200);
  }

  const shot = await rpc(ws, 'Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  fs.writeFileSync(out, Buffer.from(shot.data, 'base64'));
  console.log('已保存', out, fs.statSync(out).size, 'bytes');
  if (logs.length) console.log('--- 页面控制台 ---\n' + logs.slice(0, 40).join('\n'));
} finally {
  chrome.kill();
}
