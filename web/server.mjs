/**
 * server.mjs —— elf3 MuJoCo 网页查看器的本地静态服务器
 *
 * 同时开两个监听：
 *   HTTP  (默认 8080) —— 本机桌面浏览器直接预览（localhost 属于安全上下文，WebXR 可用）
 *   HTTPS (默认 8443) —— VR 头显必须走 https；自签证书用 tools/selfsign.mjs 纯 JS 生成，
 *                        SAN 自动包含 localhost 与本机所有内网 IP
 *
 * 用法：
 *   node server.mjs
 *   node server.mjs --http-port 8080 --https-port 8443
 *   node server.mjs --no-https
 *   node server.mjs --open
 *   node server.mjs --open --open-path "/?motion=fight_policy"   # 直接进运动仿真回放
 */
import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { ensureCert, localIPv4 } from './tools/selfsign.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const CERT_DIR = path.join(ROOT, 'certs');

// ------------------------------------------------------------------ 参数
function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return def;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}
const HTTP_PORT = Number(arg('http-port', 8080));
const HTTPS_PORT = Number(arg('https-port', 8443));
const USE_HTTPS = !process.argv.includes('--no-https');
const OPEN_BROWSER = process.argv.includes('--open');
// --open 时打开的路径（默认首页）；例如 --open-path "/?motion=fight_policy"
const OPEN_PATH = String(arg('open-path', '/')) || '/';
const LOG_REQ = process.argv.includes('--log');   // 打印每条请求（排查加载问题时用）

// ------------------------------------------------------------------ MIME
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.xml': 'application/xml; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.wasm': 'application/wasm',
  '.map': 'application/json; charset=utf-8',
  '.stl': 'model/stl',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

// ------------------------------------------------------------------ 请求处理
function handle(req, res) {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    res.writeHead(400).end('Bad Request');
    return;
  }
  if (pathname.endsWith('/')) pathname += 'index.html';
  if (pathname === '/') pathname = '/index.html';

  const filePath = path.resolve(ROOT, '.' + pathname);
  if (!filePath.startsWith(ROOT + path.sep) && filePath !== ROOT) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end(`404 Not Found: ${pathname}`);
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    if (LOG_REQ) {
      res.on('finish', () => {
        console.log(`  ${req.method} ${pathname} → ${res.statusCode} (${st.size} B)`);
      });
    }
    const headers = {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Content-Length': st.size,
      'Last-Modified': st.mtime.toUTCString(),
      // 开发用：避免浏览器缓存旧模型/旧代码
      'Cache-Control': 'no-cache, must-revalidate',
      'X-Content-Type-Options': 'nosniff',
    };
    if (ext === '.html' || ext === '.json' || ext === '.mjs' || ext === '.js' || ext === '.xml') {
      headers['Vary'] = 'Accept-Encoding';
    }
    res.writeHead(200, headers);
    if (req.method === 'HEAD') { res.end(); return; }
    fs.createReadStream(filePath).on('error', () => res.destroy()).pipe(res);
  });
}

// ------------------------------------------------------------------ 端口监听（占用则顺延）
// 绑定 :: （双栈），这样 localhost 解析到 IPv6 的 ::1 也能连上；
// 某些系统禁用 IPv6，报 EAFNOSUPPORT 时回退到 0.0.0.0。
const HOST = '::';

function listen(server, port, label, tries = 10) {
  return new Promise((resolve, reject) => {
    let attempt = 0;
    let host = HOST;
    const onError = (e) => {
      if (e.code === 'EAFNOSUPPORT' && host === HOST) {
        console.log(`   （系统不支持 IPv6（${HOST}），改用 IPv4 0.0.0.0）`);
        host = '0.0.0.0';
        server.listen(port + attempt, host);
      } else if (e.code === 'EADDRINUSE' && attempt < tries) {
        attempt++;
        server.listen(port + attempt, host);
      } else {
        server.removeListener('error', onError);
        reject(e);
      }
    };
    server.on('error', onError);
    server.once('listening', () => {
      server.removeListener('error', onError);
      const p = server.address().port;
      if (attempt > 0) console.log(`   （${label} 端口 ${port} 被占用，改用 ${p}）`);
      resolve(p);
    });
    server.listen(port, host);
  });
}

// ------------------------------------------------------------------ 主流程
const line = '─'.repeat(66);
console.log(`\n${line}\n  elf3 · MuJoCo 网页仿真 / VR 沉浸式观看 — 本地服务器\n${line}`);
console.log(`  根目录: ${ROOT}`);

const servers = [];
const httpServer = http.createServer(handle);
const httpPort = await listen(httpServer, HTTP_PORT, 'HTTP');
servers.push(httpServer);

let httpsPort = null;
if (USE_HTTPS) {
  try {
    const cert = ensureCert(CERT_DIR);
    const httpsServer = https.createServer(
      { cert: fs.readFileSync(cert.cert), key: fs.readFileSync(cert.key) },
      handle,
    );
    httpsPort = await listen(httpsServer, HTTPS_PORT, 'HTTPS');
    servers.push(httpsServer);
    console.log(`  证书: ${path.relative(ROOT, cert.cert)}${cert.reused ? '（复用）' : '（新生成）'}`);
    console.log(`  SAN : ${cert.san}`);
  } catch (e) {
    console.log(`  ! HTTPS 启动失败：${e.message}（继续以 HTTP 提供服务）`);
  }
}

const ips = localIPv4();
// 打印时带上 --open-path（例如 /?motion=fight_policy），头显里可直接照抄这条地址
const openSuffix = OPEN_PATH === '/' ? '/' : OPEN_PATH;
console.log(`\n  本机桌面浏览器：`);
console.log(`     http://localhost:${httpPort}${openSuffix}`);
if (httpsPort) console.log(`     https://localhost:${httpsPort}${openSuffix}`);

if (ips.length) {
  console.log(`\n  VR 头显（同一局域网内在头显浏览器打开）：`);
  for (const ip of ips) {
    if (httpsPort) console.log(`     https://${ip}:${httpsPort}${openSuffix}`);
    else console.log(`     http://${ip}:${httpPort}${openSuffix}   （⚠ 非 HTTPS，头显无法进入 VR）`);
  }
  if (httpsPort) {
    console.log(`\n  首次打开会提示证书不受信任 —— 选择“高级 → 继续访问”即可；`);
    console.log(`  也可以在头显上把 certs/cert.pem 装成受信任证书，之后就不再提示。`);
  }
}
console.log(`\n  Ctrl+C 停止服务。\n${line}\n`);

if (OPEN_BROWSER) {
  const url = new URL(OPEN_PATH, `http://localhost:${httpPort}/`).href;
  let command, args;
  if (process.platform === 'win32') {
    // process.env.comspec 必定是 cmd.exe 的绝对路径（OS 自动设置）。
    // 托管的 Node 构建可能没把 C:\Windows\System32 放进 PATH，所以绝不能用裸 "cmd" 去查。
    command = process.env.comspec || 'C:\\Windows\\System32\\cmd.exe';
    args = ['/c', 'start', '', url];
  } else if (process.platform === 'darwin') {
    command = 'open'; args = [url];
  } else {
    command = 'xdg-open'; args = [url];
  }
  try {
    const child = spawn(command, args, { detached: true, stdio: 'ignore' });
    // spawn() 同步返回，但执行失败（例如 ENOENT）是以异步 'error' 事件抛出的，
    // 必须处理，否则会崩掉整个服务器进程。
    child.on('error', (e) => {
      console.log(`  ! 无法自动打开浏览器（${e.message}）— 请手动打开下面的地址：`);
      console.log(`     ${url}`);
    });
    child.unref();
  } catch { /* ignore */ }
}

for (const s of servers) {
  s.on('error', (e) => console.error('服务器错误:', e.message));
}

const shutdown = () => {
  console.log('\n正在关闭…');
  for (const s of servers) s.close();
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
