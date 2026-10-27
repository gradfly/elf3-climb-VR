// 一次性下载前端依赖（three.js 及其 addons）到 web/vendor
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const V = path.resolve(__dirname, '../vendor');
const J = path.resolve(__dirname, '../jsm');

const VERSION = '0.170.0';
const BASE = `https://cdn.jsdelivr.net/npm/three@${VERSION}`;

const jobs = [
  [`${BASE}/build/three.module.js`, path.join(V, 'three.module.js')],
  [`${BASE}/examples/jsm/controls/OrbitControls.js`, path.join(J, 'controls/OrbitControls.js')],
  [`${BASE}/examples/jsm/utils/BufferGeometryUtils.js`, path.join(J, 'utils/BufferGeometryUtils.js')],
];

for (const [url, dest] of jobs) {
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  const buf = Buffer.from(await r.arrayBuffer());
  fs.writeFileSync(dest, buf);
  console.log(`${(buf.length / 1024).toFixed(0).padStart(6)} KB  ${path.relative(path.resolve(__dirname, '..'), dest)}`);
}
console.log('three.js @', VERSION, 'downloaded');
