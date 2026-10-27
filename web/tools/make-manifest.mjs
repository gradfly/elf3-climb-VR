// 生成 assets/manifest.json：列出所有需要预加载进 MuJoCo 虚拟文件系统的资源（含字节数，便于前端显示精确进度）
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.resolve(__dirname, '../assets');

const paths = ['elf3.xml', 'scene_bx.xml'];
for (const f of fs.readdirSync(path.join(ASSETS, 'meshes')).filter((f) => /\.stl$/i.test(f)).sort()) {
  paths.push(`meshes/${f}`);
}

const files = paths.map((p) => ({
  path: p,
  size: fs.statSync(path.join(ASSETS, p)).size,
}));

const manifest = {
  generatedAt: new Date().toISOString(),
  entry: 'elf3.xml',
  files,
  totalBytes: files.reduce((a, b) => a + b.size, 0),
};
fs.writeFileSync(path.join(ASSETS, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`manifest.json: ${files.length} 个文件, 共 ${(manifest.totalBytes / 1048576).toFixed(1)} MB`);
