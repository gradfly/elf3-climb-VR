/**
 * make-cert.mjs —— 生成自签证书的便捷入口
 * 实际实现见 tools/selfsign.mjs（纯 JS，无需 openssl）
 *
 * 用法：node tools/make-cert.mjs [输出目录] [--force]
 */
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { ensureCert, verify, localIPv4 } from './selfsign.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = process.argv[2] && !process.argv[2].startsWith('--')
  ? path.resolve(process.argv[2])
  : path.resolve(__dirname, '../certs');

const r = ensureCert(outDir, { force: process.argv.includes('--force') });
console.log(r.reused ? '已存在可用证书，直接复用：' : '已生成自签证书：');
console.log('  cert :', r.cert);
console.log('  key  :', r.key);
console.log('  SAN  :', r.san);
console.log('  本机内网 IP:', localIPv4().join(', ') || '(无)');
console.log('  自检 :', JSON.stringify(verify(fs.readFileSync(r.cert, 'utf8'))));
