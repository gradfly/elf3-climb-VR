/**
 * selfsign.mjs —— 纯 JavaScript 生成自签 X.509 证书（零依赖）
 *
 * 为什么不直接调 openssl？
 *   在受限沙箱/某些环境下 Node 子进程无法执行 openssl（EBUSY），
 *   而 VR 头显必须走 HTTPS。所以这里用 node:crypto + 手写 DER 编码，
 *   直接在进程内签发证书。生成结果会用 crypto.X509Certificate 自检。
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// ------------------------------------------------------------------ DER 基础
function lenBytes(n) {
  if (n < 0x80) return Buffer.from([n]);
  const bytes = [];
  let v = n;
  while (v > 0) { bytes.unshift(v & 0xff); v >>>= 8; }
  return Buffer.from([0x80 | bytes.length, ...bytes]);
}
function tlv(tag, content) {
  const body = Buffer.isBuffer(content) ? content : Buffer.from(content);
  return Buffer.concat([Buffer.from([tag]), lenBytes(body.length), body]);
}
const seq = (...parts) => tlv(0x30, Buffer.concat(parts));
const set = (...parts) => tlv(0x31, Buffer.concat(parts));

function derInt(buf) {
  let b = Buffer.isBuffer(buf) ? Buffer.from(buf) : buf;
  // 去掉多余前导 0
  while (b.length > 1 && b[0] === 0x00 && (b[1] & 0x80) === 0) b = b.subarray(1);
  // 最高位是 1 时补 0x00，保证被解析为正数
  if (b[0] & 0x80) b = Buffer.concat([Buffer.from([0x00]), b]);
  return tlv(0x02, b);
}
function derOid(oid) {
  const parts = oid.split('.').map(Number);
  const out = [parts[0] * 40 + parts[1]];
  for (let i = 2; i < parts.length; i++) {
    let v = parts[i];
    const stack = [v & 0x7f];
    v >>>= 7;
    while (v > 0) { stack.unshift((v & 0x7f) | 0x80); v >>>= 7; }
    out.push(...stack);
  }
  return tlv(0x06, Buffer.from(out));
}
const derNull = () => Buffer.from([0x05, 0x00]);
const derUtf8 = (s) => tlv(0x0c, Buffer.from(s, 'utf8'));
const derIa5 = (s) => tlv(0x16, Buffer.from(s, 'ascii'));
const derOctet = (b) => tlv(0x04, b);
const derBitString = (b) => tlv(0x03, Buffer.concat([Buffer.from([0x00]), b]));

function derTime(date) {
  const p = (n, w = 2) => String(n).padStart(w, '0');
  const Y = date.getUTCFullYear();
  if (Y >= 1950 && Y < 2050) {
    return tlv(0x17, Buffer.from(
      p(Y % 100) + p(date.getUTCMonth() + 1) + p(date.getUTCDate())
      + p(date.getUTCHours()) + p(date.getUTCMinutes()) + p(date.getUTCSeconds()) + 'Z',
      'ascii'));
  }
  return tlv(0x18, Buffer.from(
    p(Y, 4) + p(date.getUTCMonth() + 1) + p(date.getUTCDate())
    + p(date.getUTCHours()) + p(date.getUTCMinutes()) + p(date.getUTCSeconds()) + 'Z',
    'ascii'));
}

function rdn(oid, value) {
  return set(seq(derOid(oid), derUtf8(value)));
}

// ------------------------------------------------------------------ 本机 IP
export function localIPv4() {
  const out = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const info of ifaces[name] || []) {
      if (info.family === 'IPv4' && !info.internal) out.push(info.address);
    }
  }
  return [...new Set(out)];
}

// ------------------------------------------------------------------ 签发
const OID_SHA256_RSA = '1.2.840.113549.1.1.11';
const OID_CN = '2.5.4.3';
const OID_BASIC_CONSTRAINTS = '2.5.29.19';
const OID_KEY_USAGE = '2.5.29.15';
const OID_EXT_KEY_USAGE = '2.5.29.37';
const OID_SAN = '2.5.29.17';
const OID_KP_SERVER_AUTH = '1.3.6.1.5.5.7.3.1';

/**
 * @param {object} o
 * @param {string[]} [o.extraIPs]
 * @param {number}   [o.days]
 * @param {string}   [o.commonName]
 * @returns {{cert: string, key: string, san: string}} PEM 文本
 */
export function selfSign({ extraIPs = [], days = 3650, commonName = 'elf3-mujoco-web' } = {}) {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('rsa', {
    modulusLength: 2048,
    publicKeyEncoding: { type: 'spki', format: 'der' },
    privateKeyEncoding: { type: 'pkcs8', format: 'der' },
  });

  const notBefore = new Date(Date.now() - 24 * 3600 * 1000);
  const notAfter = new Date(Date.now() + days * 24 * 3600 * 1000);

  const ips = ['127.0.0.1', ...new Set([...localIPv4(), ...extraIPs])];
  const dnsNames = ['localhost', os.hostname()].filter((v, i, a) => v && a.indexOf(v) === i);

  const generalNames = Buffer.concat([
    ...dnsNames.map((d) => tlv(0x82, Buffer.from(d, 'ascii'))),      // [2] dNSName
    ...ips.map((ip) => {
      const bytes = ip.split('.').map(Number);
      return tlv(0x87, Buffer.from(bytes));                           // [7] iPAddress
    }),
  ]);

  const extensions = seq(
    seq(derOid(OID_BASIC_CONSTRAINTS), Buffer.from([0x01, 0x01, 0xff]), derOctet(seq())),
    seq(derOid(OID_KEY_USAGE), Buffer.from([0x01, 0x01, 0xff]), derOctet(derBitString(Buffer.from([0xa0])))),
    seq(derOid(OID_EXT_KEY_USAGE), derOctet(seq(derOid(OID_KP_SERVER_AUTH)))),
    seq(derOid(OID_SAN), derOctet(seq(generalNames))),
  );

  const name = seq(rdn(OID_CN, commonName));
  const sigAlg = seq(derOid(OID_SHA256_RSA), derNull());

  const tbs = seq(
    tlv(0xa0, derInt(Buffer.from([0x02]))),                            // version v3
    derInt(crypto.randomBytes(16)),
    sigAlg,
    name,
    seq(derTime(notBefore), derTime(notAfter)),
    name,
    publicKey,                                                          // SubjectPublicKeyInfo
    tlv(0xa3, extensions),                                             // [3] extensions
  );

  const signature = crypto.sign('sha256', tbs, {
    key: crypto.createPrivateKey({ key: privateKey, format: 'der', type: 'pkcs8' }),
  });

  const certificate = seq(tbs, sigAlg, derBitString(signature));

  const keyPem = pem('PRIVATE KEY', privateKey);
  const certPem = pem('CERTIFICATE', certificate);

  return {
    cert: certPem,
    key: keyPem,
    san: [...dnsNames.map((d) => `DNS:${d}`), ...ips.map((ip) => `IP:${ip}`)].join(','),
  };
}

function pem(label, der) {
  const b64 = Buffer.from(der).toString('base64').replace(/(.{64})/g, '$1\n').trimEnd();
  return `-----BEGIN ${label}-----\n${b64}\n-----END ${label}-----\n`;
}

/** 自检：把生成的证书解析回来，确认关键字段正确 */
export function verify(certPem) {
  const x = new crypto.X509Certificate(certPem);
  const problems = [];
  if (!/CN=/i.test(x.subject)) problems.push('subject 缺失');
  const san = x.subjectAltName || '';
  if (!san.includes('DNS:localhost')) problems.push('SAN 缺少 localhost');
  return { subject: x.subject, san, validTo: x.validTo, problems };
}

/**
 * 生成（或复用）证书并写入目录
 * @param {string} outDir
 * @param {{force?: boolean}} [opt]
 */
export function ensureCert(outDir, { force = false } = {}) {
  fs.mkdirSync(outDir, { recursive: true });
  const certPath = path.join(outDir, 'cert.pem');
  const keyPath = path.join(outDir, 'key.pem');

  if (!force && fs.existsSync(certPath) && fs.existsSync(keyPath)) {
    const existing = fs.readFileSync(certPath, 'utf8');
    try {
      const info = verify(existing);
      const ips = localIPv4();
      const covers = ips.every((ip) => info.san.includes(`IP:${ip}`)) && info.san.includes('DNS:localhost');
      if (covers && info.problems.length === 0 && new Date(info.validTo) > new Date()) {
        return { cert: certPath, key: keyPath, san: info.san, reused: true };
      }
    } catch { /* 重新生成 */ }
  }

  const { cert, key, san } = selfSign();
  fs.writeFileSync(certPath, cert);
  fs.writeFileSync(keyPath, key, { mode: 0o600 });
  return { cert: certPath, key: keyPath, san, reused: false };
}

// 直接执行
import { fileURLToPath } from 'node:url';
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const outDir = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve('certs');
  const r = ensureCert(outDir, { force: process.argv.includes('--force') });
  console.log(`${r.reused ? '复用' : '生成'}证书:`);
  console.log('  cert:', r.cert);
  console.log('  key :', r.key);
  console.log('  SAN :', r.san);
  console.log('  自检:', JSON.stringify(verify(fs.readFileSync(r.cert, 'utf8')), null, 2));
}
