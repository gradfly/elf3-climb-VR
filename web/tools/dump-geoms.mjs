// 逐几何体列出世界 AABB（Z 范围），用于核对腿部网格是否错位
import fs from 'node:fs';
import path from 'node:path';
import loadMujoco from '../vendor/mujoco.js';

const A = './assets';
const mj = await loadMujoco();
const vfs = new mj.MjVFS();
const mf = JSON.parse(fs.readFileSync(path.join(A, 'manifest.json'), 'utf8'));
for (const f of mf.files) {
  vfs.addBuffer(f.path, new Uint8Array(fs.readFileSync(path.join(A, f.path))));
}
const model = mj.MjModel.from_xml_string(fs.readFileSync(path.join(A, 'elf3.xml'), 'utf8'), vfs);
const data = new mj.MjData(model);
mj.mj_forward(model, data);

function stlVerts(file) {
  const b = fs.readFileSync(file);
  const n = b.readUInt32LE(80);
  const out = new Float32Array(n * 9);
  for (let t = 0, o = 84; t < n; t++, o += 50) {
    for (let v = 0; v < 3; v++) {
      const q = o + 12 + v * 12;
      out[t * 9 + v * 3] = b.readFloatLE(q);
      out[t * 9 + v * 3 + 1] = b.readFloatLE(q + 4);
      out[t * 9 + v * 3 + 2] = b.readFloatLE(q + 8);
    }
  }
  return out;
}
const cache = new Map();
function vertsOf(name) {
  if (!cache.has(name)) cache.set(name, stlVerts(path.join(A, 'meshes', name + '.STL')));
  return cache.get(name);
}

console.log('qpos0 =', Array.from(data.qpos).map((v) => +v.toFixed(4)).join(', '));
console.log('关键帧数 nkey =', model.nkey);
console.log();
console.log('gid  body                  geom                 type  Zmin      Zmax     Xpos(z)');
console.log('-'.repeat(88));

const rows = [];
for (let gi = 0; gi < model.ngeom; gi++) {
  if (model.geom_group[gi] !== 2) continue;
  const bid = model.geom_bodyid[gi];
  const body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY.value, bid);
  const gname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM.value, gi) || '(unnamed)';
  const type = model.geom_type[gi];
  const g3 = gi * 3; const g9 = gi * 9;
  let zmin = NaN; let zmax = NaN;
  if (type === 7) { // mesh
    const mn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH.value, model.geom_dataid[gi]);
    const v = vertsOf(mn);
    let lo = 1e9; let hi = -1e9;
    const P = data.geom_xpos; const M = data.geom_xmat;
    for (let i = 0; i < v.length; i += 3) {
      const Z = M[g9 + 6] * v[i] + M[g9 + 7] * v[i + 1] + M[g9 + 8] * v[i + 2] + P[g3 + 2];
      if (Z < lo) lo = Z;
      if (Z > hi) hi = Z;
    }
    zmin = lo; zmax = hi;
  } else if (type === 6) { // box
    zmin = data.geom_xpos[g3 + 2] - model.geom_size[gi * 3 + 2];
    zmax = data.geom_xpos[g3 + 2] + model.geom_size[gi * 3 + 2];
  }
  let meshName = '-';
  if (type === 7) meshName = mj.mj_id2name(model, mj.mjtObj.mjOBJ_MESH.value, model.geom_dataid[gi]) || '(null)';
  rows.push([gi, body, meshName, type, zmin, zmax, data.geom_xpos[g3 + 2]]);
}
rows.sort((a, b) => b[6] - a[6]);
const f = (v) => (Number.isFinite(v) ? v.toFixed(4).padStart(8) : '     n/a');
for (const r of rows) {
  console.log(String(r[0]).padStart(3), String(r[1]).padEnd(21), String(r[2]).slice(0, 20).padEnd(20), String(r[3]).padStart(4), f(r[4]), f(r[5]), f(r[6]));
}

