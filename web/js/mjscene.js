/**
 * mjscene.js —— 把 MuJoCo 的 mjModel/mjData 映射成 three.js 场景
 *
 * 坐标映射（MuJoCo 是 Z-up，three.js / WebXR 是 Y-up）：
 *   (x, y, z)_mj  ->  (-y, z, -x)_three
 *   用一个 Group: rotation.x = -PI/2 （Z-up → Y-up）再套一层 rotation.y = PI/2，
 *   于是机器人正面( MuJoCo +x )朝向 three 的 -Z（即默认观察方向），无镜像。
 *
 * 网格数据直接用 mjModel 里编译好的 mesh_vert / mesh_face / mesh_normal。
 *   注意：MuJoCo 编译 STL 时会把顶点“重心归零”，偏移量记在 model.mesh_pos 里。
 *   如果直接拿原始 STL 顶点、只乘 geom_xmat + geom_xpos，就会漏掉这一步，
 *   表现为部件整体错位（例如大腿 mesh_pos.z = -0.2229，腿会掉下去 22 cm）。
 *   用 model.mesh_vert 就天然是对的，而且和 MuJoCo 自己的渲染逐顶点一致。
 *
 * 每个视觉几何体单独构建 Mesh，用 data.geom_xpos / data.geom_xmat 定位。
 */

const GEOM_PLANE = 0;
const GEOM_HFIELD = 1;
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;
const GEOM_MESH = 7;

export class MjScene {
  /**
   * @param {object} o
   * @param {any}    o.THREE    three.js 命名空间
   * @param {any}    o.mujoco   MuJoCo wasm 模块
   * @param {any}    o.model    mjModel
   * @param {any}    o.data     mjData
   */
  constructor({ THREE, mujoco, model, data }) {
    this.THREE = THREE;
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;

    this.root = new THREE.Group();
    this.root.name = 'mujoco-zup';

    this.objs = new Array(model.ngeom); // gi -> THREE.Mesh | null
    this.meshGeomIds = [];
    this.envGeomIds = [];   // worldbody 下的盒子等（楼梯）
    this.crabGeomIds = [];  // worldbody 下名字以 crab_ 开头的盒子（螃蟹步台阶，独立分组）
    this.floorGeomIds = []; // worldbody 下的平面
    this.robotGeomIds = [];

    this.opts = {
      showRobot: true,
      showStairs: true,
      showCrab: false,      // 螃蟹步台阶默认隐藏：它离出生点较远，平时只会碍眼
      showFloor: true,
      showCollision: false,
    };
    this.floorOpacity = 1;  // 地面网格不透明度（半透明 VR 透视时淡化，露出真实地面）

    this.stats = { geometries: 0, triangles: 0, drawn: 0 };
    this.disposables = [];
    this._mat = new THREE.Matrix4();
    this._box = new THREE.Box3();
  }

  // ------------------------------------------------------------------ 构建
  build() {
    const { THREE, model, mujoco } = this;
    const rgba = model.geom_rgba;

    const geometryCache = new Map();
    const getMeshGeometry = (meshId) => {
      if (geometryCache.has(meshId)) return geometryCache.get(meshId);
      const g = this._buildMeshGeometry(meshId);
      this.disposables.push(g);
      this.stats.geometries++;
      this.stats.triangles += g.userData.triangles;
      geometryCache.set(meshId, g);
      return g;
    };

    const floorTexture = this._makeFloorTexture();

    for (let gi = 0; gi < model.ngeom; gi++) {
      const group = model.geom_group[gi];
      const type = model.geom_type[gi];
      const inWorld = model.geom_bodyid[gi] === 0;
      const c = [rgba[gi * 4], rgba[gi * 4 + 1], rgba[gi * 4 + 2]];
      const alpha = rgba[gi * 4 + 3];

      let geometry = null;
      let material = null;

      if (type === GEOM_MESH && group === 2) {
        geometry = getMeshGeometry(model.geom_dataid[gi]);
        material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(c[0], c[1], c[2]),
          metalness: 0.72,
          roughness: 0.34,
          envMapIntensity: 1.15,
        });
        this.robotGeomIds.push(gi);
      } else if (type === GEOM_PLANE) {
        // MuJoCo 平面：局部 XY 平面，法线 +Z。three 的 PlaneGeometry 恰好一致。
        const sx = model.geom_size[gi * 3];
        const sy = model.geom_size[gi * 3 + 1];
        const span = Math.max(sx > 0 ? sx * 2 : 80, sy > 0 ? sy * 2 : 80);
        geometry = new THREE.PlaneGeometry(span, span, 1, 1);
        material = new THREE.MeshStandardMaterial({
          map: floorTexture,
          color: 0xffffff,
          metalness: 0.0,
          roughness: 0.92,
          envMapIntensity: 0.35,
        });
        this.floorGeomIds.push(gi);
      } else if (type === GEOM_BOX) {
        const s = model.geom_size;
        geometry = new THREE.BoxGeometry(s[gi * 3] * 2, s[gi * 3 + 1] * 2, s[gi * 3 + 2] * 2);
        const isBig = Math.max(s[gi * 3], s[gi * 3 + 1], s[gi * 3 + 2]) > 0.9;
        material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(c[0], c[1], c[2]),
          metalness: isBig ? 0.05 : 0.2,
          roughness: 0.78,
          envMapIntensity: 0.5,
        });
        if (inWorld) {
          // 螃蟹步台阶按名字前缀单独分组：两套地形相距 4 m 以上，但共用同一个
          // 编译好的模型，回放时只显示当前这一段用到的那套。
          const gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM.value, gi) || '';
          if (gname.startsWith('crab_')) this.crabGeomIds.push(gi);
          else this.envGeomIds.push(gi);
        }
      } else if (type === GEOM_SPHERE) {
        geometry = new THREE.SphereGeometry(model.geom_size[gi * 3], 28, 18);
      } else if (type === GEOM_CAPSULE) {
        const r = model.geom_size[gi * 3];
        const half = model.geom_size[gi * 3 + 1];
        geometry = new THREE.CapsuleGeometry(r, half * 2, 8, 20);
        geometry.rotateX(-Math.PI / 2); // three 胶囊沿 Y，MuJoCo 沿 Z
      } else if (type === GEOM_CYLINDER) {
        const r = model.geom_size[gi * 3];
        const half = model.geom_size[gi * 3 + 1];
        geometry = new THREE.CylinderGeometry(r, r, half * 2, 28, 1, false);
        geometry.rotateX(-Math.PI / 2); // three 圆柱沿 Y，MuJoCo 沿 Z
      } else if (type === GEOM_ELLIPSOID) {
        geometry = new THREE.SphereGeometry(1, 24, 16);
        geometry.scale(
          model.geom_size[gi * 3],
          model.geom_size[gi * 3 + 1],
          model.geom_size[gi * 3 + 2],
        );
      } else {
        // hfield / sdf / 其它：本场景用不到
        this.objs[gi] = null;
        continue;
      }

      if (!material) {
        material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(c[0], c[1], c[2]),
          metalness: 0.1,
          roughness: 0.8,
          transparent: alpha < 0.999,
          opacity: alpha,
        });
      }

      const mesh = new THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = this.robotGeomIds.includes(gi) || group === 0;
      mesh.receiveShadow = true;
      {
        const bodyName = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY.value, model.geom_bodyid[gi]) || 'body';
        const meshName = type === GEOM_MESH
          ? (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH.value, model.geom_dataid[gi]) || 'mesh')
          : `t${type}`;
        mesh.name = `${gi}:${bodyName}:${meshName}`;
      }
      this.root.add(mesh);
      this.objs[gi] = mesh;
    }

    this.sync();
    this.applyOptions();
    return this;
  }

  /**
   * 用 MuJoCo 编译好的网格数据构建「索引几何体」。
   *
   * model.mesh_vert / mesh_face / mesh_normal 是全局拼接数组：
   *   mesh_vert  —— 已经做过重心归零 + scale 的顶点（mesh frame 下）
   *   mesh_face  —— 0 基三角面下标
   *   mesh_normal—— 顶点法线（MuJoCo 自己平滑/共享后的结果）
   * 这套数据和 MuJoCo 内部渲染完全一致，无需再解析 STL。
   *
   * @param {number} meshId
   * @returns {THREE.BufferGeometry}
   */
  _buildMeshGeometry(meshId) {
    const { THREE, model } = this;
    const i = meshId | 0;
    const vadr = model.mesh_vertadr[i] | 0;
    const vnum = model.mesh_vertnum[i] | 0;
    const fadr = model.mesh_faceadr[i] | 0;
    const fnum = model.mesh_facenum[i] | 0;

    const g = new THREE.BufferGeometry();

    const verts = model.mesh_vert?.slice
      ? model.mesh_vert.slice(vadr * 3, (vadr + vnum) * 3)
      : null;

    if (verts && verts.length === vnum * 3) {
      g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(verts), 3));
    } else {
      throw new Error(`mesh ${i} 顶点数据不可用（mesh_vert 缺失）`);
    }

    const faces = model.mesh_face.slice(fadr * 3, (fadr + fnum) * 3);
    g.setIndex(new THREE.BufferAttribute(
      vnum > 65535 ? Uint32Array.from(faces) : Uint16Array.from(faces),
      1,
    ));

    // 顶点法线：直接取 MuJoCo 的结果；遇到无效值再自己重算
    let normalsOk = false;
    try {
      const nadr = model.mesh_normaladr[i] | 0;
      const src = model.mesh_normal;
      if (src?.length >= (nadr + vnum) * 3) {
        const raw = src.slice(nadr * 3, (nadr + vnum) * 3);
        const out = new Float32Array(vnum * 3);
        let bad = 0;
        for (let k = 0; k < vnum; k++) {
          const x = raw[k * 3]; const y = raw[k * 3 + 1]; const z = raw[k * 3 + 2];
          const len = Math.hypot(x, y, z);
          if (!(len > 1e-6)) { bad++; continue; }
          out[k * 3] = x / len; out[k * 3 + 1] = y / len; out[k * 3 + 2] = z / len;
        }
        if (bad === 0) {
          g.setAttribute('normal', new THREE.BufferAttribute(out, 3));
          normalsOk = true;
        }
      }
    } catch { normalsOk = false; }

    if (!normalsOk) g.computeVertexNormals();

    g.computeBoundingBox();
    g.computeBoundingSphere();
    g.userData.triangles = fnum;
    g.userData.source = normalsOk ? 'mujoco-mesh' : 'mujoco-mesh+computed-normals';
    return g;
  }

  /** 地板网格贴图（程序生成，避免外部依赖） */
  _makeFloorTexture() {
    const THREE = this.THREE;
    const S = 512;
    const cv = document.createElement('canvas');
    cv.width = cv.height = S;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#1c232c';
    ctx.fillRect(0, 0, S, S);
    // 细分 8x8 小格
    ctx.strokeStyle = '#2b3846';
    ctx.lineWidth = 2;
    for (let i = 1; i < 8; i++) {
      const p = (i * S) / 8;
      ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, S); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, p); ctx.lineTo(S, p); ctx.stroke();
    }
    // 外框主线
    ctx.strokeStyle = '#3f5a72';
    ctx.lineWidth = 6;
    ctx.strokeRect(3, 3, S - 6, S - 6);

    const tex = new THREE.CanvasTexture(cv);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(40, 40);
    tex.anisotropy = 4;
    tex.colorSpace = THREE.SRGBColorSpace;
    this.disposables.push(tex);
    return tex;
  }

  // ------------------------------------------------------------------ 同步
  /** 把 data 里的世界位姿写到 three.js 对象上（静止时只需调用一次） */
  sync() {
    const { model, data } = this;
    const P = data.geom_xpos;
    const M = data.geom_xmat;
    const m = this._mat;
    const e = m.elements;

    for (let gi = 0; gi < model.ngeom; gi++) {
      const obj = this.objs[gi];
      if (!obj) continue;
      const o3 = gi * 3;
      e[0] = M[gi * 9]; e[1] = M[gi * 9 + 3]; e[2] = M[gi * 9 + 6]; e[3] = 0;
      e[4] = M[gi * 9 + 1]; e[5] = M[gi * 9 + 4]; e[6] = M[gi * 9 + 7]; e[7] = 0;
      e[8] = M[gi * 9 + 2]; e[9] = M[gi * 9 + 5]; e[10] = M[gi * 9 + 8]; e[11] = 0;
      e[12] = P[o3]; e[13] = P[o3 + 1]; e[14] = P[o3 + 2]; e[15] = 1;
      obj.matrix.copy(m);
      obj.matrixWorldNeedsUpdate = true;
    }
    return this;
  }

  /** 依据当前选项刷新可见性 */
  applyOptions() {
    const o = this.opts;
    for (const gi of this.robotGeomIds) {
      const obj = this.objs[gi];
      if (obj) obj.visible = o.showRobot;
    }
    for (const gi of this.envGeomIds) {
      const obj = this.objs[gi];
      if (obj) obj.visible = o.showStairs;
    }
    for (const gi of this.crabGeomIds) {
      const obj = this.objs[gi];
      if (obj) obj.visible = o.showCrab;
    }
    for (const gi of this.floorGeomIds) {
      const obj = this.objs[gi];
      if (obj) obj.visible = o.showFloor;
    }
    // group 3 的碰撞体
    for (let gi = 0; gi < this.model.ngeom; gi++) {
      if (this.model.geom_group[gi] !== 3) continue;
      const obj = this.objs[gi];
      if (obj) obj.visible = o.showCollision;
    }
    return this;
  }

  setOption(key, value) {
    if (!(key in this.opts)) return this;
    this.opts[key] = value;
    this.applyOptions();
    return this;
  }

  /**
   * 地面网格的不透明度（0~1）。
   * 半透明 VR（透视）时调到 0.3 左右：虚拟网格仍给地面参照，但真实地面能透出来。
   * 只作用于 worldbody 下的平面几何，机器人 / 楼梯材质不受影响。
   * @param {number} value
   */
  setFloorOpacity(value) {
    const o = Math.min(1, Math.max(0, Number(value)));
    this.floorOpacity = o;
    for (const gi of this.floorGeomIds) {
      const mat = this.objs[gi]?.material;
      if (!mat) continue;
      mat.transparent = o < 0.999;
      mat.opacity = o;
      mat.depthWrite = o >= 0.999;   // 半透明时关掉深度写入，避免挡住身后物体
      mat.needsUpdate = true;
    }
    return this;
  }

  /**
   * 机器人视觉网格在世界坐标下的最低点高度（MuJoCo 坐标，米）。
   * 逐顶点精确计算，用于把站立姿态的脚底对齐到地面。
   */
  robotMinZ() {
    const { data, model } = this;
    const P = data.geom_xpos;
    const M = data.geom_xmat;
    let minZ = Infinity;
    for (const gi of this.robotGeomIds) {
      const obj = this.objs[gi];
      if (!obj) continue;
      const pos = obj.geometry.getAttribute('position');
      if (!pos) continue;
      const o3 = gi * 3;
      const m9 = gi * 9;
      const m6 = M[m9 + 6]; const m7 = M[m9 + 7]; const m8 = M[m9 + 8];
      const pz = P[o3 + 2];
      const a = pos.array;
      for (let k = 0; k < a.length; k += 3) {
        const z = m6 * a[k] + m7 * a[k + 1] + m8 * a[k + 2] + pz;
        if (z < minZ) minZ = z;
      }
    }
    return minZ;
  }

  /**
   * 机器人视觉网格的世界包围盒（MuJoCo 坐标系，单位米）。
   * 直接用 data.geom_xpos/geom_xmat 计算，不依赖 three.js 的矩阵更新状态，
   * 也不会因为 matrixAutoUpdate=false 而被覆盖。
   */
  robotBounds() {
    const THREE = this.THREE;
    const { data } = this;
    const P = data.geom_xpos;
    const M = data.geom_xmat;
    const box = this._box.makeEmpty();
    const tmp = new THREE.Box3();
    for (const gi of this.robotGeomIds) {
      const obj = this.objs[gi];
      if (!obj) continue;
      const b = obj.geometry.boundingBox;
      const o3 = gi * 3;
      const m9 = gi * 9;
      for (let c = 0; c < 8; c++) {
        const lx = c & 1 ? b.max.x : b.min.x;
        const ly = c & 2 ? b.max.y : b.min.y;
        const lz = c & 4 ? b.max.z : b.min.z;
        const x = M[m9] * lx + M[m9 + 1] * ly + M[m9 + 2] * lz + P[o3];
        const y = M[m9 + 3] * lx + M[m9 + 4] * ly + M[m9 + 5] * lz + P[o3 + 1];
        const z = M[m9 + 6] * lx + M[m9 + 7] * ly + M[m9 + 8] * lz + P[o3 + 2];
        tmp.min.set(x, y, z);
        tmp.max.set(x, y, z);
        box.union(tmp);
      }
    }
    return box;
  }

  /** 每个视觉网格的 MuJoCo 世界轴对齐包围盒，便于逐件核对 */
  robotGeomBounds() {
    const THREE = this.THREE;
    const { data, model, mujoco } = this;
    const P = data.geom_xpos;
    const M = data.geom_xmat;
    const out = [];
    for (const gi of this.robotGeomIds) {
      const obj = this.objs[gi];
      if (!obj) continue;
      const b = obj.geometry.boundingBox;
      const o3 = gi * 3;
      const m9 = gi * 9;
      const lo = [Infinity, Infinity, Infinity];
      const hi = [-Infinity, -Infinity, -Infinity];
      for (let c = 0; c < 8; c++) {
        const lx = c & 1 ? b.max.x : b.min.x;
        const ly = c & 2 ? b.max.y : b.min.y;
        const lz = c & 4 ? b.max.z : b.min.z;
        const v = [
          M[m9] * lx + M[m9 + 1] * ly + M[m9 + 2] * lz + P[o3],
          M[m9 + 3] * lx + M[m9 + 4] * ly + M[m9 + 5] * lz + P[o3 + 1],
          M[m9 + 6] * lx + M[m9 + 7] * ly + M[m9 + 8] * lz + P[o3 + 2],
        ];
        for (let k = 0; k < 3; k++) { if (v[k] < lo[k]) lo[k] = v[k]; if (v[k] > hi[k]) hi[k] = v[k]; }
      }
      out.push({
        gi,
        body: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY.value, model.geom_bodyid[gi]),
        mesh: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH.value, model.geom_dataid[gi]),
        objName: obj.name,
        lo, hi,
      });
    }
    return out;
  }

  /** 把 object 层级挂到一个 Z-up→Y-up 的外层 Group，返回该 Group */
  wrapZUp() {
    const THREE = this.THREE;
    const flip = new THREE.Group();
    flip.rotation.x = -Math.PI / 2; // MuJoCo Z-up -> three Y-up
    flip.add(this.root);
    const yaw = new THREE.Group();
    yaw.rotation.y = Math.PI / 2;   // 让机器人正面朝 three -Z
    yaw.add(flip);
    return yaw;
  }

  dispose() {
    for (const d of this.disposables) d.dispose?.();
    for (const obj of this.objs) obj?.material?.dispose?.();
    this.disposables.length = 0;
  }
}
