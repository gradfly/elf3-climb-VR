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

/**
 * rgb → 量化后的 key 片段。
 * 必须量化：elf3.xml 里躯干写 `0.75294`、其余 31 个写 `0.752941`，经 sRGB→linear
 * 之后在 float32 下并不相等，按精确值做 key 会悄悄裂成两个材质（看不出错，但白丢一半共享）。
 */
const _q3 = (c) => `${Math.round(c[0] * 1000)},${Math.round(c[1] * 1000)},${Math.round(c[2] * 1000)}`;

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
    this._box2 = new THREE.Box3();      // robotBounds 的复用临时盒（别每帧 new）

    /**
     * 每次 sync() 要写矩阵的 geom 清单（null = 还没初始化 → 全量遍历）。
     * 由 applyOptions() 里的 _rebuildSyncIds() 重建：可见 geom ∪ 矩阵模式下的实例源。
     * 不裁剪的话 94 个 geom 里有 36 个隐藏碰撞球 + 两套隐藏地形在白白写矩阵。
     */
    this._syncIds = null;
    this._dirty = false;                // applyOptions() 置位，主循环据此补一次 sync()
    this._syncCount = 0;                // 自检用：累计 sync 次数

    /**
     * 材质共享表。32 个视觉 geom 的 rgba 其实同色，各建一份 MeshStandardMaterial
     * 会让每个 draw 都重新上传环境贴图/粗糙度等 uniform（双眼 64 个 draw）。
     * key 用**量化后**的颜色 —— 0.75294 与 0.752941 经 sRGB→linear 在 float32 下
     * 并不相等，按精确值做 key 会悄悄裂成两个材质。
     */
    this._matCache = new Map();
    this.robotMaterial = 'standard';   // 'standard' | 'lambert'（见 setRobotMaterial）

    /** 矩阵模式：机器人视觉网格的实例化副本（见 enableRobotGrid），未开启时为 null */
    this.grid = null;
    this._gridTmp = new Float32Array(16);
    this._sphere = new THREE.Sphere();
    this._frusta = null;               // 逐眼视锥（XR 里是两只眼）
    this._projScreen = new THREE.Matrix4();
  }

  // ------------------------------------------------------------------ 构建
  /** 按 key 共享材质；工厂只在第一次命中时执行 */
  _sharedMat(key, factory) {
    let m = this._matCache.get(key);
    if (!m) {
      m = factory();
      this._matCache.set(key, m);
    }
    return m;
  }

  /**
   * 切换机器人视觉网格的着色模型（只影响 32 个视觉 geom，不碰地板 / 地形 / 碰撞代理）。
   *
   *   'standard'（默认）MeshStandardMaterial + PMREM 环境贴图：金属质感、GGX 高光；
   *   'lambert'  MeshLambertMaterial：只有漫反射（逐顶点光照），没有 metalness /
   *              roughness / envMap → 机器人会变成**哑光塑料**。片元成本大约降到
   *              1/2~1/3（省掉 GGX 求值 + 两次 cubemapUV 采样 + 一大段 BRDF 数学）。
   *
   * 只换材质引用：几何、矩阵、实例化都不动；矩阵模式下 32 个 InstancedMesh 也跟着换
   * （它们共享原件材质）。用途是评估「为了帧率值不值得牺牲金属感」。
   * 启动时用 `?mat=lambert`，运行期用 `__elf.setRobotMaterial('lambert')`。
   *
   * @param {'standard'|'lambert'} kind
   * @returns {string} 实际生效的模型名
   */
  setRobotMaterial(kind) {
    const lam = kind === 'lambert';
    const THREE = this.THREE;
    for (const gi of this.robotGeomIds) {
      const obj = this.objs[gi];
      const key = obj && obj.userData.matKey;
      if (!key) continue;
      if (lam) {
        obj.material = this._sharedMat(`L|${key}`, () => {
          const std = this._matCache.get(key);
          return new THREE.MeshLambertMaterial({ color: std.color.clone() });
        });
      } else {
        obj.material = this._matCache.get(key);
      }
    }
    // 矩阵模式：实例材质是从原件传过去的引用，必须一起换
    const g = this.grid;
    if (g) {
      for (let k = 0; k < g.meshes.length; k++) {
        g.meshes[k].material = this.objs[g.geoms[k]].material;
      }
    }
    this.robotMaterial = lam ? 'lambert' : 'standard';
    return this.robotMaterial;
  }
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
      let matKey = null;      // 机器人材质在 _matCache 里的 key（切 Lambert 时要能换回来）

      if (type === GEOM_MESH && group === 2) {
        geometry = getMeshGeometry(model.geom_dataid[gi]);
        // 32 个视觉 geom 同色 → 共享一份材质（见 _matCache 的注释）
        matKey = `robot|${_q3(c)}|${alpha < 0.999 ? Math.round(alpha * 100) : 100}`;
        material = this._sharedMat(
          matKey,
          () => new THREE.MeshStandardMaterial({
            color: new THREE.Color(c[0], c[1], c[2]),
            metalness: 0.72,
            roughness: 0.34,
            envMapIntensity: 1.15,
          }),
        );
        this.robotGeomIds.push(gi);
      } else if (type === GEOM_PLANE) {
        // MuJoCo 平面：局部 XY 平面，法线 +Z。three 的 PlaneGeometry 恰好一致。
        const sx = model.geom_size[gi * 3];
        const sy = model.geom_size[gi * 3 + 1];
        const span = Math.max(sx > 0 ? sx * 2 : 80, sy > 0 ? sy * 2 : 80);
        geometry = new THREE.PlaneGeometry(span, span, 1, 1);
        // 地板是唯一需要单独改 opacity 的（半透明 VR 透视时淡化），不进共享表
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
        material = this._sharedMat(
          `box|${_q3(c)}|${isBig ? 1 : 0}|${alpha < 0.999 ? Math.round(alpha * 100) : 100}`,
          () => new THREE.MeshStandardMaterial({
            color: new THREE.Color(c[0], c[1], c[2]),
            metalness: isBig ? 0.05 : 0.2,
            roughness: 0.78,
            envMapIntensity: 0.5,
            transparent: alpha < 0.999,
            opacity: alpha,
          }),
        );
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
        // 碰撞代理（球 / 胶囊 / 圆柱）：只做调试图层，同色的共享一份就够
        material = this._sharedMat(
          `gen|${type}|${_q3(c)}|${alpha < 0.999 ? Math.round(alpha * 100) : 100}`,
          () => new THREE.MeshStandardMaterial({
            color: new THREE.Color(c[0], c[1], c[2]),
            metalness: 0.1,
            roughness: 0.8,
            transparent: alpha < 0.999,
            opacity: alpha,
          }),
        );
      }

      const mesh = new THREE.Mesh(geometry, material);
      mesh.matrixAutoUpdate = false;
      if (matKey) mesh.userData.matKey = matKey;
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
  /** 把一个 geom 的 MuJoCo 世界位姿写成 three 的本地矩阵 */
  _writeGeom(gi) {
    const obj = this.objs[gi];
    if (!obj) return;
    const { data } = this;
    const P = data.geom_xpos;
    const M = data.geom_xmat;
    const e = this._mat.elements;
    const o3 = gi * 3;
    const m9 = gi * 9;
    e[0] = M[m9]; e[1] = M[m9 + 3]; e[2] = M[m9 + 6]; e[3] = 0;
    e[4] = M[m9 + 1]; e[5] = M[m9 + 4]; e[6] = M[m9 + 7]; e[7] = 0;
    e[8] = M[m9 + 2]; e[9] = M[m9 + 5]; e[10] = M[m9 + 8]; e[11] = 0;
    e[12] = P[o3]; e[13] = P[o3 + 1]; e[14] = P[o3 + 2]; e[15] = 1;
    obj.matrix.copy(this._mat);
    obj.matrixWorldNeedsUpdate = true;
  }

  /**
   * 把 data 里的世界位姿写到 three.js 对象上（静止时只需调用一次）。
   *
   * 只写「会动」的 geom：世界体（地板 + 两套地形盒）在 mj_kinematics / mj_step 下
   * 位姿永远不变，由 _writeStatic() 在 applyOptions 之后写一次就够了。
   */
  sync() {
    const ids = this._dynamicIds;      // undefined = build() 首次调用 → 走全量
    if (ids) {
      for (let k = 0; k < ids.length; k++) this._writeGeom(ids[k]);
    } else {
      for (let gi = 0; gi < this.model.ngeom; gi++) this._writeGeom(gi);
    }
    if (this.grid) this._syncGrid();
    this._syncCount++;
    this._dirty = false;
    return this;
  }

  /** 世界体几何：位姿恒定，只在可见性/地形切换后写一次 */
  _writeStatic() {
    const ids = this._staticIds;
    if (!ids) return this;
    for (let k = 0; k < ids.length; k++) this._writeGeom(ids[k]);
    return this;
  }

  /**
   * 重建「要写矩阵的 geom」清单。
   *
   * = 当前可见的 geom ∪ （矩阵模式下）机器人视觉网格的原件。
   * 后者必须无条件纳入：矩阵模式下原件 visible=false，但 InstancedMesh 的实例矩阵
   * 就是从 this.objs[gi].matrix 读的（见 _syncGrid），漏掉会让整片实例冻在第一帧。
   */
  _rebuildSyncIds() {
    const inGrid = !!this.grid;
    const dyn = [];
    const sta = [];
    for (let gi = 0; gi < this.model.ngeom; gi++) {
      const obj = this.objs[gi];
      if (!obj) continue;
      const isRobot = this.robotGeomIds.includes(gi);
      if (!obj.visible && !(inGrid && isRobot)) continue;   // 隐藏的跳过（矩阵模式保留实例源）
      // 按 body 判：worldbody（地板 + 地形盒）的位姿恒定，机器人本体与碰撞代理会动
      if (this.model.geom_bodyid[gi] === 0) sta.push(gi);
      else dyn.push(gi);
    }
    this._dynamicIds = dyn;
    this._staticIds = sta;
    this._syncIds = dyn.concat(sta);   // 并集：自检与诊断仍按它数
    return this;
  }

  // ------------------------------------------------------------------ 矩阵模式
  /**
   * 把机器人视觉网格复制成 rows × cols 份、按网格铺开的实例（InstancedMesh）。
   *
   * 为什么用实例化而不是克隆 Mesh：96 份 × 32 个几何 = 3072 个对象 / 3072 次
   * draw call，浏览器直接跪；实例化后几何与材质只有一份，绘制批次仍等于机器人
   * 几何数（32），只是每批画 count 次。
   *
   * 实例矩阵 = 平移(offset) × 几何世界矩阵。因为 offset 是纯平移，乘出来只有
   * 平移列不同 → 每帧只改 4 个分量就够了（见 _syncGrid）。
   *
   * 排布在 MuJoCo 坐标系里给：col 沿 mj y（左右），row 沿 mj x（前后），
   * 整体以原点（单人模式那个机器人的位置）为中心。
   *
   * @param {object} [o]
   * @param {number} [o.rows]   行数（前后方向）    默认 8
   * @param {number} [o.cols]   列数（左右方向）    默认 12
   * @param {number} [o.pitchX] 列间距（米）        默认 1.08
   * @param {number} [o.pitchZ] 行间距（米）        默认 1.17
   */
  enableRobotGrid({ rows = 8, cols = 12, pitchX = 1.08, pitchZ = 1.17 } = {}) {
    this.disableRobotGrid();
    const THREE = this.THREE;
    const count = rows * cols;

    const ox = new Float32Array(count);  // MuJoCo x 偏移（前后）
    const oy = new Float32Array(count);  // MuJoCo y 偏移（左右）
    let n = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++, n++) {
        ox[n] = (r - (rows - 1) / 2) * pitchZ;
        oy[n] = (c - (cols - 1) / 2) * pitchX;
      }
    }

    const group = new THREE.Group();
    group.name = 'robot-grid';
    const meshes = [];
    const geoms = [];
    for (const gi of this.robotGeomIds) {
      const src = this.objs[gi];
      if (!src) continue;
      const im = new THREE.InstancedMesh(src.geometry, src.material, count);
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      im.frustumCulled = false;   // 实例铺满十几米，默认包围球会误判剔除
      im.castShadow = false;      // 200 份阴影没有视觉收益，只是白烧一遍 GPU
      im.receiveShadow = false;
      im.matrixAutoUpdate = false;
      im.name = `${src.name}:grid`;
      group.add(im);
      meshes.push(im);
      geoms.push(gi);
    }
    this.root.add(group);
    this.grid = {
      rows, cols, count, pitchX, pitchZ, ox, oy, group, meshes, geoms,
      // 逐实例可见性：全 1 表示不剔除（桌面 / ?cull=0）
      mask: new Uint8Array(count).fill(1),
      drawn: count,
    };
    this.sync();          // 先把实例矩阵填好，避免第一帧全在原点
    this.applyOptions();
    return this.grid;
  }

  /**
   * 逐实例视锥剔除（只在 XR 里调用）。
   *
   * InstancedMesh 的包围球罩住整片网格，three 只能「整体画 / 整体不画」，所以
   * `frustumCulled = false` 时 96 份全画。实测站在机器人前 2.6 m、水平半 FOV 48°
   * 的典型站姿下，8×12 阵型里只有约 45 份在视野内 —— 另外一半是纯白烧的
   * 96 × 20.6 万 = 1980 万三角/眼。这是矩阵模式下唯一够大的无损收益。
   *
   * 实例世界位置 = 机器人世界中心 + R_root · (ox, oy, 0)：
   * 实例矩阵是在 MuJoCo 空间里「几何世界矩阵 + 纯平移」，再整体乘 root.matrixWorld，
   * 所以偏移量要跟着 root 的旋转（Z-up→Y-up + stage 的 30° 转向）一起转。
   *
   * @param {any} camera XR 的 ArrayCamera（双目）或普通 Camera
   * @param {THREE.Vector3} robotWorld 机器人中心（three 世界坐标，上一帧值即可）
   * @param {number} radius 包围球半径：要盖住机器人（约 0.9 m）+ 一帧位姿滞后
   * @returns {number} 本帧可见份数
   */
  cullGrid(camera, robotWorld, radius = 1.7) {
    const g = this.grid;
    if (!g || !camera) return 0;
    const THREE = this.THREE;
    const cams = camera.isArrayCamera ? camera.cameras : [camera];
    if (!cams.length) return g.drawn;
    if (!this._frusta || this._frusta.length !== cams.length) {
      this._frusta = cams.map(() => new THREE.Frustum());
    }
    for (let i = 0; i < cams.length; i++) {
      this._projScreen.multiplyMatrices(cams[i].projectionMatrix, cams[i].matrixWorldInverse);
      this._frusta[i].setFromProjectionMatrix(this._projScreen);
    }

    const mw = this.root.matrixWorld.elements;
    const cx = robotWorld ? robotWorld.x : 0;
    const cy = robotWorld ? robotWorld.y : 0;
    const cz = robotWorld ? robotWorld.z : 0;
    const sp = this._sphere;
    sp.radius = radius;

    const mask = g.mask;
    let n = 0;
    for (let i = 0; i < g.count; i++) {
      const ox = g.ox[i];
      const oy = g.oy[i];
      sp.center.set(cx + mw[0] * ox + mw[4] * oy,
        cy + mw[1] * ox + mw[5] * oy,
        cz + mw[2] * ox + mw[6] * oy);
      let vis = false;
      for (let k = 0; k < this._frusta.length; k++) {
        if (this._frusta[k].intersectsSphere(sp)) { vis = true; break; }
      }
      mask[i] = vis ? 1 : 0;
      if (vis) n++;
    }
    // 一份都不可见时退回「全画」：会话第一帧 xr.getCamera() 还没被 render 更新，
    // 投影矩阵可能是单位阵，照撤会把 96 份全剔掉 → 头显里闪一下空白。
    // 反过来用户真的背对矩阵时多画一些只是浪费，不会出错。
    if (n === 0) {
      mask.fill(1);
      return g.count;
    }
    return n;
  }

  /** 关闭矩阵模式（原件重新按显示开关显示） */
  disableRobotGrid() {
    if (!this.grid) return this;
    this.root.remove(this.grid.group);
    // 只释放实例缓冲；geometry / material 是原件共享的，不能动
    for (const im of this.grid.meshes) im.dispose();
    this.grid = null;
    this.applyOptions();
    return this;
  }

  /** 刷新所有实例的 instanceMatrix（旋转部分共用，只有平移列随 offset 变） */
  _syncGrid() {
    const { count, ox, oy, meshes, geoms, mask } = this.grid;
    const tmp = this._gridTmp;
    let drawn = 0;
    for (let k = 0; k < meshes.length; k++) {
      const e = this.objs[geoms[k]].matrix.elements;
      const arr = meshes[k].instanceMatrix.array;
      tmp[0] = e[0]; tmp[1] = e[1]; tmp[2] = e[2]; tmp[3] = 0;
      tmp[4] = e[4]; tmp[5] = e[5]; tmp[6] = e[6]; tmp[7] = 0;
      tmp[8] = e[8]; tmp[9] = e[9]; tmp[10] = e[10]; tmp[11] = 0;
      tmp[15] = 1;
      // 只把可见的那些「紧凑」写到缓冲前段，然后用 count 限定绘制份数
      let n = 0;
      for (let i = 0; i < count; i++) {
        if (!mask[i]) continue;
        tmp[12] = e[12] + ox[i];
        tmp[13] = e[13] + oy[i];
        tmp[14] = e[14];
        arr.set(tmp, n * 16);
        n++;
      }
      meshes[k].count = n;
      meshes[k].instanceMatrix.needsUpdate = true;
      drawn = n;
    }
    this.grid.drawn = drawn;
  }

  /** 依据当前选项刷新可见性 */
  applyOptions() {
    const o = this.opts;
    // 矩阵模式下原件必须藏起来（画的是实例），实例显隐由同一个「机器人」开关控制
    for (const gi of this.robotGeomIds) {
      const obj = this.objs[gi];
      if (obj) obj.visible = o.showRobot && !this.grid;
    }
    if (this.grid) {
      for (const im of this.grid.meshes) im.visible = o.showRobot;
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
    this._rebuildSyncIds();
    this._writeStatic();  // 地形/地板刚变可见性，位姿要立刻写一次（它们之后不会再变）
    this._dirty = true;   // 让主循环补一次 sync()（暂停/静止时也生效）
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
    const { data } = this;
    const P = data.geom_xpos;
    const M = data.geom_xmat;
    const box = this._box.makeEmpty();
    const tmp = this._box2;   // 复用，避免每帧 new Box3()（每帧 256 次 union 的 GC 来源）
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
    this.disableRobotGrid();
    for (const d of this.disposables) d.dispose?.();
    // 材质现在是共享的，要按引用去重，否则同一个材质会被 dispose 多次
    const seen = new Set();
    for (const obj of this.objs) {
      const m = obj && obj.material;
      if (m && !seen.has(m)) { seen.add(m); m.dispose?.(); }
    }
    // 切过 Lambert 时，缓存的「另一套」材质不在任何 obj 上，也要一起释放
    for (const m of this._matCache.values()) {
      if (!seen.has(m)) { seen.add(m); m.dispose?.(); }
    }
    this._matCache.clear();
    this.disposables.length = 0;
  }
}
