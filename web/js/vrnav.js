/**
 * vrnav.js —— WebXR 沉浸式观看：站位、漫游与转向
 *
 * 设计要点
 *  - 相机保持在世界原点不动（three 会用 XR 头部姿态驱动它），
 *    「移动」通过平移/旋转装着整个模型世界的 stage 组来实现，等价于世界坐标系漫游。
 *  - 参考空间用 local-floor：用户脚踩的就是 y=0 平面，正好与 MuJoCo 地面 z=0 重合。
 *  - 进入 VR 的瞬间自动把机器人摆到用户正前方并转成正面朝向，避免用户需要自己找。
 */

const SNAP_ANGLE = Math.PI / 6; // 每次旋转 30°

export class VRNav {
  /**
   * @param {object} o
   * @param {any} o.THREE
   * @param {any} o.renderer  WebGLRenderer（已开启 xr）
   * @param {any} o.camera
   * @param {any} o.scene
   * @param {any} o.stage     承载 MuJoCo 世界的 Group
   */
  constructor({ THREE, renderer, camera, scene, stage }) {
    this.THREE = THREE;
    this.renderer = renderer;
    this.camera = camera;
    this.scene = scene;
    this.stage = stage;

    this.enabled = true;
    this.moveSpeed = 1.6;       // m/s
    this.snapTurn = true;
    this.recenterDistance = 2.6;
    this.turnCooldown = 0;
    this.follow = true;         // 回放时是否自动跟随机器人
    this.robotWorld = new THREE.Vector3();  // 机器人当前世界位置（由 main.js 每帧更新）
    // 舒适窗口：机器人在这个范围内就完全不动，避免和用户自己的摇杆移动打架
    this.followWindow = { min: 1.7, max: 4.8, front: -0.25 };

    this.controllers = [];
    this._v1 = new THREE.Vector3();
    this._v2 = new THREE.Vector3();
    this._v3 = new THREE.Vector3();
    this._up = new THREE.Vector3(0, 1, 0);
    this._q = new THREE.Quaternion();

    this.recenterRequested = false;
  }

  init() {
    const { THREE, renderer } = this;
    this.refSpaceType = 'local-floor';
    renderer.xr.setReferenceSpaceType(this.refSpaceType);
    renderer.xr.setFoveation(0.5);
    this.controllers = [];

    // —— 手柄：射线指针 + 简易手柄模型 ——
    const rayGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -1),
    ]);
    const rayMat = new THREE.LineBasicMaterial({
      color: 0x54d1ff, transparent: true, opacity: 0.65,
    });

    for (let i = 0; i < 2; i++) {
      const ctrl = renderer.xr.getController(i);
      const ray = new THREE.Line(rayGeo, rayMat);
      ray.scale.z = 1.6;
      ctrl.add(ray);

      const body = new THREE.Mesh(
        new THREE.CapsuleGeometry(0.018, 0.07, 4, 12),
        new THREE.MeshStandardMaterial({ color: 0x30363d, roughness: 0.5, metalness: 0.4 }),
      );
      body.rotation.x = Math.PI / 2;
      body.position.z = -0.005;
      ctrl.add(body);

      ctrl.addEventListener('connected', (e) => { ctrl.userData.inputSource = e.data; });
      ctrl.addEventListener('disconnected', () => { ctrl.userData.inputSource = null; });

      this.scene.add(ctrl);
      this.controllers.push(ctrl);
    }

    // 站姿起点圆环，给用户一个地面参照（固定在用户物理空间原点）
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.33, 48),
      new THREE.MeshBasicMaterial({
        color: 0x54d1ff, transparent: true, opacity: 0.4, side: THREE.DoubleSide,
      }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.006;
    ring.name = 'playspace-ring';
    this.scene.add(ring);
    this.ring = ring;

    renderer.xr.addEventListener('sessionstart', () => {
      // 会话刚建立时头部姿态还没生效，标一个待处理，下一帧真正摆位
      this.recenterRequested = true;
    });
  }

  // ------------------------------------------------------------------ 工具
  /** 头显世界坐标 */
  headPosition(out) {
    const cam = this.renderer.xr.isPresenting ? this.renderer.xr.getCamera() : this.camera;
    return cam.getWorldPosition(out ?? this._v1);
  }

  /** 头显朝向在水平面的单位向量 */
  headForwardXZ(out) {
    const cam = this.renderer.xr.isPresenting ? this.renderer.xr.getCamera() : this.camera;
    const v = cam.getWorldDirection(out ?? this._v2);
    v.y = 0;
    if (v.lengthSq() < 1e-8) v.set(0, 0, -1);
    return v.normalize();
  }

  /** 绕头显所在竖直轴旋转 stage */
  _rotateAroundHead(theta) {
    const p = this.headPosition(this._v3);
    const s = this.stage;
    const dx = s.position.x - p.x;
    const dz = s.position.z - p.z;
    const c = Math.cos(theta), sn = Math.sin(theta);
    s.position.x = p.x + (dx * c + dz * sn);
    s.position.z = p.z + (-dx * sn + dz * c);
    s.rotation.y += theta;
  }

  /**
   * 把机器人摆到用户正前方 D 米，并让它正面朝用户。
   * @param {THREE.Vector3} [robotWorld] 机器人当前世界位置；不传就用 this.robotWorld。
   *   机器人回放时会走到离世界原点好几米的地方，所以必须按它的**实际位置**摆位，
   *   否则"正对机器人"按钮会把镜头对到空地上。
   */
  recenterToRobot(robotWorld) {
    const p = this.headPosition(this._v1);
    const f = this.headForwardXZ(this._v2);
    const s = this.stage;
    const yaw = Math.atan2(f.x, f.z);
    const local = s.worldToLocal((robotWorld || this.robotWorld).clone());
    const c = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const lx = local.x * c + local.z * sn;   // R_y(yaw)·local 的水平分量
    const lz = -local.x * sn + local.z * c;
    s.rotation.set(0, yaw, 0);
    s.position.set(p.x + f.x * this.recenterDistance - lx, 0,
      p.z + f.z * this.recenterDistance - lz);
    s.updateMatrixWorld(true);
    return this;
  }

  /**
   * 回放跟随：机器人跑出舒适窗口（太近 / 太远 / 绕到身后）时慢慢把它拉回正前方；
   * 在窗口内什么都不做，用户照样可以用摇杆自由移动。
   */
  autoFollow(dt, robotWorld) {
    if (!this.enabled || !this.follow) return;
    if (!this.renderer.xr.isPresenting) return;
    const w = robotWorld || this.robotWorld;
    const p = this.headPosition(this._v1);
    const f = this.headForwardXZ(this._v2);
    const dx = w.x - p.x;
    const dz = w.z - p.z;
    const dist = Math.hypot(dx, dz);
    const front = dist > 1e-6 ? (dx * f.x + dz * f.z) / dist : 1;
    const win = this.followWindow;
    if (dist >= win.min && dist <= win.max && front > win.front) return;

    const yaw = Math.atan2(f.x, f.z);
    const local = this.stage.worldToLocal(this._v3.copy(w));
    const c = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const lx = local.x * c + local.z * sn;
    const lz = -local.x * sn + local.z * c;
    const tx = p.x + f.x * this.recenterDistance - lx;
    const tz = p.z + f.z * this.recenterDistance - lz;

    const k = 1 - Math.pow(0.05, Math.min(dt, 0.1));   // ≈0.33 s 时间常数
    const s = this.stage;
    s.position.x += (tx - s.position.x) * k;
    s.position.z += (tz - s.position.z) * k;
    let dy = yaw - s.rotation.y;
    while (dy > Math.PI) dy -= 2 * Math.PI;
    while (dy < -Math.PI) dy += 2 * Math.PI;
    s.rotation.y += dy * k;
    s.updateMatrixWorld(true);
    return this;
  }

  /** 世界回到初始位置（机器人在世界原点） */
  resetStage() {
    this.stage.rotation.set(0, 0, 0);
    this.stage.position.set(0, 0, 0);
    return this;
  }

  /** 后退/前进到机器人周围的一组预设观察位（非 VR 与 VR 都可用） */
  placeAt(angleDeg, distance) {
    const f = this.headForwardXZ(this._v2);
    // 以「用户当前朝向」为基准，把机器人放到指定方位
    const base = Math.atan2(f.x, f.z);
    const a = base + (angleDeg * Math.PI) / 180;
    const s = this.stage;
    s.rotation.set(0, a, 0);
    const p = this.headPosition(this._v1);
    s.position.set(p.x + Math.sin(a) * distance, 0, p.z + Math.cos(a) * distance);
    s.updateMatrixWorld(true);
    return this;
  }

  // ------------------------------------------------------------------ 每帧
  update(dt) {
    if (!this.enabled) return;
    const { renderer } = this;
    if (!renderer.xr.isPresenting) return;

    if (this.recenterRequested) {
      this.recenterRequested = false;
      this.recenterToRobot();
    }

    this.turnCooldown = Math.max(0, this.turnCooldown - dt);

    const session = renderer.xr.getSession();
    if (!session) return;

    let moveX = 0, moveY = 0, turn = 0;
    let recenterBtn = false, resetBtn = false;

    for (const src of session.inputSources) {
      const gp = src.gamepad;
      if (!gp) continue;
      const ax = gp.axes || [];
      // WebXR 标准：axes[0..1] 触摸板，axes[2..3] 主摇杆
      const sx = ax.length > 3 ? ax[2] : (ax[0] ?? 0);
      const sy = ax.length > 3 ? ax[3] : (ax[1] ?? 0);

      if (src.handedness === 'left') {
        moveX = sx;
        moveY = -sy;             // Quest 上摇杆前推为 -1
      } else if (src.handedness === 'right') {
        turn = sx;
      } else {
        moveX = sx; moveY = -sy; // 未知手别：当成移动
      }

      const btns = gp.buttons || [];
      const pressed = (i) => !!(btns[i] && btns[i].pressed);
      if (src.handedness === 'right') {
        if (pressed(4)) recenterBtn = true; // A / X
        if (pressed(5)) resetBtn = true;    // B / Y
      } else {
        if (pressed(4)) recenterBtn = true;
        if (pressed(5)) resetBtn = true;
      }
    }

    if (recenterBtn) this.recenterRequested = true;
    if (resetBtn) this.resetStage();

    // —— 平移 ——
    const mag = Math.hypot(moveX, moveY);
    if (mag > 0.12) {
      const f = this.headForwardXZ(this._v1);
      const right = this._v2.copy(f).cross(this._up).normalize(); // f × up = 用户右手方向
      const mv = this._v3.set(0, 0, 0)
        .addScaledVector(f, moveY)
        .addScaledVector(right, moveX);
      if (mv.lengthSq() > 1e-8) {
        mv.normalize().multiplyScalar(this.moveSpeed * dt);
        this.stage.position.sub(mv);
        this.stage.updateMatrixWorld(true);
      }
    }

    // —— 转向 ——
    if (Math.abs(turn) > 0.7 && this.turnCooldown <= 0) {
      this._rotateAroundHead(Math.sign(turn) * SNAP_ANGLE);
      this.stage.updateMatrixWorld(true);
      this.turnCooldown = 0.28;
    }
  }
}
