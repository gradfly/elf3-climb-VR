/**
 * vrnav.js —— WebXR 沉浸式观看：站位、漫游与转向
 *
 * 设计要点
 *  - 相机保持在世界原点不动（three 会用 XR 头部姿态驱动它），
 *    「移动」通过平移/旋转装着整个模型世界的 stage 组来实现，等价于世界坐标系漫游。
 *  - 参考空间用 local-floor：用户脚踩的就是 y=0 平面，正好与 MuJoCo 地面 z=0 重合。
 *  - 进入 VR 的瞬间自动把机器人摆到用户正前方并转成正面朝向，避免用户需要自己找。
 */

// 右摇杆连续转向：死区内完全不响应（挡住摇杆回中漂移），推到底的角速度见 turnSpeed。
// 不再用「每次 30° + 0.28 s 冷却」的 snap turn —— 阶跃转向在头显里一顿一顿的。
const TURN_DEADZONE = 0.15;

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
    this.turnSpeed = Math.PI / 2;   // rad/s：右摇杆推到底 90°/s，可现场改 __elf.vrNav.turnSpeed
    this.turnDeadzone = TURN_DEADZONE;
    this.recenterDistance = 2.6;
    this.follow = true;         // 回放时是否自动跟随机器人
    // 手动转向 = 用户主动接管观察角度：回调让上层关掉自动跟随（和点视角预设一致），
    // 否则连续转过大角度后 autoFollow 会慢慢把世界拉回正前方，看着像「转不动」。
    this.onManualTurn = null;
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
    // 注视点渲染（fixed foveation）这里**不设** —— 唯一写入点是 XRControls（js/xr.js），
    // 因为等级和 framebufferScale 一样必须早于 setSession 才可靠，且需要能现场调。
    // 别在这里顺手拉满 1.0：边缘着色率降到每 N×N 像素一次后，细长/小/远的几何能不能
    // 被采样到取决于亚像素对齐，左右眼网格不同 → 「单眼丢机器人」。详见 main.js 的 FFR。
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
    // 连续转向会让 rotation.y 一直累加，转几十圈后浮点精度退化；
    // 归一到 (-π, π] 与 autoFollow 的 dy 计算一致，旋转结果等价。
    const y = s.rotation.y + theta;
    s.rotation.y = (y > Math.PI || y < -Math.PI) ? Math.atan2(Math.sin(y), Math.cos(y)) : y;
  }

  /**
   * 把机器人摆到用户正前方 D 米，并让它正面朝用户。
   * @param {THREE.Vector3} [robotWorld] 机器人当前世界位置；不传就用 this.robotWorld。
   *   机器人回放时会走到离世界原点好几米的地方，所以必须按它的**实际位置**摆位，
   *   否则"正对机器人"按钮会把镜头对到空地上。
   * @param {number} [yawOff] 在「正对用户」基准上再绕竖直轴追加的转向（弧度）。
   *   0 = 看正面；Math.PI = 看背面（原地转 180°，仍在正前方）；Math.PI/2 = 看侧面。
   * @param {number} [dist] 机器人离用户的距离（米），默认 recenterDistance。
   */
  recenterToRobot(robotWorld, yawOff = 0, dist = this.recenterDistance) {
    const p = this.headPosition(this._v1);
    const f = this.headForwardXZ(this._v2);
    const s = this.stage;
    const yaw = Math.atan2(f.x, f.z) + yawOff;
    const local = s.worldToLocal((robotWorld || this.robotWorld).clone());
    const c = Math.cos(yaw);
    const sn = Math.sin(yaw);
    const lx = local.x * c + local.z * sn;   // R_y(yaw)·local 的水平分量
    const lz = -local.x * sn + local.z * c;
    s.rotation.set(0, yaw, 0);
    s.position.set(p.x + f.x * dist - lx, 0,
      p.z + f.z * dist - lz);
    // 这里不强制 updateMatrixWorld(true)：renderer.render 每帧都会做一次
    // scene.updateMatrixWorld()，而 stage 的 matrixAutoUpdate 从没关过 ——
    // 强制刷新只是把同一帧的活干第二遍（整棵子树 30+ 个对象）。
    // 需要在改完之后读世界坐标的地方（worldToLocal）自己会 updateWorldMatrix。
    return this;
  }

  /**
   * VR 里的「视角预设」：相机是头显姿态、不可改，所以改为把机器人相对用户摆到对应
   * 观察角度——与桌面 applyView 等价（桌面动相机、VR 动世界 / stage）。
   * 命名沿用 XR 面板里的 6 个视角按钮：front/back/side/squat/grid/wide。
   *   front/back/side —— 原地转动机器人，让用户分别看其正 / 背 / 侧面（仍在正前方）；
   *   squat/grid/wide —— 保持正面朝向、只改观察距离（近景 3/4 / 中景 / 远景全景）。
   * 选了固定机位就关掉自动跟随，否则下一帧 autoFollow 又会把机器人拉回正前方，
   * 按钮看着像"没反应"。
   */
  presentView(name) {
    if (!this.renderer.xr.isPresenting) return false;
    const presets = {
      front: { yaw: 0, dist: this.recenterDistance },
      back: { yaw: Math.PI, dist: this.recenterDistance },
      side: { yaw: Math.PI / 2, dist: this.recenterDistance },
      squat: { yaw: 0, dist: 2.0 },
      grid: { yaw: 0, dist: 8.5 },
      wide: { yaw: 0, dist: 12 },
    };
    const preset = presets[name];
    if (!preset) return false;
    this.recenterToRobot(undefined, preset.yaw, preset.dist);
    this.follow = false;   // 见上方说明
    return true;
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
      }
    }

    // —— 连续转向 ——
    // 相机是头显姿态、动不了，所以按角速度连续旋转 stage = 用户原地平滑转身。
    // 死区重映射：|turn| <= dead 完全不转，超过后线性放大到推到底的 turnSpeed；
    // 半推摇杆就能慢慢转，不再有 30° 阶跃与 0.28 s 冷却。
    const a = Math.abs(turn);
    if (a > this.turnDeadzone) {
      const t = Math.min(1, (a - this.turnDeadzone) / (1 - this.turnDeadzone));
      const step = Math.min(dt, 0.1);  // 卡顿/切场景时 dt 异常大，限幅避免一次跳几十度
      this._rotateAroundHead(Math.sign(turn) * this.turnSpeed * t * step);
      if (this.onManualTurn) this.onManualTurn();
    }
  }
}
