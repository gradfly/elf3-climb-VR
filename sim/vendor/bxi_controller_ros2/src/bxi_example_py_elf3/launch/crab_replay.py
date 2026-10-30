"""crab_replay.py —— 螃蟹步横移爬楼梯【轨迹回放器】

把 crab_climb.py 规划出来的 qpos 序列（.npz）灌回 MuJoCo 场景里播放。

==================== 三种模式 ====================
kin （默认）纯运动学回放：直接把 qpos 写进 data，不跑物理。
            用来检查"轨迹本身对不对"——脚有没有踩在踏面上、有没有穿模、
            身体姿态看着自不自然。这一路一定能放完，因为它不做物理。

phys 物理回放：用 PD + 重力前馈去跟踪这条 qpos 轨迹，跑真实 mj_step。
            用来回答"这条轨迹在真实物理里站不站得住"。
            预期：会摔。crab_climb.py 的体检已经算出约 43% 的帧质心在
            支撑面外（单腿支撑期最长 0.24 s），那段需要动力学/RL 兜底，
            脚本控制器补不上。摔得越晚说明轨迹越接近可行。

policy 策略回放：加载训练导出的 ONNX 策略，闭环跑真实物理。
            两套观测布局，必须与训练任务对应：
              --policy onnx          基线任务 290 维：
                lin_vel(3)+ang_vel(3)+gravity(3)+joint_pos(31)+joint_vel(31)
                +last_action(29)+command(3)+height_scan(187)
              --policy onnx-residual 残差任务 393 维：
                lin_vel(3)+ang_vel(3)+gravity(3)+joint_pos(31)+ref_err(12)
                +joint_vel(31)+last_action(29)+command(3)+phase(2)
                +ref_meta(3)+height_scan(273)
                动作走 q_target = q_ref(rise,tread,phase) + clip(scale·a, ±clip_rad)
            执行器用 mjlab 同款 <position>（kp/kv 取自 elf3_play 真值）。

用法
----
  python crab_replay.py                       # 回放默认轨迹，开窗
  python crab_replay.py --speed 0.3           # 0.3 倍速慢放
  python crab_replay.py --mode phys           # 物理回放（看它什么时候摔）
  python crab_replay.py --record out/crab     # 离屏渲染，存 PNG 序列
  python crab_replay.py --traj xxx.npz --scene xxx.xml
  python crab_replay.py --policy onnx --onnx policy.onnx   # 基线策略（290 维）
  python crab_replay.py --policy onnx-residual --onnx res.onnx --cmd-vy 0.18
                                              # 残差策略（393 维，会自己估 rise）

窗口快捷键
----------
  空格   暂停 / 继续
  R      从头开始
  [ ]    减速 / 加速
  T      切换"相机跟随躯干"
  Tab    切换 MuJoCo 自带 UI 面板

注意
----
  * 模型路径必须传【绝对路径】，MuJoCo 的 <include> 相对解析在小路径下会
    拼出重复前缀而报错。本脚本内部已统一 resolve()。
  * --record 只存 PNG，不合成 mp4 —— 环境里没有 ffmpeg。可用
    `ffmpeg -framerate 50 -i crab_%05d.png -c:v libx264 out.mp4` 自己合。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

# mujoco.viewer 不会自动挂到 mujoco 命名空间上，必须显式 import，
# 否则 mujoco.viewer 会 AttributeError。无显示环境下这个 import 也可能失败，
# 所以用 try 包住，后面再决定是否降级到离屏渲染。
try:
    import mujoco.viewer as mj_viewer                       # noqa: F401
except Exception:                                          # pragma: no cover
    mj_viewer = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import crab_ik as K                                              # noqa: E402
from bxi_example_py_elf3.policies.joints import (                # noqa: E402
    ELF3_ISAAC_PARAMETERS as P,
)
# 复用 elf3_play 的模型构建：把 <motor> 换成 mjlab 训练时用的 <position> 执行器
from elf3_play import (                                          # noqa: E402
    TrainParams,
    build_model,
    JOINT_NAMES as _JOINT_NAMES,
    _DEFAULT_JOINT_POS,
    _ACTION_SCALE,
)

DATA = HERE.parent / "data" / "mujoco_simulation"
DEFAULT_SCENE = DATA / "elf3_crab.xml"
DEFAULT_TRAJ = DATA / "crab_climb_traj.npz"

LEG_JOINTS = K.LEG_JOINTS          # 12 个腿关节（含腰），顺序见 crab_ik.py


# ---------------------------------------------------------------- 相机
def make_cam(lookat=None, dist=2.8, az=160.0, el=-14.0):
    """构造一个第三人称自由相机。

    默认 az=160 / el=-14：这台机器人朝 +X 站、沿 +Y 横移上楼，台阶剖面在
    YZ 平面上，所以虚拟相机要站在 ±X 侧才看得到"台阶"而不是一堵斜面墙。
    160 度是偏 -X、稍微带一点 +Y 的斜前方，机器人身体和台阶剖面能同时入画。
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.trackbodyid = -1
    cam.fixedcamid = -1
    if lookat is not None:
        cam.lookat[:] = np.asarray(lookat, dtype=float)
    cam.distance = float(dist)
    cam.azimuth = float(az)
    cam.elevation = float(el)
    return cam


# ---------------------------------------------------------------- 运动学回放
def run_kin(m, d, qpos, dt, *, speed=1.0, loop=False, track=True,
            record=None, every=1, torso_id=0, dist=2.8, gui=True,
            width=640, height=360, az=160.0, el=-14.0):
    """纯运动学回放。返回实际播放帧数。"""
    n = len(qpos)
    if record is not None:
        from PIL import Image
        rec = Path(record)
        rec.mkdir(parents=True, exist_ok=True)
        # 模型 XML 里 <visual><global offwidth=..> 决定了离屏帧缓冲上限
        # （本模型是 640），超了会直接报错，所以这里夹一次
        ow = int(m.vis.global_.offwidth) or 640
        oh = int(m.vis.global_.offheight) or 480
        w = int(min(width, ow))
        h = int(min(height, oh))
        print(f"  离屏分辨率 {w}x{h}（模型帧缓冲上限 {ow}x{oh}；"
              f"想更大就在 XML 的 <visual><global offwidth=..> 里改）")
        renderer = mujoco.Renderer(m, h, w)
        cam = make_cam(lookat=d.xpos[torso_id], dist=dist, az=az, el=el)
        idxs = list(range(0, n, max(1, int(every))))
        for k, i in enumerate(idxs):
            d.qpos[:] = qpos[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cam.lookat[:] = d.xpos[torso_id]
            renderer.update_scene(d, camera=cam)
            Image.fromarray(renderer.render()).save(rec / f"crab_{i:05d}.png")
            if k % 10 == 0:
                print(f"  渲染 {k}/{len(idxs)}", end="\r", flush=True)
        print(f"[OK] PNG 序列 -> {rec}  ({len(idxs)} 帧)" + " " * 12)
        print(f"     合成 mp4: ffmpeg -framerate {1.0/dt:.0f} "
              f"-i {rec.name}/crab_%05d.png -c:v libx264 crab.mp4")
        return n

    if not gui:
        # 不开窗又不录屏：快速走一遍，只确认每帧都能解算
        for i in range(n):
            d.qpos[:] = qpos[i]
            mujoco.mj_kinematics(m, d)
        print(f"[OK] 无窗口走完 {n} 帧（末帧躯干 y={d.xpos[torso_id][1]:.2f} "
              f"z={d.xpos[torso_id][2]:.2f}）")
        return n

    # ---------------- GUI ----------------
    state = {"paused": False, "speed": speed, "track": track, "restart": True}

    def on_key(key):
        if key == 32:                                   # 空格
            state["paused"] = not state["paused"]
        elif key in (ord("R"), ord("r")):
            state["restart"] = True
        elif key == ord("["):
            state["speed"] = max(0.05, state["speed"] / 1.5)
        elif key == ord("]"):
            state["speed"] = min(8.0, state["speed"] * 1.5)
        elif key in (ord("T"), ord("t")):
            state["track"] = not state["track"]

    try:
        if mj_viewer is None:
            raise RuntimeError("mujoco.viewer 不可用（缺少 GL 后端）")
        v = mj_viewer.launch_passive(m, d, key_callback=on_key)
    except Exception as e:                              # 无显示环境
        print(f"[跳过] 打不开 MuJoCo 窗口（{e}）")
        print("       改用 --record <目录> 离屏渲染出 PNG 序列。")
        return 0

    cam = make_cam(lookat=d.xpos[torso_id], dist=dist, az=az, el=el)
    t0 = time.perf_counter()
    played = 0
    with v:
        while v.is_running():
            if state["restart"]:
                t0 = time.perf_counter()
                state["restart"] = False
            if not state["paused"]:
                el = time.perf_counter() - t0
                i = int(el / (dt / state["speed"]))
                if loop:
                    i = i % n
                elif i >= n:
                    i = n - 1
                    state["paused"] = True
                played = i
            d.qpos[:] = qpos[played]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            if state["track"]:
                v.cam.lookat[:] = d.xpos[torso_id]
            v.sync()
            time.sleep(0.004)
    return played


# ---------------------------------------------------------------- 物理回放
def run_phys(m, d, qpos, dt, *, kp_scale=1.0, kd_scale=1.0, track=True,
             torso_id=0, dist=2.8, gui=True, max_time=None,
             az=160.0, el=-14.0):
    """PD + 重力前馈跟踪轨迹，跑真实物理。

    期望结果是摔倒 —— 这条轨迹有 43% 的帧质心在支撑面外。
    回报"几点几秒倒下"以及倒下时的进度，作为"离可行还差多远"的量化指标。

    力矩 = qfrc_bias(重力+科氏) + kp*e - kd*ė
    没有重力前馈时 PD 稳态误差会有 27~37 度，等于根本站不住。
    """
    n = len(qpos)
    names = list(P.layout.names)
    kp = {n_: float(v) for n_, v in zip(names, P.kp)}
    kd = {n_: float(v) for n_, v in zip(names, P.kd)}

    # 腿 + 腰用满增益；手臂/头用 0.3 倍轻 PD 钉在轨迹上（它们不承重）
    def gains(jn):
        s = (kp_scale if jn in LEG_JOINTS else 0.3 * kp_scale)
        t = (kd_scale if jn in LEG_JOINTS else 0.3 * kd_scale)
        return kp[jn] * s, kd[jn] * t

    d.qpos[:] = qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    z0 = d.xpos[torso_id][2]
    fell_at, fell_frame = None, None
    step_dt = 1.0 / 500.0                      # 物理步长，和控制周期解耦
    n_sub = max(1, int(round(dt / step_dt)))
    m.opt.timestep = step_dt

    v = None
    if gui:
        try:
            if mj_viewer is None:
                raise RuntimeError("mujoco.viewer 不可用（缺少 GL 后端）")
            v = mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = d.xpos[torso_id]
            v.cam.distance = dist
            v.cam.azimuth = az
            v.cam.elevation = el
        except Exception as e:
            print(f"[跳过] 打不开 MuJoCo 窗口（{e}），改为无窗口跑完")
            v = None

    T = max_time if max_time is not None else n * dt
    nsteps = int(T / step_dt)
    for s in range(nsteps):
        i = min(int(s / n_sub), n - 1)
        q_des = qpos[i]
        d.qfrc_applied[:] = 0.0
        # qpos 索引 != dof 地址，必须按关节表逐个换算
        for jn in names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j < 0:
                continue
            qa = m.jnt_qposadr[j]
            va = m.jnt_dofadr[j]
            kk, dd = gains(jn)
            d.qfrc_applied[va] = (d.qfrc_bias[va]
                                  + kk * (q_des[qa] - d.qpos[qa])
                                  - dd * d.qvel[va])
        mujoco.mj_step(m, d)

        if fell_at is None and (d.xpos[torso_id][2] < z0 - 0.35
                                or not np.isfinite(d.qpos[:3]).all()):
            fell_at = s * step_dt
            fell_frame = i
        if v is not None:
            if track:
                v.cam.lookat[:] = d.xpos[torso_id]
            v.sync()
            if not v.is_running():
                break
        if fell_at is not None and fell_at + 1.0 < s * step_dt:
            break

    if v is not None:
        v.close()
    if fell_at is None:
        print(f"[OK] 物理回放跑完 {T:.1f} s 没倒（进度 {100.0:.0f}%）")
    else:
        print(f"[结果] 第 {fell_at:.2f} s 倒下，对应轨迹第 {fell_frame}/{n} 帧"
              f"（进度 {100.0 * fell_frame / n:.0f}%）")
        print("       这正是 RL 要补的缺口：单腿支撑期需要动力学平衡。")
    return fell_at


# ---------------------------------------------------------------- 策略回放（ONNX）
# 观测布局必须与 train/mjlab_tasks/crab/env_cfg.py 的 actor 组严格一致：
#   base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
#   + joint_pos(31) + joint_vel(31) + last_action(29) + command(3) + height_scan(187)
#   = 290
_GRID_XS = np.arange(-0.8, 0.8 + 0.05, 0.1)      # 17
_GRID_YS = np.arange(-0.5, 0.5 + 0.05, 0.1)      # 11
_GY, _GX = np.meshgrid(_GRID_XS, _GRID_YS, indexing="xy")   # mjlab: 先 y 后 x
_SCAN_OFFSETS = np.stack([_GX.ravel(), _GY.ravel()], axis=1)
_SCAN_DOWN = np.array([0.0, 0.0, -1.0])
_SCAN_GMASK = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # 只打 group 0（地形）
_RAY_GEOMID = np.zeros(1, dtype=np.int32)                    # mj_ray 输出缓冲
_RAY_NORMAL = np.zeros(3, dtype=np.float64)


def _sensor_vec(m, d, name, fallback):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, fallback)
    if sid < 0:
        raise KeyError(f"模型里没有传感器 {name} / {fallback}")
    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
    return d.sensordata[adr:adr + dim].copy()


def _yaw_only(r_mat):
    """取旋转矩阵的偏航分量（与 mjlab ray_alignment='yaw' 一致）。"""
    x_axis = r_mat[:, 0]
    n = float(np.linalg.norm(x_axis[:2]))
    xp = x_axis[:2] / n if n > 1e-6 else np.array([1.0, 0.0])
    return np.array([[xp[0], -xp[1], 0.0],
                     [xp[1],  xp[0], 0.0],
                     [0.0,    0.0,    1.0]])


def run_policy(m, d, *, onnx_path, cmd=(0.0, 0.25, 0.0), steps=2000,
               dt=0.02, decimation=10, torso_id=0, dist=2.8, az=160.0, el=-14.0,
               track=True, record=None, every=4, gui=True, width=640, height=360,
               spawn_y=0.30, max_dist=5.0):
    """加载 ONNX 策略闭环跑仿真。返回最终躯干 y 位移。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    # 机器人几何体挪到 group 1，height_scan 只打地形（group 0）
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != 0:
            m.geom_group[g] = 1

    # 31 个铰链关节（按模型顺序，与训练观测一致）
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    j_names = [m.joint(j).name for j in hinge]
    q_adr = np.array([m.jnt_qposadr[j] for j in hinge])
    v_adr = np.array([m.jnt_dofadr[j] for j in hinge])
    default_all = np.zeros(len(hinge))
    default29 = np.asarray(_DEFAULT_JOINT_POS, dtype=float)
    scale29 = np.asarray(_ACTION_SCALE, dtype=float)
    for i, n in enumerate(j_names):
        if n in _JOINT_NAMES:
            default_all[i] = default29[_JOINT_NAMES.index(n)]

    # 初始位姿：地面 y=spawn_y，朝 +X，默认站姿
    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.0, spawn_y, 1.0339]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, n in enumerate(j_names):
        d.qpos[q_adr[i]] = default_all[i]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    ctrl_lo = np.full(29, -np.inf)
    ctrl_hi = np.full(29, np.inf)
    for i, n in enumerate(_JOINT_NAMES):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0 and m.jnt_limited[j]:
            ctrl_lo[i], ctrl_hi[i] = m.jnt_range[j]

    last_action = np.zeros(29)
    cmd_vec = np.asarray(cmd, dtype=float)

    def height_scan():
        r = d.xmat[torso_id].reshape(3, 3)
        yaw = _yaw_only(r)
        p0 = d.xpos[torso_id]
        vals = np.empty(_SCAN_OFFSETS.shape[0])
        for i, (ox, oy) in enumerate(_SCAN_OFFSETS):
            p = p0 + yaw @ np.array([ox, oy, 0.0])
            dist = mujoco.mj_ray(m, d, p, _SCAN_DOWN, _SCAN_GMASK,
                                 True, -1, _RAY_GEOMID, _RAY_NORMAL)
            vals[i] = dist if dist >= 0.0 else max_dist
        return vals / max_dist

    def build_obs():
        lin = _sensor_vec(m, d, "torso_lin_vel", "imu_lin_vel")
        ang = _sensor_vec(m, d, "torso_ang_vel", "imu_ang_vel")
        r = d.xmat[torso_id].reshape(3, 3)
        grav = r.T @ np.array([0.0, 0.0, -1.0])
        q = d.qpos[q_adr] - default_all
        qd = d.qvel[v_adr]
        return np.concatenate([lin, ang, grav, q, qd, last_action, cmd_vec, height_scan()])

    obs = build_obs()
    print(f"[policy] ONNX {Path(onnx_path).name}   obs_dim={obs.size}   "
          f"cmd=(vx={cmd_vec[0]:.2f}, vy={cmd_vec[1]:.2f}, wz={cmd_vec[2]:.2f})")

    renderer = cam = None
    v = None
    if record is not None:
        from PIL import Image
        rec = Path(record)
        rec.mkdir(parents=True, exist_ok=True)
        w = int(min(width, int(m.vis.global_.offwidth) or 640))
        h = int(min(height, int(m.vis.global_.offheight) or 480))
        renderer = mujoco.Renderer(m, h, w)
        cam = make_cam(lookat=d.xpos[torso_id], dist=dist, az=az, el=el)
        print(f"  离屏 {w}x{h} -> {rec}")
    elif gui:
        try:
            if mj_viewer is None:
                raise RuntimeError("mujoco.viewer 不可用")
            v = mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = d.xpos[torso_id]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, az, el
        except Exception as e:
            print(f"[跳过] 打不开窗口（{e}），改为无窗口运行")

    y0 = float(d.xpos[torso_id][1])
    z0 = float(d.xpos[torso_id][2])
    dt_sim = float(m.opt.timestep) * decimation
    t_wall = time.perf_counter()

    for k in range(steps):
        out = sess.run(None, {inp_name: obs[None].astype(np.float32)})
        action = np.asarray(out[0]).reshape(-1)[:29].astype(float)
        last_action = action.copy()

        target = np.clip(default29 + scale29 * action, ctrl_lo, ctrl_hi)
        d.ctrl[:] = target
        for _ in range(decimation):
            mujoco.mj_step(m, d)
        obs = build_obs()

        if renderer is not None and k % max(1, every) == 0:
            from PIL import Image
            cam.lookat[:] = d.xpos[torso_id]
            renderer.update_scene(d, camera=cam)
            Image.fromarray(renderer.render()).save(rec / f"policy_{k:05d}.png")
        elif v is not None:
            if track:
                v.cam.lookat[:] = d.xpos[torso_id]
            v.sync()
            if not v.is_running():
                break
            # 实时播放：控制在 50 Hz，落后就补睡
            want = t_wall + (k + 1) * dt_sim
            sleep = want - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

        if k % 100 == 0:
            print(f"  t={k * dt_sim:5.1f}s  y={d.xpos[torso_id][1]:.2f}  "
                  f"z={d.xpos[torso_id][2]:.2f}", end="\r", flush=True)

    if v is not None:
        v.close()
    dy = float(d.xpos[torso_id][1]) - y0
    dz = float(d.xpos[torso_id][2]) - z0
    print(f"\n[OK] 跑完 {steps} 控制步（{steps * dt_sim:.1f}s）  "
          f"Δy={dy:+.2f} m  Δz={dz:+.2f} m  末帧 z={d.xpos[torso_id][2]:.2f}")
    if renderer is not None:
        print(f"     PNG -> {rec}  合成: ffmpeg -framerate {1.0 / (dt_sim * max(1, every)):.0f} "
              f"-i policy_%05d.png -pix_fmt yuv420p policy.mp4")
    return dy


# ---------------------------------------------------------------- 残差策略回放
# 观测布局必须与 train/mjlab_tasks/crab/env_cfg_residual.py 的 ACTOR_ORDER 严格一致：
#   base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(31)
#   + joint_pos_ref_err(12) + joint_vel(31) + actions(29) + command(3)
#   + phase(2) + ref_meta(3) + height_scan(273)  =  393
# 动作：q_target = q_ref(rise,tread,phase) + clip(scale·a, ±clip_rad)
RES_SCAN_SIZE = (1.2, 2.0)          # 前后 ±0.6 / 左右 ±1.0（与训练一致）
RES_SCAN_RES = 0.1                  # -> 13 x 21 = 273
RES_PHASE_SHIFT = 33.0 / 90.0       # phase=0 落在双支撑起点（表帧 33）
RES_PERIOD_FRAMES = 90
RES_HOLD_STEPS = 25                 # 前 0.5 s 冻结相位，与训练一致
RES_V_RATE = (0.6, 1.6)
# ⚠️ 必须与 train/mjlab_tasks/crab/env_cfg_residual.py 里 ResidualGaitActionCfg 的
# leg_clip_rad / waist_clip_rad / arm_clip_rad 完全一致，否则回放跑的不是训练时的策略。
# 2026-10-30：这里原本是 0.30/0.15/0.15，而训练早已改成 0.55/0.20/0.20。
# 训练日志显示约 60% 的动作维度顶在限幅上，回放把腿从 0.55 砍到 0.30 = 把残差
# 拦腰截断，结果站 1.8 s 就塌，但同一 checkpoint 在 mjlab 里 level 9 是 0/8 摔。
# 改训练侧限幅时记得同步这里；临时可用 --clip-legs 等覆盖。
_CLIP_LEGS, _CLIP_WAIST, _CLIP_ARMS = 0.55, 0.20, 0.20
_WAIST_JOINTS = ("waist_y_joint", "waist_x_joint", "waist_z_joint")


def _clip_rad_for(name: str) -> float:
    if name in K.LEG_JOINTS:
        return _CLIP_LEGS
    if name in _WAIST_JOINTS:
        return _CLIP_WAIST
    return _CLIP_ARMS


# 393 维观测的分块边界，必须与 env_cfg_residual.ACTOR_ORDER 一致
_OBS_BLOCKS = (
    ("lin_vel", 0, 3), ("ang_vel", 3, 6), ("gravity", 6, 9),
    ("joint_pos", 9, 40), ("ref_err", 40, 52), ("joint_vel", 52, 83),
    ("actions", 83, 112), ("command", 112, 115), ("phase", 115, 117),
    ("ref_meta", 117, 120), ("height_scan", 120, 393),
)


def _print_obs_blocks(obs, tag: str = "") -> None:
    """按训练侧的观测分块打印均值/极值，用来一眼看出哪一块拼错了。"""
    print(f"  --- 观测分块 {tag} (dim={obs.size}) ---")
    for name, a, b in _OBS_BLOCKS:
        if b > obs.size:
            print(f"    {name:11s} 越界，跳过（obs 只有 {obs.size} 维）")
            break
        seg = obs[a:b]
        print(f"    {name:11s} [{a:3d}:{b:3d}] n={b - a:3d} "
              f"mean={seg.mean():+7.3f} min={seg.min():+7.3f} max={seg.max():+7.3f}")


def _find_repo_file(rel: str):
    """从脚本位置向上找仓库内文件；也支持 ELF3_REPO 环境变量。"""
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / rel).is_file():
            return p / rel
    env = os.environ.get("ELF3_REPO")
    if env and (Path(env) / rel).is_file():
        return Path(env) / rel
    return None


def _load_ref_table():
    """直接按文件加载 ref_table.py（它只依赖 numpy/torch，不 import mjlab）。"""
    import importlib.util
    tbl = _find_repo_file("train/assets/ref_tables/crab_ref_v1.npz")
    mod = _find_repo_file("train/mjlab_tasks/crab/ref_table.py")
    if tbl is None or mod is None:
        raise SystemExit(
            "[错误] 找不到参考表或 ref_table.py。\n"
            "       期望 train/assets/ref_tables/crab_ref_v1.npz 与 train/mjlab_tasks/crab/ref_table.py\n"
            "       若仓库不在脚本上层目录，设 ELF3_REPO=<仓库根>")
    spec = importlib.util.spec_from_file_location("crab_ref_table", mod)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RefTable(tbl)


def run_policy_residual(m, d, *, onnx_path, cmd=(0.0, 0.20, 0.0), steps=2000,
                        dt=0.02, decimation=10, torso_id=0, dist=2.8, az=160.0,
                        el=-14.0, track=True, record=None, every=4, gui=True,
                        width=640, height=360, spawn_y=0.30, spawn_z=1.0339,
                        max_dist=5.0, table=None, dump_obs=None,
                        save_traj=None, stop_on_fall=False):
    """残差策略闭环：q_target = q_ref(相位) + 限幅残差。返回躯干 y 位移。"""
    import onnxruntime as ort
    import torch

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"[policy] ONNX {Path(onnx_path).name}  输入 {inp.name} {inp.shape}")

    tb = table if table is not None else _load_ref_table()
    names29 = list(_JOINT_NAMES)
    scale29 = np.asarray(_ACTION_SCALE, dtype=float)
    default29 = np.asarray(_DEFAULT_JOINT_POS, dtype=float)
    clip29 = np.array([_clip_rad_for(n) for n in names29])
    leg_pos = np.array([names29.index(n) for n in K.LEG_JOINTS])

    # 机器人几何体 -> group 1；height_scan 只打地形（group 0）
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != 0:
            m.geom_group[g] = 1

    # 31 个铰链关节（模型顺序）与 29 个执行器关节（_JOINT_NAMES 顺序）
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    j_names = [m.joint(j).name for j in hinge]
    q_adr = np.array([m.jnt_qposadr[j] for j in hinge])
    v_adr = np.array([m.jnt_dofadr[j] for j in hinge])
    default_all = np.zeros(len(hinge))
    for i, n in enumerate(j_names):
        if n in names29:
            default_all[i] = default29[names29.index(n)]

    mid = mujoco.mj_name2id
    q29_adr = np.array([m.jnt_qposadr[mid(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names29])
    v29_adr = np.array([m.jnt_dofadr[mid(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names29])
    act_joint = [m.actuator(i).trnid[0] for i in range(m.nu)]
    act_names = [m.joint(j).name for j in act_joint]
    assert act_names == names29, f"执行器顺序与 _JOINT_NAMES 不一致：{act_names[:3]}..."

    # 扫描网格：y 外层、x 内层（与 mjlab GridPattern 一致）
    nx = int(round(RES_SCAN_SIZE[0] / RES_SCAN_RES)) + 1     # 13
    ny = int(round(RES_SCAN_SIZE[1] / RES_SCAN_RES)) + 1     # 21
    xs = -RES_SCAN_SIZE[0] / 2 + np.arange(nx) * RES_SCAN_RES
    ys = -RES_SCAN_SIZE[1] / 2 + np.arange(ny) * RES_SCAN_RES
    gy, gx = np.meshgrid(ys, xs, indexing="ij")              # [ny, nx]
    scan_off = np.stack([gx.ravel(), gy.ravel()], axis=1)    # 行 = iy*nx+ix
    down = np.array([0.0, 0.0, -1.0])
    gmask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    gid = np.zeros(1, dtype=np.int32)
    nrm = np.zeros(3, dtype=np.float64)

    def scan_raw():
        """返回米制高度（躯干 z - 命中 z），miss -> max_dist。"""
        r = d.xmat[torso_id].reshape(3, 3)
        yaw = _yaw_only(r)
        p0 = d.xpos[torso_id]
        out = np.empty(scan_off.shape[0])
        for i, (ox, oy) in enumerate(scan_off):
            p = p0 + yaw @ np.array([ox, oy, 0.0])
            dist = mujoco.mj_ray(m, d, p, down, gmask, True, -1, gid, nrm)
            out[i] = dist if dist >= 0.0 else max_dist
        return out

    # ---- 初始状态：参考表 phase=0（含双支撑平移）的姿势 ----
    phase = 0.0
    rise, tread = 0.0, float(tb.mid_tread)
    v_ref = float(tb.v_ref[0, tb.T // 2])
    q_ref = tb.select(torch.tensor([rise]), torch.tensor([tread]),
                      torch.tensor([phase + RES_PHASE_SHIFT]))[0][0].numpy()

    d.qpos[:] = 0.0
    d.qpos[0:3] = [0.0, spawn_y, spawn_z]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for i, n in enumerate(j_names):
        d.qpos[q_adr[i]] = q_ref[names29.index(n)] if n in names29 else 0.0
    d.qvel[:] = 0.0
    d.ctrl[:] = q_ref
    mujoco.mj_forward(m, d)

    last_action = np.zeros(29)
    cmd_vec = np.asarray(cmd, dtype=float)

    def ref_meta():
        return np.array([rise * 5.0, tread * 2.5, v_ref * 5.0])

    def build_obs():
        lin = _sensor_vec(m, d, "torso_lin_vel", "imu_lin_vel")
        ang = _sensor_vec(m, d, "torso_ang_vel", "imu_ang_vel")
        r = d.xmat[torso_id].reshape(3, 3)
        grav = r.T @ np.array([0.0, 0.0, -1.0])
        q_all = d.qpos[q_adr] - default_all
        qd_all = d.qvel[v_adr]
        q29 = d.qpos[q29_adr]
        err = q29[leg_pos] - q_ref[leg_pos]
        ang_p = 2.0 * np.pi * ((phase + RES_PHASE_SHIFT) % 1.0)
        ph = np.array([np.sin(ang_p), np.cos(ang_p)])
        hs = scan_raw() / max_dist
        return np.concatenate([lin, ang, grav, q_all, err, qd_all,
                               last_action, cmd_vec, ph, ref_meta(), hs])

    obs = build_obs()
    print(f"  残差模式：obs_dim={obs.size}（训练 393）  表 {tb.R}x{tb.T} "
          f"phase_shift={RES_PHASE_SHIFT:.3f}")
    if inp.shape and len(inp.shape) == 2 and inp.shape[1] not in (-1, obs.size):
        raise SystemExit(f"[错误] ONNX 输入维度 {inp.shape[1]} != 观测 {obs.size}")
    print(f"  残差限幅 legs={_CLIP_LEGS:.2f} waist={_CLIP_WAIST:.2f} "
          f"arms={_CLIP_ARMS:.2f} rad"
          f"   <- 必须与训练 env_cfg_residual 的 *_clip_rad 一致")
    if dump_obs:
        np.save(dump_obs, obs)
        print(f"  首帧观测已存 {dump_obs}")
        _print_obs_blocks(obs, tag="k=0 复位")

    renderer = cam = None
    v = None
    if record is not None:
        from PIL import Image
        rec = Path(record)
        rec.mkdir(parents=True, exist_ok=True)
        w = int(min(width, int(m.vis.global_.offwidth) or 640))
        h = int(min(height, int(m.vis.global_.offheight) or 480))
        renderer = mujoco.Renderer(m, h, w)
        cam = make_cam(lookat=d.xpos[torso_id], dist=dist, az=az, el=el)
        print(f"  离屏 {w}x{h} -> {rec}")
    elif gui:
        try:
            if mj_viewer is None:
                raise RuntimeError("mujoco.viewer 不可用")
            v = mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = d.xpos[torso_id]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = dist, az, el
        except Exception as e:
            print(f"[跳过] 打不开窗口（{e}），改为无窗口运行")

    y0 = float(d.xpos[torso_id][1])
    z0 = float(d.xpos[torso_id][2])
    dt_sim = float(m.opt.timestep) * decimation
    t_wall = time.perf_counter()
    fell_at = None

    traj_qpos = []                  # 每控制步一份 qpos，供 --save-traj 烘焙到网页
    for k in range(steps):
        traj_qpos.append(d.qpos.copy())
        out = sess.run(None, {inp.name: obs[None].astype(np.float32)})
        raw = np.asarray(out[0]).reshape(-1)[:29].astype(float)
        # 与训练一致：先限幅到 action 空间，再乘 scale（等效残差 <= clip_rad）
        action = np.clip(raw, -clip29 / scale29, clip29 / scale29)
        # 训练侧 mdp.last_action 取的是 env.action_manager.action，即**策略原始输出**
        # （截幅之前）。实测约 60% 的维度顶在限幅上，回填截幅后的值会让观测分布
        # 系统性偏低，归一化之后是明显的偏移 —— 所以这里回填 raw。
        last_action = raw.copy()

        if k in (25, 100) or (dump_obs and k in (0, 50, 200)):
            sat = float((np.abs(raw) > (clip29 / scale29)).mean()) * 100.0
            print(f"\n  t={k * 0.02:5.2f}s  mean|a|={np.abs(raw).mean():.3f}  "
                  f"max|a|={np.abs(raw).max():.3f}  饱和={sat:.0f}%")
            _print_obs_blocks(obs, tag=f"k={k}")

        delta = np.clip(scale29 * action, -clip29, clip29)
        target = q_ref + delta
        for i, n in enumerate(names29):
            j = mid(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if m.jnt_limited[j]:
                target[i] = float(np.clip(target[i], *m.jnt_range[j]))
        d.ctrl[:] = target
        for _ in range(decimation):
            mujoco.mj_step(m, d)

        # 地形估计（每 5 步）+ 相位推进（与训练同逻辑）
        if k % 5 == 0:
            hs_raw = scan_raw()
            with torch.no_grad():
                rr, tt, ok = tb.estimate(torch.tensor(hs_raw[None], dtype=torch.float32),
                                         nx=nx, ny=ny, size_y=float(RES_SCAN_SIZE[1]),
                                         res=float(RES_SCAN_RES))
            rr, tt, ok = float(rr[0]), float(tt[0]), bool(ok[0])
            if ok:
                rise = 0.5 * rr + 0.5 * rise
                tread = tt
            else:
                rise = 0.9 * rise
            rise = float(np.clip(rise, float(tb.rise[0]), float(tb.rise[-1])))
            v_ref = float(tb.select(torch.tensor([rise]), torch.tensor([tread]),
                                    torch.tensor([0.0]))[1][0])
        rate = float(np.clip(abs(cmd_vec[1]) / max(v_ref, 1e-3), *RES_V_RATE))
        if k < RES_HOLD_STEPS:
            rate = 0.0
        phase = (phase + rate / RES_PERIOD_FRAMES) % 1.0
        q_ref = tb.select(torch.tensor([rise]), torch.tensor([tread]),
                          torch.tensor([(phase + RES_PHASE_SHIFT) % 1.0]))[0][0].numpy()

        obs = build_obs()

        if renderer is not None and k % max(1, every) == 0:
            from PIL import Image
            cam.lookat[:] = d.xpos[torso_id]
            renderer.update_scene(d, camera=cam)
            Image.fromarray(renderer.render()).save(rec / f"policy_{k:05d}.png")
        elif v is not None:
            if track:
                v.cam.lookat[:] = d.xpos[torso_id]
            v.sync()
            if not v.is_running():
                break
            want = t_wall + (k + 1) * dt_sim
            slp = want - time.perf_counter()
            if slp > 0:
                time.sleep(slp)

        z = float(d.xpos[torso_id][2])
        if fell_at is None and (z < 0.55 or float(d.xpos[torso_id][1]) < spawn_y - 0.5):
            fell_at = k * dt_sim
            if stop_on_fall:
                print(f"\n  [停止] 第 {fell_at:.2f} s 判定摔倒，按 --stop-on-fall 截断")
                break
        if k % 100 == 0:
            print(f"  t={k * dt_sim:5.1f}s  y={d.xpos[torso_id][1]:.2f}  "
                  f"z={z:.2f}  rise={rise:.2f}  tread={tread:.2f}  "
                  f"phase={(phase + RES_PHASE_SHIFT) % 1.0:.2f}", end="\r", flush=True)

    if v is not None:
        v.close()
    dy = float(d.xpos[torso_id][1]) - y0
    dz = float(d.xpos[torso_id][2]) - z0
    print(f"\n[OK] 残差回放 {steps} 控制步（{steps * dt_sim:.1f}s）  "
          f"Δy={dy:+.2f} m  Δz={dz:+.2f} m  末帧 z={d.xpos[torso_id][2]:.2f}")
    if save_traj:
        q = np.asarray(traj_qpos, dtype=np.float64)
        out_p = Path(save_traj)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_p), qpos=q, dt=float(dt_sim))
        print(f"[输出] 轨迹 {out_p}: {q.shape[0]} 帧 x nq={q.shape[1]}，"
              f"dt={dt_sim:.4f}s（{q.shape[0] * dt_sim:.1f}s）")
        print(f"       烘焙到网页：python web/tools/bake_crab.py --skip-gen --skip-plan "
              f"--traj {out_p} --out policy_crab")
    if fell_at is not None:
        print(f"[!] 第 {fell_at:.2f} s 判定为摔倒（z<0.55 或偏离 >0.5 m）")
    if renderer is not None:
        print(f"     PNG -> {rec}  合成: ffmpeg -framerate {1.0 / (dt_sim * max(1, every)):.0f} "
              f"-i policy_%05d.png -pix_fmt yuv420p policy.mp4")
    return dy


# ---------------------------------------------------------------- 入口
def load_traj(path: Path):
    z = np.load(str(path), allow_pickle=True)
    return z["qpos"], float(z["dt"]) if "dt" in z.files else 0.02


def main() -> None:
    # --clip-* 会覆盖下面这几个模块级常量（argparse 的 default 也要读它们，
    # 所以 global 必须声明在函数最前面，不能挪到 args 解析之后）
    global _CLIP_LEGS, _CLIP_WAIST, _CLIP_ARMS

    ap = argparse.ArgumentParser(
        description="螃蟹步爬楼梯轨迹回放器",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default=str(DEFAULT_TRAJ), help="轨迹 npz")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE), help="场景 xml")
    ap.add_argument("--mode", default="kin", choices=["kin", "phys"],
                    help="kin=纯运动学（默认）；phys=PD 跟踪真实物理")
    ap.add_argument("--speed", type=float, default=1.0, help="播放倍速")
    ap.add_argument("--loop", action="store_true", help="循环播放")
    ap.add_argument("--no-track", action="store_true", help="相机不跟随躯干")
    ap.add_argument("--dist", type=float, default=2.8, help="相机距离 m")
    ap.add_argument("--az", type=float, default=160.0,
                    help="相机方位角 deg（160=斜前方，能同时看到身体和台阶剖面）")
    ap.add_argument("--el", type=float, default=-14.0, help="相机俯仰角 deg")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="离屏渲染存 PNG 序列到该目录（不开窗）")
    ap.add_argument("--every", type=int, default=1, help="录屏时每 N 帧存 1 帧")
    ap.add_argument("--width", type=int, default=640, help="录屏宽度 px（受模型帧缓冲限制）")
    ap.add_argument("--height", type=int, default=360, help="录屏高度 px")
    ap.add_argument("--no-gui", action="store_true", help="不开窗口，直接跑完")
    # phys 专用
    ap.add_argument("--kp-scale", type=float, default=1.0, help="phys: 比例增益缩放")
    ap.add_argument("--kd-scale", type=float, default=1.0, help="phys: 微分增益缩放")
    ap.add_argument("--max-time", type=float, default=None, help="phys: 最长仿真时间 s")
    # policy 专用
    ap.add_argument("--policy", default="none", choices=["none", "onnx", "onnx-residual"],
                    help="onnx=基线策略(290 维)；onnx-residual=残差策略(393 维)")
    ap.add_argument("--onnx", default=None, help="policy: ONNX 策略路径")
    ap.add_argument("--cmd-vy", type=float, default=0.25, help="policy: 侧向速度指令 m/s")
    ap.add_argument("--spawn-y", type=float, default=0.30, help="policy: 初始 y 位置 m")
    ap.add_argument("--spawn-z", type=float, default=1.0339,
                    help="policy: 初始基座高度 m（残差任务训练也是 1.0339）")
    ap.add_argument("--steps", type=int, default=2000, help="policy: 控制步数（50Hz，2000=40s）")
    ap.add_argument("--clip-legs", type=float, default=_CLIP_LEGS,
                    help="policy: 腿残差限幅 rad，必须与训练 leg_clip_rad 一致")
    ap.add_argument("--clip-waist", type=float, default=_CLIP_WAIST,
                    help="policy: 腰残差限幅 rad，必须与训练 waist_clip_rad 一致")
    ap.add_argument("--clip-arms", type=float, default=_CLIP_ARMS,
                    help="policy: 臂残差限幅 rad，必须与训练 arm_clip_rad 一致")
    ap.add_argument("--dump-obs", default=None,
                    help="policy: 打印观测分块统计并存首帧 npy（排障用）")
    ap.add_argument("--save-traj", default=None,
                    help="policy: 把每控制步的 qpos 存成 npz（给 web/tools/bake_crab.py 烘焙网页）")
    ap.add_argument("--stop-on-fall", action="store_true",
                    help="policy: 判定摔倒就停止仿真（录网页用，避免末尾躺地上的画面）")
    args = ap.parse_args()

    _CLIP_LEGS, _CLIP_WAIST, _CLIP_ARMS = args.clip_legs, args.clip_waist, args.clip_arms

    if args.policy in ("onnx", "onnx-residual"):
        if not args.onnx:
            raise SystemExit(f"[错误] --policy {args.policy} 需要 --onnx <policy.onnx>")
        onnx_path = Path(args.onnx)
        if not onnx_path.is_file():
            raise SystemExit(f"[错误] 找不到 ONNX {onnx_path}")
        params = TrainParams()
        # build_model 会把 <motor> 换成 mjlab 训练用的 <position> 执行器，并保留地形
        m = build_model(params, scene="stairs", xml_path=Path(args.scene).resolve())
        m.opt.timestep = 0.002          # 与训练一致：500 Hz 物理 x 10 步 -> 50 Hz 控制
        # MuJoCo 的离屏帧缓冲默认只有 640x480，mujoco.Renderer 一旦超过就抛
        # ValueError；而 run_policy* 里又是 `min(width, offwidth)`，于是
        # --width 1280 会被**静默**夹成 640。这里先把上限抬到请求值。
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(args.width))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(args.height))
        d = mujoco.MjData(m)
        torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        print(f"场景 {Path(args.scene).name}   策略 {onnx_path.name}")
        runner = run_policy_residual if args.policy == "onnx-residual" else run_policy
        kw = dict(onnx_path=onnx_path, cmd=(0.0, args.cmd_vy, 0.0),
                  steps=args.steps, decimation=10, torso_id=torso,
                  dist=args.dist, az=args.az, el=args.el,
                  track=not args.no_track, record=args.record,
                  every=max(1, args.every), gui=not args.no_gui,
                  width=args.width, height=args.height, spawn_y=args.spawn_y)
        if args.policy == "onnx-residual":
            kw["spawn_z"] = args.spawn_z
            kw["dump_obs"] = args.dump_obs
            kw["save_traj"] = args.save_traj
            kw["stop_on_fall"] = args.stop_on_fall
        runner(m, d, **kw)
        return

    tp = Path(args.traj)
    if not tp.is_file():
        raise SystemExit(f"[错误] 找不到轨迹 {tp}\n"
                         f"       先跑: python crab_climb.py（或 run_crab.py）")
    qpos, dt = load_traj(tp)
    m = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    print(f"场景 {Path(args.scene).name}   轨迹 {tp.name}   "
          f"{len(qpos)} 帧 / {len(qpos) * dt:.1f} s   dt={dt}")
    print(f"模式 {args.mode}" + (f"  倍速 {args.speed}" if args.mode == "kin" else ""))

    if args.mode == "kin":
        run_kin(m, d, qpos, dt, speed=args.speed, loop=args.loop,
                track=not args.no_track, record=args.record,
                every=args.every, torso_id=torso, dist=args.dist,
                gui=not args.no_gui, width=args.width, height=args.height,
                az=args.az, el=args.el)
    else:
        run_phys(m, d, qpos, dt, kp_scale=args.kp_scale, kd_scale=args.kd_scale,
                 track=not args.no_track, torso_id=torso, dist=args.dist,
                 gui=not args.no_gui, max_time=args.max_time, az=args.az, el=args.el)


if __name__ == "__main__":
    main()
