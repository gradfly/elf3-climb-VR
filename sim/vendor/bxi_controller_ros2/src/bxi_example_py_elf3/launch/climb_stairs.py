"""闭环楼梯测试：MuJoCo 渲染深度图 -> bxi 官方框架深度策略 -> PD 力矩控制 -> 上楼梯。

数据流完全复刻 state.py 的 ROS 部署链路（只是把 ROS 话题换成 MuJoCo 传感器/渲染器）：
  IMU gyro(torso_ang_vel, 体坐标系角速度) + framequat(torso_quat)
  head_depth_camera 渲染 848x480 深度图(米)
  -> project_depth(官方 FOV 裁剪+最近邻缩放) -> rot90 -> (36,48) 中间形状
  -> HumanoidGaitOriginCameraPolicyIsaaclab.step
  -> 29 维关节目标 -> tau = kp*(q*-q) - kd*dq -> motor 执行器

地形：universal_test compact obstacles
  x=4.80 低楼梯(10cm 级高 x 28cm 进深, 3上+平台+2下) | x=8.00 高楼梯(15cm 级高)
用法:
  python climb_stairs.py --policy dagger2 --cmd-vx 0.5 --duration 25
  python climb_stairs.py --policy dagger_new_position --viewer native
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WS = Path(r"E:/mujoco/climb stairs")
FW = WS / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3"
# depth.py / depth_projection.py / assets/*.onnx 现已归入 mod 目录（2026-10-13 搬迁后）
MOD = FW / "mods" / "com.bxi.normal_depth"
sys.path.insert(0, str(FW))
sys.path.insert(0, str(MOD))

import mujoco  # noqa: E402

# sensor_msgs 仅被 depth_projection.py 用于类型标注；Windows 无 ROS，注入 stub。
if "sensor_msgs" not in sys.modules:
    import types as _types
    _sm = _types.ModuleType("sensor_msgs")
    _sm_msg = _types.ModuleType("sensor_msgs.msg")
    _sm_msg.CameraInfo = object  # type: ignore[attr-defined]
    _sm.msg = _sm_msg  # type: ignore[attr-defined]
    sys.modules["sensor_msgs"] = _sm
    sys.modules["sensor_msgs.msg"] = _sm_msg

try:  # 原生查看器（可选依赖，仅 native 模式需要）
    import mujoco.viewer as mj_viewer  # noqa: E402
except Exception:  # pragma: no cover
    mj_viewer = None

from bxi_example_py_elf3.framework.inference import (  # noqa: E402
    InferenceFrame,
    ModelSpec,
)
from bxi_example_py_elf3.framework.joints import JointStateBuffer  # noqa: E402
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS  # noqa: E402

import depth  # noqa: E402  (mod 目录 mods/com.bxi.normal_depth 里的策略封装)
from depth_projection import ProjectionSpec, project_depth  # noqa: E402

SCENE = FW / "data" / "mujoco_simulation" / "elf3.xml"
# 注意：vendor 的 elf3.xml 第 354 行已自带
#   <include file="terrains/universal_test/universal_test_terrain_compact.xml"/>
# 即它本身就是「机器人 + 紧凑障碍地形」的完整场景，直接加载即可。

# ---- 两个 profile 的投影参数（与 plugin.py 完全一致） ----
PROJECTION_SPECS = {
    "origin_camera": dict(spec=ProjectionSpec(
        output_width=36, output_height=48,
        horizontal_fov_deg=45.2, vertical_fov_deg=58.0616969,
        minimum_m=0.2, maximum_m=3.0,
    ), cls="origin"),
    "depth_walk": dict(spec=ProjectionSpec(
        output_width=64, output_height=36,
        horizontal_fov_deg=89.24, vertical_fov_deg=58.06,
        minimum_m=0.2, maximum_m=2.5,
    ), cls="legacy"),
}
POLICIES = {
    "dagger2": ("assets/dagger2.onnx", "origin_camera", "head_depth_camera"),
    "dagger_new_position": ("assets/dagger_new_position.onnx", "origin_camera", "head_depth_camera"),
    # legacy depth_walk 策略：俯角 70° 的躯干相机（盯脚前地面），8 帧深度历史
    "normal_depth": ("assets/normal_depth.onnx", "depth_walk", "body_depth_camera"),
}

# 楼梯路标（compact_obstacles.xml）
LANDMARKS = [
    (4.80, "低楼梯第1级(10cm)"),
    (5.36, "低楼梯第3级/平台"),
    (6.93, "低楼梯下完"),
    (8.00, "高楼梯第1级(15cm)"),
    (8.85, "高楼梯平台"),
    (9.85, "高楼梯下完"),
    (11.10, "壕沟区入口"),
    (13.20, "终点线"),
]


class DepthCameraBridge:
    """MuJoCo 渲染 -> 官方 project_depth -> rot90，输出策略侧中间形状。"""

    def __init__(self, model, cam_name, spec, raw_h=480, raw_w=848):
        self.spec = spec
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            raise KeyError(f"模型里没有相机 {cam_name}")
        self.cam_id = cid
        model.vis.global_.offwidth = max(raw_w, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(raw_h, model.vis.global_.offheight)
        self.renderer = mujoco.Renderer(model, height=raw_h, width=raw_w)
        self.renderer.enable_depth_rendering()
        # 由 MuJoCo 相机几何反推内参（方形像素）：fy = (H/2)/tan(fovy/2)
        fovy = model.cam_fovy[cid] * np.pi / 180.0
        fy = 0.5 * raw_h / np.tan(fovy / 2.0)
        self.cam_info = SimpleNamespace(
            width=raw_w, height=raw_h,
            k=[fy, 0.0, raw_w / 2.0, 0.0, fy, raw_h / 2.0, 0.0, 0.0, 1.0],
        )

    def read(self, data):
        self.renderer.update_scene(data, camera=self.cam_id)
        raw = self.renderer.render().astype(np.float32)  # (H,W) 米, 0=背景
        # MuJoCo 深度渲染中 0 表示「无命中」(天空/远处)。RealSense 的习惯是
        # 无回波=超量程。若把 0 直接传下去，project_depth 会把它原样保留，
        # 策略侧 clip 到 0.2m —— 相当于「面前 0.2m 有一堵墙」，机器人会拒绝前进。
        # 所以这里把背景替换成超远距，交给 project_depth clip 到 maximum_m。
        raw[raw <= 0.0] = 1.0e3
        projected, _crop = project_depth(raw, self.cam_info, self.spec)
        return np.ascontiguousarray(np.rot90(projected, k=-1).astype(np.float32))


def build_joint_maps(model, policy_names):
    qadr, vadr, aadr = [], [], []
    for n in policy_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise KeyError(f"模型缺少关节 {n}")
        qadr.append(model.jnt_qposadr[jid])
        vadr.append(model.jnt_dofadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if aid < 0:
            raise KeyError(f"模型缺少执行器 {n}")
        aadr.append(aid)
    return np.array(qadr), np.array(vadr), np.array(aadr)


def sensor_adr(model, name, dim):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise KeyError(f"模型缺少传感器 {name}")
    adr = model.sensor_adr[sid]
    return slice(adr, adr + dim)


def run(args):
    onnx_rel, profile, cam_name = POLICIES[args.policy]
    spec_cfg = PROJECTION_SPECS[profile]

    # ---------- 模型 ----------
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    gyro = sensor_adr(model, "torso_ang_vel", 3)      # 体坐标系角速度 rad/s
    quat_s = sensor_adr(model, "torso_quat", 4)       # wxyz

    # ---------- 策略 ----------
    spec = ModelSpec.portable_onnx(str(MOD / onnx_rel), input_names=(), output_names=())
    if spec_cfg["cls"] == "origin":
        pol = depth.HumanoidGaitOriginCameraPolicyIsaaclab(spec)
    else:
        pol = depth.HumanoidGaitDepthPolicyIsaaclab(spec)
    layout = pol.joint_contract.observation
    names = list(layout.names)
    jbuf = JointStateBuffer(layout, dtype=np.float32)

    qadr, vadr, aadr = build_joint_maps(model, names)
    params = ELF3_ISAAC_PARAMETERS
    kp, kd, q_default = params.kp, params.kd, params.default_position

    # 关节限位（按策略顺序）
    jnt_lo = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in names])
    jnt_hi = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in names])

    # 头部两个非策略关节：轻量 PD 保持 0（相机朝前）
    head_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                 for n in ("head_z_joint", "head_y_joint")]
    head_qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("head_z_joint", "head_y_joint")]
    head_vadr = [model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("head_z_joint", "head_y_joint")]

    ctrl_lo = model.actuator_ctrlrange[aadr, 0]
    ctrl_hi = model.actuator_ctrlrange[aadr, 1]

    # ---------- 深度相机桥 ----------
    bridge = DepthCameraBridge(model, cam_name, spec_cfg["spec"])

    # ---------- 初始化：默认站姿 ----------
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [args.spawn_x, 0.0, 1.10]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = q_default
    mujoco.mj_forward(model, data)

    # 观测器
    # 观测器 / 录像
    viewer = None
    if args.viewer == "native":
        if mj_viewer is None:
            raise RuntimeError("mujoco.viewer 不可用，请用 --viewer headless")
        viewer = mj_viewer.launch_passive(model, data)

    video_frames = []
    rgb_renderer = None
    track_cid = -1
    if args.record:
        model.vis.global_.offwidth = max(1280, model.vis.global_.offwidth)
        model.vis.global_.offheight = max(720, model.vis.global_.offheight)
        rgb_renderer = mujoco.Renderer(model, height=720, width=1280)
        track_cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
        try:
            from PIL import Image, ImageDraw
            _pil = (Image, ImageDraw)
        except Exception:
            _pil = None
    else:
        _pil = None

    # ---------- 主循环 ----------
    dt = 0.02                       # 策略频率 50 Hz
    substeps = max(1, round(dt / model.opt.timestep))  # 物理子步(模型 0.002s -> 10)
    n_steps = int(args.duration / dt)
    render_every = max(1, round(args.depth_period / dt))
    stand_steps = int(args.stand_time / dt)
    print(f"控制: 策略 {1/dt:.0f}Hz x 物理子步 {substeps}"
          f" (PD @ {substeps/model.opt.timestep:.0f}Hz), 深度渲染 {1/(render_every*dt):.0f}Hz")

    # 先渲染一帧并 reset 策略
    depth_img = bridge.read(data)
    frame_id = 1
    jbuf.update(position=data.qpos[qadr].astype(np.float32),
                velocity=data.qvel[vadr].astype(np.float32))
    frame = InferenceFrame(
        joints=jbuf.view,
        quat_wxyz=np.array(data.sensordata[quat_s], dtype=np.float32),
        angular_velocity=np.array(data.sensordata[gyro], dtype=np.float32),
        command=np.zeros(3, dtype=np.float32),
        depth=depth_img,
        depth_frame_id=frame_id,
    )
    pol.reset(frame)

    traj = []
    t0 = time.time()
    fell = False
    q_des = q_default.copy()

    for step in range(n_steps):
        # --- 命令：先站稳，再给前进速度 ---
        if step < stand_steps:
            cmd = np.zeros(3, dtype=np.float32)
        else:
            cmd = np.array([args.cmd_vx, args.cmd_vy, args.cmd_vz], dtype=np.float32)
            if args.yaw_hold:
                # 航向保持：按 yaw 误差微调转向命令（速度命令式 locomotion
                # 的标准做法），避免策略步态不对称导致绕圈走不到楼梯
                w, x_, y_, z_ = data.qpos[3:7]
                yaw = np.arctan2(2.0 * (w * z_ + x_ * y_),
                                 1.0 - 2.0 * (y_ * y_ + z_ * z_))
                cmd[2] = float(np.clip(-2.0 * yaw, -0.5, 0.5))

        # --- 深度图（按 depth_period 限频） ---
        if step % render_every == 0:
            depth_img = bridge.read(data)
            frame_id += 1

        # --- 观测 -> 策略 ---
        jbuf.update(position=data.qpos[qadr].astype(np.float32),
                    velocity=data.qvel[vadr].astype(np.float32))
        frame.quat_wxyz = np.array(data.sensordata[quat_s], dtype=np.float32)
        frame.angular_velocity = np.array(data.sensordata[gyro], dtype=np.float32)
        frame.command = cmd
        frame.depth = depth_img
        frame.depth_frame_id = frame_id
        out = pol.step(frame, dt, advance=True)
        q_des = np.asarray(out.joints.position, dtype=np.float64).copy()
        q_des = np.clip(q_des, jnt_lo, jnt_hi)

        # --- PD 力矩控制：每个物理子步重算（等价真机电机环频率） ---
        for _ in range(substeps):
            q = data.qpos[qadr]
            dq = data.qvel[vadr]
            tau = kp * (q_des - q) - kd * dq
            data.ctrl[aadr] = np.clip(tau, ctrl_lo, ctrl_hi)
            for a, qa, va in zip(head_acts, head_qadr, head_vadr):
                data.ctrl[a] = np.clip(10.0 * (0.0 - data.qpos[qa]) - 1.0 * data.qvel[va],
                                       -21.0, 21.0)
            mujoco.mj_step(model, data)

        x, y, z = data.qpos[0:3]
        traj.append((step * dt, x, y, z))

        # --- 录像（跟随相机 + 状态叠加） ---
        if rgb_renderer is not None and step % args.record_every == 0:
            if track_cid >= 0:
                model.cam_pos[track_cid] = [x - 2.2, y + 1.6, z + 0.7]
                rgb_renderer.update_scene(data, camera=track_cid)
            else:
                rgb_renderer.update_scene(data, camera=-1)
            frame_rgb = rgb_renderer.render()
            if _pil is not None:
                img = _pil[0].fromarray(frame_rgb)
                dr = _pil[1].Draw(img)
                dr.rectangle([16, 14, 480, 66], fill=(0, 0, 0, 160))
                dr.text((24, 20),
                        f"{args.policy}  t={step*dt:5.1f}s  x={x:5.2f}  z={z:4.2f}",
                        fill=(255, 220, 80))
                dr.text((24, 42),
                        f"cmd_vx={args.cmd_vx}  "
                        f"{'CLIMBING' if 4.6 < x < 7.2 or 7.8 < x < 10.2 else 'walk'}",
                        fill=(120, 255, 140))
                frame_rgb = np.asarray(img)
            video_frames.append(frame_rgb)

        # --- 摔倒判定 ---
        if z < 0.55 and step > stand_steps:
            fell = True
            break

        if args.viewer == "native" and viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()

        # --- 进度打印 ---
        if step % (50 * 2) == 0:
            passed = [name for lx, name in LANDMARKS if x > lx]
            nxt = next(((lx, name) for lx, name in LANDMARKS if x <= lx), None)
            d = depth_img
            print(f"t={step*dt:6.2f}s  x={x:6.3f} y={y:+6.3f} z={z:5.3f}  "
                  f"|tau|max={np.abs(data.ctrl[aadr]).max():6.1f}  "
                  f"depth[min/mean/max]={d.min():.2f}/{d.mean():.2f}/{d.max():.2f}  "
                  f"已过: {passed[-1] if passed else '出生点'}  "
                  f"下一路标: {nxt[1] if nxt else '全部完成'}")

    wall = time.time() - t0
    traj = np.array(traj)
    x, z = traj[-1, 1], traj[-1, 3]
    max_x = traj[:, 1].max()

    print("\n" + "=" * 66)
    print(f"策略        : {args.policy}  (cmd_vx={args.cmd_vx})")
    print(f"仿真时长    : {traj[-1,0]:.2f} s ({len(traj)} 步)  实际耗时 {wall:.1f} s")
    print(f"结束位置    : x={x:.3f} y={traj[-1,2]:+.3f} z={z:.3f}   最远 x={max_x:.3f}")
    print(f"是否摔倒    : {'是 (z<0.55)' if fell else '否'}")
    for lx, name in LANDMARKS:
        reached = max_x > lx
        print(f"  [{'✓' if reached else ' '}] x>{lx:5.2f}  {name}")
    print("=" * 66)

    # 保存轨迹
    out_npz = WS / "vendor" / f"climb_traj_{args.policy}.npz"
    np.savez(out_npz, traj=traj, q_des=q_des,
             policy=args.policy, cmd=[args.cmd_vx, args.cmd_vy, args.cmd_vz])
    print(f"轨迹已保存: {out_npz}")

    # 保存视频
    if video_frames:
        import imageio.v2 as imageio
        out_mp4 = WS / "vendor" / f"climb_{args.policy}.mp4"
        fps = round(1.0 / (args.record_every * dt))
        imageio.mimwrite(str(out_mp4), [f.astype(np.uint8) for f in video_frames],
                         fps=fps, quality=8, macro_block_size=None)
        print(f"视频已保存: {out_mp4}  ({len(video_frames)} 帧 @ {fps}fps)")

    if viewer is not None:
        viewer.close()
    pol.close()
    return fell, max_x


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="dagger2",
                    choices=list(POLICIES), help="策略 onnx")
    ap.add_argument("--cmd-vx", type=float, default=0.5, help="前进速度命令 m/s")
    ap.add_argument("--cmd-vy", type=float, default=0.0, help="横移速度命令 m/s")
    ap.add_argument("--cmd-vz", type=float, default=0.0, help="转向速度命令 rad/s")
    ap.add_argument("--duration", type=float, default=25.0, help="仿真时长 s")
    ap.add_argument("--stand-time", type=float, default=2.0, help="起步前站稳时间 s")
    ap.add_argument("--depth-period", type=float, default=0.04,
                    help="深度图渲染周期 s（策略默认更新周期 0.05）")
    ap.add_argument("--spawn-x", type=float, default=0.0, help="出生点 x（平地区 0~4m）")
    ap.add_argument("--yaw-hold", action=argparse.BooleanOptionalAction, default=True,
                    help="航向保持（推荐开，防步态漂移）")
    ap.add_argument("--record", action=argparse.BooleanOptionalAction, default=False,
                    help="录制跟随相机视频 mp4")
    ap.add_argument("--record-every", type=int, default=2,
                    help="每 N 个策略步录 1 帧（2 -> 25fps）")
    ap.add_argument("--viewer", default="headless", choices=["headless", "native"])
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
