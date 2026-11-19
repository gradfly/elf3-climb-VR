"""本机 MuJoCo 闭环验证三份自训蹲起 ONNX（down / up / hold）。

与 `mods/com.bxi.squat_actions/verify_policy.py` 的区别：
  - verify_policy.py 只做合成观测下的数值自检（obs 拼装 / 后处理对拍）；
  - 本脚本**真正闭环**：从 mj data 构造观测 → ONNX 推理 → 残差后处理 →
    <position> 执行器下发 → mj_step，观察机器人能否蹲下 / 站起 / 保持。

用法（在 FW 目录下，或任意目录均可，脚本用 __file__ 定位）：
    python launch/verify_squat_sim.py                       # 全流程 DOWN→HOLD→UP（headless）
    python launch/verify_squat_sim.py --policy down         # 只跑下蹲
    python launch/verify_squat_sim.py --policy up           # 只跑起立
    python launch/verify_squat_sim.py --policy hold         # 只跑蹲底保持
    python launch/verify_squat_sim.py --viewer native       # 开原生窗口看
    python launch/verify_squat_sim.py --record --gif out.gif  # 离屏录像 / GIF

关键一致性措施（否则结论不可信）：
  1. 用 elf3_squat.xml（平地、29 关节顺序 == ELF3_POLICY_JOINTS、armature 与训练一致），
     通过 elf3_play.build_model 把执行器换成 mjlab 训练同款内置 <position kp kd>
     （kp/kd 取 ONNX metadata，EFFORT=4·scale·kp，implicitfast 积分），d.ctrl=q_target。
  2. 残差后处理完全复用 SquatResidualPolicy.step()，不重写。
  3. 提供两种观测模式：
       --obs-mode deploy : 与真机一致，base_lin_vel 填 0、基座高度填 0（默认，检验真机近似是否可行）
       --obs-mode train  : 喂仿真的 torso_lin_vel / 基座高度真值（检验策略本身在训练分布下是否成立）
     两者结果对比即可回答「真机填 0 近似会不会毁掉策略」。
  4. 初始姿态用源轨迹 squat_extreme_{down,up}_traj.npz 的完整 38 维 qpos
     （含蹲姿前倾 18.5° 的根节点四元数、COM 稳定裕度 +9.65cm），与训练 reset 精确对齐。
     只用 q_ref+z 拼姿态会因四元数符号/根节点位置错误，导致开局就倒（已踩坑）。

已知结论（2026-11-19 本机实测，详见 docs/squat_onnx_sim_verify.md）：
  - down 闭环成立（1.034→0.523 m，跟踪误差 ~12°）；
  - hold 能蹲住但缓慢下沉 ~8 cm/1.2 s；
  - up 起立到 ~0.99 m（目标 1.034）后失稳，且开环 q_ref 起立更早倒 → 属参考轨迹/柔顺 kp
    的动力学问题，非 ONNX 策略本身；deploy 近似比 train 真值明显更差。
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import types
import importlib.util
from pathlib import Path

import numpy as np
import mujoco

_FW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FW))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # elf3_play 在 launch/

DATA = _FW / "data" / "mujoco_simulation"
ASSETS = _FW / "mods" / "com.bxi.squat_actions" / "assets"
REF = ASSETS / "squat_ref_v1.npz"
XML = DATA / "elf3_squat.xml"


# ------------------------------------------------------------------ mod 加载
def _load_policy_module():
    """以合成包方式加载新 Mod 的适配层（不依赖 colcon / ROS）。"""
    def ensure(name):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
        return sys.modules[name]

    for n in ("com", "com.bxi"):
        ensure(n)
    pkg = ensure("com.bxi.squat_actions")
    pkg.__path__ = [str(ASSETS.parent)]
    spec = importlib.util.spec_from_file_location(
        "com.bxi.squat_actions.squat_residual_policy",
        str(ASSETS.parent / "squat_residual_policy.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 模型
def build_squat_model(kp: np.ndarray, kd: np.ndarray, action_scale: np.ndarray,
                      default_pos: np.ndarray) -> mujoco.MjModel:
    """复用 elf3_play.build_model 生成 mjlab 等价模型（已验证的 <position> 执行器路径）。

    它把 <motor> 换成内置 <position kp kd forcerange>，设 implicitfast 积分、
    armature、EFFORT=4·scale·kp，与训练一致；d.ctrl = q_target 即可。
    用 elf3_squat.xml 作为基座（平地、29 关节顺序已核对），scene="stairs" 让
    build_model 不改动地形 include（elf3_squat.xml 本身无地形，就是平地）。
    """
    import elf3_play as ep

    p = ep.TrainParams()
    p.stiffness = np.asarray(kp, dtype=np.float64)
    p.damping = np.asarray(kd, dtype=np.float64)
    p.action_scale = np.asarray(action_scale, dtype=np.float64)
    p.default_joint_pos = np.asarray(default_pos[:29], dtype=np.float64)
    p.source = "squat ONNX metadata"
    return ep.build_model(p, scene="stairs", xml_path=XML, effort="derived")


# ------------------------------------------------------------------ 主逻辑
def main() -> int:
    ap = argparse.ArgumentParser(description="ELF3 三份蹲起 ONNX 本机闭环验证")
    ap.add_argument("--policy", choices=["down", "up", "hold", "all"], default="all",
                    help="all=按 DOWN→HOLD→UP 顺序连跑（默认）")
    ap.add_argument("--obs-mode", choices=["deploy", "train"], default="deploy",
                    help="deploy=真机近似(lin_vel/z 填0)；train=仿真真值")
    ap.add_argument("--viewer", choices=["headless", "native"], default="headless")
    ap.add_argument("--record", action="store_true", help="离屏渲染 PNG 序列")
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--gif", default=None, help="输出 GIF（需 pillow）")
    ap.add_argument("--gain-scale", type=float, default=1.0,
                    help="kp/kd 统一缩放（复现真机首跑降刚度用 0.3；默认 1.0=训练原值）")
    ap.add_argument("--render-every", type=int, default=5)
    ap.add_argument("--diag-every", type=int, default=0,
                    help=">0 时每 N 步打印 z/tilt/q误差/z_ref（调试用）")
    ap.add_argument("--tilt-fall", type=float, default=60.0,
                    help="躯干倾角摔倒阈值（度）；设 999 可关闭 tilt 摔倒")
    ap.add_argument("--up-trim", type=int, default=0,
                    help="up 提前完成帧数（真机 end_frame_trim 扫参用）：frame>=last-N 即停，输出切走点姿态")
    ap.add_argument("--segmented", action="store_true",
                    help="all 模式下每段 reset 到源轨迹首帧（旧默认）；"
                         "默认连续——段间保留上一段末态，复现真机状态机切换，量化切换瞬态")
    args = ap.parse_args()

    import squat_replay as sr  # 稳定裕度诊断

    sp = _load_policy_module()
    from bxi_example_py_elf3.framework.inference.api import InferenceFrame
    from bxi_example_py_elf3.framework.joints import JointStateView
    from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

    layout = ELF3_POLICY_JOINTS
    names = list(layout.names)
    dof = layout.dof_num

    # 源轨迹含完整 38 维 qpos（根节点 xyz/quat + 29 关节 + 2 头关节），
    # 用它初始化才能精确复现训练 reset 姿态（蹲姿前倾 18.5°，根节点 quat y=+0.1607）。
    src_qpos = {}
    for key, fn in (("down", "squat_extreme_down_traj.npz"),
                    ("up", "squat_extreme_up_traj.npz")):
        src = np.load(DATA / fn, allow_pickle=True)
        src_qpos[key] = np.asarray(src["qpos"], dtype=np.float64)

    # 策略实例（三份都建，all 模式会依次切换）
    policies = {
        m: sp.SquatResidualPolicy(str(ASSETS / f"squat_{m}.onnx"), str(REF),
                                  mode=m, gain_scale=args.gain_scale)
        for m in ("down", "up", "hold")
    }
    kp_train = np.asarray(policies["down"]._parameters.kp, dtype=np.float64)
    kd_train = np.asarray(policies["down"]._parameters.kd, dtype=np.float64)
    scale_train = np.asarray(policies["down"]._parameters.action_scale, dtype=np.float64)
    default_train = np.asarray(policies["down"]._parameters.default_position, dtype=np.float64)
    if args.gain_scale != 1.0:
        kp_train = kp_train * args.gain_scale
        kd_train = kd_train * args.gain_scale
    model = build_squat_model(kp_train, kd_train, scale_train, default_train)
    data = mujoco.MjData(model)
    torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    qadr = np.array([model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    vadr = np.array([model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_ang_vel")
    quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_quat")
    linvel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_lin_vel")
    if min(gyro_id, quat_id, linvel_id) < 0:
        raise RuntimeError("elf3_squat.xml 缺少 torso_* 传感器")
    gyro_sl = slice(model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + 3)
    quat_sl = slice(model.sensor_adr[quat_id], model.sensor_adr[quat_id] + 4)
    lin_sl = slice(model.sensor_adr[linvel_id], model.sensor_adr[linvel_id] + 3)

    dt = sp.DT  # 0.02 s，50 Hz
    substeps = max(1, round(dt / model.opt.timestep))

    # 录制
    viewer = None
    renderer = None
    rec_dir = Path(args.record_dir) if args.record_dir else (
        Path("render/squat_verify") if args.record else None)
    frames = []
    if args.viewer == "native":
        from mujoco import viewer as _mjv
        viewer = _mjv.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.0, 0.0, 0.6]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 270.0, -5.0
    elif rec_dir or args.gif:
        renderer = mujoco.Renderer(model, height=480, width=640)
        rec_cam = mujoco.MjvCamera()
        rec_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        rec_cam.distance, rec_cam.azimuth, rec_cam.elevation = 1.5, 200.0, -8.0

    # 要跑的策略序列。all 默认连续模式（段间保留末态，复现真实切换）；
    # --segmented 回到旧的"每段 reset 到源轨迹首帧"。
    if args.policy == "all":
        seq = ["down", "hold", "up"]
    else:
        seq = [args.policy]
    continuous = args.policy == "all" and not args.segmented

    results = {}
    last_target = None  # 上一段最后下发的 target（连续模式 blend 起点）
    for seg_i, mode in enumerate(seq):
        pol = policies[mode]
        # 策略状态复位（真机：down/up 进场 reset_policy；hold 是独立新实例）。
        # 连续模式下机器人 qpos/qvel 不重置，保留上一段末态 —— 复现真实切换。
        pol.reset_to_start()
        if continuous and seg_i > 0:
            print(f"\n---- 连续切换 [{seq[seg_i-1]}] → [{mode}]：保留末态，"
                  f"z={float(data.xpos[torso_bid][2]):.3f} ----")
        else:
            # 训练 reset 姿态 = 源轨迹对应帧的完整 qpos（hold 冻结在 down 末帧）。
            traj_key = "down" if mode == "hold" else mode
            init_frame = pol.frame if mode != "hold" else (pol.frames - 1)
            qpos0 = src_qpos[traj_key][init_frame]
            mujoco.mj_resetData(model, data)
            data.qpos[:] = qpos0
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        mujoco.mj_forward(model, data)
        seg_z0 = float(data.xpos[torso_bid][2])  # 本段起始高度

        n_steps = pol.frames if mode != "hold" else 60  # hold 跑 1.2 s
        up_trim_stop = (pol.frames - 1 - args.up_trim) if (mode == "up" and args.up_trim > 0) else None
        z_hist, err_hist, sat_hist = [], [], []
        fell_at = None
        # 切换 blend 仿真：连续模式下切换后 0.3s（15 帧）把两侧 target 做
        # smootherstep 插值，对应真机 end_transition=dual_running_blend。
        # blend_from = 上一段最后下发的 target（freeze）。
        blend_from = None
        blend_frames = 0
        if continuous and seg_i > 0 and last_target is not None:
            blend_from = np.asarray(last_target, dtype=np.float64).copy()
            blend_frames = 15  # 0.3 s / 0.02 s

        for k in range(n_steps):
            joints = JointStateView(layout, data.qpos[qadr].astype(np.float32),
                                    data.qvel[vadr].astype(np.float32), 0)
            frame = InferenceFrame(
                joints=joints,
                quat_wxyz=data.sensordata[quat_sl].astype(np.float32),
                angular_velocity=data.sensordata[gyro_sl].astype(np.float32),
                command=np.zeros(3, dtype=np.float32),
            )
            out = pol.step(frame, dt, advance=True)
            target = np.asarray(out.joints.position, dtype=np.float64).copy()

            # 观测模式覆盖：step() 内已按 deploy 填 0；train 模式用仿真真值重算 obs 再推一次
            if args.obs_mode == "train":
                pol._obs[0, 0:3] = data.sensordata[lin_sl]  # base_lin_vel
                z_meas = float(data.qpos[2])
                idx = pol.frame
                pol._obs[0, 129] = z_meas - float(pol._z_ref[idx])
                outputs = pol._backend.run(pol._inputs)
                raw = np.asarray(outputs["actions"]).reshape(-1)
                a = np.clip(raw, -pol._clip_rad / pol._scale, pol._clip_rad / pol._scale)
                np.add(pol._q_ref[idx], pol._scale * a, out=target)
                np.copyto(pol._previous_action, raw)

            # 关节限位
            for i, n in enumerate(names):
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if model.jnt_limited[j]:
                    target[i] = float(np.clip(target[i], *model.jnt_range[j]))
            # 切换 blend（dual_running_blend 0.3s smootherstep 仿真）：
            # 切换后前 15 帧在「上一段最后 target」与「本策略当帧 target」间插值。
            if blend_frames > 0:
                t_frac = min(1.0, (k + 1) / 15.0)
                s = t_frac * t_frac * t_frac * (t_frac * (6.0 * t_frac - 15.0) + 10.0)
                target = blend_from + s * (target - blend_from)
            # <general affine> 执行器内部做 PD：ctrl = q_target。
            # 头关节保持 0（不在 29 维策略里）。
            data.ctrl[:29] = target
            data.ctrl[29:31] = 0.0
            last_target = np.asarray(target, dtype=np.float64).copy()
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            z_now = float(data.xpos[torso_bid][2])
            z_hist.append(z_now)
            idx = min(pol.frame, pol.frames - 1)
            err_hist.append(float(np.max(np.abs(
                data.qpos[qadr] - pol._q_ref[idx]))))
            # 饱和率（用上一帧 raw 不可得，这里用 target 与 q_ref 偏差近似）
            sat_hist.append(float(np.mean(np.abs(
                target - pol._q_ref[idx]) > 0.01)))

            if viewer is not None:
                viewer.cam.lookat[:] = [0.0, 0.0, z_now - 0.1]
                viewer.sync()
                if not viewer.is_running():
                    break
            elif renderer is not None and k % args.render_every == 0:
                rec_cam.lookat[:] = [0.0, 0.0, z_now - 0.05]
                renderer.update_scene(data, camera=rec_cam)
                img = renderer.render()
                if rec_dir:
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    from PIL import Image
                    Image.fromarray(img).save(rec_dir / f"squat_{mode}_{k:04d}.png")
                frames.append(img)

            # 摔倒判据（蹲姿目标 z≈0.523 m，不能用固定 z<0.5）：
            #   1) 躯干倾角 > 60°（真翻倒）；或
            #   2) 躯干高度比当前参考帧低 > 0.15 m（掉出参考轨迹 = 塌了/没跟上）。
            # 用躯干 up 向量与竖直方向夹角（比 quat w 更准，参考表同款定义）
            up_now = data.xmat[torso_bid].reshape(3, 3)[:, 2]
            tilt = float(np.degrees(np.arccos(np.clip(up_now[2], -1.0, 1.0))))
            z_ref_now = float(pol._z_ref[min(pol.frame, pol.frames - 1)])
            margin = sr.signed_margin(data.subtree_com[torso_bid][:2],
                                      sr.foot_hull(model, data))
            if args.diag_every and k % args.diag_every == 0:
                print(f"    k={k:3d} z={z_now:.3f} z_ref={z_ref_now:.3f} "
                      f"tilt={tilt:5.1f}° qerr={np.degrees(err_hist[-1]):.1f}° "
                      f"裕度={margin*100:+.1f}cm")
            elif continuous and k < 10:
                # 连续模式：段头 10 帧逐帧打印，量化切换瞬态
                print(f"    [{mode} k={k:2d}] z={z_now:.3f} tilt={tilt:5.1f}° "
                      f"qerr={np.degrees(err_hist[-1]):.1f}° 裕度={margin*100:+.1f}cm")

            # up 提前完成（end_frame_trim 扫参）：真机 finished(trim) 等价点
            if up_trim_stop is not None and pol.frame >= up_trim_stop:
                print(f"    [up-trim={args.up_trim}] 切走点 frame={pol.frame}: "
                      f"z={z_now:.3f} tilt={tilt:5.1f}° 裕度={margin*100:+.1f}cm "
                      f"qerr={np.degrees(err_hist[-1]):.1f}°")
                break

            if tilt > args.tilt_fall or z_now < z_ref_now - 0.15:
                fell_at = k
                break

        z_arr = np.asarray(z_hist)
        err_arr = np.asarray(err_hist)
        z_end = float(z_arr[-1]) if len(z_arr) else seg_z0
        trim_stopped = (up_trim_stop is not None
                        and pol.frame >= up_trim_stop and fell_at is None)
        results[mode] = dict(
            z0=seg_z0, z_end=z_end,
            z_min=float(z_arr.min()) if len(z_arr) else seg_z0,
            z_max=float(z_arr.max()) if len(z_arr) else seg_z0,
            z_delta=z_end - seg_z0,   # down 应为负（降），up 应为正（升）
            z_ref_end=float(pol._z_ref[min(pol.frame, pol.frames - 1)]),
            qerr_max=float(err_arr.max()) if len(err_arr) else 0.0,
            qerr_mean=float(err_arr.mean()) if len(err_arr) else 0.0,
            steps=len(z_hist), fell_at=fell_at,
            finished=pol.finished(), trim_stopped=trim_stopped,
        )
        if fell_at is not None and continuous:
            print(f"[{mode}] 连续模式下摔倒，终止后续段")
            break

    if viewer is not None:
        viewer.close()

    if args.gif and frames:
        from PIL import Image
        dur = int(round(1000 * dt * args.render_every))
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(args.gif, save_all=True, append_images=pil_frames[1:],
                           duration=dur, loop=0, disposal=2)
        print(f"[GIF] {args.gif}  ({len(frames)} 帧)")

    print("\n" + "=" * 68)
    print(f"观测模式: {args.obs_mode}  kp/kd 缩放: {args.gain_scale}")
    print("=" * 68)
    for mode, r in results.items():
        if r["fell_at"] is not None:
            status = "摔倒"
        elif r.get("trim_stopped"):
            status = "trim停"
        elif r["finished"] or mode == "hold":
            status = "完成"
        else:
            status = "未完成"
        print(f"[{mode:4s}] {status:4s} 步数 {r['steps']:3d}  "
              f"躯干 z {r['z0']:.3f}→{r['z_end']:.3f} m "
              f"(参考末值 {r['z_ref_end']:.3f}, Δz {r['z_delta']*100:+.1f} cm)  "
              f"q误差 mean/max {np.degrees(r['qerr_mean']):.1f}°/"
              f"{np.degrees(r['qerr_max']):.1f}°"
              + (f"  摔倒@{r['fell_at']}" if r["fell_at"] is not None else ""))

    # 结论：each 段是否达成目标 + 末态是否贴合参考
    print("\n---- 判定 ----")
    def reached(r, sign):
        # sign=-1 down（降幅>0.3m）；sign=+1 up（升幅>0.3m）；0 hold（贴参考±0.05）
        if r["fell_at"] is not None:
            return False
        if sign < 0:
            return r["z_delta"] < -0.30
        if sign > 0:
            return r["z_delta"] > 0.30
        return abs(r["z_end"] - r["z_ref_end"]) < 0.05

    if args.policy == "all":
        down, hold, up = results["down"], results["hold"], results["up"]
        d_ok, h_ok, u_ok = reached(down, -1), reached(hold, 0), reached(up, 1)
        print(f"下蹲 squat_down : {'OK' if d_ok else '未达标'}  (Δz {down['z_delta']*100:+.1f} cm)")
        print(f"保持 squat_hold : {'OK' if h_ok else '未达标'}  (末 z {hold['z_end']:.3f} vs 参考 {hold['z_ref_end']:.3f})")
        print(f"起立 squat_up   : {'OK' if u_ok else '未达标'}  (Δz {up['z_delta']*100:+.1f} cm)")
        if d_ok and u_ok:
            print("结论: 三策略闭环成立（可上真机进一步验证）")
        elif args.obs_mode == "deploy":
            print("建议: deploy 近似未达标，用 --obs-mode train 复测；"
                  "若 train 成立说明真机需补 odom/基座高度估计或改观测重训")
        else:
            print("建议: train 真值仍未达标，先排查增益/初始姿态/参考表，再考虑重训")
    else:
        r = results[args.policy]
        sign = {"down": -1, "up": 1, "hold": 0}[args.policy]
        print("通过" if reached(r, sign) else "未达标")

    # 关闭策略后端
    for pol in policies.values():
        try:
            pol.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
