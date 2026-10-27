"""bake_climb.py —— 把「爬楼梯」的策略仿真录制成网页可回放的运动数据

和 bake_motion.py 是同一套思路，区别只在"跑哪条链路"：

  * bake_motion.py —— 平地打拳：elf3_play.py 的 MuJoCo + policy.onnx 动作跟踪。
  * bake_climb.py  —— 爬楼梯：climb_stairs.py 的深度视觉闭环
                      （MuJoCo 渲染深度图 -> normal_depth.onnx -> PD 力矩 -> mj_step），
                      地形是 universal_test 紧凑障碍（10cm 低楼梯 / 15cm 高楼梯 / 壕沟）。

浏览器里的 MuJoCo WASM 没有 onnxruntime，算不了策略，所以这里在本机把整段跑完，
只把每个控制步（50 Hz）的 `qpos` 录下来，网页端按时间插值回放。

关节顺序对账（重要）
--------------------
物理仿真跑的是官方完整模型 `data/mujoco_simulation/elf3.xml`（nq=38：自由基座 7 +
29 个策略关节 + head_z / head_y 两个头部关节）；网页端用的是老版
`web/assets/elf3.xml`（nq=36，没有那两个头部关节）。两者的前 36 个 qpos
在这里是按关节名逐个对齐后再写出的 —— 顺序完全一致时就是原样截取，
将来任一侧改了关节顺序也能自动重排，不会录出"关节乱掉"的回放。

用法（在项目根目录或 web/ 下都可以）：
  python web/tools/bake_climb.py                      # 默认 32 s，楼梯前 1.2 m 出生
  python web/tools/bake_climb.py --duration 20        # 只录前 20 s
  python web/tools/bake_climb.py --cmd-vx 0.4         # 走慢一点
  python web/tools/bake_climb.py --spawn-x 0          # 从世界原点出发（先走 4.6 m 平地）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent      # web/tools
WEB = HERE.parent                           # web/
ASSETS = WEB / "assets"

DEFAULT_WS = Path(r"E:/mujoco/climb stairs")


def setup_climb_env(ws: Path):
    """把 climb stairs 工作区的框架 / mod 接进来，返回 climb_stairs 模块。"""
    fw = ws / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3"
    mod = fw / "mods" / "com.bxi.normal_depth"
    launch = fw / "launch"
    if not (launch / "climb_stairs.py").is_file():
        raise SystemExit(
            f"[错误] 找不到爬楼梯闭环脚本：{launch / 'climb_stairs.py'}\n"
            f"       用 --ws 指定 climb stairs 工作区根目录（当前 {ws}）。"
        )
    for p in (str(launch), str(fw), str(mod)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import climb_stairs  # noqa: E402  （它自己会注入 sensor_msgs stub）

    return climb_stairs, fw, mod


def build_qpos_remap(src_model, dst_model):
    """按关节名建立 src->dst 的 qpos 重排索引，返回长度 dst.nq 的下标数组。"""
    import mujoco

    src = {}
    for j in range(src_model.njnt):
        name = mujoco.mj_id2name(src_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        src[name] = (int(src_model.jnt_qposadr[j]), int(src_model.jnt_type[j]))

    width = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
             int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}

    ix = []
    for j in range(dst_model.njnt):
        name = mujoco.mj_id2name(dst_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name not in src:
            raise SystemExit(f"[错误] 网页模型里的关节 {name} 在仿真模型里不存在，无法对齐")
        sadr, stype = src[name]
        dadr = int(dst_model.jnt_qposadr[j])
        dnum = width[int(dst_model.jnt_type[j])]
        snum = width[stype]
        if snum != dnum:
            raise SystemExit(f"[错误] 关节 {name} 的自由度不一致（仿真 {snum} vs 网页 {dnum}）")
        ix.extend(range(sadr, sadr + dnum))
    return np.asarray(ix, dtype=np.int64)


def save(name: str, frames: np.ndarray, meta: dict) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    bin_name = f"motion_{name}.bin"
    arr = np.ascontiguousarray(frames, dtype="<f4")
    (ASSETS / bin_name).write_bytes(arr.tobytes())
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        **meta,
        "bin": bin_name,
        "generated": stamp,
        # 网页端拿它当缓存版本号：数据一重录，URL 就变
        "version": f"{stamp}-{arr.shape[0]}",
        "nframes": int(arr.shape[0]),
        "nq": int(arr.shape[1]),
        "duration": round(float(arr.shape[0]) / meta["fps"], 3),
        "qposFormat": "float32-le, row-major, (nframes, nq)",
    }
    (ASSETS / f"motion_{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    size = (ASSETS / bin_name).stat().st_size
    print(f"\n[输出] {bin_name}: {arr.shape[0]} 帧 × {arr.shape[1]} = "
          f"{size / 1048576:.2f} MB，时长 {meta['duration']:.1f} s")
    print(f"[输出] motion_{name}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ws", default=str(DEFAULT_WS), help="climb stairs 工作区根目录")
    ap.add_argument("--out", default="climb_policy", help="输出名前缀（默认 climb_policy）")
    ap.add_argument("--policy", default="normal_depth", help="深度策略名（默认 normal_depth）")
    ap.add_argument("--cmd-vx", type=float, default=0.5, help="前进速度命令 m/s")
    ap.add_argument("--duration", type=float, default=32.0, help="录制时长 s")
    ap.add_argument("--spawn-x", type=float, default=3.5, help="出生点 x（0~4 m 是干净区）")
    ap.add_argument("--stand-time", type=float, default=2.0, help="起步前站稳时间 s")
    ap.add_argument("--depth-period", type=float, default=0.04, help="深度图渲染周期 s")
    ap.add_argument("--stop-x", type=float, default=13.4,
                    help="越过该 x 后收尾停录（默认刚好过终点线 13.2）")
    ap.add_argument("--no-yaw-hold", action="store_true", help="关闭航向保持")
    ap.add_argument("--print-every", type=float, default=2.0, help="每 N 秒打印一次进度")
    args = ap.parse_args()

    import mujoco  # noqa: E402

    t_start = time.time()
    ws = Path(args.ws)
    cs, fw, mod = setup_climb_env(ws)

    # ---------------------------------------------------------------- 模型
    scene = fw / "data" / "mujoco_simulation" / "elf3.xml"
    if not scene.is_file():
        raise SystemExit(f"[错误] 找不到仿真场景 {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    web_xml = ASSETS / "elf3.xml"
    web_model = mujoco.MjModel.from_xml_path(str(web_xml))
    ix = build_qpos_remap(model, web_model)
    assert ix.shape[0] == web_model.nq, (ix.shape[0], web_model.nq)
    print(f"[信息] 仿真模型 nq={model.nq} -> 网页模型 nq={web_model.nq}"
          f"（{'原样截取' if np.array_equal(ix, np.arange(web_model.nq)) else '按关节名重排'}）")

    # ---------------------------------------------------------------- 策略
    try:
        onnx_rel, profile, cam_name = cs.POLICIES[args.policy]
    except KeyError:
        raise SystemExit(f"[错误] 未知策略 {args.policy}，可选：{list(cs.POLICIES)}")
    spec_cfg = cs.PROJECTION_SPECS[profile]
    from bxi_example_py_elf3.framework.inference import (  # noqa: E402
        InferenceFrame, ModelSpec,
    )
    from bxi_example_py_elf3.framework.joints import JointStateBuffer  # noqa: E402
    from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS  # noqa: E402
    import depth  # noqa: E402

    spec = ModelSpec.portable_onnx(str(mod / onnx_rel), input_names=(), output_names=())
    pol = (depth.HumanoidGaitOriginCameraPolicyIsaaclab(spec) if spec_cfg["cls"] == "origin"
           else depth.HumanoidGaitDepthPolicyIsaaclab(spec))
    layout = pol.joint_contract.observation
    names = list(layout.names)
    jbuf = JointStateBuffer(layout, dtype=np.float32)

    qadr, vadr, aadr = cs.build_joint_maps(model, names)
    params = ELF3_ISAAC_PARAMETERS
    kp, kd, q_default = params.kp, params.kd, params.default_position
    jnt_lo = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)][0] for n in names])
    jnt_hi = np.array([model.jnt_range[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)][1] for n in names])
    head_acts = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                 for n in ("head_z_joint", "head_y_joint")]
    head_qadr = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("head_z_joint", "head_y_joint")]
    head_vadr = [model.jnt_dofadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("head_z_joint", "head_y_joint")]
    ctrl_lo = model.actuator_ctrlrange[aadr, 0]
    ctrl_hi = model.actuator_ctrlrange[aadr, 1]

    gyro = cs.sensor_adr(model, "torso_ang_vel", 3)
    quat_s = cs.sensor_adr(model, "torso_quat", 4)

    bridge = cs.DepthCameraBridge(model, cam_name, spec_cfg["spec"])
    print(f"[信息] 策略 {args.policy} / 相机 {cam_name} / 深度 {spec_cfg['spec'].output_width}"
          f"x{spec_cfg['spec'].output_height} / 关节 {len(names)} 个")

    # ---------------------------------------------------------------- 初始化
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [args.spawn_x, 0.0, 1.10]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = q_default
    mujoco.mj_forward(model, data)

    dt = 0.02
    substeps = max(1, round(dt / model.opt.timestep))
    n_steps = int(args.duration / dt)
    render_every = max(1, round(args.depth_period / dt))
    stand_steps = int(args.stand_time / dt)
    print(f"[信息] 控制 {1/dt:.0f} Hz × 物理子步 {substeps}"
          f"（PD @ {1/model.opt.timestep:.0f} Hz）| 深度渲染 {1/(render_every*dt):.0f} Hz"
          f" | 出生 x={args.spawn_x} | 计划 {n_steps} 步")

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

    raw = np.zeros((n_steps + 1, model.nq), dtype=np.float64)
    raw[0] = data.qpos
    xs, zs = [float(data.qpos[0])], [float(data.qpos[2])]
    fell, reached = False, n_steps
    q_des = q_default.copy()
    print_every = max(1, int(args.print_every / dt))

    for step in range(n_steps):
        if step < stand_steps:
            cmd = np.zeros(3, dtype=np.float32)
        else:
            cmd = np.array([args.cmd_vx, 0.0, 0.0], dtype=np.float32)
            if not args.no_yaw_hold:
                w, x_, y_, z_ = data.qpos[3:7]
                yaw = np.arctan2(2.0 * (w * z_ + x_ * y_), 1.0 - 2.0 * (y_ * y_ + z_ * z_))
                cmd[2] = float(np.clip(-2.0 * yaw, -0.5, 0.5))

        if step % render_every == 0:
            depth_img = bridge.read(data)
            frame_id += 1

        jbuf.update(position=data.qpos[qadr].astype(np.float32),
                    velocity=data.qvel[vadr].astype(np.float32))
        frame.quat_wxyz = np.array(data.sensordata[quat_s], dtype=np.float32)
        frame.angular_velocity = np.array(data.sensordata[gyro], dtype=np.float32)
        frame.command = cmd
        frame.depth = depth_img
        frame.depth_frame_id = frame_id
        out = pol.step(frame, dt, advance=True)
        q_des = np.clip(np.asarray(out.joints.position, dtype=np.float64).copy(), jnt_lo, jnt_hi)

        for _ in range(substeps):
            tau = kp * (q_des - data.qpos[qadr]) - kd * data.qvel[vadr]
            data.ctrl[aadr] = np.clip(tau, ctrl_lo, ctrl_hi)
            for a, qa, va in zip(head_acts, head_qadr, head_vadr):
                data.ctrl[a] = np.clip(10.0 * (0.0 - data.qpos[qa]) - 1.0 * data.qvel[va],
                                       -21.0, 21.0)
            mujoco.mj_step(model, data)

        raw[step + 1] = data.qpos
        x, z = float(data.qpos[0]), float(data.qpos[2])
        xs.append(x)
        zs.append(z)

        if step % print_every == 0 or step == n_steps - 1:
            passed = [n for lx, n in cs.LANDMARKS if x > lx]
            print(f"  t={step*dt:5.1f}s  x={x:6.2f}  z={z:5.2f}  "
                  f"|tau|max={np.abs(data.ctrl[aadr]).max():5.1f}  "
                  f"已过：{passed[-1] if passed else '出生点'}", flush=True)

        if z < 0.55 and step > stand_steps:
            fell, reached = True, step + 1
            print(f"  [警告] 第 {step + 1} 步摔倒（z={z:.2f} m），提前收尾")
            break
        if x > args.stop_x:
            reached = step + 1
            print(f"  [完成] 已越过 x={args.stop_x}（终点线 13.2），收尾停录")
            break

    frames = raw[:reached + 1]
    web_frames = frames[:, ix]

    traj_x = np.array(xs)
    stats = {
        "policy": args.policy,
        "camera": cam_name,
        "cmdVx": args.cmd_vx,
        "spawnX": args.spawn_x,
        "stepCount": int(len(frames)),
        "fell": bool(fell),
        "maxX": round(float(traj_x.max()), 3),
        "finalX": round(float(traj_x[-1]), 3),
        "rootZMin": round(float(np.min(zs)), 4),
        "rootZMax": round(float(np.max(zs)), 4),
        # 台阶/壕沟路标：x 与该处躯干最高点（看是否真的踩上去了）
        "landmarks": [{"x": lx, "name": nm,
                       "passed": bool(traj_x.max() > lx)}
                      for lx, nm in cs.LANDMARKS],
    }

    save(args.out, web_frames, {
        "fps": 50.0,
        "dt": dt,
        "model": "elf3.xml",
        "scene": "universal_test 紧凑地形（10cm 低楼梯 / 15cm 高楼梯 / 壕沟）",
        "policy": f"{args.policy}.onnx",
        "camera": cam_name,
        "jointNames": list(names),
        "jointQposAdr": [int(a) for a in np.array([
            web_model.jnt_qposadr[mujoco.mj_name2id(
                web_model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])],
        "name": args.out,
        "kind": "climb",
        "label": f"爬楼梯 · {args.policy}",
        "source": "climb_stairs.py 深度视觉闭环（MuJoCo 深度渲染 → "
                  f"{args.policy}.onnx → PD → mj_step）",
        "stats": stats,
    })

    print("\n" + "=" * 62)
    print(f"策略 {args.policy} | 前进命令 {args.cmd_vx} m/s | 出生 x={args.spawn_x}")
    for lm in stats["landmarks"]:
        print(f"  [{'✓' if lm['passed'] else ' '}] x>{lm['x']:5.2f}  {lm['name']}")
    print(f"最远 x={stats['maxX']}  躯干 z 范围 {stats['rootZMin']}~{stats['rootZMax']}"
          f"  摔倒：{'是' if fell else '否'}")
    print(f"总用时 {time.time() - t_start:.0f} s")
    print("=" * 62)
    pol.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
