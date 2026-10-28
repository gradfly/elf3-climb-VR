"""crab_replay.py —— 螃蟹步横移爬楼梯【轨迹回放器】

把 crab_climb.py 规划出来的 qpos 序列（.npz）灌回 MuJoCo 场景里播放。

==================== 两种模式 ====================
kin （默认）纯运动学回放：直接把 qpos 写进 data，不跑物理。
            用来检查"轨迹本身对不对"——脚有没有踩在踏面上、有没有穿模、
            身体姿态看着自不自然。这一路一定能放完，因为它不做物理。

phys 物理回放：用 PD + 重力前馈去跟踪这条 qpos 轨迹，跑真实 mj_step。
            用来回答"这条轨迹在真实物理里站不站得住"。
            预期：会摔。crab_climb.py 的体检已经算出约 43% 的帧质心在
            支撑面外（单腿支撑期最长 0.24 s），那段需要动力学/RL 兜底，
            脚本控制器补不上。摔得越晚说明轨迹越接近可行。

用法
----
  python crab_replay.py                       # 回放默认轨迹，开窗
  python crab_replay.py --speed 0.3           # 0.3 倍速慢放
  python crab_replay.py --mode phys           # 物理回放（看它什么时候摔）
  python crab_replay.py --record out/crab     # 离屏渲染，存 PNG 序列
  python crab_replay.py --traj xxx.npz --scene xxx.xml

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


# ---------------------------------------------------------------- 入口
def load_traj(path: Path):
    z = np.load(str(path), allow_pickle=True)
    return z["qpos"], float(z["dt"]) if "dt" in z.files else 0.02


def main() -> None:
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
    args = ap.parse_args()

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
