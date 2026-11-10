"""俯卧爬行动作序列回放器。

    python prone_replay.py --mode kin --loop          # 运动学回放（看动作本身）
    python prone_replay.py --mode phys                # PD + 真物理（看动力学可行性）
    python prone_replay.py --record logs/prone_crawl/frames   # 离屏出 PNG 序列

kin：逐帧把 qpos 写进去做正运动学，纯"摆姿势"，不受重力/接触影响 —— 用来验收
     动作序列本身（姿态、接触、推进方向）。
phys：用训练同款 kp/kd（elf3_play.TrainParams）做重力前馈 PD 跟踪，跑真物理。
     **四肢一律用满增益**（crab_replay.run_phys 把手臂降到 0.3 倍是给站立设计的，
     爬行时手臂承重，降增益必然塌）。开环 PD 跟不上接触切换，会掉接触/漂高，
     这是预期结果，也正是 RL 要补的缺口 —— 所以 phys 只给诊断数字，不给 PASS/FAIL。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_prone.xml"
TRAJ = DATA / "prone_crawl_traj.npz"

LIMBS = [("l_arm", "l", True), ("r_leg", "r", False),
         ("r_arm", "r", True), ("l_leg", "l", False)]


def contact_z_list(m: mujoco.MjModel, d: mujoco.MjData) -> dict[str, float]:
    """四端接触点最低高度（米）。与 prone_crawl.Kin.min_z 同一套解析式。"""
    out = {}
    GE = mujoco.mjtObj.mjOBJ_GEOM
    for name, side, arm in LIMBS:
        want = (f"{side}_wrist_z_collision",) if arm else tuple(
            f"{side}_foot{i}_collision" for i in range(1, 8))
        zs = []
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, GE, g) or ""
            if nm not in want:
                continue
            pos = d.geom_xpos[g]
            r = float(m.geom_size[g][0])
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = d.geom_xmat[g].reshape(3, 3)
                zs.append(pos[2] - (float(m.geom_size[g][1]) * abs(R[2, 2]) + r))
            else:
                zs.append(pos[2] - r)
        out[name] = float(min(zs)) if zs else 9.9
    return out


# ---------------------------------------------------------------- 运动学
def run_kin(args, qpos, dt, meta) -> int:
    import crab_replay as CR          # 复用相机/录制/按键逻辑，不改原文件
    m = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    # 出生点圆盘是个 site，这个 MuJoCo 版本会把它渲染成一块纯黑椭圆（不是阴影），
    # 正好压在身体下方特别难看，直接透明掉。想找出生点看地面网格/光照中心即可。
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    print(f"[kin] {len(qpos)} 帧 / {len(qpos) * dt:.2f} s   "
          f"俯卧躯干高 {qpos[-1, 2] * 100:.1f} cm   速度 "
          f"{meta.get('speed', 0):.3f} m/s")
    CR.run_kin(m, d, qpos, dt, speed=args.speed, loop=args.loop,
               track=not args.no_track, record=args.record, every=args.every,
               torso_id=torso, dist=args.dist, gui=not args.no_gui,
               width=args.width, height=args.height, az=args.az, el=args.el)
    return 0


# ---------------------------------------------------------------- 物理
def run_phys(args, qpos, dt, meta) -> int:
    import crab_replay as CR
    from elf3_play import TrainParams

    P = TrainParams()
    kp = {n: float(v) * args.kp_scale for n, v in zip(P.joint_names, P.stiffness)}
    kd = {n: float(v) * args.kd_scale for n, v in zip(P.joint_names, P.damping)}

    m = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    n = len(qpos)
    step = 1.0 / 500.0                 # 与训练一致：500 Hz 物理 x 10 步 = 50 Hz 控制
    n_sub = max(1, int(round(dt / step)))
    m.opt.timestep = step

    d.qpos[:] = qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    z0 = float(d.xpos[torso][2])

    # 接触高度参考：把轨迹逐帧 FK 一遍，作为"应该有多低"的基准
    ref_z = []
    for i in range(n):
        d.qpos[:] = qpos[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        ref_z.append(contact_z_list(m, d))
    d.qpos[:] = qpos[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    v = None
    if not args.no_gui:
        try:
            v = CR.mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = d.xpos[torso]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = args.dist, args.az, args.el
        except Exception as e:
            print(f"[跳过] 打不开窗口（{e}），改为无窗口跑完")

    flags = None
    if "contact_flags" in np.load(args.traj, allow_pickle=True).files:
        flags = np.load(args.traj, allow_pickle=True)["contact_flags"]

    T = args.max_time if args.max_time else n * dt
    nsteps = int(T / step)
    print(f"[phys] {n} 帧 / {n * dt:.2f} s  PD 跟踪（500 Hz 物理）"
          f"  开环预期会掉接触，看数字即可")

    trk = []
    err_by_joint: dict[str, list[float]] = {n: [] for n in P.joint_names}
    trace = []                                  # (控制步, 四端实际接触高)
    next_print = 0.0
    for s in range(nsteps):
        i = min(int(s / n_sub), n - 1)
        q_des = qpos[i]
        d.qfrc_applied[:] = 0.0
        for jn in P.joint_names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j < 0:
                continue
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            tau = (d.qfrc_bias[va]
                   + kp[jn] * (q_des[qa] - d.qpos[qa])
                   - kd[jn] * d.qvel[va])
            # 用 XML 的执行器力矩上限夹一次（qfrc_applied 本身不受 actuatorfrcrange
            # 约束，不夹会高估机器人能力）
            lo, hi = m.jnt_actfrcrange[j]
            if hi > 0:
                tau = float(np.clip(tau, lo, hi))
            d.qfrc_applied[va] = tau
        mujoco.mj_step(m, d)
        trk.append(float(np.abs(d.qpos[7:] - q_des[7:]).max()))
        for jn in P.joint_names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            qa = m.jnt_qposadr[j]
            err_by_joint[jn].append(abs(float(d.qpos[qa] - q_des[qa])))

        t = s * step
        if s % n_sub == 0:
            trace.append((i, contact_z_list(m, d), float(d.qpos[0])))
        if t >= next_print:
            next_print += 0.5
            cz = contact_z_list(m, d)
            print(f"  t={t:5.2f}s 躯干z={d.xpos[torso][2]:.3f}(目标 "
                  f"{qpos[i][2]:.3f}) 基座x={d.qpos[0]:.3f}(目标 {qpos[i][0]:.3f}) "
                  + "  ".join(f"{k}={cz[k] * 1000:+.0f}mm" for k, *_ in LIMBS)
                  + f"  关节误差max={np.degrees(trk[-1]):.1f}°")
        if v is not None:
            if not args.no_track:
                v.cam.lookat[:] = d.xpos[torso]
            v.sync()
            if not v.is_running():
                break
        if not v.is_running() if v is not None else False:
            break

    if v is not None:
        v.close()

    # ---- 汇总 ----
    n_done = len(trk) // n_sub
    print("\n[phys 汇总] 开环 PD 跟踪的差距（= RL 要补的缺口）：")
    print(f"  跟踪到 {n_done}/{n} 帧（{n_done / n * 100:.0f}%）")
    print(f"  关节跟踪误差 |max| 中位 {np.degrees(np.median(trk)):.1f}°  "
          f"95% {np.degrees(np.percentile(trk, 95)):.1f}°")
    worst = sorted(err_by_joint.items(), key=lambda kv: -np.median(kv[1]))[:3]
    print("  误差最大的关节：" + "  ".join(
        f"{k}={np.degrees(np.median(v)):.1f}°" for k, v in worst))
    print(f"  躯干高度 起始 {z0:.3f} m -> 结束 {d.xpos[torso][2]:.3f} m"
          f"（轨迹该在 {qpos[min(n_done, n - 1)][2]:.3f}）")
    drift = float(d.qpos[0] - qpos[min(n_done, n - 1)][0])
    print(f"  X 向漂移 {drift:+.3f} m（相对轨迹；负=没爬动/后退）")
    et = float(meta.get("entry_time", 0.0))
    i_loop = int(round(et / dt))
    xs = {i: x for i, _cz, x in trace}
    if i_loop in xs and xs:
        x_end = xs[max(xs)]
        want = float(meta.get("speed", 0.0)) * (max(xs) - i_loop) * dt
        got = x_end - xs[i_loop]
        print(f"  循环段实际前进 {got:+.3f} m（轨迹目标 {want:.3f} m）"
              f" -> 开环推进率 {got / want * 100:.0f}%" if want > 1e-6 else "")
    if flags is not None and trace:
        hit = []
        for k, (name, *_rest) in enumerate(LIMBS):
            want = np.array([flags[i, k] == 1 for i, _cz, _x in trace])
            if not want.any():
                hit.append(0.0)
                continue
            act = np.array([abs(cz[name]) for _, cz, _x in trace])
            hit.append(float(np.mean(act[want] < 0.010)))
        print("  四肢贴地保持率（支撑相里仍在 10 mm 内）: "
              + "  ".join(f"{LIMBS[k][0]}={vv * 100:.0f}%"
                          for k, vv in enumerate(hit)))
    print("  说明：开环 PD 天生追不上接触切换，掉接触/漂高是常态；"
          "要看动作序列本身请用 --mode kin。")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="俯卧爬行动作序列回放器")
    ap.add_argument("--traj", default=str(TRAJ))
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--mode", default="kin", choices=["kin", "phys"])
    ap.add_argument("--speed", type=float, default=1.0, help="kin: 播放倍速")
    ap.add_argument("--loop", action="store_true", help="kin: 循环播放")
    ap.add_argument("--no-track", action="store_true", help="相机不跟随")
    ap.add_argument("--dist", type=float, default=2.2, help="相机距离 m")
    ap.add_argument("--az", type=float, default=105.0, help="相机方位角 deg（110 左右是 3/4 侧后视）")
    ap.add_argument("--el", type=float, default=-30.0,
                    help="相机俯仰角 deg。**负值才是俯视**（-30 = 从斜上方 30° 往下看，默认）；"
                         "正值会把相机放到地面以下（仰视，画面只剩天空）")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="kin: 离屏渲染 PNG 序列到该目录")
    ap.add_argument("--every", type=int, default=1, help="录屏时每 N 帧存 1 帧")
    ap.add_argument("--width", type=int, default=800, help="录屏宽 px")
    ap.add_argument("--height", type=int, default=450, help="录屏高 px")
    ap.add_argument("--no-gui", action="store_true", help="不开窗口")
    # phys
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--max-time", type=float, default=None)
    args = ap.parse_args()

    tp = Path(args.traj)
    if not tp.is_file():
        raise SystemExit(f"[错误] 找不到轨迹 {tp}\n       先跑: python prone_crawl.py")
    if not Path(args.scene).is_file():
        raise SystemExit(f"[错误] 找不到场景 {args.scene}\n"
                         f"       先跑: python prone_crawl.py --write-scene")
    z = np.load(tp, allow_pickle=True)
    qpos, dt = z["qpos"], float(z["dt"])
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}

    if args.mode == "kin":
        raise SystemExit(run_kin(args, qpos, dt, meta))
    raise SystemExit(run_phys(args, qpos, dt, meta))


if __name__ == "__main__":
    main()
