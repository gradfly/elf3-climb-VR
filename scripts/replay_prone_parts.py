"""依次回放 prone 三个独立动作 npz（站立→趴姿 / 爬行 / 趴姿→站立）。

用法
----
    python scripts/replay_prone_parts.py                       # 三段依次回放
    python scripts/replay_prone_parts.py --single crawl        # 只放爬行段
    python scripts/replay_prone_parts.py --mode phys          # PD + 真物理跟踪
    python scripts/replay_prone_parts.py --preset fast        # 用 fast 档素材
    python scripts/replay_prone_parts.py --loop                # 三段循环
    python scripts/replay_prone_parts.py --no-gui              # 只跑不解算

依赖
----
    正式 replay 用 launch/ 下的 crab_replay.run_kin + elf3_play.TrainParams。
    本脚本把 launch/ 目录加到 sys.path，import 时会自动找到。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
LAUNCH = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
          / "bxi_example_py_elf3" / "launch")
EXPORT = REPO / "logs" / "export"
DATA = LAUNCH.parent / "data" / "mujoco_simulation"

sys.path.insert(0, str(LAUNCH))


PARTS_ORDER = ["stand2crawl", "crawl", "crawl2stand"]
PART_DESC = {
    "stand2crawl": "站立 → 蹲下 → 双手撑地 → 双脚后伸 → 趴姿",
    "crawl":       "俯卧向前爬行",
    "crawl2stand": "趴姿 → 站立",
}


def _load_npz(path: Path) -> tuple[np.ndarray, float, dict]:
    """读一个 npz，返回 (qpos, dt, meta)。"""
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到 {path}")
    with np.load(path, allow_pickle=True) as z:
        qpos = np.asarray(z["qpos"], dtype=np.float64).copy()
        dt = float(z["dt"])
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return qpos, dt, meta


def run_kin(m: mujoco.MjModel, d: mujoco.MjData, torso_id: int,
            segs: list[tuple[str, np.ndarray, float]], args) -> None:
    """逐段运动学回放，复用 crab_replay.run_kin 的相机/循环逻辑。"""
    import crab_replay as CR

    # 合并成一个大序列（段间无间断），方便一次性 loop
    big_q = np.concatenate([q for _, q, _dt in segs], axis=0)
    dt = segs[0][2]

    # 先打印段信息
    t_acc = 0.0
    print(f"\n[kin] 合并 {len(segs)} 段，共 {len(big_q)} 帧，"
          f"总时长 {len(big_q)*dt:.2f}s")
    for name, q, _dt in segs:
        meta = None
        print(f"  - {name:14s}  f{t_acc:5.0f}~f{t_acc+len(q)-1:5.0f}  "
              f"{len(q)}帧  z {q[0,2]:.3f}→{q[-1,2]:.3f}  "
              f"x {q[0,0]:.3f}→{q[-1,0]:.3f}")
        t_acc += len(q)

    CR.run_kin(m, d, big_q, dt, speed=args.speed, loop=args.loop,
               track=not args.no_track, record=args.record, every=args.every,
               torso_id=torso_id, dist=args.dist, gui=not args.no_gui,
               width=args.width, height=args.height, az=args.az, el=args.el)


def run_phys(m: mujoco.MjModel, d: mujoco.MjData, torso_id: int,
             segs: list[tuple[str, np.ndarray, float]], args) -> None:
    """逐段 PD + 真物理跟踪。"""
    from elf3_play import TrainParams

    P = TrainParams()
    kp = {n: float(v) * args.kp_scale for n, v in zip(P.joint_names, P.stiffness)}
    kd = {n: float(v) * args.kd_scale for n, v in zip(P.joint_names, P.damping)}

    step = 1.0 / 500.0
    n_sub = max(1, int(round(0.02 / step)))
    m.opt.timestep = step

    # 初始化成第一段第一帧
    name0, q0, _ = segs[0]
    d.qpos[:] = q0[0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    v = None
    if not args.no_gui:
        try:
            import crab_replay as CR
            v = CR.mj_viewer.launch_passive(m, d)
            v.cam.lookat[:] = d.xpos[torso_id]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = args.dist, args.az, args.el
        except Exception as e:
            print(f"[跳过] 打不开窗口（{e}），改为无窗口跑完")

    loop = True
    while loop:
        for seg_i, (name, q, dt) in enumerate(segs):
            if not args.loop and seg_i > 0 and v is None:
                pass
            n = len(q)
            print(f"\n[phys] 段 {seg_i+1}/{len(segs)}: {name}  {n} 帧 / {n*dt:.2f}s")
            errs = []
            for i in range(n):
                q_des = q[i]
                d.qfrc_applied[:] = 0.0
                for jn in P.joint_names:
                    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                    if j < 0:
                        continue
                    qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
                    tau = (d.qfrc_bias[va]
                           + kp[jn] * (q_des[qa] - d.qpos[qa])
                           - kd[jn] * d.qvel[va])
                    lo, hi = m.jnt_actfrcrange[j]
                    if hi > 0:
                        tau = float(np.clip(tau, lo, hi))
                    d.qfrc_applied[va] = tau
                mujoco.mj_step(m, d)
                errs.append(float(np.abs(d.qpos[7:36] - q_des[7:36]).max()))

                if v is not None:
                    v.cam.lookat[:] = d.xpos[torso_id]
                    v.sync()
                    if not v.is_running():
                        loop = False
                        break

                if (i + 1) % 50 == 0 or i + 1 == n:
                    print(f"    f{i+1:4d}/{n}  关节误差max={np.degrees(errs[-1]):.1f}°  "
                          f"基座x={d.qpos[0]:.3f} 躯干z={d.xpos[torso_id][2]:.3f}")
            if not loop:
                break

        if not args.loop or v is None:
            break
        # 循环：回到第一段首帧（物理上可能掉地上，但能循环起来）
        print("\n[phys] loop 重启，回到第一段首帧")
        d.qpos[:] = segs[0][1][0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    if v is not None:
        v.close()

    # 汇总
    all_errs = []
    for _name, q, _dt in segs:
        # 粗略重新算一遍（上面循环时 errs 变量被覆盖了最后一段）
        pass
    # 重新单段精确计算一次（不 loop）
    print("\n[phys 汇总]")
    for name, q, dt in segs:
        d.qpos[:] = q[0]
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        seg_errs = []
        for i in range(len(q)):
            q_des = q[i]
            d.qfrc_applied[:] = 0.0
            for jn in P.joint_names:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if j < 0:
                    continue
                qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
                tau = (d.qfrc_bias[va]
                       + kp[jn] * (q_des[qa] - d.qpos[qa])
                       - kd[jn] * d.qvel[va])
                lo, hi = m.jnt_actfrcrange[j]
                if hi > 0:
                    tau = float(np.clip(tau, lo, hi))
                d.qfrc_applied[va] = tau
            mujoco.mj_step(m, d)
            seg_errs.append(float(np.abs(d.qpos[7:36] - q_des[7:36]).max()))
        errs_deg = np.degrees(seg_errs)
        print(f"  {name:14s}  中位 {np.median(errs_deg):.1f}°  "
              f"95% {np.percentile(errs_deg, 95):.1f}°  "
              f"末帧 x={d.qpos[0]:.3f}  z={d.xpos[torso_id][2]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=("normal", "fast"), default="normal")
    ap.add_argument("--single", choices=PARTS_ORDER, default=None,
                    help="只回放单个动作；默认依次回放三段")
    ap.add_argument("--mode", choices=("kin", "phys"), default="kin")
    ap.add_argument("--loop", action="store_true", help="kin/phys 都支持循环")
    ap.add_argument("--record", default=None, metavar="DIR")
    ap.add_argument("--every", type=int, default=1)
    # 相机
    ap.add_argument("--dist", type=float, default=2.2)
    ap.add_argument("--az", type=float, default=105.0)
    ap.add_argument("--el", type=float, default=-30.0)
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=450)
    ap.add_argument("--speed", type=float, default=1.0)
    # phys
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--src-dir", default=str(EXPORT))
    args = ap.parse_args()

    # 确定要加载的段
    parts = [args.single] if args.single else PARTS_ORDER
    suf = "" if args.preset == "normal" else "_fast"

    src_dir = Path(args.src_dir).expanduser()
    if not src_dir.is_absolute():
        src_dir = (REPO / src_dir).resolve()

    scene = DATA / "elf3_prone.xml"
    if not scene.is_file():
        raise SystemExit(f"[错误] 找不到场景 {scene}")

    # 加载各段
    segs = []
    for name in parts:
        npz = src_dir / f"prone_{name}{suf}.npz"
        if not npz.exists():
            raise SystemExit(f"[错误] 找不到 {npz}。先跑: "
                             f"python scripts/build_prone_parts.py --preset {args.preset}")
        q, dt, meta = _load_npz(npz)
        print(f"加载 {npz.name}: {len(q)} 帧 / {len(q)*dt:.2f}s  "
              f"z {q[0,2]:.3f} → {q[-1,2]:.3f}  x {q[0,0]:.3f} → {q[-1,0]:.3f}  "
              f"[{PART_DESC.get(name, '')}]")
        segs.append((name, q, dt))

    m = mujoco.MjModel.from_xml_path(str(scene))
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    if args.mode == "kin":
        run_kin(m, d, torso_id, segs, args)
    else:
        run_phys(m, d, torso_id, segs, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
