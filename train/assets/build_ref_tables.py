#!/usr/bin/env python3
"""离线生成「侧向步态参考表」，供残差 RL 使用。

思路
----
crab_climb.py 的关节轨迹只依赖 (rise, tread) 三个标量，与台阶在世界中的位置无关。
所以可以离线枚举 (rise, tread)，对每组跑一次稳态两周期（90 帧 @50Hz = 1.8 s），
把 29 个关节角存成表；训练时按地形高度估计 rise、按踏面估计 tread，选表插值。

关键实现细节
------------
* 复用 crab_climb.solve_step，帧数传 46 再丢掉最后一帧 -> 每步 45 帧，周期严格 90 帧。
* 从第 2 步（k=2，G1->S1）开始取两个步（k=2,3），跳过 HOLD 与首步，
  避开 plan_all 把起步 GATHER 髋高拉向邻步造成的过伸。
* 负 rise 组不生成：`build_platforms` 会在末端补一块 top=0 的地面，下楼时最后一级
  会变成 +1.76 m 的假台阶让 `plan_all` 直接失败；而下楼本来就不在训练范围内。

用法
----
    python train/assets/build_ref_tables.py                 # 全量 23x4
    python train/assets/build_ref_tables.py --limit 2       # 只跑前 2 组，冒烟
    python train/assets/build_ref_tables.py --out xxx.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch"
DATA = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
sys.path.insert(0, str(LAUNCH))
sys.path.insert(0, str(LAUNCH.parent))

import mujoco  # noqa: E402

import crab_climb as C  # noqa: E402

# 29 个可控关节（与 train/mjlab_tasks/crab/robot_cfg.py 完全一致）
JOINT_NAMES_29 = [
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint",
    "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint",
    "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
]

STEP_TIME = 0.9
DT = 0.02
CLEARANCE = 0.055
N_FRAME_PER_STEP = int(round(STEP_TIME / DT))   # 45
SOLVE_N = N_FRAME_PER_STEP + 1                  # 46：u=i/45，丢末帧
FIRST_STEADY_STEP = 2                           # k=0,1 是起步，k>=2 稳态
N_STEPS_PER_CYCLE = 2                           # 一个周期两步（先 L 后 R）
N_TREAD_GEN = 8
PERIOD_FRAMES = N_FRAME_PER_STEP * N_STEPS_PER_CYCLE   # 90


def synth_platforms(rise: float, tread: float) -> list:
    """自己拼平台序列：起点地面 + N 级台阶，**不补末端地面**。

    不能用 crab_climb.build_platforms：它固定在末尾补一块 top=0 的 floor_end，
    上楼时最后一级会变成 (0 - N*rise) 的巨大假下台阶，plan_all 直接失败。
    我们只需要中间几级稳态，不需要末端。
    """
    pls = [C.Platform("floor_start", -C.FLOOR_DEPTH, 0.0, 0.0)]
    for i in range(N_TREAD_GEN):
        pls.append(C.Platform(f"t{i}", i * tread, (i + 1) * tread, (i + 1) * rise))
    return pls


def model_path() -> Path:
    robot_only = DATA / "elf3_robot_only.xml"
    return robot_only if robot_only.exists() else DATA / "elf3_crab.xml"


def extract_q29(d: np.ndarray, model) -> np.ndarray:
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in JOINT_NAMES_29]
    return d[adr]


def smooth_circular(q: np.ndarray, k: int) -> np.ndarray:
    """周期序列的循环滑动平均（k 为奇数）。用于抹掉摆动腿松约束时的单帧尖峰。"""
    if k <= 1:
        return q
    pad = k // 2
    qp = np.pad(q, ((pad, pad), (0, 0)), mode="wrap")
    ker = np.ones(k, dtype=np.float64) / k
    return np.stack([np.convolve(qp[:, j], ker, mode="valid")
                     for j in range(q.shape[1])], axis=1).astype(np.float32)


def build_one(cl, model, rise: float, tread: float, smooth: int = 5):
    """返回 (q[90,29], root_y[90], root_z[90], swing[90], dy_cycle, diag)。"""
    pls = synth_platforms(rise, tread)
    sts = C.build_stances(pls)
    if not C.plan_all(sts):
        return None

    frames_q, frames_y, frames_z, swings = [], [], [], []
    worst_err = 0.0
    k_last = FIRST_STEADY_STEP + N_STEPS_PER_CYCLE
    for k in range(FIRST_STEADY_STEP, k_last):
        a, b = sts[k], sts[k + 1]
        mv = "L" if abs(b.y["L"] - a.y["L"]) > 1e-9 else "R"
        picked = None
        for lam, rw in C.LADDER:
            qs, dgs, worst = C.solve_step(cl, a, b, SOLVE_N, CLEARANCE, lam, rw)
            if worst < 1e-3:
                picked = (qs, dgs)
                break
        if picked is None:
            qs, dgs = qs, dgs          # 最后一档结果
        worst_err = max(worst_err, worst)
        for qq, dg in zip(qs[:-1], dgs[:-1]):        # 丢末帧（与下一步首帧重复）
            frames_q.append(extract_q29(qq, model))
            frames_y.append(float(qq[1]))
            frames_z.append(float(qq[2]))
            swings.append(1 if (dg["phase"] == "SWING" and mv == "L")
                          else 2 if (dg["phase"] == "SWING" and mv == "R") else 0)

    # 下一周期首帧，用于跨周期前进量
    a, b = sts[k_last], sts[k_last + 1]
    qs, _, _ = C.solve_step(cl, a, b, SOLVE_N, CLEARANCE, *C.LADDER[0])
    y_next = float(qs[0][1])
    dy_cycle = y_next - frames_y[0]

    q = smooth_circular(np.asarray(frames_q, dtype=np.float32), smooth)
    return q, np.asarray(frames_y, np.float32), np.asarray(frames_z, np.float32), \
        np.asarray(swings, np.int8), float(dy_cycle), float(worst_err)


def check(q, swing, dy, tread, worst) -> tuple[bool, list[str]]:
    msgs = []
    ok = True
    if q.shape != (PERIOD_FRAMES, 29):
        msgs.append(f"shape={q.shape}"); ok = False
    if not np.isfinite(q).all():
        msgs.append("nan"); ok = False
    if worst > 1e-3:
        msgs.append(f"IK err {worst*1e3:.2f} mm"); ok = False
    d1 = np.abs(np.diff(q, axis=0))
    d2 = np.abs(np.diff(q, axis=0, n=2))
    if d1.max() > 0.40:
        msgs.append(f"speed {d1.max():.3f} rad/frame"); ok = False
    # 真正的「断点」表现为二阶差尖峰；抬腿快是平滑的钟形加速，二阶差很小
    if d2.max() > 0.12:
        msgs.append(f"jerk {d2.max():.3f} rad"); ok = False
    if dy <= 0.02:
        msgs.append(f"dy={dy:.3f}"); ok = False
    # 一个周期应前进约一个 tread（两步各半个踏面）
    if abs(dy - tread) > 0.25 * tread:
        msgs.append(f"dy {dy:.3f} vs tread {tread:.2f}")
    if swing.max() == 0:
        msgs.append("no swing"); ok = False
    return ok, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "train/assets/ref_tables/crab_ref_v1.npz"))
    ap.add_argument("--rise-max", type=float, default=0.22)
    ap.add_argument("--rise-step", type=float, default=0.02)
    ap.add_argument("--treads", default="0.24,0.30,0.36,0.40")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 组（冒烟）")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--smooth", type=int, default=5, help="循环平滑窗口（奇数，1=不平滑）")
    args = ap.parse_args()

    rises = np.round(np.arange(0.0, args.rise_max + 1e-9, args.rise_step), 4)
    treads = np.array([float(x) for x in args.treads.split(",")], dtype=np.float32)

    mp = model_path()
    print(f"[model] {mp.name}")
    m = mujoco.MjModel.from_xml_path(str(mp))
    d = mujoco.MjData(m)
    cl = C.CrabClimber(m, d)

    R, T = len(rises), len(treads)
    q_all = np.zeros((R, T, PERIOD_FRAMES, 29), np.float32)
    y_all = np.zeros((R, T, PERIOD_FRAMES), np.float32)
    z_all = np.zeros((R, T, PERIOD_FRAMES), np.float32)
    sw_all = np.zeros((R, T, PERIOD_FRAMES), np.int8)
    dy_all = np.zeros((R, T), np.float32)
    ok_all = np.zeros((R, T), bool)
    err_all = np.zeros((R, T), np.float32)

    t0 = time.perf_counter()
    count = 0
    total = R * T
    for i, r in enumerate(rises):
        for j, t in enumerate(treads):
            count += 1
            if args.limit and count > args.limit:
                break
            res = build_one(cl, m, float(r), float(t), smooth=args.smooth)
            if res is None:
                print(f"[{count:3d}/{total}] rise={r:+.2f} tread={t:.2f}  NO SOLUTION")
                continue
            q, y, z, sw, dy, worst = res
            good, msgs = check(q, sw, dy, float(t), worst)
            q_all[i, j], y_all[i, j], z_all[i, j], sw_all[i, j] = q, y, z, sw
            dy_all[i, j], ok_all[i, j], err_all[i, j] = dy, good, worst
            flag = "ok  " if good else "FAIL"
            extra = ("  " + "; ".join(msgs)) if msgs else ""
            print(f"[{count:3d}/{total}] rise={r:+.2f} tread={t:.2f}  {flag} "
                  f"dy={dy:.3f} v={dy/1.8:.3f} err={worst*1e3:.2f}mm{extra}")
        if args.limit and count > args.limit:
            break

    dt = time.perf_counter() - t0
    print(f"\n[done] {count} 组，用时 {dt:.1f}s，通过 {ok_all.sum()}/{count}")

    # 敏感性回归：rise 0.12 vs 0.22（取最接近的网格）
    try:
        i12 = int(np.argmin(np.abs(rises - 0.12)))
        i22 = int(np.argmin(np.abs(rises - 0.22)))
        j = int(np.argmin(np.abs(treads - 0.30)))
        diff = float(np.abs(q_all[i12, j] - q_all[i22, j]).max())
        print(f"[check] rise {rises[i12]:+.2f} vs {rises[i22]:+.2f} (tread 0.30) "
              f"max joint diff = {diff:.3f} rad（预期 0.35~0.55）")
    except Exception as e:
        print("[check] 敏感性回归跳过:", e)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        q=q_all, root_y=y_all, root_z=z_all, swing=sw_all,
        rise=rises.astype(np.float32), tread=treads,
        dy_cycle=dy_all, v_ref=(dy_all / (PERIOD_FRAMES * DT)).astype(np.float32),
        ok=ok_all, err_pos_max=err_all,
        joint_names=np.array(JOINT_NAMES_29),
        dt=np.float32(DT), period_frames=np.int32(PERIOD_FRAMES),
        step_time=np.float32(STEP_TIME), version=np.int32(1),
    )
    print(f"[OK] 写出 {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
