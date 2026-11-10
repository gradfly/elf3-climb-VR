"""动作 1 下蹲段「腿部电机抖动」的可复现判据（docs/prone_crawl.md §3.4）。

    python scripts/check_prone_shake.py                  # 默认 fast 档
    python scripts/check_prone_shake.py --preset normal
    python scripts/check_prone_shake.py --traj X.npz

背景：深蹲时脚目标几乎不动、腿近竖直，hip_x / hip_z 在任务 Jacobian 上退化，
逐帧 IK 会在解流形上左右漂移。真机上的表现不是「走不到位」，而是蹲着的时候
腿部电机小幅来回抽动 —— 用最大穿地 / 角加速度这些常规判据都查不出来，
得直接看关节角一阶差分的**变号率**和二阶差分 rms。

处理见 prone_crawl.smooth_leg_joints()：对 f0~PLANT_U 窗口的这 4 个关节做
端点羽化高斯低通（sigma=2.0），再由 ground_clamp 补一次接触。

判据（2026-11-09 标定，改动作 1 或调 leg_shake_sigma 后都要重跑）：
  * 下蹲段 u∈[0,0.24]：hip_x/z 二阶差分 rms ≤ 0.5°、|d2|max ≤ 2.0°、
    一阶变号率 ≤ 15%（未平滑时是 1.29/1.20°、5.15/4.74°、26/29%）
  * 撑地段 u∈[0.24,0.44]：hip_x/z 二阶差分 rms ≤ 1.5°
  * hip_y / knee_y 的 rmsD1 ≥ 0.5°：确认只磨掉了冗余自由度的抖动，
    没有把下蹲这条主屈伸曲线本身抹钝（sigma 调太大时这里会先报警）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = (ROOT / "sim" / "vendor" / "bxi_controller_ros2" / "src"
        / "bxi_example_py_elf3" / "data" / "mujoco_simulation")

LEG_JOINTS = [f"{s}_{a}_joint" for s in "lr"
              for a in ("hip_y", "hip_x", "hip_z", "knee_y", "ankle_y")]
REDUNDANT = [f"{s}_{a}_joint" for s in "lr" for a in ("hip_x", "hip_z")]
MAIN = [f"{s}_{a}_joint" for s in "lr" for a in ("hip_y", "knee_y")]


def stats(v_deg: np.ndarray, i0: int, i1: int) -> tuple[float, float, float]:
    d1 = np.diff(v_deg, axis=0)[i0:i1]
    d2 = np.diff(v_deg, n=2, axis=0)[i0:i1]
    flip = (float(np.mean(np.sign(d1[1:]) * np.sign(d1[:-1]) < 0)) * 100
            if len(d1) > 1 else 0.0)
    return (float(np.sqrt(np.mean(d2 ** 2))), float(np.abs(d2).max()), flip)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=["fast", "normal"])
    ap.add_argument("--traj", default=None)
    args = ap.parse_args()

    traj = Path(args.traj) if args.traj else (
        DATA / f"prone_standup{'_fast' if args.preset == 'fast' else ''}_traj.npz")
    z = np.load(traj, allow_pickle=True)
    q = z["qpos"]
    names = [str(x) for x in z["joint_names"]] if "joint_names" in z else None
    n = len(q)
    col = {nm: 7 + names.index(nm) for nm in LEG_JOINTS} if names else None
    if col is None:
        raise SystemExit("npz 里没有 joint_names，无法定位关节列")

    i0, i1, i2 = 0, int(round(0.24 * (n - 1))), int(round(0.44 * (n - 1)))
    print(f"轨迹 {traj.name}  {n} 帧   下蹲段 f0~{i1}   撑地段 f{i1}~{i2}")
    print(f"{'关节':<20}{'下蹲 rmsD2':>12}{'|d2|max':>10}{'变号%':>8}"
          f"{'撑地 rmsD2':>12}{'下蹲 rmsD1':>12}")
    bad = []
    d1_all = {}
    for nm in LEG_JOINTS:
        v = np.degrees(q[:, col[nm]])
        a, mx, fl = stats(v, i0, i1)
        b, _, _ = stats(v, i1, i2)
        d1 = float(np.sqrt(np.mean(np.diff(v, axis=0)[i0:i1] ** 2)))
        d1_all[nm] = d1
        print(f"{nm:<20}{a:12.3f}{mx:10.2f}{fl:8.1f}{b:12.3f}{d1:12.3f}")
        if nm in REDUNDANT:
            if a > 0.5:
                bad.append(f"{nm} 下蹲 rmsD2 {a:.3f}° > 0.5°（还在抖）")
            if mx > 2.0:
                bad.append(f"{nm} 下蹲 |d2|max {mx:.2f}° > 2.0°（有尖峰）")
            if fl > 15.0:
                bad.append(f"{nm} 下蹲变号率 {fl:.1f}% > 15%（来回抽动）")
            if b > 1.5:
                bad.append(f"{nm} 撑地 rmsD2 {b:.3f}° > 1.5°")
        if nm in MAIN and d1 < 0.5:
            bad.append(f"{nm} 下蹲 rmsD1 {d1:.3f}° < 0.5°（下蹲曲线被抹钝了）")

    print()
    if bad:
        print("!! 抖动判据未通过：")
        for b in bad:
            print("   ", b)
        return 1
    print("OK：下蹲段腿部无抖动，且主屈伸曲线保持（未被过度平滑）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
