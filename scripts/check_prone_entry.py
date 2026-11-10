"""动作 1（站立 → 俯卧）过渡段的分阶段实测表：写/核对 docs/prone_crawl.md §3.4 用。

    python scripts/check_prone_entry.py                 # 默认 fast 档
    python scripts/check_prone_entry.py --preset normal
    python scripts/check_prone_entry.py --traj X.npz --frames 60 110 184

v9 四阶段：**蹲下 → 双手撑地 → 双脚向后伸 → 趴姿**。除数值判据外，
这个脚本还直接核对「顺序」本身（这才是 v8 观感不协调的根因）：
  * u<0.31 脚踝 x 必须几乎不动（蹲下时脚钉住）
  * u=0.44 手已贴地（|手最低| ≤ 10 mm），脚 x 还在 −0.12 m 以内
  * 0.44~0.74 脚踝 x 单调后移 ≥ 0.45 m，且手最低点全程保持贴地
  * u<0.85 膝中心必须低于髋（反过来就是 IK 掉进「大腿水平前伸」坐姿分支）
  * 膝胶囊最低 > −10 mm（不下跪、不穿地）、踝胶囊最低 > −10 mm（脚贴地）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = (ROOT / "sim" / "vendor" / "bxi_controller_ros2" / "src"
          / "bxi_example_py_elf3" / "launch")
DATA = LAUNCH.parent / "data" / "mujoco_simulation"
sys.path.insert(0, str(ROOT / "scripts"))
import check_prone_crawl as ck  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast", choices=["fast", "normal"])
    ap.add_argument("--traj", default=None)
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="直接给帧号（默认按 ENTRY_STAGES 的 u 取）")
    args = ap.parse_args()

    traj = Path(args.traj) if args.traj else (
        DATA / f"prone_standup{'_fast' if args.preset == 'fast' else ''}_traj.npz")
    q = np.load(traj, allow_pickle=True)["qpos"]
    n = len(q)
    kin = ck.Kin(ck.SCENE)
    bid = lambda nm: mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, nm)

    if args.frames:
        frames = [min(i, n - 1) for i in args.frames]
    else:
        frames = [min(int(round(u * (n - 1))), n - 1)
                  for u in (0.00, 0.11, 0.24, 0.31, 0.38, 0.44, 0.54,
                            0.64, 0.74, 0.88, 1.00)]

    print(f"轨迹 {traj.name}  n={n}  dt={float(np.load(traj, allow_pickle=True)['dt'])}")
    print(f"{'u':>5} {'帧':>4} {'base':>7} {'pitch':>6} {'髋高':>7} "
          f"{'膝low':>7} {'踝low':>7} {'手low':>7} {'髋x':>8} {'踝x':>8}")
    bad = []
    for i in frames:
        u = i / max(1, n - 1)
        kin.set(q[i])
        pitch = np.degrees(2 * np.arctan2(q[i][5], q[i][3]))
        hip = kin.d.xpos[bid("l_hip_y_link")]
        knee = kin.body_min_z(bid("l_knee_y_link")) * 1000
        # 碰撞胶囊挂在 ankle_x_link（7 个：脚掌/脚趾）与 wrist_z_link 上，
        # ankle_y_link / wrist_y_link 是纯视觉 body，body_min_z 会返回哨兵 9.9
        ankle = kin.body_min_z(bid("l_ankle_x_link")) * 1000
        hand = min(kin.body_min_z(bid(b)) * 1000
                   for b in ("l_wrist_z_link", "r_wrist_z_link"))
        print(f"{u:5.2f} {i:4d} {q[i][2]:7.3f} {pitch:6.0f} {hip[2] * 1000:7.0f} "
              f"{knee:7.1f} {ankle:7.1f} {hand:7.0f} {hip[0] * 1000:8.0f} "
              f"{kin.d.xpos[bid('l_ankle_x_link')][0] * 1000:8.0f}")
        if knee < -10 or ankle < -10:
            bad.append((i, round(u, 2), "膝/踝穿地", round(knee, 1), round(ankle, 1)))
        # 「大腿水平前伸」的坐姿分支指纹：蹲/前倾段膝中心被抬到髋以上（§3.4 第一条硬约束）。
        # u>0.85 之后是俯卧收腿，膝本来就翘在髋上方（蟹式），不再适用这条。
        kcz = kin.d.xpos[bid("l_knee_y_link")][2]
        if u < 0.85 and kcz > hip[2] - 0.02:
            bad.append((i, round(u, 2), "膝中心高于髋（坐姿分支）",
                        round(kcz * 1000), round(hip[2] * 1000)))
    # ---- 顺序判据：v9 的「先撑地、后伸脚」必须能从世界坐标里读出来 ----
    def _at(u: float):
        i = min(int(round(u * (n - 1))), n - 1)
        kin.set(q[i])
        return (kin.d.xpos[bid("l_ankle_x_link")][0],
                min(kin.body_min_z(bid(b)) for b in
                    ("l_wrist_z_link", "r_wrist_z_link")))

    ax0, _ = _at(0.00)
    ax31, _ = _at(0.31)
    ax44, hand44 = _at(0.44)
    ax54, hand54 = _at(0.54)
    ax64, hand64 = _at(0.64)
    ax74, hand74 = _at(0.74)
    print("== 顺序核对（v9）==")
    print(f"  蹲下段脚位移 u0→0.31      : {(ax31-ax0)*1000:+7.1f} mm  "
          f"（应 |·| < 120）")
    print(f"  撑地时 u=0.44 手最低      : {hand44*1000:+7.1f} mm  "
          f"（应 |·| ≤ 10，即已撑住）")
    print(f"  撑地时 u=0.44 脚 x        : {ax44*1000:+7.1f} mm  "
          f"（应 > −120，即还没后伸）")
    print(f"  后伸量 u0.44→0.74         : {(ax74-ax44)*1000:+7.1f} mm  "
          f"（应 ≤ −450）")
    mono = ax44 > ax54 > ax64 > ax74
    print(f"  后伸单调 0.44/0.54/0.64/0.74: "
          f"{[round(v*1000) for v in (ax44, ax54, ax64, ax74)]} mm  "
          f"{'单调 ✔' if mono else '!! 非单调'}")
    hands = max(abs(hand44), abs(hand54), abs(hand64), abs(hand74))
    print(f"  后伸期手离地峰值          : {hands*1000:7.1f} mm  "
          f"（应 ≤ 10，即手是固定支点）")

    if abs(ax31 - ax0) > 0.12:
        bad.append(("顺序", "蹲下段脚提前移动", round((ax31-ax0)*1000, 1)))
    if abs(hand44) > 0.010:
        bad.append(("顺序", "u=0.44 手未撑地", round(hand44*1000, 1)))
    if ax44 < -0.12:
        bad.append(("顺序", "撑地时脚已后伸", round(ax44*1000, 1)))
    if ax74 - ax44 > -0.45 or not mono:
        bad.append(("顺序", "后伸不足或不单调", round((ax74-ax44)*1000, 1)))
    if hands > 0.010:
        bad.append(("顺序", "后伸期手离地", round(hands*1000, 1)))

    print()
    if bad:
        print("!! 违规：")
        for b in bad:
            print("   ", b)
        return 1
    print("OK：顺序正确（蹲下脚不动 → 手先撑地 → 脚单调后伸）、膝/踝无穿地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
