"""测量脚本动作里「手」与身体其它部位的最小间距。

为什么需要它：手是 2026-11-10 才加进模型里的（纯视觉 mesh，挂在 wrist_z_link
上），而正步/敬礼的手臂轨迹是按「腕」这个点设计的 —— 手从腕再往外伸 0.119 m，
原来留的余量可能不够。这个脚本用顶点最近邻距离量化到底贴多近。

用法::

    python scripts/check_hand_clearance.py                    # 默认查正步两套轨迹
    python scripts/check_hand_clearance.py --traj xxx.npz --limit 10

判据：两段 mesh 的顶点最小距离 < 5 mm 视为接触（穿插）。由于是点采样，
数值只是上界，所以对最贴的若干帧再做全量顶点精确复核。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
DEFAULT_TRAJS = ["goose_step_traj.npz", "goose_step_inplace_traj.npz"]
LIMIT_MM = 5.0


def load_model(xml: pathlib.Path):
    import mujoco

    cwd = os.getcwd()
    os.chdir(xml.parent)
    try:
        return mujoco.MjModel.from_xml_path(xml.name)
    finally:
        os.chdir(cwd)


def mesh_points(m, name: str):
    """mesh 顶点在**所属 body** 坐标系里的坐标（含 mesh_pos/quat 与 geom pos/quat）。

    MuJoCo 编译时会把 STL 重定心并旋到主惯性轴，偏移记在 ``mesh_pos``、旋转记在
    ``mesh_quat``（主轴符号任意）。漏掉这两项，量出来的位姿是错的。
    """
    import mujoco

    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, name)
    if mid < 0:
        return None, -1
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a : a + n] * np.array(m.mesh_scale[mid])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.mesh_quat[mid])
    v = v @ R.reshape(3, 3).T + m.mesh_pos[mid]
    for g in range(m.ngeom):
        if m.geom_dataid[g] != mid:
            continue
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        v = v @ Rg.reshape(3, 3).T + m.geom_pos[g]
        return v, int(m.geom_bodyid[g])
    return v, -1


def min_dist(A: np.ndarray, B: np.ndarray) -> float:
    """两组点的最小欧氏距离（分块算，避免 2500×2500 一次性占内存）。"""
    best = np.inf
    for i in range(0, len(A), 256):
        chunk = A[i : i + 256]
        d2 = ((chunk[:, None, :] - B[None, :, :]) ** 2).sum(-1).min()
        best = min(best, float(d2))
    return float(np.sqrt(best))


def voxel_downsample(pts: np.ndarray, cell: float = 0.006) -> np.ndarray:
    """体素抽稀：每个 6 mm 立方体只留一个点（曲面覆盖均匀，比随机抽样可复现）。

    距离只会被高估 ``cell`` 数量级（点变稀 → 最近邻变远），所以粗扫结果
    只用来挑可疑帧，判定仍在全量顶点上复核。
    """
    key = np.floor(pts / cell).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="*", default=DEFAULT_TRAJS)
    ap.add_argument("--limit", type=float, default=LIMIT_MM, help="判定接触的阈值 mm")
    ap.add_argument("--xml", default="elf3_squat.xml")
    ap.add_argument("--worst", type=int, default=6, help="列出最贴的若干帧")
    args = ap.parse_args()

    import mujoco

    m = load_model(DATA / args.xml)
    d = mujoco.MjData(m)

    # 手 + 手臂链（含对侧）+ 大腿 + 腰：这几组是唯一可能撞上的
    names = []
    for s in "lr":
        names += [f"{s}_hand", f"{s}_wrist_z_link", f"{s}_elbow_y_link",
                  f"{s}_shoulder_z_link", f"{s}_hip_x_link"]
    names.append("waist_z_link")
    P = {}
    for n in names:
        v, b = mesh_points(m, n)
        if v is not None:
            P[n] = (v, b)
    if "l_hand" not in P or "r_hand" not in P:
        print("[ERR] 模型里没有手部 mesh，先跑 scripts/add_hand_mesh.py")
        return 2

    pairs = [("l_hand", "r_hand"),
             ("l_hand", "r_elbow_y_link"), ("r_hand", "l_elbow_y_link"),
             ("l_hand", "r_wrist_z_link"), ("r_hand", "l_wrist_z_link"),
             ("l_hand", "r_shoulder_z_link"), ("r_hand", "l_shoulder_z_link"),
             ("l_hand", "l_elbow_y_link"), ("r_hand", "r_elbow_y_link"),
             ("l_hand", "l_hip_x_link"), ("r_hand", "r_hip_x_link"),
             ("l_hand", "waist_z_link"), ("r_hand", "waist_z_link")]
    pairs = [p for p in pairs if p[0] in P and p[1] in P]

    def world(n: str) -> np.ndarray:
        v, b = P[n]
        return v @ d.xmat[b].reshape(3, 3).T + d.xpos[b]

    failed = 0
    for tf in args.traj:
        path = DATA / tf
        if not path.exists():
            print(f"[SKIP] {tf} 不存在")
            continue
        q = np.load(path)["qpos"]
        print(f"\n=== {tf}  {len(q)} 帧 ===")

        # 1) 粗扫：世界系体素抽稀到 10 mm，只算「手对手/手对同侧腿/手对躯干」
        #    三组（其余组合实测远未接近），挑出最贴的若干帧。
        coarse_pairs = [("l_hand", "r_hand"),
                        ("l_hand", "l_hip_x_link"), ("r_hand", "r_hip_x_link"),
                        ("l_hand", "waist_z_link"), ("r_hand", "waist_z_link")]
        coarse_pairs = [p for p in coarse_pairs if p in pairs]
        coarse = []
        for k in range(len(q)):
            d.qpos[:] = q[k]
            mujoco.mj_kinematics(m, d)
            w = {n: voxel_downsample(world(n), 0.010) for n in P}
            coarse.append(min(min_dist(w[a], w[b]) for a, b in coarse_pairs) * 1000)
        coarse = np.asarray(coarse)
        order = np.argsort(coarse)[: args.worst]
        print(f"  粗扫最小 {coarse.min():.1f} mm @帧 {int(coarse.argmin())}")

        # 2) 对最贴的帧做全量精确复核
        print(f"  {'帧':>5}  {'最小间距 mm':>10}  最贴的一对")
        worst_exact = np.inf
        for k in order:
            d.qpos[:] = q[k]
            mujoco.mj_kinematics(m, d)
            w = {n: world(n) for n in P}
            vals = {p: min_dist(w[p[0]], w[p[1]]) * 1000 for p in pairs}
            pk = min(vals, key=vals.get)
            worst_exact = min(worst_exact, vals[pk])
            print(f"  {int(k):>5}  {vals[pk]:>10.2f}  {pk[0]} ↔ {pk[1]}")
        bad = worst_exact < args.limit
        print(f"  => 最小 {worst_exact:.2f} mm  "
              f"{'[FAIL] 手撞到别的部件' if bad else '[PASS]'}")
        failed += int(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
