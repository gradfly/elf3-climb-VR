"""把三个 prone 动作 npz 烘焙成训练用的参考表。

输入（由 scripts/build_prone_parts.py 生成）：
    logs/export/prone_stand2crawl[_fast].npz    站立 -> 趴姿
    logs/export/prone_crawl[_fast].npz          趴姿向前爬行
    logs/export/prone_crawl2stand[_fast].npz    趴姿 -> 站立

输出：train/assets/ref_tables/prone_ref_v1[_fast].npz，每个动作一组键：

    <key>/q      [T, 29]  29 个可控关节角（JOINT_NAMES 顺序）
    <key>/z      [T]      基座高度（米）
    <key>/xy     [T, 2]   基座水平位置（米）—— 爬行会前进，必须跟踪
    <key>/up     [T, 3]   躯干「本体系重力反方向」（= 躯干朝向）
    <key>/pitch  [T]      前倾角（度，仅日志）
    <key>/time   [T]      时间戳（秒）
    <key>/dt     标量     帧间隔（0.02 s = 50 Hz）
    <key>/names  [29]     关节名（自检用）
    <key>/meta           源文件 / 起止高度 / 总时长

与 build_squat_ref.py 的差别：多存一份 `xy`。蹲起是原地动作（xy 恒为 0），
爬行不是 —— 用「水平位移」当奖励时必须知道参考此刻应该在哪。

用法：
    python train/assets/build_prone_ref.py
    python train/assets/build_prone_ref.py --preset fast
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXPORT = REPO / "logs/export"
OUT = HERE / "ref_tables/prone_ref_v1.npz"

# 与 crab/robot_cfg.py 的 JOINT_NAMES 完全一致（含顺序）
JOINT_NAMES = [
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

KEYS = ("stand2crawl", "crawl", "crawl2stand")


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> 旋转矩阵（world <- body），支持批量 [T,4]。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_one(key: str, preset: str) -> dict:
    suf = "" if preset == "normal" else "_fast"
    src = EXPORT / f"prone_{key}{suf}.npz"
    if not src.exists():
        raise SystemExit(
            f"[ERR] 找不到 {src}\n"
            f"      先生成动作：python scripts/build_prone_parts.py"
            + ("" if preset == "normal" else f" --preset {preset}")
        )
    d = np.load(src, allow_pickle=True)
    qpos = np.asarray(d["qpos"], dtype=np.float64)          # [T,38]
    names = [str(s) for s in d["joint_names"]]
    dt = float(d["dt"])

    if names != JOINT_NAMES:
        raise SystemExit(f"[ERR] {key}: 轨迹关节名与 JOINT_NAMES 不一致\n{names}")

    q = qpos[:, 7:36].astype(np.float32)
    z = qpos[:, 2].astype(np.float32)
    xy = qpos[:, :2].astype(np.float32)

    R = quat_to_mat(qpos[:, 3:7])
    up = np.einsum("tji,j->ti", R, np.array([0.0, 0.0, 1.0])).astype(np.float32)
    pitch = np.degrees(np.arctan2(-up[:, 0], up[:, 2])).astype(np.float32)

    t = (np.arange(len(q)) * dt).astype(np.float32)
    dq = float(np.abs(np.diff(q, axis=0)).max())
    print(f"[{key}] {len(q)} 帧 / {t[-1]:.2f} s  "
          f"z {z[0]:.3f} -> {z[-1]:.3f} m  "
          f"x {xy[0,0]:+.3f} -> {xy[-1,0]:+.3f} m  "
          f"单帧最大跳变 {dq:.4f} rad")

    return dict(
        q=q, z=z, xy=xy, up=up, pitch=pitch, time=t,
        dt=np.float32(dt), names=np.array(JOINT_NAMES),
        meta=np.array(
            dict(key=key, source=src.name, preset=preset,
                 z0=float(z[0]), z_end=float(z[-1]),
                 x0=float(xy[0, 0]), x_end=float(xy[-1, 0]),
                 duration=float(t[-1]), frames=int(len(q))),
            dtype=object,
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="烘焙 prone 参考表")
    ap.add_argument("--preset", choices=("normal", "fast"), default="normal")
    ap.add_argument("--only", choices=KEYS, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    keys = [args.only] if args.only else list(KEYS)
    payload = {}
    for key in keys:
        one = build_one(key, args.preset)
        for k, v in one.items():
            payload[f"{key}/{k}"] = v

    if args.out:
        out = Path(args.out)
    else:
        out = OUT if args.preset == "normal" else OUT.with_name("prone_ref_v1_fast.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[OK] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print("     键：", ", ".join(sorted(payload)))


if __name__ == "__main__":
    main()
