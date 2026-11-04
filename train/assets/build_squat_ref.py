"""把「下蹲 / 起立」脚本轨迹整理成训练用的参考表。

输入是 `launch/squat_extreme_{down,up}_traj.npz`（qpos 38 维、50 Hz），
输出 `train/assets/ref_tables/squat_ref_v1.npz`，只保留训练真正需要的东西：

    q     [T, 29]  29 个可控关节角（JOINT_NAMES 顺序，与动作空间一一对应）
    z     [T]      基座高度（米，世界系）—— 蹲多深就靠它量
    up    [T, 3]   躯干的「本体系重力反方向」单位向量（= 前倾角）
    time  [T]      时间戳（秒）
    dt    标量     帧间隔（0.02 s = 50 Hz，正好等于控制频率）
    names [29]     关节名（自检用）
    meta          segment / 起止高度 / 总时长 / 源文件

为什么要预烘焙，而不是训练时直接读原 npz
----------------------------------------
原 npz 的 `qpos` 是 38 维（free joint 7 + 29 个可控 + head_z/head_y 两个被动），
而 `joint_names` 只给了 29 个 —— 训练侧要靠名字反查 qpos 列号，一旦机器人 XML
的关节顺序变了就会静默错位（这类错位在 mjlab 里表现为「策略怎么都学不会」，
极难查）。这里在**离线**用 mujoco 读一遍 XML 把它钉死，训练侧只吃成品表。

用法：
    python train/assets/build_squat_ref.py            # 两份都做
    python train/assets/build_squat_ref.py --only down
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SIMDATA = (REPO / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3"
           / "data/mujoco_simulation")
OUT = HERE / "ref_tables/squat_ref_v1.npz"

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

SOURCES = {
    "down": "squat_extreme_down_traj.npz",
    "up": "squat_extreme_up_traj.npz",
}


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> 旋转矩阵 R（world <- body），支持批量 [T,4]。"""
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


def build_one(tag: str) -> dict:
    src = SIMDATA / SOURCES[tag]
    if not src.exists():
        raise SystemExit(f"[ERR] 找不到 {src}（先在 launch/ 下跑 split_traj.py）")
    d = np.load(src, allow_pickle=True)
    qpos = np.asarray(d["qpos"], dtype=np.float64)          # [T,38]
    names = [str(s) for s in d["joint_names"]]
    dt = float(d["dt"])
    meta = dict(d["meta"].item()) if "meta" in d else {}

    # 29 个可控关节：elf3_robot_only.xml 里就是 qpos[7:36]（head_z/head_y 在 36/37）
    if names != JOINT_NAMES:
        raise SystemExit(f"[ERR] {tag}: 轨迹关节名与 JOINT_NAMES 不一致\n{names}")
    q = qpos[:, 7:36].astype(np.float32)
    z = qpos[:, 2].astype(np.float32)

    R = quat_to_mat(qpos[:, 3:7])                            # [T,3,3]
    up = np.einsum("tji,j->ti", R, np.array([0.0, 0.0, 1.0]))  # R^T e_z
    up = up.astype(np.float32)
    pitch = np.degrees(np.arctan2(-up[:, 0], up[:, 2])).astype(np.float32)

    t = np.arange(len(q)) * dt
    print(f"[{tag}] {len(q)} 帧 / {t[-1]:.2f} s  "
          f"z {z[0]:.3f} -> {z[-1]:.3f} m  "
          f"前倾 {pitch[0]:+.1f}° -> {pitch[-1]:+.1f}°（峰 {pitch.max():+.1f}°）")
    # 逐帧跳变量自检：正常轨迹单帧关节变化 < 0.1 rad
    dq = np.abs(np.diff(q, axis=0)).max()
    print(f"      单帧最大关节跳变 {dq:.4f} rad，末帧速度 "
          f"{np.abs(q[-1] - q[-2]).max() / dt:.3f} rad/s")
    if dq > 0.30:
        print(f"      [!] 单帧跳变 {dq:.3f} rad 偏大，检查轨迹是否有断点")

    return dict(q=q, z=z, up=up, pitch=pitch, time=t.astype(np.float32),
                dt=np.float32(dt), names=np.array(JOINT_NAMES),
                meta=np.array(dict(meta, segment=tag,
                                   source=src.name,
                                   z0=float(z[0]), z_end=float(z[-1]),
                                   duration=float(t[-1]),
                                   pitch_end=float(pitch[-1])), dtype=object))


def main() -> None:
    ap = argparse.ArgumentParser(description="烘焙下蹲/起立参考表")
    ap.add_argument("--only", choices=sorted(SOURCES), default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tags = [args.only] if args.only else list(SOURCES)
    payload = {}
    for tag in tags:
        one = build_one(tag)
        for k, v in one.items():
            payload[f"{tag}/{k}"] = v

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[OK] {out}  ({out.stat().st_size / 1024:.1f} KB)")
    print("     键：", ", ".join(sorted(payload)))


if __name__ == "__main__":
    main()
