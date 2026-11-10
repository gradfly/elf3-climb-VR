"""把「站→蹲→站」的循环轨迹切成两段独立动作。

    python split_traj.py squat_extreme_cycle_traj.npz
        -> squat_extreme_down_traj.npz  站姿 → 蹲姿（下蹲）
        -> squat_extreme_up_traj.npz    蹲姿 → 站姿（起立）

两段的接缝帧是同一个蹲姿（down 的末帧 = up 的首帧），可以首尾相接播放。
相位时间从 npz 的 meta 里读（t_pre / t_desc / t_hold），不写死，所以
squat_stand / squat_deep / squat_extreme 的循环轨迹都能切。

    python split_traj.py --hold 0.4 xxx_cycle_traj.npz    # 下蹲末尾多留 0.4 s 蹲姿
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    for base in (DATA, HERE, Path.cwd()):
        cand = base / p.name
        if cand.exists():
            return cand
    raise SystemExit(f"[错误] 找不到 {path}（数据目录 {DATA}）")


def torso_z(qpos: np.ndarray) -> np.ndarray:
    """基座高度直接就是 qpos[:, 2]（free joint 的 z）。"""
    return qpos[:, 2]


def cut(src: Path, outdir: Path, hold: float) -> tuple[Path, Path]:
    data = np.load(src, allow_pickle=True)
    qpos = data["qpos"]
    dt = float(data["dt"])
    meta = dict(data["meta"].item()) if "meta" in data else {}
    names = data["joint_names"] if "joint_names" in data else None

    t_pre = float(meta.get("t_pre", 0.0))
    t_desc = float(meta.get("t_desc", 0.0))
    t_hold = float(meta.get("t_hold", 0.0))
    t_sq = t_pre + t_desc                      # 蹲到位
    t_up = t_sq + t_hold                       # 开始起身

    i_sq = min(int(np.floor(t_sq / dt)) + 1, len(qpos) - 1)   # 含蹲到位那一帧
    i_up = min(int(np.floor(t_up / dt)), len(qpos) - 1)       # 保持段最后一帧
    n_hold = int(round(hold / dt))
    i_sq = min(i_sq + n_hold, i_up)            # 下蹲末尾多带几帧保持

    segs = {}
    segs["down"] = (qpos[:i_sq + 1], dict(meta, segment="down",
                                          t_hold=round(hold, 3), t_rise=0.0))
    segs["up"] = (qpos[i_up:], dict(meta, segment="up",
                                    t_pre=0.0, t_desc=0.0, t_hold=0.0))

    stem = src.stem.replace("_cycle", "").replace("_traj", "")
    outs = []
    for tag, (q, m) in segs.items():
        t = np.arange(len(q)) * dt
        m["duration"] = float(t[-1])
        m["start_z"] = float(torso_z(q)[0])
        m["end_z"] = float(torso_z(q)[-1])
        m["source"] = src.name
        if tag == "up":
            m["t_rise"] = float(t[-1])
        out = outdir / f"{stem}_{tag}_traj.npz"
        payload = dict(time=t, qpos=q, dt=dt, meta=np.array(m, dtype=object))
        if names is not None:
            payload["joint_names"] = names
        np.savez(out, **payload)
        outs.append(out)
        print(f"[OK] {out.name}  {len(q)} 帧 / {t[-1]:.2f} s   "
              f"躯干 {m['start_z']:.3f} → {m['end_z']:.3f} m")
    return outs[0], outs[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="把循环轨迹切成 下蹲 / 起立 两段")
    ap.add_argument("traj", help="循环轨迹 npz（站→蹲→站）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认与源文件同目录）")
    ap.add_argument("--hold", type=float, default=0.0,
                    help="下蹲段末尾额外保留的蹲姿保持时间（s）")
    args = ap.parse_args()

    src = resolve(args.traj)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    cut(src, outdir, args.hold)


if __name__ == "__main__":
    main()
