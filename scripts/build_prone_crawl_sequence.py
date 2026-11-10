"""把贴地爬行的「动作 1 + 动作 2」拼成一条连续的动作序列 npz。

背景
----
`run_crawl.sh --split` 会产出两份独立轨迹（默认 normal 档，50 Hz）：

    prone_standup_traj.npz   250 帧 / 5.00 s   站立 -> 蹲下 -> 双手撑地 -> 双脚后伸 -> 趴姿
    prone_gait_traj.npz      400 帧 / 8.00 s   俯卧向前爬行 4 个循环

拆段是为了避开接缝处 514 rad/s² 的角加速度尖峰。split 版生成时，
动作 1 的末 48 帧由 `blend_tail_to()` 五次 Hermite 对齐到动作 2 首帧，
实测两者 qpos 差 < 1.4e-4 rad，所以**直接拼接就是连续的**（C0）。
本脚本丢掉动作 2 的首帧（那一帧与动作 1 末帧几乎完全重复，留着会多一次
20 ms 定格），其余按时间顺序接上；gait 重复时按每循环的前进量累加 x，
否则第二个循环会让机器人瞬移回原点。

输出保留源 npz 的全部键（38 维 qpos / time / dt / joint_names /
limb_names / contact_flags / preset / action / meta），`meta.sequence` 里
记录分段边界与接缝残差，所以既能给 `prone_replay.py` 直接回放，
也能给 `web/tools/bake_crawl.py` 烘焙成网页 motion。

用法
----
    python scripts/build_prone_crawl_sequence.py                 # normal 档
    python scripts/build_prone_crawl_sequence.py --preset fast  # 快档
    python scripts/build_prone_crawl_sequence.py --cycles 2     # 爬行段播两遍

产物：``<src-dir>/prone_crawl_sequence[_fast]_traj.npz``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
               / "bxi_example_py_elf3" / "data" / "mujoco_simulation")


def load_part(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到 {path}\n"
                         f"       先跑 bash run_crawl.sh --no-replay [--preset fast]")
    with np.load(path, allow_pickle=True) as z:
        need = ("qpos", "time", "dt", "joint_names", "limb_names",
                "contact_flags", "preset", "action", "meta")
        missing = [k for k in need if k not in z.files]
        if missing:
            raise SystemExit(f"[错误] {path.name} 缺键 {missing}")
        d = dict(
            qpos=np.asarray(z["qpos"], dtype=np.float64).copy(),
            dt=float(z["dt"]),
            names=[str(s) for s in z["joint_names"]],
            limbs=[str(s) for s in z["limb_names"]],
            flags=np.asarray(z["contact_flags"]).copy(),
            preset=str(z["preset"]),
            action=str(z["action"]),
            meta=json.loads(str(z["meta"])),
        )
    if d["qpos"].ndim != 2 or d["qpos"].shape[1] != 38:
        raise SystemExit(f"[错误] {path.name} qpos 形状 {d['qpos'].shape}，期望 [T,38]")
    if len(d["names"]) != 29:
        raise SystemExit(f"[错误] {path.name} joint_names 有 {len(d['names'])} 个，期望 29")
    if abs(d["dt"] - 0.02) > 1e-9:
        raise SystemExit(f"[错误] {path.name} dt={d['dt']}，期望 0.02")
    d["path"] = path
    return d


def build(a: dict, b: dict, cycles: int) -> dict:
    errs = []
    if a["names"] != b["names"]:
        errs.append("两段的 joint_names 顺序不一致，不能直接拼")
    if a["limbs"] != b["limbs"]:
        errs.append("两段的 limb_names 不一致")
    if a["preset"] != b["preset"]:
        errs.append(f"预设不同：动作 1={a['preset']} / 动作 2={b['preset']}")
    if a["flags"].shape[1:] != b["flags"].shape[1:]:
        errs.append("两段的 contact_flags 列数不一致")
    seam = float(np.abs(a["qpos"][-1] - b["qpos"][0]).max())
    if seam > 5e-3:
        errs.append(f"接缝残差 {seam:.4g} > 5e-3：动作 1 末帧没有对齐到动作 2 首帧，"
                    f"请重跑 run_crawl.sh --split（--blend-frames 48）")
    if errs:
        raise SystemExit("[错误] " + "\n       ".join(errs))

    qa, qb = a["qpos"], b["qpos"]
    fa, fb = a["flags"], b["flags"]
    travel = float(qb[-1, 0] - qb[0, 0])       # 一个 gait 循环的 X 向推进
    parts, qparts, fparts = [], [qa], [fa]
    parts.append(dict(action="standup", frames=int(len(qa)),
                      start_frame=0, end_frame=int(len(qa)) - 1,
                      duration_s=round(len(qa) * a["dt"], 3),
                      description="站立 → 蹲下 → 双手撑地 → 双脚后伸 → 趴姿"))
    for k in range(cycles):
        seg = qb[1:].copy()
        fseg = fb[1:].copy()
        if k > 0:                                # 第 2 个循环起累加位移，避免瞬移回原点
            seg[:, 0] += k * travel
        qparts.append(seg)
        fparts.append(fseg)
        start = sum(len(x) for x in qparts[:-1])
        parts.append(dict(action="gait", cycle=k + 1, frames=int(len(seg)),
                          start_frame=int(start),
                          end_frame=int(start + len(seg) - 1),
                          duration_s=round(len(seg) * b["dt"], 3),
                          description="俯卧向前爬行（4 个步态循环）"))

    qpos = np.concatenate(qparts, axis=0)
    flags = np.concatenate(fparts, axis=0)
    n = len(qpos)
    time = np.arange(n, dtype=np.float64) * a["dt"]

    # 数值连续性报告（不是判据：动作 2 本身的万向锁抖动就有 ~0.31 rad/帧）
    d1 = np.abs(np.diff(qpos[:, 7:36], axis=0)).max()
    d2 = np.abs(np.diff(qpos[:, 7:36], n=2, axis=0)).max() / a["dt"] ** 2
    i_seam = len(qa) - 1
    seam_acc = float(np.abs(np.diff(qpos[:, 7:36], n=2, axis=0))[
        i_seam - 1].max()) / a["dt"] ** 2

    meta = dict(a["meta"])
    meta["cycles"] = int(b["meta"].get("cycles", 4)) * cycles
    meta["sequence"] = dict(
        source_parts=[p["path"].name for p in (a, b)],
        parts=parts,
        seam_frame=int(len(qa)),
        seam_max_diff_rad=round(seam, 8),
        seam_ang_acc_rad_s2=round(seam_acc, 1),
        gait_cycles=int(cycles),
        gait_travel_per_cycle_m=round(travel, 4),
        total_travel_x_m=round(float(qpos[-1, 0] - qpos[0, 0]), 4),
        dropped_duplicate_frames=1,
        note=("动作 1 末帧由 blend_tail_to(48) 对齐到动作 2 首帧，"
              "拼接时丢掉动作 2 首帧以免 20 ms 定格；播完冻结，不循环回站立。"),
    )

    return dict(
        qpos=qpos, time=time, dt=a["dt"],
        joint_names=np.asarray(a["names"]),
        limb_names=np.asarray(a["limbs"]),
        contact_flags=flags,
        preset=a["preset"], action="full",
        meta=json.dumps(meta, ensure_ascii=False),
        stats=dict(n=n, d1=float(d1), d2=float(d2), seam=seam,
                   seam_acc=seam_acc, travel=travel),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--preset", choices=("normal", "fast"), default="normal")
    ap.add_argument("--cycles", type=int, default=1,
                    help="gait 段重复次数（默认 1，每份 gait 自带 4 个步态循环）")
    ap.add_argument("--out", default=None, help="输出 npz 路径")
    args = ap.parse_args()
    if args.cycles < 1:
        raise SystemExit("[错误] --cycles 至少为 1")

    src = Path(args.src_dir).expanduser()
    if not src.is_absolute():
        src = (REPO / src).resolve()
    suffix = "" if args.preset == "normal" else "_fast"
    a = load_part(src / f"prone_standup{suffix}_traj.npz")
    b = load_part(src / f"prone_gait{suffix}_traj.npz")
    out = Path(args.out) if args.out else src / f"prone_crawl_sequence{suffix}_traj.npz"

    r = build(a, b, args.cycles)
    np.savez(
        out,
        qpos=r["qpos"], time=r["time"], dt=np.float64(r["dt"]),
        joint_names=r["joint_names"], limb_names=r["limb_names"],
        contact_flags=r["contact_flags"],
        preset=r["preset"], action=r["action"], meta=r["meta"],
    )

    s = r["stats"]
    seq = json.loads(r["meta"])["sequence"]
    print(f"输入 A：{a['path'].name}  {len(a['qpos'])} 帧 / {a['action']}")
    print(f"输入 B：{b['path'].name}  {len(b['qpos'])} 帧 / {b['action']}"
          f"（重复 {args.cycles} 次）")
    print(f"接缝　：f{len(a['qpos'])-1} → f{len(a['qpos'])}，"
          f"qpos 最大差 {s['seam']:.2e}，接缝角加速度 {s['seam_acc']:.1f} rad/s²")
    print(f"输出　：{out}")
    print(f"        {s['n']} 帧 × 38 维 @ {1/r['dt']:.0f} Hz = "
          f"{s['n']*r['dt']:.2f} s，X 向推进 {seq['total_travel_x_m']:.3f} m")
    print(f"        单帧最大关节跳变 {s['d1']:.4f} rad，"
          f"全程二阶差分峰值 {s['d2']:.1f} rad/s²"
          f"（动作 2 自带的万向锁抖动，非拼缝）")
    print("分段　：" + " | ".join(
        f"{p['action']}{'#'+str(p['cycle']) if 'cycle' in p else ''} "
        f"{p['frames']}帧 f{p['start_frame']}~{p['end_frame']}"
        for p in seq["parts"]))
    print("\n回放：python sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/"
          f"launch/prone_replay.py --traj {out.as_posix()} --mode kin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
