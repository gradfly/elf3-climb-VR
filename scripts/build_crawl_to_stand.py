"""构建「爬行 → 站立」完整动作序列 npz + 渲染 GIF。

已有素材（都是 50 Hz, dt=0.02）：
  prone_gait_traj.npz      俯卧向前爬行 4 个循环（normal=400帧, fast=170帧）
  prone_standup_traj.npz   站立 → 蹲下 → 双手撑地 → 双脚后伸 → 趴姿（250帧）

新序列分三段：
  [爬行段] gait 的前 ``--cycles`` 个循环（默认 2）
  [过渡段] gait 末帧 → standup 末帧（趴姿静止） Hermite 插值，``--blend-frames`` 帧
  [站立段] reversed standup（趴 → 站）

为什么需要过渡段：
  gait 末帧是爬行中的某个姿态，和 standup 末帧（趴姿静止）差 ~0.96 rad，
  直接拼接会跳一下。加 20 帧 Hermite 过渡（~0.4 s）即可平滑回趴姿。

用法
----
    python scripts/build_crawl_to_stand.py
    python scripts/build_crawl_to_stand.py --preset fast --cycles 1
    python scripts/build_crawl_to_stand.py --no-gif        # 只产 npz

产物
----
    logs/export/prone_crawl2stand[_fast].npz
    logs/export/prone_crawl2stand[_fast].gif
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
DEFAULT_OUT = REPO / "logs" / "export"


def linear_transition(q_start: np.ndarray, q_end: np.ndarray,
                      n_frames: int) -> np.ndarray:
    """从 q_start 线性插值到 q_end，返回 n_frames 帧（含端点 t=0 和 t=1）。

    调用方负责决定是否丢掉端点。
    """
    t = np.linspace(0, 1, n_frames, endpoint=True)  # 包含端点
    return q_start + np.outer(t, q_end - q_start)


def load_part(path: Path, required_keys: tuple[str, ...]) -> dict:
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到 {path}\n"
                         f"       先跑 bash run_crawl.sh --no-replay [--preset fast]")
    with np.load(path, allow_pickle=True) as z:
        missing = [k for k in required_keys if k not in z.files]
        if missing:
            raise SystemExit(f"[错误] {path.name} 缺键 {missing}")
        d = dict(
            qpos=np.asarray(z["qpos"], dtype=np.float64).copy(),
            dt=float(z["dt"]),
            names=[str(s) for s in z["joint_names"]],
            limbs=[str(s) for s in z["limb_names"]],
            flags=np.asarray(z["contact_flags"]).copy() if "contact_flags" in z else None,
            preset=str(z["preset"]) if "preset" in z else "",
            action=str(z["action"]) if "action" in z else "",
            meta=json.loads(str(z["meta"])) if "meta" in z else {},
        )
    d["path"] = path
    return d


def build(crawl: dict, standup: dict, cycles: int, blend_frames: int) -> dict:
    """拼出 crawl → transition → reversed standup 三段序列。"""
    if crawl["names"] != standup["names"]:
        raise SystemExit("[错误] 两段 joint_names 顺序不一致")

    qa, qb = crawl["qpos"], standup["qpos"]
    fa, fb = crawl["flags"], standup["flags"]
    dt = crawl["dt"]

    # 1. 爬行段：取 gait 的前 N 个循环
    # normal gait: 400帧=4循环, 每循环100帧
    # fast gait: 170帧=4循环, 每循环42.5帧（取整处理）
    total_frames = len(qa)
    meta = crawl["meta"]
    total_cycles = int(meta.get("cycles", 4))
    cycle_frames = total_frames // total_cycles
    n_crawl = cycle_frames * cycles
    if n_crawl > total_frames:
        raise SystemExit(f"[错误] --cycles {cycles} 超过 gait 可用循环数 {total_cycles}")
    crawl_q = qa[:n_crawl].copy()
    crawl_f = fa[:n_crawl].copy() if fa is not None else None

    # gait 前进量（用于修正 x 坐标，保证爬行末帧 x 是累加的）
    # gait 的 x 坐标本身就是累加过的，不用额外处理

    # 2. 过渡段：只对关节维 [7:36] 插值，基座位置/四元数保持 crawl 末帧
    #    这样过渡段不会把机器人从爬行终点拉回起点附近的趴姿位置。
    q_start = crawl_q[-1]
    q_end = qb[-1]
    dx = float(q_start[0] - q_end[0])       # standup 末帧需要整体 x 偏移量

    if blend_frames > 0:
        all_trans = linear_transition(q_start, q_end, blend_frames)
        trans_q = all_trans[1:].copy()
        # 覆盖：基座位置和四元数固定为 crawl 末帧（保持爬行终点位置）
        trans_q[:, :7] = q_start[None, :7]
        # flags 过渡（同样丢掉首帧）
        if fa is not None:
            f0 = crawl_f[-1].astype(np.float64)
            f1 = fb[-1].astype(np.float64)
            t_arr = np.linspace(0, 1, blend_frames, endpoint=True)[1:]
            trans_f = (f0[None, :] * (1 - t_arr[:, None])
                       + f1[None, :] * t_arr[:, None]) > 0.5
        else:
            trans_f = None
    else:
        trans_q = np.empty((0, len(q_start)), dtype=np.float64)
        trans_f = None

    # 3. 站立段：reversed standup（趴→站），跳过 standup 末帧（已作为过渡段末帧）
    #    整体基座 x 偏移 dx，让站立从爬行终点位置原地站起，不回退。
    standup_rev_q = qb[-2::-1].copy()
    standup_rev_q[:, 0] += dx               # x 整体偏移
    standup_rev_f = fb[-2::-1].copy() if fb is not None else None

    # 4. 拼接
    parts = [crawl_q]
    if len(trans_q) > 0:
        parts.append(trans_q)
    parts.append(standup_rev_q)
    qpos = np.concatenate(parts, axis=0)

    if fa is not None:
        f_parts = [crawl_f]
        if trans_f is not None:
            f_parts.append(trans_f)
        f_parts.append(standup_rev_f)
        flags = np.concatenate(f_parts, axis=0)
    else:
        flags = None

    n = len(qpos)
    time = np.arange(n, dtype=np.float64) * dt

    # 数值连续性检查
    d1 = np.abs(np.diff(qpos[:, 7:36], axis=0)).max() if n > 1 else 0
    d2 = np.abs(np.diff(qpos[:, 7:36], n=2, axis=0)).max() / dt ** 2 if n > 2 else 0

    # 接缝残差（拼接后各段边界的实际帧差）
    seams = {}
    if len(trans_q) > 0:
        seams["crawl→trans"] = float(np.abs(crawl_q[-1] - trans_q[0]).max())
        seams["trans→standup"] = float(np.abs(trans_q[-1] - standup_rev_q[0]).max())
    else:
        seams["crawl→standup"] = float(np.abs(crawl_q[-1] - standup_rev_q[0]).max())

    # transition 要跨过的总差（参考值）
    span_crawl_to_standup = float(np.abs(q_start - q_end).max())

    travel_x = float(qpos[-1, 0] - qpos[0, 0])
    z_start = float(qpos[0, 2])
    z_end = float(qpos[-1, 2])

    # 构建 meta
    seq_parts = []
    s0 = 0
    seq_parts.append(dict(
        action="crawl", frames=int(len(crawl_q)),
        start_frame=0, end_frame=int(len(crawl_q)) - 1,
        duration_s=round(len(crawl_q) * dt, 3),
        description=f"俯卧爬行 {cycles} 个循环（每循环 {cycle_frames} 帧）",
    ))
    s0 += len(crawl_q)
    if len(trans_q) > 0:
        seq_parts.append(dict(
            action="transition", frames=int(len(trans_q)),
            start_frame=int(s0), end_frame=int(s0 + len(trans_q) - 1),
            duration_s=round(len(trans_q) * dt, 3),
            description="线性过渡：爬行末帧 → 趴姿静止",
        ))
        s0 += len(trans_q)
    seq_parts.append(dict(
        action="standup", frames=int(len(standup_rev_q)),
        start_frame=int(s0), end_frame=int(s0 + len(standup_rev_q) - 1),
        duration_s=round(len(standup_rev_q) * dt, 3),
        description="趴姿 → 站立（反向播放 prone_standup）",
    ))

    meta_out = dict(
        source_parts=[crawl["path"].name, standup["path"].name],
        parts=seq_parts,
        gait_cycles=int(cycles),
        blend_frames=int(blend_frames),
        seams=seams,
        span_crawl_to_standup_rad=round(span_crawl_to_standup, 6),
        total_travel_x_m=round(travel_x, 4),
        z_start=round(z_start, 4),
        z_end=round(z_end, 4),
        note=("爬行→线性过渡→反向standup；"
              f"爬行末帧到趴姿静止差 {span_crawl_to_standup:.3f} rad，"
              f"用 {blend_frames} 帧线性插值平滑。"),
    )

    out = dict(
        qpos=qpos, time=time, dt=dt,
        joint_names=np.asarray(crawl["names"]),
        limb_names=np.asarray(crawl["limbs"]),
        contact_flags=flags,
        preset=crawl["preset"],
        action="crawl_to_stand",
        meta=json.dumps(meta_out, ensure_ascii=False),
        stats=dict(
            n=n, d1=float(d1), d2=float(d2),
            seams=seams, span=span_crawl_to_standup,
            travel_x=travel_x, z_start=z_start, z_end=z_end,
            cycle_frames=cycle_frames,
        ),
    )
    return out


def render_gif(result: dict, out_path: Path) -> None:
    """用 MuJoCo 离屏渲染 gif。"""
    import mujoco
    from PIL import Image

    import sys
    launch = REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3" / "launch"
    sys.path.insert(0, str(launch))
    import crab_replay as CR  # noqa: E402

    xml_path = DEFAULT_SRC / "elf3_prone.xml"
    if not xml_path.exists():
        print(f"[warn] 找不到 {xml_path}，跳过 GIF 渲染")
        return

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    q = result["qpos"]
    dt = result["dt"]
    every = 3  # 每 3 帧取 1 帧，gif 不会太密

    gr = mujoco.Renderer(m, 360, 640)
    imgs = []
    for i in range(0, len(q), every):
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        cam = CR.make_cam(lookat=d.xpos[torso], dist=1.9, az=105.0, el=-30.0)
        gr.update_scene(d, camera=cam)
        imgs.append(Image.fromarray(gr.render()).convert(
            "P", palette=Image.ADAPTIVE, colors=128))

    duration_ms = int(dt * every * 1000)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=True)
    print(f"  GIF: {out_path}  {len(imgs)} 帧 @ {1000/duration_ms:.1f} fps")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--preset", choices=("normal", "fast"), default="normal")
    ap.add_argument("--cycles", type=int, default=2,
                    help="爬行段取 gait 的前 N 个循环（默认 2，每循环 normal=100帧 fast=42帧）")
    ap.add_argument("--blend-frames", type=int, default=20,
                    help="gait末帧 → 趴姿静止 的线性过渡帧数（默认 20，0=不插值）")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--no-gif", action="store_true", help="只产 npz，不渲染 GIF")
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser()
    if not src.is_absolute():
        src = (REPO / src).resolve()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (REPO / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    suf = "" if args.preset == "normal" else "_fast"

    crawl = load_part(src / f"prone_gait{suf}_traj.npz",
                      ("qpos", "dt", "joint_names", "limb_names", "contact_flags", "preset", "action", "meta"))
    standup = load_part(src / f"prone_standup{suf}_traj.npz",
                        ("qpos", "dt", "joint_names", "limb_names", "contact_flags", "preset", "action", "meta"))

    print(f"gait: {crawl['path'].name}  {len(crawl['qpos'])} 帧 "
          f"meta.cycles={crawl['meta'].get('cycles')}")
    print(f"standup: {standup['path'].name}  {len(standup['qpos'])} 帧")

    r = build(crawl, standup, args.cycles, args.blend_frames)

    # 输出 npz
    out_npz = outdir / f"prone_crawl2stand{suf}.npz"
    np.savez(
        out_npz,
        qpos=r["qpos"], time=r["time"], dt=np.float64(r["dt"]),
        joint_names=r["joint_names"], limb_names=r["limb_names"],
        contact_flags=r["contact_flags"],
        preset=r["preset"], action=r["action"], meta=r["meta"],
    )

    s = r["stats"]
    print(f"\n=== 拼接结果 ===")
    print(f"输出 npz: {out_npz}")
    print(f"  总帧数: {s['n']} × 38 维 @ {1/r['dt']:.0f} Hz = {s['n']*r['dt']:.2f} s")
    print(f"  基座 z: {s['z_start']:.3f} → {s['z_end']:.3f} m (爬行末→站立)")
    print(f"  X 向推进: {s['travel_x']:.4f} m")
    print(f"  单帧最大关节跳变: {s['d1']:.4f} rad")
    print(f"  过渡段总跨度: {s['span']:.4f} rad ({args.blend_frames} 帧线性插值)")
    for k, v in s["seams"].items():
        print(f"  接缝 {k}: {v:.4e} rad")

    seq = json.loads(r["meta"])
    print(f"\n分段:")
    for p in seq["parts"]:
        print(f"  {p['action']:12s} f{p['start_frame']}~f{p['end_frame']}  "
              f"{p['frames']:4d} 帧  {p['duration_s']:.2f} s  "
              f"{p['description']}")

    # 渲染 gif
    if not args.no_gif:
        out_gif = outdir / f"prone_crawl2stand{suf}.gif"
        render_gif(r, out_gif)
    else:
        print("\n[skip] GIF 渲染")

    print(f"\n回放：python sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/"
          f"launch/prone_replay.py --traj {out_npz.as_posix()} --mode kin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
