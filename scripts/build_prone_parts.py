"""把 prone 系列动作拆成三个独立 npz + gif，每个末尾冻结 1s。

三个动作（基座 x 连续，可以依次回放）：

  Action 1 : stand2crawl     站立 → 蹲下 → 双手撑地 → 双脚后伸 → 趴姿
  Action 2 : crawl            俯卧向前爬行（前 N 个循环）
  Action 3 : crawl2stand     趴姿 → 站立（反向播放 prone_standup）

素材：
  prone_standup{, _fast}_traj.npz   250 帧  站立→趴姿（z 1.034→0.100）
  prone_gait{, _fast}_traj.npz      400 / 170 帧  趴姿爬行 4 循环

用法
----
    python scripts/build_prone_parts.py
    python scripts/build_prone_parts.py --preset fast --cycles 2 --freeze-s 1.5
    python scripts/build_prone_parts.py --no-gif

产物：
    logs/export/prone_{action}{_suffix}.npz
    logs/export/prone_{action}{_suffix}.gif
    其中 action ∈ {stand2crawl, crawl, crawl2stand}
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


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到 {path}")
    with np.load(path, allow_pickle=True) as z:
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


def _freeze(qpos: np.ndarray, flags: np.ndarray | None,
            n_frames: int) -> tuple[np.ndarray, np.ndarray | None]:
    """追加 n_frames 冻结帧（重复末帧）。"""
    frozen_q = np.tile(qpos[-1:], (n_frames, 1))
    out_q = np.concatenate([qpos, frozen_q], axis=0)
    frozen_f = None
    out_f = None
    if flags is not None:
        frozen_f = np.tile(flags[-1:], (n_frames, 1))
        out_f = np.concatenate([flags, frozen_f], axis=0)
    return out_q, out_f


def _build_one(action: str, qpos: np.ndarray, flags: np.ndarray | None,
               src_names: str, desc: str, freeze_s: float,
               preset: str, dt: float, original_action: str) -> dict:
    """给一个动作追加冻结帧并组装成 npz-ready dict。"""
    n_freeze = max(0, int(round(freeze_s / dt)))
    if n_freeze > 0:
        qpos, flags = _freeze(qpos, flags, n_freeze)

    n = len(qpos)
    time = np.arange(n, dtype=np.float64) * dt

    # z 方向描述
    z_start = float(qpos[0, 2])
    z_end = float(qpos[-1, 2])
    travel_x = float(qpos[-1, 0] - qpos[0, 0])

    meta = dict(
        source_action=original_action,
        description=desc,
        frames_before_freeze=n - n_freeze,
        freeze_frames=n_freeze,
        freeze_duration_s=round(n_freeze * dt, 4),
        frames_total=n,
        duration_s=round((n - 1) * dt, 4),
        dt=dt,
        hz=round(1.0 / dt, 3),
        z_start=round(z_start, 4),
        z_end=round(z_end, 4),
        travel_x_m=round(travel_x, 4),
        qpos_layout=("0:3 xyz | 3:7 quat | 7:36 29 joints | 36:38 passive"),
        note=("基座位置 qpos[:7] 是参考轨迹的位置，物理仿真时由积分器生成，"
              "不能直接赋值；关节参考只看 qpos[7:36]。"),
    )

    return dict(
        action=action,
        qpos=qpos, time=time, dt=dt,
        joint_names=np.asarray(src_names),
        meta=json.dumps(meta, ensure_ascii=False),
        flags=flags, preset=preset,
    )


def build_all(src: dict, gait: dict, cycles: int, freeze_s: float) -> list[dict]:
    """构建三个动作。"""
    # 先算 gait 的周期
    meta_gait = gait["meta"]
    total_cycles = int(meta_gait.get("cycles", 4))
    cycle_frames = len(gait["qpos"]) // total_cycles
    n_gait = cycle_frames * cycles
    if n_gait > len(gait["qpos"]):
        raise SystemExit(f"[错误] --cycles {cycles} 超过可用 {total_cycles}")

    dt = gait["dt"]
    preset = gait["preset"]

    # Action 1 : standup 原素材 = 站立→趴姿
    a1 = _build_one(
        action="stand2crawl",
        qpos=src["qpos"].copy(),
        flags=src["flags"].copy() if src["flags"] is not None else None,
        src_names=src["names"],
        desc="站立 → 蹲下 → 双手撑地 → 双脚后伸 → 趴姿",
        freeze_s=freeze_s, preset=preset, dt=dt,
        original_action="prone_standup",
    )

    # Action 2 : gait 截取前 N 循环，修正 x 让首帧 = Action 1 末帧 x +1 帧
    x_target_1 = float(a1["qpos"][-1, 0])
    q2 = gait["qpos"][:n_gait].copy()
    f2 = gait["flags"][:n_gait].copy() if gait["flags"] is not None else None
    dx2 = x_target_1 - float(q2[0, 0])
    q2[:, 0] += dx2

    a2 = _build_one(
        action="crawl",
        qpos=q2, flags=f2,
        src_names=gait["names"],
        desc=f"俯卧向前爬行 {cycles} 个循环（每循环 {cycle_frames} 帧）",
        freeze_s=freeze_s, preset=preset, dt=dt,
        original_action="prone_gait",
    )

    # Action 3 : reversed standup，修正 x 让首帧 = Action 2 末帧 x +1 帧
    # 注意 standup 的原末帧是趴姿，首帧是站立
    # reversed: 原倒数第二帧开始倒序 → 趴姿 → ... → 站立
    # 趴姿首帧 x = standup[-1, 0] = 0.14（原素材里趴姿静止位置）
    q3 = src["qpos"][-2::-1].copy()
    f3 = src["flags"][-2::-1].copy() if src["flags"] is not None else None
    x_target_2 = float(a2["qpos"][-1 - int(round(freeze_s / dt)), 0])  # 冻结前末帧 x
    # a2 的冻结帧是爬行到的位置，趴姿静止；冻结前末帧就是 Action 3 该接上的
    dx3 = x_target_2 - float(q3[0, 0])
    q3[:, 0] += dx3

    a3 = _build_one(
        action="crawl2stand",
        qpos=q3, flags=f3,
        src_names=src["names"],
        desc=f"趴姿 → 站立（反向播放 prone_standup，整体 x 偏移 dx3={dx3:.4f}）",
        freeze_s=freeze_s, preset=preset, dt=dt,
        original_action="prone_standup(reversed)",
    )

    # 数值校验：Action 1 末帧 → Action 2 首帧 x 连续
    seam12 = float(abs(a1["qpos"][-1 - int(round(freeze_s/dt)), 0] - a2["qpos"][0, 0]))
    # 冻结帧期间基座位置不变，所以用冻结前末帧比首帧才对
    seam23 = float(abs(a2["qpos"][-1 - int(round(freeze_s/dt)), 0] - a3["qpos"][0, 0]))

    print(f"  [x连续性] a1冻结前末帧x={a1['qpos'][-1 - int(round(freeze_s/dt)), 0]:.4f} → "
          f"a2首帧x={a2['qpos'][0, 0]:.4f}  差={seam12:.6f}")
    print(f"  [x连续性] a2冻结前末帧x={a2['qpos'][-1 - int(round(freeze_s/dt)), 0]:.4f} → "
          f"a3首帧x={a3['qpos'][0, 0]:.4f}  差={seam23:.6f}")

    return [a1, a2, a3]


def _make_gif(result: dict, xml_path: Path, out_path: Path) -> None:
    import mujoco
    from PIL import Image

    import sys
    launch = REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3" / "launch"
    sys.path.insert(0, str(launch))
    import crab_replay as CR  # noqa: E402

    if not xml_path.exists():
        print(f"    [skip] gif (xml 不存在)")
        return
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    q = result["qpos"]
    dt = result["dt"]
    every = max(1, int(round(0.04 / dt)))  # gif ~25 fps
    fps_target = 25

    gr = mujoco.Renderer(m, 360, 640)
    imgs = []
    for i in range(0, len(q), every):
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        cam = CR.make_cam(lookat=d.xpos[torso], dist=1.9, az=105.0, el=-30.0)
        gr.update_scene(d, camera=cam)
        imgs.append(Image.fromarray(gr.render()).convert("P", palette=Image.ADAPTIVE, colors=128))

    duration_ms = int(1000 / fps_target)
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration_ms, loop=0, optimize=True)
    print(f"    gif: {out_path.name}  {len(imgs)} 帧 @ {fps_target} fps")


def save_one(r: dict, outdir: Path, no_gif: bool, xml_path: Path, suf: str) -> None:
    npz_path = outdir / f"prone_{r['action']}{suf}.npz"
    payload = dict(
        qpos=r["qpos"], time=r["time"], dt=np.float64(r["dt"]),
        joint_names=r["joint_names"],
        contact_flags=r["flags"] if r["flags"] is not None else np.empty((0, 0)),
        preset=r["preset"], action=r["action"],
        meta=r["meta"],
    )
    np.savez(npz_path, **payload)
    meta = json.loads(r["meta"])
    print(f"    [{r['action']:14s}] {npz_path.name}  "
          f"{meta['frames_total']} 帧 ({meta['duration_s']:.2f}s)  "
          f"含冻结 {meta['freeze_frames']} 帧")

    if not no_gif:
        _make_gif(r, xml_path, outdir / f"prone_{r['action']}{suf}.gif")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--preset", choices=("normal", "fast"), default="normal")
    ap.add_argument("--cycles", type=int, default=2,
                    help="爬行段取 gait 的前 N 个循环（默认 2）")
    ap.add_argument("--freeze-s", type=float, default=1.0,
                    help="每个动作末尾冻结秒数（默认 1.0 s）")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser()
    if not src.is_absolute():
        src = (REPO / src).resolve()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (REPO / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    suf = "" if args.preset == "normal" else "_fast"
    xml_path = src / "elf3_prone.xml"

    standup = _load(src / f"prone_standup{suf}_traj.npz")
    gait = _load(src / f"prone_gait{suf}_traj.npz")
    print(f"standup={len(standup['qpos'])}帧 gait={len(gait['qpos'])}帧 "
          f"cycles={args.cycles} freeze={args.freeze_s}s\n")

    results = build_all(standup, gait, args.cycles, args.freeze_s)

    print(f"\n输出目录: {outdir}")
    for r in results:
        save_one(r, outdir, args.no_gif, xml_path, suf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
