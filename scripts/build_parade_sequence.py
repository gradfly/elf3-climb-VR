"""把「走正步」和「敬礼」拼成一条连续的动作序列 npz。

背景
----
``launch/goose_step.py`` 与 ``launch/salute.py`` 各产出一份 38 维 qpos 轨迹：

    goose_step_traj.npz   341 帧 / 6.82 s   立正 -> 4 个正步周期 -> 收步立正，x: 0 -> 3.152 m
    salute_traj.npz       226 帧 / 4.52 s   立正 -> 抬手 -> 敬礼保持 -> 礼毕 -> 收势，x 恒 0

两段不能直接接，有两处硬伤：

1. **姿态断差 0.92 rad**：正步首尾用的是 ``action_lib.stand_q()``（肩 y=0.2、
   肘 0.6，小臂前伸 34°），敬礼首尾用的是 ``salute.attention_q()``（肩 y=0.04、
   肘 1.52，小臂垂直）。直接拼会在一帧内跳 53°。
2. **位置瞬移 3.152 m**：敬礼段基座 x=0，正步已经走到 3.152 m。

做法
----
    A 正步 341 帧   原样保留（末尾自带 25 帧立正，正好是"收步站定"的一拍）
    B 过渡  50 帧   A 末帧 -> 敬礼首帧，smootherstep 关节插值（两端零速度）
    C 敬礼 225 帧   丢掉与 B 末帧重复的首帧，整段基座 x 平移 +3.152 m
    D 定格  25 帧   重复 C 末帧，画面停在立正

合计 641 帧 = 12.82 s，``loop=False``（末帧是立正，不能回卷到首帧）。

**关键细节：过渡段只插值关节，基座 x/y/z 与四元数沿用正步末帧。**
``lerp_qpos`` 是 38 维整行插值，若把基座一起插，B 段 1 s 内机器人会从
x=3.152 倒着滑回 0（正步白走了）。同理 C 段的 x 偏移是逐帧累加常量，
不是插值。

输出保留源格式（qpos / time / dt / joint_names / action / meta），
``meta.sequence`` 记分段边界与接缝残差（供 ``check_parade_actions.py`` 与
``web/tools/bake_actions.py`` 消费），``meta.parts_meta`` 存两段原始 meta
（两边的 ``frames`` / ``dt`` 键名相同，不能平铺）。

用法
----
    python scripts/build_parade_sequence.py                 # 正步 1 遍 + 敬礼 1 遍
    python scripts/build_parade_sequence.py --salute-cycles 2
    python scripts/build_parade_sequence.py --blend-frames 75 --hold-frames 50

产物：``<src-dir>/parade_sequence_traj.npz``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LAUNCH = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
          / "bxi_example_py_elf3" / "launch")
DEFAULT_SRC = LAUNCH.parent / "data" / "mujoco_simulation"
WEB_XML = REPO / "web" / "assets" / "elf3.xml"

sys.path.insert(0, str(LAUNCH))
sys.path.insert(0, str(REPO / "web" / "tools"))

from action_lib import JOINT_NAMES, smootherstep                   # noqa: E402
from bake_squat import build_qpos_remap, foot_min_z, visual_geoms  # noqa: E402


def repo_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def load_part(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到 {path}\n"
                         f"       先跑 launch/goose_step.py 与 launch/salute.py")
    with np.load(path, allow_pickle=True) as z:
        need = ("qpos", "time", "dt", "joint_names", "action", "meta")
        missing = [k for k in need if k not in z.files]
        if missing:
            raise SystemExit(f"[错误] {path.name} 缺键 {missing}")
        d = dict(qpos=np.asarray(z["qpos"], dtype=np.float64).copy(),
                 dt=float(z["dt"]),
                 names=[str(s) for s in z["joint_names"]],
                 action=str(z["action"]),
                 meta=json.loads(str(z["meta"])))
    errs = []
    if d["qpos"].ndim != 2 or d["qpos"].shape[1] != 38:
        errs.append(f"qpos 形状 {d['qpos'].shape}，期望 [T,38]")
    if d["names"] != JOINT_NAMES:
        errs.append(f"joint_names 与 action_lib 不一致（{len(d['names'])} 个）")
    if abs(d["dt"] - 0.02) > 1e-9:
        errs.append(f"dt={d['dt']}，期望 0.02")
    if errs:
        raise SystemExit(f"[错误] {path.name} 自检失败：\n  - " + "\n  - ".join(errs))
    d["path"] = path
    return d


def blend_row(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """关节空间 smootherstep 插值，**基座（x/y/z + 四元数）原样沿用 a**。

    见模块 docstring：基座若一起插，过渡段会把机器人从正步终点倒着拖回原点。
    两段都是 pitch=0 的立正姿态，四元数恒为 identity，直接沿用即可。
    """
    r = a * (1.0 - s) + b * s
    r[0], r[1], r[2] = a[0], a[1], a[2]
    r[3:7] = a[3:7] / np.linalg.norm(a[3:7])
    return r


def build(goose: dict, salute: dict, n_blend: int, n_hold: int,
          cycles: int) -> dict:
    qa, qb = goose["qpos"], salute["qpos"]
    dt = goose["dt"]

    # 敬礼段整段要平移到的横向位置：正步末帧的基座 x（y / z 本来就相同）
    x_off = float(qa[-1, 0]) - float(qb[0, 0])

    # ---- B 过渡：A 末帧 -> 敬礼首帧（立正）----
    # s 取 k/n_blend（k=1..n_blend），末帧严格等于敬礼首帧；首帧只比 A 末帧
    # 前进 smootherstep(1/50)≈1e-4，是"平滑起步"而不是重复帧。
    blend = np.stack([blend_row(qa[-1], qb[0], smootherstep(k / n_blend))
                      for k in range(1, n_blend + 1)])

    # ---- C 敬礼：丢掉与 B 末帧重复的首帧，整段累加 x 偏移 ----
    sal = np.concatenate([qb[1:]] * cycles, axis=0)
    sal[:, 0] += x_off

    # ---- D 定格：重复敬礼末帧（立正）----
    hold = np.repeat(sal[-1:], n_hold, axis=0)

    qpos = np.concatenate([qa, blend, sal, hold], axis=0)
    n = len(qpos)
    time = np.arange(n, dtype=np.float64) * dt

    # 接缝残差（A->B / B->C / C->D 各一处）
    seams = dict(
        goose_to_blend=float(np.abs(qa[-1] - blend[0]).max()),
        blend_to_salute=float(np.abs(blend[-1] - sal[0]).max()),
        salute_to_hold=float(np.abs(sal[-1] - hold[0]).max()),
    )

    parts = [
        dict(action="goose_walk", frames=int(len(qa)), start_frame=0,
             end_frame=int(len(qa)) - 1, duration_s=round(len(qa) * dt, 3),
             description="立正 → 起步 → 4 个正步周期 → 收步立正"),
        dict(action="attention_blend", frames=int(len(blend)),
             start_frame=int(len(qa)), end_frame=int(len(qa) + len(blend)) - 1,
             duration_s=round(len(blend) * dt, 3),
             description="正步收势手臂 → 敬礼立正（smootherstep 关节插值）"),
        dict(action="salute", frames=int(len(sal)),
             start_frame=int(len(qa) + len(blend)),
             end_frame=int(len(qa) + len(blend) + len(sal)) - 1,
             duration_s=round(len(sal) * dt, 3),
             description="立正 → 抬手 → 敬礼保持 → 礼毕 → 收势"),
        dict(action="hold_attention", frames=int(len(hold)),
             start_frame=n - len(hold), end_frame=n - 1,
             duration_s=round(len(hold) * dt, 3),
             description="立正定格（播完冻结）"),
    ]

    jl = qpos[:, 7:36]
    d1 = float(np.abs(np.diff(jl, axis=0)).max())
    meta = {
        "label": "脚本动作 · 正步 + 敬礼（连贯）",
        "loop": False,
        "in_place": False,
        "dt": dt,
        "sequence": {
            "parts": parts,
            "seams": {k: round(v, 8) for k, v in seams.items()},
            "blend_frames": int(n_blend),
            "hold_frames": int(n_hold),
            "salute_cycles": int(cycles),
            "goose_x_end_m": round(float(qa[-1, 0]), 4),
            "salute_x_offset_m": round(x_off, 4),
            "total_travel_x_m": round(float(qpos[-1, 0] - qpos[0, 0]), 4),
            "dropped_duplicate_frames": 1,
            "source_npz": [repo_rel(goose["path"]), repo_rel(salute["path"])],
            "note": ("正步末帧与敬礼首帧的手臂姿态不同（stand_q vs attention_q，"
                     "肘差 0.92 rad），用 smoothstep 过渡避免单帧跳 53°；"
                     "敬礼段基座 x 累加正步终点位移，不回弹。"),
        },
        "parts_meta": {"goose": goose["meta"], "salute": salute["meta"]},
    }
    return dict(qpos=qpos, time=time, dt=dt, parts=parts, meta=meta,
                seams=seams, d1=d1, stats=dict(n=n, d1=d1,
                                               x_off=x_off,
                                               travel=float(qpos[-1, 0] - qpos[0, 0])))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--blend-frames", type=int, default=50,
                    help="正步 -> 敬礼立正的过渡帧数（默认 50 = 1.0 s）")
    ap.add_argument("--hold-frames", type=int, default=25,
                    help="末帧立正定格帧数（默认 25 = 0.5 s）")
    ap.add_argument("--salute-cycles", type=int, default=1,
                    help="敬礼段重复次数（默认 1；敬礼本身首尾同帧，可直接接）")
    ap.add_argument("--foot-tol", type=float, default=0.012,
                    help="允许的脚底离地/穿地量（米）")
    ap.add_argument("--out", default=None, help="输出 npz 路径")
    args = ap.parse_args()
    if args.blend_frames < 2 or args.hold_frames < 1 or args.salute_cycles < 1:
        raise SystemExit("[错误] --blend-frames ≥2，--hold-frames ≥1，--salute-cycles ≥1")

    src = Path(args.src_dir).expanduser()
    if not src.is_absolute():
        src = (REPO / src).resolve()
    out = Path(args.out) if args.out else src / "parade_sequence_traj.npz"

    goose = load_part(src / "goose_step_traj.npz")
    salute = load_part(src / "salute_traj.npz")
    r = build(goose, salute, args.blend_frames, args.hold_frames,
              args.salute_cycles)

    # ---- 落地自检：与 web/tools/bake_actions.py 同一套公式 ----
    import mujoco  # noqa: E402
    model = mujoco.MjModel.from_xml_path(str(WEB_XML))
    data = mujoco.MjData(model)
    gids = visual_geoms(model)
    ix = build_qpos_remap(model, [str(s) for s in JOINT_NAMES])
    lows = np.asarray([foot_min_z(model, data, row[ix], gids) for row in r["qpos"]])
    lo, hi = float(lows.min()), float(lows.max())

    # ---- 基座连续性：过渡段绝不能出现"倒退" ----
    x = r["qpos"][:, 0]
    back = float(np.diff(x).min())

    print(f"输入 A：{goose['path'].name}  {len(goose['qpos'])} 帧 / {goose['action']}"
          f"  x -> {goose['qpos'][-1, 0]:.3f} m")
    print(f"输入 B：{salute['path'].name}  {len(salute['qpos'])} 帧 / {salute['action']}"
          f"（敬礼 {args.salute_cycles} 遍）")
    print("分段　：" + " | ".join(
        f"{p['action']} {p['frames']}帧 f{p['start_frame']}~{p['end_frame']}"
        for p in r["parts"]))
    s = r["seams"]
    print(f"接缝　：A→B {s['goose_to_blend']:.2e} / B→C {s['blend_to_salute']:.2e}"
          f" / C→D {s['salute_to_hold']:.2e}（qpos 最大分量差）")
    print(f"落地　：视觉最低点 {lo*100:+.2f} ~ {hi*100:+.2f} cm"
          f"（容差 ±{args.foot_tol*100:.1f} cm）")
    print(f"基座　：x {x[0]:+.3f} -> {x[-1]:+.3f} m，敬礼段偏移 {r['stats']['x_off']:+.3f} m，"
          f"单帧最大回退 {back*1000:+.3f} mm")
    print(f"输出　：{out}\n        {r['stats']['n']} 帧 × 38 维 @ {1/r['dt']:.0f} Hz = "
          f"{r['stats']['n']*r['dt']:.2f} s，单帧最大关节跳变 {r['d1']:.4f} rad")

    if s["blend_to_salute"] > 1e-6:
        raise SystemExit(f"[错误] 过渡末帧没对上敬礼首帧（{s['blend_to_salute']:.3e}）")
    if lo < -args.foot_tol or hi > args.foot_tol:
        raise SystemExit("[错误] 脚底没贴在 z=0 上：网页回放会明显穿地/悬空")
    if back < -1e-6:
        raise SystemExit(f"[错误] 基座出现 {abs(back)*1000:.2f} mm 后退"
                         f"（过渡段把基座一起插值了？见 build_parade_sequence 的 blend_row）")

    np.savez(out,
             qpos=r["qpos"], time=r["time"], dt=np.float64(r["dt"]),
             joint_names=np.asarray(JOINT_NAMES),
             action="parade_sequence",
             meta=json.dumps(r["meta"], ensure_ascii=False))

    # ---- 落盘后回归 ----
    with np.load(out, allow_pickle=True) as z:
        q = z["qpos"]
        m = json.loads(str(z["meta"]))
        okk = (q.shape == r["qpos"].shape and str(z["action"]) == "parade_sequence"
               and m["loop"] is False and len(m["sequence"]["parts"]) == 4)
    print(f"[{'OK' if okk else 'FAIL'}] {out.name}: {q.shape[0]} 帧 × {q.shape[1]} = "
          f"{out.stat().st_size/1024:.1f} KB")
    print("\n烘焙：python web/tools/bake_actions.py")
    print("网页：http://localhost:8080/?motion=parade_seq")
    return 0 if okk else 1


if __name__ == "__main__":
    sys.exit(main())
