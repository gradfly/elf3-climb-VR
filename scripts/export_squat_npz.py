"""导出「下蹲 / 起立」各自一份独立的参考 npz（与 ONNX 一一对应）。

仓库里原本有两套数据，都不适合直接交给真机：

  * ``squat_extreme_{down,up}_traj.npz``：38 维 qpos 全量轨迹，是 MuJoCo 回放用的
    「输入素材」，没有 29 关节参考 / up 向量 / 部署契约；
  * ``train/assets/ref_tables/squat_ref_v1.npz``：训练侧烘焙表，down / up 用
    ``"down/q"`` 这类前缀**合并在一个文件里**，且只有 29 关节、没有 qpos。

本脚本把两者合并成**每个模式一份、自包含**的文件，默认输出到 ``logs/export/``，
正好和 ``train/export_onnx.py`` 的 ``squat_down.onnx`` / ``squat_up.onnx`` 配对：

    logs/export/squat_down.npz
    logs/export/squat_up.npz
    logs/export/squat_ref_manifest.json

每份 npz 的键（扁平，不需要前缀）：

    q              [T, 29]  float32  29 个可控关节参考角（JOINT_NAMES 顺序）
    qpos           [T, 38]  float64  MuJoCo 全量 qpos，可直接喂 squat_replay.py
    z              [T]      float32  基座高度（米）
    up             [T, 3]   float32  躯干 up 向量（R^T e_z）
    pitch          [T]      float32  躯干前倾角（度）
    time           [T]      float32  时间戳（秒）
    dt             标量      float32  0.02（50 Hz）
    names          [29]     <U18     关节名（与 q 的列一一对应）
    joint_names    [29]     <U18     同上，别名，兼容 squat_replay.py
    meta           标量 dict          部署契约（见下）

``meta`` 里对部署有用的字段：``mode`` / ``task`` / ``frames`` / ``duration_s`` /
``control_hz`` / ``z_start`` / ``z_end`` / ``freeze_at_end`` / ``qpos_layout`` /
``action_formula`` / ``clip_rad`` / ``progress_formula`` / ``source`` / ``ref_table``。

**纯 numpy，不依赖 mjlab / mujoco / GPU**，本机和服务器都能直接跑。

用法：
    python scripts/export_squat_npz.py                     # 两份都导
    python scripts/export_squat_npz.py --mode down
    python scripts/export_squat_npz.py --outdir sim/vendor/onnx
    python scripts/export_squat_npz.py --no-verify        # 跳过交叉校验（不推荐）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SIMDATA = (REPO / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3"
           / "data/mujoco_simulation")
REF_TABLE = REPO / "train/assets/ref_tables/squat_ref_v1.npz"
CFG_FILE = REPO / "train/mjlab_tasks/crab/squat_env_cfg.py"
DEFAULT_OUTDIR = REPO / "logs/export"

MODES = {
    "down": dict(
        src="squat_extreme_down_traj.npz",
        task="Elf3-Squat-Down-Play",
        desc="站姿 → 极限蹲",
    ),
    "up": dict(
        src="squat_extreme_up_traj.npz",
        task="Elf3-Squat-Up-Play",
        desc="极限蹲 → 站姿",
    ),
}

# 与 train/assets/build_squat_ref.py 的 JOINT_NAMES 完全一致（含顺序）
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

QPOS_LAYOUT = ("0:3 基座位置(xyz) | 3:7 基座四元数(w,x,y,z) | "
               "7:36 29 个可控关节(=q) | 36:38 head_z/head_y(被动)")


def _static_clip() -> dict:
    """从 squat_env_cfg.py 静态读出三组限幅，避免 import mjlab。"""
    fallback = dict(leg=0.35, waist=0.15, arm=0.12)
    try:
        text = CFG_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback
    out = dict(fallback)
    pat = r"(leg|waist|arm)_clip_rad\s*=\s*([0-9]*\.?[0-9]+)"
    for key, val in re.findall(pat, text):
        out[key] = float(val)
    return out


def _repo_rel(path: Path) -> str:
    """尽量给仓库相对路径；--outdir / --ref 在仓库外时退回绝对路径。"""
    p = str(path).replace("\\", "/")
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_one(tag: str, src_dir: Path, ref_path: Path) -> dict:
    """把一个模式的原始轨迹 + 训练参考表读进来，返回未落盘的数据字典。"""
    src = src_dir / MODES[tag]["src"]
    if not src.exists():
        raise SystemExit(f"[ERR] 找不到原始轨迹 {src}\n"
                         f"      先在 launch/ 下跑 split_traj.py，或用 --src-dir 指定目录")
    if not ref_path.exists():
        raise SystemExit(f"[ERR] 找不到训练参考表 {ref_path}\n"
                         f"      补救：python train/assets/build_squat_ref.py")

    with np.load(src, allow_pickle=True) as d:
        qpos = np.asarray(d["qpos"], dtype=np.float64)
        dt = float(d["dt"])
        src_names = [str(s) for s in d["joint_names"]] if "joint_names" in d else []
        raw_meta = dict(d["meta"].item()) if "meta" in d else {}

    with np.load(ref_path, allow_pickle=True) as r:
        need = [f"{tag}/{k}" for k in ("q", "z", "up", "pitch", "time", "dt", "names")]
        missing = [k for k in need if k not in r.files]
        if missing:
            raise SystemExit(f"[ERR] {ref_path.name} 缺键 {missing}")
        ref = {k: np.asarray(r[f"{tag}/{k}"]).copy() for k in
               ("q", "z", "up", "pitch", "time")}
        ref["dt"] = float(r[f"{tag}/dt"])
        ref["names"] = [str(s) for s in r[f"{tag}/names"]]

    return dict(tag=tag, src=src, qpos=qpos, dt=dt, src_names=src_names,
                raw_meta=raw_meta, ref=ref)


class Checker:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((name, bool(ok), detail))

    def close(self, name: str, a, b, tol: float, detail: str = "") -> None:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            self.add(name, False, f"形状不一致 {a.shape} vs {b.shape}")
            return
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.add(name, err <= tol, f"max|Δ|={err:.2e} (容差 {tol:.0e}) {detail}".strip())

    @property
    def ok(self) -> bool:
        return all(r[1] for r in self.rows)

    def report(self) -> None:
        for name, ok, detail in self.rows:
            print(f"  [{'OK' if ok else 'FAIL':4s}] {name}" + (f"  {detail}" if detail else ""))


def verify(items: list[dict]) -> Checker:
    """交叉校验：原始 qpos、训练参考表、两份轨迹的接缝必须自洽。"""
    c = Checker()
    for it in items:
        tag, qpos, ref, dt = it["tag"], it["qpos"], it["ref"], it["dt"]
        T = len(qpos)
        c.add(f"{tag}: qpos 是 [T,38]", qpos.ndim == 2 and qpos.shape[1] == 38,
              f"实际 {qpos.shape}")
        c.add(f"{tag}: 关节名与 JOINT_NAMES 一致",
              it["src_names"] == JOINT_NAMES == ref["names"],
              f"{len(it['src_names'])} 个")
        c.add(f"{tag}: 帧数一致", T == len(ref["q"]) == len(ref["z"]),
              f"qpos={T} ref={len(ref['q'])}")
        c.add(f"{tag}: dt 一致 (0.02)", abs(dt - ref["dt"]) < 1e-9 and abs(dt - 0.02) < 1e-9,
              f"qpos={dt} ref={ref['dt']}")
        c.close(f"{tag}: q == qpos[:,7:36]", ref["q"], qpos[:, 7:36], 2e-6)
        c.close(f"{tag}: z == qpos[:,2]", ref["z"], qpos[:, 2], 2e-6)
        c.close(f"{tag}: time == arange(T)*dt", ref["time"], np.arange(T) * dt, 2e-6)
        dq = float(np.abs(np.diff(qpos[:, 7:36], axis=0)).max())
        c.add(f"{tag}: 单帧关节跳变 < 0.30 rad", dq < 0.30, f"实测 {dq:.4f} rad")
        if tag == "down":
            c.add("down: 基座高度单调下降", bool(np.all(np.diff(ref["z"]) <= 1e-5)),
                  f"{ref['z'][0]:.3f} → {ref['z'][-1]:.3f} m")
        else:
            c.add("up: 基座高度单调上升", bool(np.all(np.diff(ref["z"]) >= -1e-5)),
                  f"{ref['z'][0]:.3f} → {ref['z'][-1]:.3f} m")

    if len(items) == 2:
        d, u = (next(i for i in items if i["tag"] == t) for t in ("down", "up"))
        c.close("接缝: down 末帧 qpos == up 首帧 qpos", d["qpos"][-1], u["qpos"][0], 1e-9)
        c.close("接缝: down 末帧 q == up 首帧 q", d["ref"]["q"][-1], u["ref"]["q"][0], 1e-6)
        c.add("接缝: 蹲姿高度一致",
              abs(float(d["ref"]["z"][-1]) - float(u["ref"]["z"][0])) < 1e-5,
              f"{float(d['ref']['z'][-1]):.4f} / {float(u['ref']['z'][0]):.4f} m")
    return c


def export_one(it: dict, outdir: Path, prefix: str, ref_path: Path) -> dict:
    tag, qpos, ref = it["tag"], it["qpos"], it["ref"]
    T = len(qpos)
    # 顶层 dt 用原始轨迹的 float64 0.02（部署侧要拿它累加时间轴）；
    # q/z/time 仍逐位取自训练参考表，避免部署与训练出现浮点漂移。
    dt = float(it["dt"])
    clip = _static_clip()
    names = np.array(JOINT_NAMES)

    meta = dict(it["raw_meta"])
    meta.update(
        mode=tag,
        task=MODES[tag]["task"],
        desc=MODES[tag]["desc"],
        frames=T,
        duration_s=round((T - 1) * dt, 4),
        control_hz=round(1.0 / dt, 3),
        z_start=float(ref["z"][0]),
        z_end=float(ref["z"][-1]),
        pitch_end=float(ref["pitch"][-1]),
        freeze_at_end=True,
        qpos_layout=QPOS_LAYOUT,
        action_formula=("q_target = q[frame] + ACTION_SCALE_MAP * "
                        "clip(a, ±clip_rad/scale) * residual_scale"),
        clip_rad=clip,
        action_scale_source="train/mjlab_tasks/crab/robot_cfg.py:ACTION_SCALE_MAP",
        progress_formula=("p = frame/(frames-1); "
                          "traj_progress_sincos = [sin(pi*p), cos(pi*p)]"),
        obs_from_ref=("traj_joint_err = q_meas - q[frame]; "
                      "traj_height_err = z_meas - z[frame]"),
        source=it["src"].name,
        ref_table=_repo_rel(ref_path),
        exported_by="scripts/export_squat_npz.py",
    )

    payload = dict(
        q=np.asarray(ref["q"], dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float64),
        z=np.asarray(ref["z"], dtype=np.float32),
        up=np.asarray(ref["up"], dtype=np.float32),
        pitch=np.asarray(ref["pitch"], dtype=np.float32),
        time=np.asarray(ref["time"], dtype=np.float32),
        dt=np.float64(dt),
        names=names,
        joint_names=names,                      # squat_replay.py 读这个键
        meta=np.array(meta, dtype=object),
    )

    out = outdir / f"{prefix}_{tag}.npz"
    if out.exists():
        prev = out.with_suffix(".prev.npz")
        out.replace(prev)
        print(f"  [info] 旧文件已归档为 {prev.name}")
    np.savez_compressed(out, **payload)
    return dict(
        mode=tag,
        task=MODES[tag]["task"],
        file=_repo_rel(out),
        size_bytes=out.stat().st_size,
        sha256=_sha256(out),
        frames=T,
        dt=dt,
        duration_s=round((T - 1) * dt, 4),
        z_start=round(float(ref["z"][0]), 4),
        z_end=round(float(ref["z"][-1]), 4),
        keys=sorted(payload),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出蹲下/起立各自独立的参考 npz")
    ap.add_argument("--mode", choices=("down", "up", "all"), default="all")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--prefix", default="squat", help="输出文件名前缀")
    ap.add_argument("--src-dir", default=str(SIMDATA), help="原始 38 维 qpos 轨迹所在目录")
    ap.add_argument("--ref", default=str(REF_TABLE), help="训练侧合并参考表")
    ap.add_argument("--no-verify", action="store_true", help="跳过交叉校验（不推荐）")
    args = ap.parse_args(argv)

    tags = list(MODES) if args.mode == "all" else [args.mode]
    src_dir = Path(args.src_dir).expanduser().resolve()
    ref_path = Path(args.ref).expanduser().resolve()
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (REPO / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"源轨迹目录：{src_dir}")
    print(f"训练参考表：{ref_path}")
    print(f"输出目录　：{outdir}\n")

    items = [load_one(t, src_dir, ref_path) for t in tags]

    if not args.no_verify:
        print("=== 交叉校验 ===")
        chk = verify(items)
        chk.report()
        if not chk.ok:
            print("\n[ERR] 校验未通过，已中止导出。请先修数据源，不要 --no-verify 蒙混过去。")
            return 1
        print("  全部通过 ✔\n")
    else:
        print("[warn] 已跳过交叉校验（--no-verify）\n")

    print("=== 导出 ===")
    files = []
    for it in items:
        info = export_one(it, outdir, args.prefix, ref_path)
        files.append(info)
        print(f"  [OK] {info['file']}  {info['frames']} 帧 / {info['duration_s']} s  "
              f"z {info['z_start']:.3f} → {info['z_end']:.3f} m  "
              f"({info['size_bytes'] / 1024:.1f} KB)")

    man = dict(
        generated_by="scripts/export_squat_npz.py",
        ref_table=_repo_rel(ref_path),
        verified=not args.no_verify,
        files=files,
    )
    man_path = outdir / f"{args.prefix}_ref_manifest.json"
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"  [OK] {_repo_rel(man_path)}（清单 / 校验和）\n")
    print("配对的 ONNX：train/export_onnx.py --squat {down,up}")
    print("真机读取：q[frame] / z[frame] + 20 ms 帧计数器，"
          "到 frames-1 停住（等价于 freeze_frame）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
