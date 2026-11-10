"""bake_actions.py — 把「走正步 / 敬礼」脚本动作烘焙成网页可回放资源。

输入是 ``launch/goose_step.py`` / ``launch/salute.py`` 产出的 38 维 qpos npz：

    data/mujoco_simulation/goose_step_traj.npz          前进正步（播完冻结）
    data/mujoco_simulation/goose_step_inplace_traj.npz  原地正步（无缝循环）
    data/mujoco_simulation/salute_traj.npz              敬礼（无缝循环）

网页模型 ``web/assets/elf3.xml`` 是 nq=36（没有 head_z / head_y），按关节名
重映射后逐帧算视觉网格最低点做落地自检，再写成 float32 行优先二进制：

    assets/motion_goose_walk.bin/.json
    assets/motion_goose_inplace.bin/.json
    assets/motion_salute.bin/.json

与 bake_squat.py 一样是**纯运动学回放**：网页端只做 mj_kinematics + 插值，
不跑 mj_step、没有 onnxruntime，所以不会摔倒。

    python web/tools/bake_actions.py
    python web/tools/bake_actions.py --z-offset 0.01
    python web/tools/bake_actions.py --foot-tol 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # web/tools
WEB = HERE.parent
ASSETS = WEB / "assets"
REPO = WEB.parent
WEB_XML = ASSETS / "elf3.xml"
DEFAULT_SRC = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
               / "bxi_example_py_elf3" / "data" / "mujoco_simulation")

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from bake_climb import save                                   # noqa: E402
from bake_squat import build_qpos_remap, foot_min_z, visual_geoms  # noqa: E402

# (npz 文件名, 网页 motion 名, 标签, loop, freezeAtEnd, 机位)
# 机位用 parade：网页场景把机器人正面旋到了 three 的 −Z，所以站在 −Z 一侧
# 才是正面。旧配置给敬礼用 squat（+X+Z）实际是从背面看，已改掉。
ACTIONS = [
    ("goose_step_traj.npz", "goose_walk", "脚本动作 · 走正步（前进）",
     False, True, "parade"),
    ("goose_step_inplace_traj.npz", "goose_inplace", "脚本动作 · 原地正步（循环）",
     True, False, "parade"),
    ("salute_traj.npz", "salute", "脚本动作 · 敬礼（右手军礼）",
     True, False, "parade"),
]


def repo_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def load_traj(path: Path) -> dict:
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
    if len(d["names"]) != 29:
        errs.append(f"joint_names 有 {len(d['names'])} 个，期望 29")
    if abs(d["dt"] - 0.02) > 1e-9:
        errs.append(f"dt={d['dt']}，期望 0.02")
    if d["qpos"].ndim == 2 and d["qpos"].shape[1] == 38:
        jump = float(np.abs(np.diff(d["qpos"][:, 7:36], axis=0)).max())
        d["jump"] = jump
        if jump >= 0.30:
            errs.append(f"单帧关节跳变 {jump:.3f} rad ≥ 0.30")
    if errs:
        raise SystemExit(f"[错误] {path.name} 自检失败：\n  - " + "\n  - ".join(errs))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--z-offset", type=float, default=0.0)
    ap.add_argument("--foot-tol", type=float, default=0.012,
                    help="允许的脚底离地/穿地量（米）")
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser()
    if not src.is_absolute():
        src = (REPO / src).resolve()
    t0 = time.time()
    print(f"源目录　：{src}\n网页模型：{WEB_XML}")

    import mujoco  # noqa: E402
    model = mujoco.MjModel.from_xml_path(str(WEB_XML))
    data = mujoco.MjData(model)
    geom_ids = visual_geoms(model)
    if not geom_ids:
        raise SystemExit("[错误] 网页模型里找不到 group 2 的视觉 MESH")

    jobs = []
    for fname, name, label, loop, freeze, camera in ACTIONS:
        d = load_traj(src / fname)
        ix = build_qpos_remap(model, d["names"])
        frames = d["qpos"][:, ix].astype(np.float64).copy()
        if args.z_offset:
            frames[:, 2] += args.z_offset
        jobs.append(dict(d=d, name=name, label=label, loop=loop,
                         freeze=freeze, camera=camera, frames=frames,
                         path=src / fname))

    print(f"[信息] npz nq=38 -> 网页模型 nq={model.nq}（按关节名重映射）")

    print("\n=== 逐帧落地自检 ===")
    for j in jobs:
        lows = np.asarray([foot_min_z(model, data, row, geom_ids)
                           for row in j["frames"]])
        j["lo"], j["hi"] = float(lows.min()), float(lows.max())
        print(f"  {j['name']:16s} {len(lows):3d} 帧  "
              f"最低点 {j['lo']*100:+.2f} ~ {j['hi']*100:+.2f} cm  "
              f"单帧跳变 {j['d']['jump']:.3f} rad")
        if j["lo"] < -args.foot_tol or j["hi"] > args.foot_tol:
            raise SystemExit(
                f"[错误] {j['name']} 脚底没有贴在 z=0 上（容差 ±"
                f"{args.foot_tol*100:.1f} cm）。用 --z-offset 补偿，或检查轨迹。")

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(1, model.njnt)]
    fps = round(1.0 / jobs[0]["d"]["dt"], 3)

    print("\n=== 写出 ===")
    for j in jobs:
        f = np.ascontiguousarray(j["frames"], dtype=np.float32)
        src_meta = j["d"]["meta"]
        stats = dict(
            frames=int(len(f)),
            durationS=round(len(f) / fps, 3),
            fps=fps,
            footMinZCm=round(j["lo"] * 100, 2),
            footMaxZCm=round(j["hi"] * 100, 2),
            maxJointJumpRad=round(j["d"]["jump"], 4),
            travelX=round(float(j["frames"][-1, 0] - j["frames"][0, 0]), 4),
        )
        if j["name"].startswith("goose"):
            stats.update(kickLiftCm=round(float(src_meta.get("kick_lift", 0)) * 100, 1),
                         baseSpeed=src_meta.get("speed"),
                         leanDeg=src_meta.get("lean_deg"),
                         balanceShiftCm=round(
                             float(src_meta.get("balance_shift", 0)) * 100, 2),
                         comResidualMm=round(
                             float(src_meta.get("com_hip_residual", 0)) * 1000, 2),
                         kneeSupportDeg=src_meta.get("knee_support_deg"),
                         kneeSwingDeg=src_meta.get("knee_swing_deg"),
                         hipRiseCm=src_meta.get("hip_rise_cm"),
                         elbowFrontDeg=src_meta.get("elbow_front_deg"),
                         stanceBackDeg=src_meta.get("stance_back_deg"),
                         stanceBackCm=round(
                             float(src_meta.get("stance_back_m", 0)) * 100, 1),
                         armFwdDeg=src_meta.get("arm_fwd_deg"),
                         armMedial=src_meta.get("arm_medial"))
        else:
            stats.update(wristToHeadMm=src_meta.get("wrist_to_head_mm"),
                         wristErrMm=src_meta.get("wrist_err_mm"),
                         wristRightCm=round(
                             float(src_meta.get("wrist_right_m", 0)) * 100, 1),
                         wristY_m=src_meta.get("wrist_y_m"),
                         leftArmDown=src_meta.get("left_forearm_down"))
        meta = dict(
            fps=fps, dt=j["d"]["dt"], model="elf3.xml",
            scene="平地（回放时自动隐藏两套台阶）", policy=None,
            camera=j["camera"], jointNames=joint_names,
            kind="parade", name=j["name"], label=j["label"],
            loop=j["loop"], freezeAtEnd=j["freeze"],
            action=j["d"]["action"],
            source=("launch/goose_step.py / launch/salute.py 逐帧解析 IK 生成，"
                    "纯运动学装配（无策略网络、不跑 mj_step）"),
            zOffset=float(args.z_offset),
            refNpz=repo_rel(j["path"]),
            stats=stats,
        )
        save(j["name"], f, meta)

    print("\n=== 回归 ===")
    ok_all = True
    for j in jobs:
        meta = json.loads((ASSETS / f"motion_{j['name']}.json").read_text(encoding="utf-8"))
        nbytes = (ASSETS / meta["bin"]).stat().st_size
        frames = np.ascontiguousarray(j["frames"], dtype=np.float32)
        ok = (meta["nq"] == model.nq and meta["nframes"] == len(frames)
              and meta["kind"] == "parade" and meta["loop"] == j["loop"]
              and nbytes == meta["nq"] * meta["nframes"] * 4)
        if j["loop"]:
            seam = float(np.abs(frames[0] - frames[-1]).max())
            # 循环版允许"末帧 -> 首帧"有一帧的正常推进；只有敬礼要求严格闭合
            if j["name"] == "salute":
                ok = ok and seam < 1e-9
            print(f"  [{'OK' if ok else 'FAIL'}] {j['name']}: {len(frames)} 帧 × "
                  f"{meta['nq']} = {nbytes/1024:.1f} KB, loop={meta['loop']}, "
                  f"首尾差 {seam:.2e}")
        else:
            print(f"  [{'OK' if ok else 'FAIL'}] {j['name']}: {len(frames)} 帧 × "
                  f"{meta['nq']} = {nbytes/1024:.1f} KB, loop=False（播完冻结）")
        ok_all = ok_all and ok

    print(f"\n耗时 {time.time()-t0:.1f} s")
    print("浏览器打开：http://localhost:8080/?motion=goose_walk")
    print("            http://localhost:8080/?motion=goose_inplace")
    print("            http://localhost:8080/?motion=salute")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
