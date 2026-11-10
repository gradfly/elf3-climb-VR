"""bake_crawl.py — 把「贴地爬行」动作序列烘焙成网页 WASM 可回放资源。

输入是 ``scripts/build_prone_crawl_sequence.py`` 合成的 38 维 npz（动作 1
站立→蹲下→双手撑地→双脚后伸→趴姿 + 动作 2 俯卧向前爬行）：

    sim/vendor/.../data/mujoco_simulation/prone_crawl_sequence_traj.npz
    sim/vendor/.../data/mujoco_simulation/prone_crawl_sequence_fast_traj.npz

网页模型 ``web/assets/elf3.xml`` 是 nq=36（少 head_z / head_y 两个被动关节），
按关节名重映射后写成 float32 行优先二进制：

    assets/motion_crawl.bin / .json      （默认 normal 档）
    assets/motion_crawl_fast.bin / .json （--name crawl_fast）

与 bake_squat.py / bake_actions.py 一样是**纯运动学回放**：网页端只做
mj_kinematics + 插值，不跑 mj_step、没有 onnxruntime，所以不会摔倒。

和蹲起/正步回放的两个不同点（别照抄它们的判据）
------------------------------------------------
1. **不做「脚底全程贴地」自检**：趴姿时脚在身后抬起，接触点是手 + 躯干，
   视觉最低点在 0 附近靠的是腕/躯干而不是脚。这里只查
   *站姿段*（前 60 帧）脚底贴地，以及全程最低点不低于 -12 cm（防整体陷地）。
2. **手 mesh 视觉穿地是源轨迹的已知遗留**：撑地瞬间左右手 mesh 最低约
   -8.9 cm、趴姿段约 -5 cm。sim 端 ``elf3_prone.xml`` 用同一份 STL、同样的
   ``geom_pos/quat``，实测数字完全一致（碰撞代理只有腕部 r=0.05 球，
   数值验收不查视觉网格）。这是动作设计问题，不是 38→36 重映射出错；
   本脚本把它记进 ``stats`` 而不是擅自抬高基座或改手腕姿态（那会让脚离地、
   或者与真机轨迹不一致）。

用法
----
    python web/tools/bake_crawl.py
    python web/tools/bake_crawl.py --preset fast --name crawl_fast
    python web/tools/bake_crawl.py --src <自定义 npz> --name crawl_custom
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
from bake_squat import build_qpos_remap, visual_geoms         # noqa: E402


def repo_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def load_sequence(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"[错误] 找不到 {path}\n"
                         f"       先跑 python scripts/build_prone_crawl_sequence.py")
    with np.load(path, allow_pickle=True) as z:
        need = ("qpos", "time", "dt", "joint_names", "meta", "action", "preset")
        missing = [k for k in need if k not in z.files]
        if missing:
            raise SystemExit(f"[错误] {path.name} 缺键 {missing}")
        d = dict(qpos=np.asarray(z["qpos"], dtype=np.float64).copy(),
                 dt=float(z["dt"]),
                 names=[str(s) for s in z["joint_names"]],
                 action=str(z["action"]),
                 preset=str(z["preset"]),
                 meta=json.loads(str(z["meta"])))
    errs = []
    if d["qpos"].ndim != 2 or d["qpos"].shape[1] != 38:
        errs.append(f"qpos 形状 {d['qpos'].shape}，期望 [T,38]")
    if len(d["names"]) != 29:
        errs.append(f"joint_names 有 {len(d['names'])} 个，期望 29")
    if abs(d["dt"] - 0.02) > 1e-9:
        errs.append(f"dt={d['dt']}，期望 0.02（50 Hz）")
    if d["action"] != "full":
        errs.append(f"action={d['action']}，期望 full（合并序列）")
    if "sequence" not in d["meta"]:
        errs.append("meta 里没有 sequence 分段信息")
    if errs:
        raise SystemExit(f"[错误] {path.name} 自检失败：\n  - " + "\n  - ".join(errs))
    d["path"] = path
    return d


def frame_lows(model, data, row, geom_ids, hand_ids):
    """按网页 MjScene.robotMinZ() 的同一套公式，返回 (整体最低点, 手 mesh 最低点)。

    只用 mesh_vert + geom_xmat/xpos，**不要**再加 model.mesh_pos ——
    MuJoCo 编译 STL 时已把质心偏移折进 geom_xpos。
    """
    import mujoco

    data.qpos[:] = row
    mujoco.mj_kinematics(model, data)
    lo, lo_hand = float("inf"), float("inf")
    for g in geom_ids:
        mid = int(model.geom_dataid[g])
        a = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        v = model.mesh_vert[a:a + n]
        rot = data.geom_xmat[g].reshape(3, 3)
        z = float((v @ rot.T)[:, 2].min() + data.geom_xpos[g][2])
        lo = min(lo, z)
        if g in hand_ids:
            lo_hand = min(lo_hand, z)
    return lo, lo_hand


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=("normal", "fast"), default="normal")
    ap.add_argument("--src", default=None, help="合并序列 npz（默认按 --preset 找）")
    ap.add_argument("--name", default=None,
                    help="motion 名（默认 crawl；fast 档建议 crawl_fast）")
    ap.add_argument("--label", default=None)
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="整段基座 z 的额外偏移（米，默认 0；贴地动作一般不要动）")
    ap.add_argument("--stand-frames", type=int, default=60,
                    help="开头多少帧按「站姿脚贴地」检查（默认 60 = 1.2 s）")
    ap.add_argument("--stand-tol", type=float, default=0.015,
                    help="站姿段允许的视觉最低点偏差（米）")
    args = ap.parse_args()

    suffix = "" if args.preset == "normal" else "_fast"
    src = Path(args.src) if args.src else \
        DEFAULT_SRC / f"prone_crawl_sequence{suffix}_traj.npz"
    if not src.is_absolute():
        src = (REPO / src).resolve()
    name = args.name or ("crawl" if args.preset == "normal" else "crawl_fast")
    label = args.label or ("参考动作 · 贴地爬行（蹲下→撑地→后伸→爬行）"
                           if args.preset == "normal" else
                           "参考动作 · 贴地爬行·快档（0.5 m/s）")
    t0 = time.time()

    import mujoco  # noqa: E402
    model = mujoco.MjModel.from_xml_path(str(WEB_XML))
    data = mujoco.MjData(model)
    geom_ids = visual_geoms(model)
    if not geom_ids:
        raise SystemExit("[错误] 网页模型里找不到 group 2 的视觉 MESH")
    hand_ids = {g for g in geom_ids
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH,
                                      int(model.geom_dataid[g])) or "") in
                ("l_hand", "r_hand")}
    if len(hand_ids) != 2:
        raise SystemExit(f"[错误] 手 mesh geom 数 {len(hand_ids)}，期望 2")

    d = load_sequence(src)
    ix = build_qpos_remap(model, d["names"])
    frames = d["qpos"][:, ix].astype(np.float64).copy()
    # 生成器写 qpos 时把四元数截到 4 位（0.7071），实测范数偏 7.5e-5。
    # 网页端 mj_kinematics 不会替你归一化，非单位四元数会让旋转矩阵轻微缩放，
    # 所以烘焙时统一归一化（源 npz 保持原样，只改写进 bin 的 36 维）。
    qn = np.linalg.norm(frames[:, 3:7], axis=1, keepdims=True)
    quat_fix = float(np.abs(qn - 1.0).max())
    frames[:, 3:7] /= qn
    if args.z_offset:
        frames[:, 2] += args.z_offset
    n = len(frames)
    print(f"源序列　：{src}")
    print(f"网页模型：{WEB_XML}（nq={model.nq}，按关节名 38 → 36 重映射）")
    print(f"输出名　：motion_{name}.bin / .json（{n} 帧 / {n*d['dt']:.2f} s）")
    print(f"[信息] 基座四元数归一化修正量 {quat_fix:.2e}（源 npz 保留 4 位小数）")

    # ---- 自检 ----
    errs = []
    quat_norm = np.linalg.norm(frames[:, 3:7], axis=1)
    if float(np.abs(quat_norm - 1.0).max()) > 1e-5:
        errs.append(f"自由基座四元数未归一化（最大偏差 "
                    f"{np.abs(quat_norm - 1.0).max():.2e}）")
    if abs(float(frames[0, 2]) - 1.0339) > 0.02:
        errs.append(f"首帧不是站姿（z={frames[0, 2]:.4f}）")
    if frames[-1, 2] > 0.16:
        errs.append(f"末帧不是趴姿（z={frames[-1, 2]:.4f}）")
    jump = float(np.abs(np.diff(frames[:, 7:], axis=0)).max())
    if jump > 0.50:
        errs.append(f"单帧关节跳变 {jump:.3f} rad > 0.50（重映射可能错位）")
    print(f"[信息] 单帧最大关节跳变 {jump:.4f} rad（动作 2 自带万向锁抖动，见说明）")

    print("\n=== 逐帧视觉落地自检 ===")
    lows = np.empty(n)
    hand_lows = np.empty(n)
    for i in range(n):
        lows[i], hand_lows[i] = frame_lows(model, data, frames[i],
                                           geom_ids, hand_ids)
    k = min(args.stand_frames, n)
    stand_lo, stand_hi = float(lows[:k].min()), float(lows[:k].max())
    lo, hi = float(lows.min()), float(lows.max())
    print(f"  站姿段 f0~{k-1}：视觉最低 {stand_lo*100:+.2f} ~ {stand_hi*100:+.2f} cm "
          f"（脚贴地）")
    print(f"  全程　　　　：视觉最低 {lo*100:+.2f} cm @ f{int(lows.argmin())}、"
          f"最高 {hi*100:+.2f} cm")
    print(f"  手 mesh 最低：{hand_lows.min()*100:+.2f} cm @ "
          f"f{int(hand_lows.argmin())}（源轨迹固有，sim 端一致）")
    if stand_lo < -args.stand_tol or stand_hi > args.stand_tol:
        errs.append(f"站姿段视觉最低点 {stand_lo*100:+.2f}~{stand_hi*100:+.2f} cm "
                    f"超出 ±{args.stand_tol*100:.1f} cm")
    if lo < -0.12:
        errs.append(f"全程最低点 {lo*100:+.2f} cm < -12 cm（整体陷地）")
    if args.z_offset:
        print(f"  [注意] 用了 --z-offset {args.z_offset} m，脚会整体离地/下沉")
    if errs:
        raise SystemExit("[错误] " + "\n       ".join(errs))

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(1, model.njnt)]
    fps = round(1.0 / d["dt"], 3)
    seq = d["meta"]["sequence"]
    stats = dict(
        frames=n,
        durationS=round(n / fps, 3),
        fps=fps,
        standupFrames=int(seq["parts"][0]["frames"]),
        gaitFrames=int(n - seq["parts"][0]["frames"]),
        seamFrame=int(seq["seam_frame"]),
        gaitCycles=int(seq["gait_cycles"]),
        visualMinZCm=round(lo * 100, 2),
        visualMinZFrame=int(lows.argmin()),
        handMinZCm=round(float(hand_lows.min()) * 100, 2),
        standMinZCm=round(stand_lo * 100, 2),
        standMaxZCm=round(stand_hi * 100, 2),
        maxJointJumpRad=round(jump, 4),
        seamMaxDiffRad=seq["seam_max_diff_rad"],
        seamAngAccRadS2=seq["seam_ang_acc_rad_s2"],
        travelX=round(float(frames[-1, 0] - frames[0, 0]), 4),
        speed=d["meta"].get("speed"),
    )
    meta = dict(
        fps=fps, dt=d["dt"], model="elf3.xml",
        scene="平地（回放时自动隐藏两套台阶）", policy=None,
        camera="crawl", jointNames=joint_names,
        kind="crawl", name=name, label=label,
        loop=False, freezeAtEnd=True, action="full",
        source=("scripts/build_prone_crawl_sequence.py 把动作 1（站立→蹲下→双手"
                "撑地→双脚后伸→趴姿）与动作 2（俯卧向前爬行）拼成一条序列；"
                "launch/prone_crawl.py 逐帧 IK 生成，纯运动学装配"
                "（无策略网络、不跑 mj_step）"),
        zOffset=float(args.z_offset),
        refNpz=repo_rel(src),
        stats=stats,
    )
    print("\n=== 写出 ===")
    save(name, np.ascontiguousarray(frames, dtype=np.float32), meta)

    print("\n=== 回归 ===")
    got = json.loads((ASSETS / f"motion_{name}.json").read_text(encoding="utf-8"))
    nbytes = (ASSETS / got["bin"]).stat().st_size
    arr = np.fromfile(ASSETS / got["bin"], dtype="<f4").reshape(-1, got["nq"])
    ok = (got["nq"] == model.nq and got["nframes"] == n
          and got["kind"] == "crawl" and got["loop"] is False
          and nbytes == got["nq"] * got["nframes"] * 4
          and np.allclose(arr, frames, atol=2e-6)
          and np.array_equal(got["jointNames"], joint_names))
    print(f"  [{'OK' if ok else 'FAIL'}] {n} 帧 × {got['nq']} = "
          f"{nbytes/1024:.1f} KB，loop=False（播完冻结），"
          f"bin 与源 npz 逐元素一致")
    print(f"\n耗时 {time.time()-t0:.1f} s")
    print(f"浏览器打开：http://localhost:8080/?motion={name}&scene=single")
    print("改完 assets 记得跑：node web/tools/make-manifest.mjs")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
