"""bake_squat.py —— 把「下蹲 / 起立」参考轨迹烘焙成网页 WASM 可回放的资源

输入是 ``scripts/export_squat_npz.py`` 产出的两份自包含 npz：

    logs/export/squat_down.npz     qpos [124, 38]  站姿 -> 极限蹲
    logs/export/squat_up.npz       qpos [102, 38]  极限蹲 -> 站姿

网页端跑的是 MuJoCo 官方 WASM，模型是 ``web/assets/elf3.xml``（nq=36，没有
head_z / head_y 两个被动关节），并且**没有 onnxruntime**。所以这里不现算策略，
只把每个控制步（50 Hz）的全量 qpos 按关节名对齐后写成 float32 二进制：

    assets/motion_squat_down.bin  / .json    单段下蹲，播完冻结在蹲姿
    assets/motion_squat_up.bin    / .json    单段起立，播完冻结在站姿
    assets/motion_squat_cycle.bin / .json    下蹲 + 保持 + 起立 + 保持，无缝循环

三段都是**纯运动学参考轨迹**（和「参考动作 · 打拳」同类），不是 RL 策略闭环
仿真结果：网页端只做 mj_kinematics + 插值，不跑 mj_step，也不会摔倒。
想看策略实际跑出来的动作，要在服务器把 MuJoCo + ONNX 闭环的 qpos 录下来，
再用本脚本的 --src-dir 指向那份 npz（键名一致即可）。

两个必须做的对账（脚本里都会断言）
--------------------------------
1. **38 -> 36 维重映射**：npz 的 qpos 是
   ``0:7 自由基座 | 7:36 29 关节 | 36:38 head_z/head_y``；网页模型少了最后
   两个关节。按 ``web/assets/elf3.xml`` 的关节名逐个查 npz 的 ``names`` 生成索引，
   顺序将来变了也不会录出「关节乱掉」的回放。
2. **落地自检**：按网页端 ``MjScene.robotMinZ()`` 的**同一套公式**
   （``mesh_vert`` 顶点直接乘 geom_xmat + geom_xpos，**不要再加 mesh_pos**，
   因为 MuJoCo 编译时已经把质心偏移折进 geom_xpos 了）逐帧算视觉网格最低点。
   实测这批轨迹每帧 minZ≈0，即脚底正好贴地，所以默认 z 偏移为 0；
   换模型一旦偏了会直接报错，而不是在网页里看到脚陷进地板。

用法（项目根目录或任意目录都行）：
  python web/tools/bake_squat.py
  python web/tools/bake_squat.py --hold-frames 75       # 循环版蹲姿/站姿各停 1.5 s
  python web/tools/bake_squat.py --src-dir logs/export  # 指定 npz 目录
  python web/tools/bake_squat.py --z-offset 0.02        # 整体抬高 2 cm（一般不用）
  python web/tools/bake_squat.py --no-cycle             # 只出 down / up 两段

生成后在浏览器里打开 ``http://localhost:8080/?motion=squat_cycle`` 即可观看。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # web/tools
WEB = HERE.parent                               # web/
ASSETS = WEB / "assets"
REPO = WEB.parent
DEFAULT_SRC = REPO / "logs" / "export"
WEB_XML = ASSETS / "elf3.xml"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from bake_climb import save                      # noqa: E402  共用 bin/json 写出格式

LABELS = {
    "down": "参考动作 · 下蹲（站姿 → 极限蹲）",
    "up": "参考动作 · 起立（极限蹲 → 站姿）",
    "cycle": "参考动作 · 蹲下起立循环",
}


def repo_rel(p: Path) -> str:
    """尽量给仓库相对路径；--src-dir 在仓库外时退回绝对路径。"""
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


# ---------------------------------------------------------------- 读取与对账
def load_traj(tag: str, src_dir: Path) -> dict:
    p = src_dir / f"squat_{tag}.npz"
    if not p.is_file():
        raise SystemExit(f"[错误] 找不到 {p}\n"
                         f"       先跑：python scripts/export_squat_npz.py")
    with np.load(p, allow_pickle=True) as z:
        need = ("qpos", "q", "z", "pitch", "time", "dt", "names")
        missing = [k for k in need if k not in z.files]
        if missing:
            raise SystemExit(f"[错误] {p.name} 缺键 {missing}（需要 export_squat_npz.py 的格式）")
        d = dict(qpos=np.asarray(z["qpos"], dtype=np.float64).copy(),
                 q=np.asarray(z["q"], dtype=np.float32).copy(),
                 z=np.asarray(z["z"], dtype=np.float32).copy(),
                 pitch=np.asarray(z["pitch"], dtype=np.float32).copy(),
                 dt=float(z["dt"]),
                 names=[str(s) for s in z["names"]])
    q, qpos = d["q"], d["qpos"]
    errs = []
    if qpos.ndim != 2 or qpos.shape[1] != 38:
        errs.append(f"qpos 形状 {qpos.shape}，期望 [T,38]")
    if len(d["names"]) != 29:
        errs.append(f"names 有 {len(d['names'])} 个，期望 29")
    if abs(d["dt"] - 0.02) > 1e-9:
        errs.append(f"dt={d['dt']}，期望 0.02（50 Hz）")
    if qpos.ndim == 2 and qpos.shape[1] == 38:
        if q.shape != (len(qpos), 29):
            errs.append(f"q 形状 {q.shape} 与 qpos 帧数不符")
        elif not np.allclose(q, qpos[:, 7:36], atol=2e-6):
            errs.append("q 与 qpos[:,7:36] 不一致")
        if not np.allclose(d["z"], qpos[:, 2], atol=2e-6):
            errs.append("z 与 qpos[:,2] 不一致")
        jump = float(np.abs(np.diff(qpos[:, 7:36], axis=0)).max())
        if jump >= 0.30:
            errs.append(f"单帧关节跳变 {jump:.3f} rad ≥ 0.30")
    if errs:
        raise SystemExit(f"[错误] {p.name} 自检失败：\n  - " + "\n  - ".join(errs))
    d["path"] = p
    return d


def build_qpos_remap(model, names: list[str]) -> np.ndarray:
    """网页模型 36 个 qpos 各自从 npz 的 38 维 qpos 里取哪一列。"""
    import mujoco

    ix: list[int] = []
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        typ = int(model.jnt_type[j])
        width = {int(mujoco.mjtJoint.mjJNT_FREE): 7,
                 int(mujoco.mjtJoint.mjJNT_BALL): 4,
                 int(mujoco.mjtJoint.mjJNT_HINGE): 1,
                 int(mujoco.mjtJoint.mjJNT_SLIDE): 1}[typ]
        if typ == int(mujoco.mjtJoint.mjJNT_FREE):
            ix.extend(range(7))                       # 自由基座 (x,y,z,qw,qx,qy,qz)
            continue
        if width != 1:
            raise SystemExit(f"[错误] 网页模型关节 {j} 是 {width} 自由度，脚本只支持铰链")
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jname not in names:
            raise SystemExit(f"[错误] 网页模型关节 {jname} 不在 npz 的 names 里")
        ix.append(7 + names.index(jname))
        if adr != len(ix) - 1:
            raise SystemExit(f"[错误] 关节 {jname} 的 qposadr={adr} 与预期 {len(ix)-1} 不符")
    ix = np.asarray(ix, dtype=np.int64)
    if len(ix) != model.nq:
        raise SystemExit(f"[错误] 重映射长度 {len(ix)} != 网页模型 nq={model.nq}")
    return ix


def visual_geoms(model) -> list[int]:
    """与 js/mjscene.js 的 robotGeomIds 完全一致：group 2 的 MESH。"""
    import mujoco
    return [g for g in range(model.ngeom)
            if model.geom_group[g] == 2
            and model.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH)]


def foot_min_z(model, data, qpos36: np.ndarray, geom_ids: list[int]) -> float:
    """按网页 MjScene.robotMinZ() 的公式算视觉网格最低点（Z-up，米）。

    注意：只用 mesh_vert + geom_xmat/xpos，**不要**再加 model.mesh_pos ——
    MuJoCo 编译 STL 时把质心偏移折进了 geom_xpos，重复加会整体低约 7.6 cm。
    """
    data.qpos[:] = qpos36
    import mujoco
    mujoco.mj_kinematics(model, data)
    lo = float("inf")
    for g in geom_ids:
        mid = int(model.geom_dataid[g])
        a = int(model.mesh_vertadr[mid])
        n = int(model.mesh_vertnum[mid])
        v = model.mesh_vert[a:a + n]
        rot = data.geom_xmat[g].reshape(3, 3)
        lo = min(lo, float((v @ rot.T)[:, 2].min() + data.geom_xpos[g][2]))
    return lo


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=str(DEFAULT_SRC),
                    help="squat_down.npz / squat_up.npz 所在目录")
    ap.add_argument("--hold-frames", type=int, default=50,
                    help="cycle 版蹲姿/站姿各保持多少帧（默认 50 = 1 s）")
    ap.add_argument("--z-offset", type=float, default=0.0,
                    help="整段轨迹基座 z 的额外偏移（米，默认 0）")
    ap.add_argument("--no-cycle", action="store_true", help="不生成无缝循环版")
    ap.add_argument("--foot-tol", type=float, default=0.012,
                    help="允许的脚底离地/穿地量（米，默认 1.2 cm）")
    args = ap.parse_args()

    src_dir = Path(args.src_dir).expanduser()
    if not src_dir.is_absolute():
        src_dir = (REPO / src_dir).resolve()
    t0 = time.time()
    print(f"源目录　：{src_dir}")
    print(f"网页模型：{WEB_XML}")

    down = load_traj("down", src_dir)
    up = load_traj("up", src_dir)
    names = down["names"]
    if names != up["names"]:
        raise SystemExit("[错误] down / up 的关节名顺序不一致")

    # 接缝：down 末帧 == up 首帧、up 末帧 == down 首帧 —— 后者保证 cycle 能无缝循环
    e1 = float(np.abs(down["qpos"][-1] - up["qpos"][0]).max())
    e2 = float(np.abs(up["qpos"][-1] - down["qpos"][0]).max())
    print(f"接缝自检：down[-1]↔up[0] max|Δ|={e1:.2e} / "
          f"up[-1]↔down[0] max|Δ|={e2:.2e}")
    if e1 > 1e-9 or e2 > 1e-9:
        raise SystemExit("[错误] 蹲起首尾接缝不闭合，循环回放会跳帧；先修数据源")

    import mujoco  # noqa: E402
    model = mujoco.MjModel.from_xml_path(str(WEB_XML))
    ix = build_qpos_remap(model, names)
    print(f"[信息] npz nq=38 -> 网页模型 nq={model.nq}"
          f"（{'原样截取前 36 列' if np.array_equal(ix, np.arange(model.nq)) else '按关节名重排'}）")

    # 逐帧落地自检（用网页同一公式，不做任何隐式 z 修正）
    data = mujoco.MjData(model)
    geom_ids = visual_geoms(model)
    if not geom_ids:
        raise SystemExit("[错误] 网页模型里找不到 group 2 的视觉 MESH")
    f_down = down["qpos"][:, ix].copy()
    f_up = up["qpos"][:, ix].copy()
    if args.z_offset:
        f_down[:, 2] += args.z_offset
        f_up[:, 2] += args.z_offset
        print(f"[信息] 已应用 --z-offset {args.z_offset:+.4f} m")
    lows = []
    for frames in (f_down, f_up):
        rows = [foot_min_z(model, data, row, geom_ids) for row in frames]
        lows.append(np.asarray(rows))
    all_lows = np.concatenate(lows)
    lo, hi = float(all_lows.min()), float(all_lows.max())
    print(f"落地自检：{len(all_lows)} 帧视觉最低点 {lo*100:+.2f} ~ {hi*100:+.2f} cm"
          f"（容差 ±{args.foot_tol*100:.1f} cm）")
    if lo < -args.foot_tol or hi > args.foot_tol:
        raise SystemExit("[错误] 脚底没有贴在 z=0 上：网页回放会明显穿地/悬空。\n"
                         "       用 --z-offset 手动补偿，或检查轨迹与模型是否配套")

    # ---- 三段回放数据 ----
    hold = max(1, int(args.hold_frames))
    hold_squat = np.repeat(f_down[-1:], hold, axis=0)   # 蹲姿保持
    hold_stand = np.repeat(f_up[-1:], hold, axis=0)     # 站姿保持
    f_cycle = np.concatenate([f_down, hold_squat[1:], f_up[1:], hold_stand[1:]], axis=0)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(1, model.njnt)]
    fps = round(1.0 / down["dt"], 3)
    common = dict(
        fps=fps,
        dt=down["dt"],
        model="elf3.xml",
        scene="平地（选这段回放时自动隐藏两套台阶）",
        policy=None,
        camera="squat",
        jointNames=joint_names,
        kind="squat",
        source=("scripts/export_squat_npz.py 的 29 关节参考轨迹（纯运动学，无策略网络、"
                "不跑 mj_step）+ web/tools/bake_squat.py 烘焙"),
        zOffset=float(args.z_offset),
        refNpz=[repo_rel(down["path"]), repo_rel(up["path"])],
        stats=dict(
            framesDown=int(len(f_down)),
            framesUp=int(len(f_up)),
            holdFrames=hold,
            fps=fps,
            zStand=round(float(down["z"][0]), 4),
            zSquat=round(float(down["z"][-1]), 4),
            pitchSquatDeg=round(float(down["pitch"][-1]), 2),
            footMinZCm=round(lo * 100, 2),
            footMaxZCm=round(hi * 100, 2),
            seamMaxAbs=max(e1, e2),
        ),
    )

    outputs = [
        ("squat_down", f_down, dict(label=LABELS["down"], loop=False,
                                    freezeAtEnd=True, mode="down")),
        ("squat_up", f_up, dict(label=LABELS["up"], loop=False,
                                freezeAtEnd=True, mode="up")),
    ]
    if not args.no_cycle:
        outputs.append(("squat_cycle", f_cycle, dict(label=LABELS["cycle"], loop=True,
                                                     freezeAtEnd=False, mode="cycle")))

    print("\n=== 写出 ===")
    for name, frames, extra in outputs:
        meta = {**common, "name": name, **extra,
                "stats": {**common["stats"], "frames": int(len(frames)),
                          "durationS": round(len(frames) / fps, 3)}}
        save(name, np.asarray(frames, dtype=np.float32), meta)

    # ---- 落盘后回归：bin 长度 / json 字段 ----
    print("\n=== 回归 ===")
    for name, frames, extra in outputs:
        meta = json.loads((ASSETS / f"motion_{name}.json").read_text(encoding="utf-8"))
        nbytes = (ASSETS / meta["bin"]).stat().st_size
        ok = (meta["nq"] == model.nq and meta["nframes"] == len(frames)
              and meta["kind"] == "squat" and meta["loop"] is extra["loop"]
              and nbytes == meta["nq"] * meta["nframes"] * 4)
        if extra["mode"] == "cycle":
            ok = ok and bool(np.array_equal(frames[0], frames[-1]))
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {meta['nframes']} 帧 × "
              f"{meta['nq']} = {nbytes/1024:.1f} KB, loop={meta['loop']}"
              + ("（首尾同帧 ✔）" if extra["mode"] == "cycle" else ""))
        if not ok:
            return 1

    print(f"\n耗时 {time.time()-t0:.1f} s")
    print("浏览器打开：http://localhost:8080/?motion=squat_cycle")
    print("（或 ?motion=squat_down / squat_up 单段播放，播完冻结在末帧）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
