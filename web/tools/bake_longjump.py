"""bake_longjump.py —— 把「高跳」参考轨迹烘焙成网页 WASM 可回放的资源

输入是 long_jump_real.py 产出的自驱轨迹（纯 PD + 摆臂 + 蹬腿，零冲量注入）：
    sim/.../data/mujoco_simulation/long_jump_real_hi.npz
    keys: time (T,)  qpos (T,38)  phase (T,)  dt  joint_names (29,)  meta

网页端跑官方 MuJoCo WASM，模型是 web/assets/elf3.xml（nq=36，没有
head_z / head_y 两个被动关节），也没有 onnxruntime —— 所以不现算策略，
只把每个控制步（50 Hz）的全量 qpos 按关节名对齐后写成 float32 二进制：

    assets/motion_longjump_hi.bin  / .json

两段必须做的对账（脚本里都会断言）
--------------------------------
1. **38 -> 36 维重映射**：npz 的 qpos 是
   ``0:7 自由基座 | 7:36 29 关节 | 36:38 head_z/head_y``；网页模型少了最后
   两个关节。按 web/assets/elf3.xml 的关节名逐个查 npz 的 names 生成索引，
   顺序将来变了也不会录出「关节乱掉」的回放。
2. **落地 / 腾空自检**：和 squat 不同，高跳是**真的会离地**的动作，所以
   不要求每帧脚底贴 z=0。只检查：
     * 没有 NaN / inf；
     * 任何时刻脚都不穿地板（foot_min_z >= -tol）；
     * 首 / 末帧脚底贴地（站姿，foot_min_z ∈ ±tol），保证回放首尾是站着的；
     * 报告腾空段最小脚高、质心峰值高度、腾空帧数等统计量。

用法（项目根目录或任意目录都行）：
  python web/tools/bake_longjump.py
  python web/tools/bake_longjump.py --src /path/to/long_jump_real_hi.npz
  python web/tools/bake_longjump.py --name longjump_hi --foot-tol 0.03
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
DEFAULT_SRC = (REPO / "sim" / "vendor" / "bxi_controller_ros2" /
               "src" / "bxi_example_py_elf3" / "data" / "mujoco_simulation" /
               "long_jump_real_hi.npz")
WEB_XML = ASSETS / "elf3.xml"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from bake_squat import build_qpos_remap, visual_geoms, foot_min_z, save  # noqa: E402


def repo_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(p.resolve()).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="源 npz 轨迹")
    ap.add_argument("--name", default="longjump_hi", help="输出名前缀")
    ap.add_argument("--foot-tol", type=float, default=0.03,
                    help="允许的脚底穿地量（米，默认 3 cm）")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    if not src.is_file():
        raise SystemExit(f"[错误] 找不到源轨迹 {src}\n"
                         f"       先跑：python sim/.../launch/long_jump_real.py "
                         f"--arm-scale 1.8 --kp 16 --lean 8 --anchor -0.10")
    with np.load(src, allow_pickle=True) as z:
        qpos = np.asarray(z["qpos"], dtype=np.float64).copy()
        dt = float(z["dt"])
        names = [str(s) for s in z["joint_names"]]
        meta_src = z.get("meta", None)
        meta_src = dict(meta_src.item()) if meta_src is not None else {}

    t0 = time.time()
    errs = []
    if qpos.ndim != 2 or qpos.shape[1] != 38:
        errs.append(f"qpos 形状 {qpos.shape}，期望 [T,38]")
    if len(names) != 29:
        errs.append(f"joint_names 有 {len(names)} 个，期望 29")
    if abs(dt - 0.02) > 1e-9:
        errs.append(f"dt={dt}，期望 0.02（50 Hz）")
    if not np.all(np.isfinite(qpos)):
        errs.append("qpos 含 NaN / inf")
    if errs:
        raise SystemExit("[错误] 源轨迹自检失败：\n  - " + "\n  - ".join(errs))

    import mujoco  # noqa: E402
    model = mujoco.MjModel.from_xml_path(str(WEB_XML))
    ix = build_qpos_remap(model, names)
    print(f"[信息] npz nq=38 -> 网页模型 nq={model.nq}"
          f"（{'原样截取前 36 列' if np.array_equal(ix, np.arange(model.nq)) else '按关节名重排'}）")

    frames = qpos[:, ix].astype(np.float64)

    # ---------------------------------------------------------- 腾空 / 落地自检
    data = mujoco.MjData(model)
    geom_ids = visual_geoms(model)
    if not geom_ids:
        raise SystemExit("[错误] 网页模型里找不到 group 2 的视觉 MESH")
    lows = np.asarray([foot_min_z(model, data, row, geom_ids) for row in frames])
    lo, hi = float(lows.min()), float(lows.max())
    if lo < -args.foot_tol:
        raise SystemExit(f"[错误] 第 {int(lows.argmin())} 帧脚穿透地板 "
                         f"{lo*100:.1f} cm（容差 ±{args.foot_tol*100:.1f} cm）；"
                         "轨迹与网页模型不配套")
    if abs(lows[0]) > args.foot_tol or abs(lows[-1]) > args.foot_tol:
        raise SystemExit(f"[错误] 首 / 末帧脚底离地 {lows[0]*100:+.1f} / "
                         f"{lows[-1]*100:+.1f} cm，回放首尾不是站姿")
    flight = int(np.sum(lows > 0.05))
    print(f"腾空自检：{len(lows)} 帧视觉最低点 {lo*100:+.2f} ~ {hi*100:+.2f} cm"
          f"（容差 ±{args.foot_tol*100:.1f} cm，{flight} 帧离地 >5 cm）")

    z = frames[:, 2]
    x = frames[:, 0]
    peak_com_z = float(z.max())
    start_com_z = float(z[0])
    jump_rel = peak_com_z - start_com_z
    travel_x = float(x[-1] - x[0])
    fps = round(1.0 / dt, 3)
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                   for j in range(1, model.njnt)]

    meta = dict(
        fps=fps,
        dt=dt,
        model="elf3.xml",
        scene="平地（选这段回放时自动隐藏障碍）",
        policy=None,
        camera="jump",
        kind="jump",
        name=args.name,
        label="参考动作 · 高跳（自驱摆臂 + 蹬腿）",
        loop=False,
        freezeAtEnd=True,
        source=("long_jump_real.py 纯自驱闭环仿真（PD + 摆臂 + 蹬腿，"
                "零冲量注入）录制的 29 关节 qpos，经 web/tools/bake_longjump.py "
                "按关节名重排成网页 36 维 + 逐帧落地自检"),
        zOffset=0.0,
        refNpz=[repo_rel(src)],
        stats=dict(
            nframes=int(len(frames)),
            fps=fps,
            durationS=round(len(frames) / fps, 3),
            startComZ=round(start_com_z, 4),
            peakComZ=round(peak_com_z, 4),
            jumpHeightRelCm=round(jump_rel * 100, 1),
            travelXm=round(travel_x, 3),
            footMinZCm=round(lo * 100, 2),
            footMaxZCm=round(hi * 100, 2),
            flightFrames=flight,
            flightS=round(flight / fps, 3),
        ),
    )
    save(args.name, np.asarray(frames, dtype=np.float32), meta)

    # ---------------------------------------------------------- 落盘后回归
    j = json.loads((ASSETS / f"motion_{args.name}.json").read_text(encoding="utf-8"))
    nbytes = (ASSETS / j["bin"]).stat().st_size
    need = j["nq"] * j["nframes"] * 4
    ok = (j["nq"] == model.nq and j["nframes"] == len(frames)
          and j["kind"] == "jump" and j["loop"] is False
          and nbytes == need)
    print(f"\n[回归] {'OK' if ok else 'FAIL'} {args.name}: "
          f"{j['nframes']} 帧 × {j['nq']} = {nbytes/1024:.1f} KB, "
          f"loop={j['loop']}, 相对站立跳高 {jump_rel*100:.1f} cm, "
          f"前进 {travel_x*100:.0f} cm, 腾空 {flight} 帧 ({flight/fps:.2f} s)")
    if not ok:
        return 1
    print(f"\n耗时 {time.time()-t0:.1f} s")
    print("浏览器打开：http://localhost:8080/?motion=longjump_hi")
    print("（播完冻结在末帧站姿，想重看点「从头播放」）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
