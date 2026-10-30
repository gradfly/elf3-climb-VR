"""bake_crab.py —— 把「螃蟹步横移上楼」烘焙成网页可回放的资源

和 bake_climb.py / bake_motion.py 是同一套三件事，区别在于这次的轨迹**不是**策略
网络跑出来的，而是 `crab_climb.py` 的四相位准静态脚本控制器解出来的关节轨迹：

  1) 生成地形      gen_crab_stairs.py   -> crab_stairs_obstacles.xml + 真值表
  2) 规划轨迹      crab_climb.py        -> crab_climb_traj.npz（每 20 ms 一帧 qpos）
  3) 烘焙到网页    本脚本               -> assets/motion_crab.{bin,json}
                                          assets/scene_crab.xml（台阶几何）

为什么台阶也要搬过来
--------------------
网页里 MuJoCo 是 WASM，模型在启动时一次编译完成，不能中途换地形。原来的
`scene_bx.xml` 只有沿 +X 排布的 universal_test 紧凑障碍；螃蟹步的楼梯沿 +Y 上升，
是另一套几何。所以这里把它抽成 `scene_crab.xml`，由 `scene_bx.xml` 的
`<worldbody>` 末尾 include 进来，网页端按 geom 名前缀 `crab_` 单独分组显隐
（见 js/mjscene.js 的 crabGeomIds / showCrab）。

关节顺序对账（同 bake_climb）
----------------------------
仿真模型 `elf3_crab.xml` 的 nq=38（自由基座 7 + 29 策略关节 + head_z/head_y），
网页模型 `web/assets/elf3.xml` 的 nq=36（没有两个头部关节）。按关节名逐个对齐
后再写出，顺序变了也不会录出"关节乱掉"的回放。

用法（项目根目录或任意目录都行）：
  python web/tools/bake_crab.py                    # 全链：生成地形 + 规划轨迹 + 烘焙
  python web/tools/bake_crab.py --seed 7 --rise-max 0.18    # 换一套地形重录
  python web/tools/bake_crab.py --skip-gen --skip-plan      # 只重烘焙已有轨迹
  python web/tools/bake_crab.py --traj some.npz             # 指定轨迹文件
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # web/tools
WEB = HERE.parent                               # web/
ASSETS = WEB / "assets"

# 工作区 = 包含 vendor/ 的目录（本仓库里是 <repo>/sim）
WS = HERE.parent.parent / "sim"
FW = WS / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3"
LAUNCH = FW / "launch"
DATA = FW / "data" / "mujoco_simulation"
TERRAIN = DATA / "terrains" / "universal_test"

OBSTACLES_XML = TERRAIN / "crab_stairs_obstacles.xml"
TERRAIN_JSON = TERRAIN / "crab_stairs.json"
TRAJ_NPZ = DATA / "crab_climb_traj.npz"
SCENE_XML = ASSETS / "scene_crab.xml"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from bake_climb import build_qpos_remap, save            # noqa: E402


# ---------------------------------------------------------------- 子流程
def run_script(name: str, args: list[str]) -> None:
    cmd = [sys.executable, str(LAUNCH / name), *[str(a) for a in args]]
    print("$ python " + " ".join([name, *[str(a) for a in args]]), flush=True)
    r = subprocess.run(cmd, cwd=str(LAUNCH))
    if r.returncode != 0:
        raise SystemExit(f"[错误] {name} 失败（exit {r.returncode}）")


def gen_terrain(a) -> None:
    args = ["--seed", a.seed, "--n-up", a.n_up, "--n-down", a.n_down,
            "--rise-min", a.rise_min, "--rise-max", a.rise_max,
            "--tread-min", a.tread_min, "--tread-max", a.tread_max,
            "--write-entry", "--write-scene"]
    if a.landing_depth is not None:
        args += ["--landing-depth", a.landing_depth]
    if a.no_landing:
        args += ["--no-landing"]
    run_script("gen_crab_stairs.py", args)


def plan_traj(a) -> None:
    run_script("crab_climb.py", ["--out", a.traj, "--step-time", a.step_time])


# ---------------------------------------------------------------- 网页台阶几何
GEOM_RE = re.compile(r'^\s*<geom\s+name="(crab_[^"]+)"', re.I)


def write_scene_crab(seed: int) -> int:
    """从生成的台阶 XML 里抽出 crab_* 的 geom，写成网页端可 include 的片段。

    只搬 geom、不搬 site：js/mjscene.js 只渲染 geom，site 搬过去也看不见，
    徒增模型里的元素。scene_bx.xml 的 <worldbody> 末尾 include 本文件。
    """
    if not OBSTACLES_XML.is_file():
        raise SystemExit(f"[错误] 找不到台阶 XML：{OBSTACLES_XML}\n"
                         f"       先跑 gen_crab_stairs.py（或去掉 --skip-gen）")
    lines, n = [], 0
    for ln in OBSTACLES_XML.read_text(encoding="utf-8").splitlines():
        if GEOM_RE.match(ln):
            lines.append("  " + ln.strip())
            n += 1
    if not n:
        raise SystemExit(f"[错误] {OBSTACLES_XML} 里没有 crab_* 的 geom")

    text = (
        "<mujocoinclude>\n"
        "  <!-- 自动生成，勿手工编辑。\n"
        "       来源：sim/.../terrains/universal_test/crab_stairs_obstacles.xml\n"
        f"       由 web/tools/bake_crab.py 生成（seed={seed}）。\n"
        "       本片段被 scene_bx.xml 的 <worldbody> 末尾 include，材质 ut_concrete\n"
        "       已在那里定义；网页端按名字前缀 crab_ 单独分组显隐。 -->\n"
        + "\n".join(lines) + "\n"
        "</mujocoinclude>\n"
    )
    SCENE_XML.write_text(text, encoding="utf-8")
    return n


def refresh_manifest() -> None:
    """重写 assets/manifest.json（与 tools/make-manifest.mjs 的逻辑一致）。

    不依赖 node：网页启动时必须把所有 include 到的 XML 都灌进 MuJoCo 的虚拟
    文件系统，漏了 scene_crab.xml 就会在编译模型时报 include not found。
    """
    paths = ["elf3.xml", "scene_bx.xml"]
    if SCENE_XML.is_file():
        paths.append(SCENE_XML.name)
    meshes = sorted((ASSETS / "meshes").glob("*.stl"),
                    key=lambda p: p.name.lower())
    paths += [f"meshes/{p.name}" for p in meshes]

    files = [{"path": p, "size": (ASSETS / p).stat().st_size} for p in paths]
    manifest = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "entry": "elf3.xml",
        "files": files,
        "totalBytes": sum(f["size"] for f in files),
    }
    (ASSETS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[输出] manifest.json: {len(files)} 个文件 / "
          f"{manifest['totalBytes'] / 1048576:.1f} MB")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="crab", help="输出名前缀（默认 crab）")
    ap.add_argument("--traj", default=str(TRAJ_NPZ), help="轨迹 npz 路径")
    ap.add_argument("--policy-name", default=None,
                    help="策略名（如 elf3_crab_residual.onnx）；非空时网页按「策略仿真」介绍")
    ap.add_argument("--label", default="螃蟹步横移上楼",
                    help="网页里显示的名称（策略回放可填「策略仿真 · 螃蟹步横移上楼」）")
    ap.add_argument("--skip-gen", action="store_true", help="不重生成地形")
    ap.add_argument("--skip-plan", action="store_true", help="不重跑轨迹规划")
    # 地形参数（透传给 gen_crab_stairs.py）
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-up", type=int, default=6)
    ap.add_argument("--n-down", type=int, default=3)
    ap.add_argument("--rise-min", type=float, default=0.12)
    ap.add_argument("--rise-max", type=float, default=0.25)
    ap.add_argument("--tread-min", type=float, default=0.20)
    ap.add_argument("--tread-max", type=float, default=0.40)
    ap.add_argument("--landing-depth", type=float, default=None)
    ap.add_argument("--no-landing", action="store_true")
    # 步态参数（透传给 crab_climb.py）
    ap.add_argument("--step-time", type=float, default=0.9, help="每步时长 s")
    args = ap.parse_args()

    t0 = time.time()
    args.traj = str(Path(args.traj).resolve())

    if not args.skip_gen:
        gen_terrain(args)
    if not args.skip_plan:
        plan_traj(args)

    traj = Path(args.traj)
    if not traj.is_file():
        raise SystemExit(f"[错误] 找不到轨迹 {traj}（去掉 --skip-plan 重跑）")

    # 台阶几何必须先落地：scene_bx.xml 已经 include 了它，网页模型一加载就要用。
    n_geoms = write_scene_crab(args.seed)
    print(f"[输出] {SCENE_XML.name}: {n_geoms} 个台阶 geom")

    import mujoco                                    # noqa: E402

    sim_xml = DATA / "elf3_crab.xml"
    if not sim_xml.is_file():
        raise SystemExit(f"[错误] 找不到仿真场景 {sim_xml}（去掉 --skip-gen 重跑）")
    sim_model = mujoco.MjModel.from_xml_path(str(sim_xml))
    web_model = mujoco.MjModel.from_xml_path(str(ASSETS / "elf3.xml"))
    ix = build_qpos_remap(sim_model, web_model)
    assert ix.shape[0] == web_model.nq, (ix.shape[0], web_model.nq)
    print(f"[信息] 仿真模型 nq={sim_model.nq} -> 网页模型 nq={web_model.nq}"
          f"（{'原样截取' if np.array_equal(ix, np.arange(web_model.nq)) else '按关节名重排'}）")

    z = np.load(traj)
    qpos = np.asarray(z["qpos"], dtype=np.float64)
    dt = float(z["dt"]) if "dt" in z.files else 0.02
    if qpos.ndim != 2 or qpos.shape[1] != sim_model.nq:
        raise SystemExit(f"[错误] 轨迹 qpos 形状 {qpos.shape} 与仿真模型 nq="
                         f"{sim_model.nq} 不符")
    frames = qpos[:, ix]

    # ---- 统计（HUD 与说明文字用）----
    meta_t = {}
    if TERRAIN_JSON.is_file():
        meta_t = json.loads(TERRAIN_JSON.read_text(encoding="utf-8"))
    treads = meta_t.get("treads", [])
    ups = [t for t in treads if t["section"] == "up"]
    rises = [abs(t["rise"]) for t in ups] or [0.0]
    treads_d = [t["depth"] for t in treads] or [0.0]
    root_y = qpos[:, 1]
    root_z = qpos[:, 2]

    stats = {
        "terrainSeed": int(args.seed),
        "steps": len(treads),
        "riseMin": round(float(min(rises)), 3),
        "riseMax": round(float(max(rises)), 3),
        "treadMin": round(float(min(treads_d)), 3),
        "treadMax": round(float(max(treads_d)), 3),
        "maxTop": round(float(meta_t.get("max_top", 0.0)), 3),
        "travelY": round(float(root_y[-1] - root_y[0]), 3),
        "rootYMin": round(float(root_y.min()), 3),
        "rootYMax": round(float(root_y.max()), 3),
        "rootZMin": round(float(root_z.min()), 3),
        "rootZMax": round(float(root_z.max()), 3),
    }

    save(args.out, np.asarray(frames, dtype=np.float32), {
        "fps": round(1.0 / dt, 3),
        "dt": dt,
        "model": "elf3.xml",
        "scene": f"crab_stairs（seed={args.seed}，楼梯沿 +Y 上升）",
        "policy": args.policy_name,
        "camera": None,
        "jointNames": [mujoco.mj_id2name(web_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                       for j in range(web_model.njnt)][1:],
        "name": args.out,
        "kind": "crab",
        "label": args.label,
        "source": "crab_climb.py 四相位准静态脚本控制器（闭式 IK + DLS 精修，"
                  "无策略网络）+ gen_crab_stairs.py 随机台阶",
        "terrain": {"entry": SCENE_XML.name, "seed": int(args.seed)},
        "stats": stats,
    })
    refresh_manifest()

    print("\n" + "=" * 62)
    print(f"台阶 {stats['steps']} 级 | 级高 {stats['riseMin']}~{stats['riseMax']} m"
          f" | 进深 {stats['treadMin']}~{stats['treadMax']} m | 最高 {stats['maxTop']} m")
    print(f"横移 {stats['travelY']} m（y {stats['rootYMin']} -> {stats['rootYMax']}）"
          f" | 躯干 z {stats['rootZMin']}~{stats['rootZMax']} m")
    print(f"帧数 {len(frames)} @ {1 / dt:.0f} fps = {len(frames) / (1 / dt):.1f} s")
    print(f"耗时 {time.time() - t0:.0f} s")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
