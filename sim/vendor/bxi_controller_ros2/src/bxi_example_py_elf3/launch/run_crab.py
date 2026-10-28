"""run_crab.py —— 螃蟹步横移爬楼梯【一键启动脚本】

一条命令跑完整条链路：

    ┌─────────┐   ┌──────────┐   ┌──────────┐
    │ gen  地形 │ → │ plan 轨迹 │ → │ replay 回放│
    └─────────┘   └──────────┘   └──────────┘
    gen_crab_stairs   crab_climb      crab_replay
    (随机台阶 XML/JSON) (四相位 IK)    (MuJoCo 窗口)

==================== 三种改参数的方式（优先级从低到高）====================
  1. 改仓库根目录的 crab_config.json          —— 最常用，改一次长期生效
  2. 命令行 --preset easy                     —— 切换预设难度
  3. 命令行 --set terrain.seed=7              —— 临时改某一项
  4. 命令行快捷参数 --seed 7 --step-time 1.2   —— 同上，更好记

  优先级：内置默认 < crab_config.json < --preset < --set < 快捷参数

==================== 常用命令 ====================
  python run_crab.py                          # 一键跑完并弹出回放窗口
  python run_crab.py --preset easy            # 简单地形
  python run_crab.py --seed 7                 # 换个随机地形
  python run_crab.py --stages gen,plan        # 只生成地形+规划，不弹窗
  python run_crab.py --show-config            # 打印当前会用到的全部参数
  python run_crab.py --mode phys              # 回放改用物理模式（会摔）
  python run_crab.py --record out/crab        # 离屏渲染 PNG 序列

  参数全表见 --help，或看 crab_config.json 里的字段。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                       # launch/
DATA = HERE.parent / "data" / "mujoco_simulation"
TERRAIN_DIR = DATA / "terrains" / "universal_test"

# launch -> [0]=launch [1]=bxi_example_py_elf3 [2]=src [3]=bxi_controller_ros2
#          [4]=vendor [5]=sim [6]=仓库根
REPO = HERE.parents[6]
DEFAULT_CONFIG = REPO / "crab_config.json"

# ---------------------------------------------------------------- 默认值
DEFAULTS = {
    "terrain": {
        "seed": 42,
        "n_up": 6,                 # 上级数
        "n_down": 3,               # 下级数（0 = 不要下楼段）
        "landing": True,           # 上/下楼之间要不要中间平台
        "landing_depth": 0.30,     # 平台进深 m（太深会把站姿撑到超 ankle_x 限位）
        "rise_min": 0.12,          # 级高下限 m（低于 12cm 深度图分辨率不够）
        "rise_max": 0.25,          # 级高上限 m（25cm 需要 10.4° 骨盆侧倾）
        "tread_min": 0.20,         # 踏面进深下限 m
        "tread_max": 0.40,         # 踏面进深上限 m
        "width": 1.2,              # 楼梯净宽 m（沿 X）
        "out_xml": "crab_stairs_obstacles.xml",
        "out_json": "crab_stairs.json",
    },
    "gait": {
        "step_time": 0.9,          # 每步时长 s（越大越慢越稳）
        "dt": 0.02,                # 控制周期 s（50 Hz）
        "clearance": 0.055,        # 摆腿越过踏面时的离地余量 m
        "hold": 0.5,               # 起步站稳时间 s
        "scene": "elf3_crab.xml",
        "out": "crab_climb_traj.npz",
    },
    "replay": {
        "enabled": True,           # 跑完要不要弹回放窗口
        "mode": "kin",             # kin=运动学（必成功） / phys=物理（会摔）
        "speed": 1.0,              # 播放倍速
        "loop": False,
        "track": True,             # 相机跟随躯干
        "dist": 2.8,               # 相机距离 m
        "az": 160.0,               # 相机方位角 deg（160 = 斜前方，看得到台阶剖面）
        "el": -14.0,               # 相机俯仰角 deg
        "record": None,            # 给个目录就离屏渲染存 PNG（不弹窗）
        "every": 1,                # 录屏每 N 帧存 1 帧
    },
    "run": {
        "stages": ["gen", "plan", "replay"],
        "check_only": False,       # True = 只做体检不写 npz
        "dry_run": False,          # True = 只打印要执行的命令
    },
}

# ---------------------------------------------------------------- 预设难度
# 只覆盖 terrain 里的字段
PRESETS = {
    "easy":   {"rise_min": 0.12, "rise_max": 0.16,
               "tread_min": 0.30, "tread_max": 0.40,
               "landing_depth": 0.34},
    "normal": {"rise_min": 0.12, "rise_max": 0.25,
               "tread_min": 0.20, "tread_max": 0.40,
               "landing_depth": 0.30},
    "hard":   {"rise_min": 0.20, "rise_max": 0.25,
               "tread_min": 0.20, "tread_max": 0.28,
               "landing_depth": 0.26},
    # 计划里的 L4 降级档：级高压到 15cm、进深提到 28cm 以上
    "l4":     {"rise_min": 0.12, "rise_max": 0.15,
               "tread_min": 0.28, "tread_max": 0.32,
               "landing_depth": 0.30},
}

HINT = {
    "seed": "随机种子，换一个数就是换一套台阶",
    "n_up": "上级数",
    "n_down": "下级数，0 = 只有上楼",
    "landing": "上下楼之间是否插一个平台",
    "landing_depth": "平台进深 m，>0.40 会把 SPLIT 站姿撑到超 ankle_x 限位",
    "rise_min": "级高下限 m，<0.12 深度图分辨率不够看台阶",
    "rise_max": "级高上限 m，0.25 需要 10.4deg 骨盆侧倾（限位 15deg）",
    "tread_min": "踏面进深下限 m，脚掌 Y 向只有 8cm 宽，<0.20 很勉强",
    "tread_max": "踏面进深上限 m",
    "width": "楼梯净宽 m（沿 X）",
    "step_time": "每步时长 s，调大 = 走慢一点 = 更容易站住",
    "dt": "控制周期 s，50Hz=0.02",
    "clearance": "摆腿离地余量 m，台阶边缘有毛刺就调大",
    "hold": "起步站稳时间 s",
    "speed": "回放倍速，0.3 慢放便于看细节",
    "dist": "相机距离 m",
    "az": "相机方位角 deg，台阶沿 Y 上升，所以要站在 ±X 侧才看得见剖面",
    "el": "相机俯仰角 deg",
}


# ---------------------------------------------------------------- 配置合并
def deep_copy(d):
    if isinstance(d, dict):
        return {k: deep_copy(v) for k, v in d.items()}
    if isinstance(d, list):
        return list(d)
    return d


def deep_merge(base, over):
    out = deep_copy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deep_copy(v)
    return out


def get(cfg, dotted):
    cur = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def setv(cfg, dotted, val):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = val


def coerce(s: str):
    """把命令行字符串转成 int/float/bool/None/str。"""
    low = s.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_config(path: Path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"[错误] 配置文件 {path} 解析失败: {e}")


def print_config(cfg, title="生效参数"):
    print(f"=== {title} ===")
    for grp in ("terrain", "gait", "replay", "run"):
        if grp not in cfg:
            continue
        print(f"  [{grp}]")
        for k, v in cfg[grp].items():
            hint = HINT.get(k, "")
            print(f"    {k:<16} = {str(v):<22} {hint}")
    print()


# ---------------------------------------------------------------- 流水线
def sh(cmd: list[str], dry: bool):
    pretty = " ".join(str(c) for c in cmd[1:])
    # 必须 flush：否则父进程的输出会攒到退出时才吐，孩子进程的日志反而排到前面
    print(f"$ python {pretty}", flush=True)
    if dry:
        return 0
    r = subprocess.run(cmd, cwd=str(HERE))
    return r.returncode


def stage_gen(cfg, dry):
    t = cfg["terrain"]
    cmd = [sys.executable, "gen_crab_stairs.py",
           "--seed", str(t["seed"]),
           "--n-up", str(t["n_up"]),
           "--n-down", str(t["n_down"]),
           "--rise-min", str(t["rise_min"]),
           "--rise-max", str(t["rise_max"]),
           "--tread-min", str(t["tread_min"]),
           "--tread-max", str(t["tread_max"]),
           "--landing-depth", str(t["landing_depth"]),
           "--width", str(t["width"]),
           "--out-xml", str(t["out_xml"]),
           "--out-json", str(t["out_json"]),
           "--write-entry", "--write-scene"]
    if not t["landing"]:
        cmd.append("--no-landing")
    return sh(cmd, dry)


def stage_plan(cfg, dry):
    g = cfg["gait"]
    t = cfg["terrain"]
    cmd = [sys.executable, "crab_climb.py",
           "--scene", str(DATA / g["scene"]),
           "--terrain-json", str(TERRAIN_DIR / t["out_json"]),
           "--out", str(DATA / g["out"]),
           "--step-time", str(g["step_time"]),
           "--dt", str(g["dt"]),
           "--clearance", str(g["clearance"]),
           "--hold", str(g["hold"])]
    if cfg["run"]["check_only"]:
        cmd.append("--check-only")
    return sh(cmd, dry)


def stage_replay(cfg, dry):
    r = cfg["replay"]
    g = cfg["gait"]
    if not r["enabled"]:
        print("$ (replay 已关闭：replay.enabled=false 或 --no-replay)")
        return 0
    cmd = [sys.executable, "crab_replay.py",
           "--scene", str(DATA / g["scene"]),
           "--traj", str(DATA / g["out"]),
           "--mode", str(r["mode"]),
           "--speed", str(r["speed"]),
           "--dist", str(r["dist"]),
           "--az", str(r["az"]), "--el", str(r["el"])]
    if r["loop"]:
        cmd.append("--loop")
    if not r["track"]:
        cmd.append("--no-track")
    if r["record"]:
        cmd += ["--record", str(r["record"]), "--every", str(r["every"])]
    return sh(cmd, dry)


STAGES = {"gen": stage_gen, "plan": stage_plan, "replay": stage_replay}


# ---------------------------------------------------------------- 入口
def main() -> None:
    # 行缓冲：保证流水线日志和子进程日志按真实顺序出现
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description="螃蟹步横移爬楼梯 —— 一键启动（地形 → 轨迹 → 回放）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="改参数：编辑仓库根目录 crab_config.json，或用 --set 组.键=值\n"
               "例：  python run_crab.py --set terrain.seed=7 --set gait.step_time=1.2")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS),
                    help="预设难度（只改地形）：easy/normal/hard/l4")
    ap.add_argument("--set", action="append", default=[], metavar="组.键=值",
                    help="覆盖任意配置项，可重复，如 --set terrain.seed=7")
    ap.add_argument("--stages", default=None,
                    help="要跑的阶段，逗号分隔，如 gen,plan（默认 gen,plan,replay）")
    ap.add_argument("--show-config", action="store_true",
                    help="只打印生效参数，不执行")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--check-only", action="store_true", help="plan 阶段只体检不写 npz")
    ap.add_argument("--no-replay", action="store_true", help="跑完不弹回放窗口")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="离屏渲染存 PNG 序列到该目录")
    ap.add_argument("--mode", default=None, choices=["kin", "phys"],
                    help="回放模式：kin=运动学 / phys=物理（会摔）")
    # ---- 常用快捷参数（等价于 --set）----
    ap.add_argument("--seed", type=int, default=None, help="= --set terrain.seed")
    ap.add_argument("--n-up", type=int, default=None, help="= --set terrain.n_up")
    ap.add_argument("--n-down", type=int, default=None, help="= --set terrain.n_down")
    ap.add_argument("--rise-min", type=float, default=None, help="= --set terrain.rise_min")
    ap.add_argument("--rise-max", type=float, default=None, help="= --set terrain.rise_max")
    ap.add_argument("--tread-min", type=float, default=None, help="= --set terrain.tread_min")
    ap.add_argument("--tread-max", type=float, default=None, help="= --set terrain.tread_max")
    ap.add_argument("--step-time", type=float, default=None, help="= --set gait.step_time")
    ap.add_argument("--speed", type=float, default=None, help="= --set replay.speed")
    args = ap.parse_args()

    cfg = deep_merge(DEFAULTS, load_config(Path(args.config)))
    if args.preset:
        cfg = deep_merge(cfg, {"terrain": PRESETS[args.preset]})
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"[错误] --set 要写成 组.键=值，收到 {item!r}")
        k, v = item.split("=", 1)
        setv(cfg, k.strip(), coerce(v))

    shortcut = {
        "terrain.seed": args.seed, "terrain.n_up": args.n_up,
        "terrain.n_down": args.n_down, "terrain.rise_min": args.rise_min,
        "terrain.rise_max": args.rise_max, "terrain.tread_min": args.tread_min,
        "terrain.tread_max": args.tread_max, "gait.step_time": args.step_time,
        "replay.speed": args.speed, "replay.mode": args.mode,
        "replay.record": args.record,
    }
    for k, v in shortcut.items():
        if v is not None:
            setv(cfg, k, v)
    if args.check_only:
        cfg["run"]["check_only"] = True
    if args.no_replay:
        cfg["replay"]["enabled"] = False
    if args.dry_run:
        cfg["run"]["dry_run"] = True
    if args.stages:
        cfg["run"]["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]

    cfg["run"]["stages"] = [s for s in cfg["run"]["stages"] if s in STAGES]
    if not cfg["run"]["stages"]:
        raise SystemExit(f"[错误] 阶段名只能是 {sorted(STAGES)} 里的组合")

    print()
    print("螃蟹步横移爬楼梯 —— 一键流水线")
    print(f"  配置文件 {Path(args.config).name}"
          + (f"   预设 {args.preset}" if args.preset else ""))
    print(f"  阶段     {' -> '.join(cfg['run']['stages'])}")
    print()
    print_config(cfg)

    if args.show_config:
        return

    dry = cfg["run"]["dry_run"]
    for s in cfg["run"]["stages"]:
        print(f"----- [{s}] -----")
        rc = STAGES[s](cfg, dry)
        if rc != 0:
            raise SystemExit(f"\n[失败] 阶段 {s} 退出码 {rc}")
        print()

    print("全部阶段完成。")
    if "replay" not in cfg["run"]["stages"]:
        print("想看回放： python run_crab.py --stages replay")
    elif not cfg["replay"]["enabled"]:
        print("想看回放： python run_crab.py --stages replay")


if __name__ == "__main__":
    main()
