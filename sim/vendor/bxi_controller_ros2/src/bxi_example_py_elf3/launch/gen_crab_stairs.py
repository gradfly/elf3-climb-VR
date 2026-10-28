"""螃蟹步横移楼梯地形生成器。

坐标系约定（与 elf3.xml 一致）：
    机器人朝 +X 站立，左腿在 +Y 侧。
    楼梯沿 +Y 逐级升高 —— 即机器人的【侧向】是上楼方向。
    每级是一个从 z=0 到 z=top 的实心 box，相邻级在 y 上首尾相接。

几何（MuJoCo 的 size 是半尺寸）：
    第 i 级占据 y in [y0, y1]、x in [-w/2, w/2]、z in [0, top]：
        pos  = (0, (y0+y1)/2, top/2)
        size = (w/2, (y1-y0)/2, top/2)

不做前后错位，改用骨盆侧倾 waist_x（重要结论）：
    脚掌 24 cm(前后 X) x 8 cm(左右 Y)，髋间距 27.2 cm。
    - 窄踏面(20 cm)不需要错位：双脚按踏面中心 ±off 放置即可不重叠（见 foot_placement）。
    - 级高才是真瓶颈：骨盆水平时，平脚 + 踝限位 [-50,+45]deg 把级高压到 22.7 cm。
      解法是 waist_x 侧倾（限位 ±15deg，在 torso_link 下游所以不影响相机）：
      侧倾让上坡侧髋抬高、下坡侧髋降低，两条腿的 rho 约束同时放松。
      25 cm 级高只需 10.4deg；同样效果若靠前后错位则需要 31 cm/侧（双脚叉开 62 cm）。
    真值表里每级都带 tilt(rad) 与 off(m)，控制器直接取用。

用法:
    python gen_crab_stairs.py --seed 42 --n-up 6 --n-down 3 --landing
    python gen_crab_stairs.py --seed 7 --rise-max 0.15      # 降级难度 L4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                       # launch/
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import crab_ik as K                                          # noqa: E402

DATA = HERE.parent / "data" / "mujoco_simulation"            # data/mujoco_simulation/
TERRAIN_DIR = DATA / "terrains" / "universal_test"

DEFAULT_XML = TERRAIN_DIR / "crab_stairs_obstacles.xml"
DEFAULT_JSON = TERRAIN_DIR / "crab_stairs.json"

# 出生区：机器人站在 y=0 的平地上，楼梯从 y=START_Y 开始
START_Y = 1.5
# 出生高度（ISAAC 默认位形下脚底贴地实测值）
SPAWN_Z = 1.0339


NOMINAL_SEP = 0.30      # 双脚 Y 间距的代表值（SPLIT 时两脚各在自己踏面中心）


def plan_tread(rise: float, depth: float):
    """给一级台阶做运动学规划。返回 (tilt_rad, off_m, 髋高窗口 or None)。

    tilt  : SPLIT 相位需要的 waist_x 侧倾（rad，正 = +Y 髋抬高）
    off   : 双脚同踏面时相对踏面中心的 Y 偏移
    注：tilt 用代表间距算，控制器拿到真实 sep 后应用 crab_ik.split_window 复核。
    """
    r = abs(float(rise))
    if r <= 1e-6:
        return 0.0, K.foot_placement(float(depth)), (0.451, 0.678)
    t = K.tilt_for_rise(r, sep=NOMINAL_SEP)
    if t is None:
        return None
    tilt, _w, rng = t
    return float(tilt), K.foot_placement(float(depth)), (float(rng[0]), float(rng[1]))


def clip_rise(rise: float, rise_max: float) -> float:
    """可行域裁剪：采样出的级高若超出上限就压回上限。"""
    return float(min(abs(rise), rise_max))


def sample_rises(rng, n: int, total: float, lo: float, hi: float):
    """采样 n 个级高，每个落在 [lo, hi] 且【总和恰为 total】。

    下楼段必须用这个：下楼总降高必须正好等于上楼总升高，否则楼梯末端
    会悬在半空（机器人走到最后一级就掉下 0.68 m 的悬崖）。
    做法：先均匀采样，再朝 lo 或 hi 做线性重标定（保持区间与总和）。
    """
    if n <= 0 or not (n * lo - 1e-9 <= total <= n * hi + 1e-9):
        return None
    u = rng.uniform(lo, hi, n)
    s = float(u.sum())
    if abs(s - total) < 1e-12:
        return u
    if s >= total:                       # 偏大 -> 朝 lo 收缩
        k = (total - n * lo) / (s - n * lo)
        u = lo + (u - lo) * k
    else:                                # 偏小 -> 朝 hi 拉伸
        k = (n * hi - total) / (n * hi - s)
        u = hi - (hi - u) * k
    return np.clip(u, lo, hi)


def generate(
    seed: int = 0,
    n_up: int = 6,
    n_down: int = 3,
    landing: bool = True,
    width: float = 1.2,
    rise_min: float = 0.12,
    rise_max: float = 0.25,
    tread_min: float = 0.20,
    tread_max: float = 0.40,
    landing_depth: float = 0.30,
    start_y: float = START_Y,
    material: str = "ut_concrete",
    friction: str = "1.05 0.01 0.001",
):
    """生成螃蟹步楼梯。返回 (xml 片段字符串, 真值表 dict)。"""
    rng = np.random.default_rng(seed)

    lines = []
    treads = []
    y = float(start_y)
    z = 0.0

    lines.append("<mujocoinclude>")
    lines.append("  <worldbody>")
    lines.append(f"    <!-- crab stairs: seed={seed} 楼梯沿 +Y 升高，机器人朝 +X 侧向横移 -->")
    lines.append(
        f'    <site name="crab_start" type="box" pos="0 {start_y - 0.35:.3f} 0.006" '
        f'size="0.9 0.06 0.006" rgba="0.1 0.8 0.25 0.55"/>'
    )

    def emit(name: str, y0: float, y1: float, top: float, section: str, rise: float):
        cy = (y0 + y1) / 2.0
        # top == 0 的"台阶"就是地面本身：不能再建一个零厚度的 box
        # （MuJoCo 会报 size 2 must be positive），但仍要记进真值表，
        # 因为控制器需要知道"这里有一步下到地面"。
        if top > 1e-4:
            lines.append(
                f'    <geom name="{name}" type="box" '
                f'pos="0 {cy:.4f} {top / 2.0:.4f}" '
                f'size="{width / 2.0:.4f} {(y1 - y0) / 2.0:.4f} {top / 2.0:.4f}" '
                f'material="{material}" friction="{friction}"/>'
            )
        plan = plan_tread(rise, y1 - y0)
        if plan is None:
            raise SystemExit(
                f"[错误] {name}: 级高 {rise:.3f} m 超出侧倾限位能覆盖的范围，"
                f"请调低 --rise-max")
        tilt, off, hwin = plan
        treads.append(
            {
                "name": name,
                "section": section,
                "y0": round(float(y0), 4),
                "y1": round(float(y1), 4),
                "top": round(float(top), 4),
                "rise": round(float(rise), 4),
                "depth": round(float(y1 - y0), 4),
                "center": round(float((y0 + y1) / 2.0), 4),
                "off": round(float(off), 4),
                "tilt": round(float(tilt), 4),
                "hip_h_window": [round(hwin[0], 4), round(hwin[1], 4)],
            }
        )

    # ---- 上行段 ----
    for i in range(n_up):
        d = float(rng.uniform(tread_min, tread_max))
        r = clip_rise(float(rng.uniform(rise_min, rise_max)), rise_max)
        z += r
        emit(f"crab_up_{i}", y, y + d, z, "up", r)
        y += d

    # ---- 中间平台 ----
    # 平台不能太深：SPLIT 站姿的脚间距 = (进深_i + 进深_i+1)/2 + 两脚间距，
    # 太深会把脚撑到 |Y| > rho*sin20deg，撞上 ankle_x 的 ±20deg 硬限位。
    if landing and n_down > 0:
        d = float(rng.uniform(landing_depth, landing_depth + 0.08))
        emit("crab_landing", y, y + d, z, "landing", 0.0)
        y += d

    # ---- 下行段 ----
    # 关键：下楼总降高必须 == 上楼总升高 z，末端才能与地面齐平。
    # 级数由总高反推（按级高上限估下限），再采样一组"和恰为 z"的级高。
    if n_down > 0:
        need = int(np.ceil(z / rise_max))
        if need * rise_min > z:                       # 级数太少 -> 单级会超限
            need = int(np.floor(z / rise_min))
        if need < 1:
            need = 1
        if need != n_down:
            print(f"  [调整] 下楼级数 {n_down} -> {need}"
                  f"（总降高 {z:.3f} m，级高须落在 {rise_min:.2f}~{rise_max:.2f}）")
        rises = sample_rises(rng, need, z, rise_min, rise_max)
        if rises is None:
            raise SystemExit(f"[错误] 无法把 {z:.3f} m 拆成 {need} 级 "
                             f"{rise_min:.2f}~{rise_max:.2f} 的台阶")
        for i, r in enumerate(rises):
            d = float(rng.uniform(tread_min, tread_max))
            z -= float(r)
            emit(f"crab_dn_{i}", y, y + d, z, "down", -float(r))
            y += d
        z = 0.0

    y_end = y
    lines.append(
        f'    <site name="crab_finish" type="box" pos="0 {y_end + 0.30:.3f} 0.006" '
        f'size="0.9 0.06 0.006" rgba="0.1 0.35 1.0 0.55"/>'
    )
    lines.append("  </worldbody>")
    lines.append("</mujocoinclude>")

    meta = {
        "seed": int(seed),
        "width": float(width),
        "rise_range": [float(rise_min), float(rise_max)],
        "tread_range": [float(tread_min), float(tread_max)],
        "n_up": int(n_up),
        "n_down": int(n_down),
        "landing": bool(landing),
        "start_y": float(start_y),
        "end_y": float(y_end),
        "spawn": {"x": 0.0, "y": 0.0, "z": SPAWN_Z},
        "max_top": round(float(z if n_down == 0 else max(t["top"] for t in treads)), 4),
        "treads": treads,
    }
    return "\n".join(lines) + "\n", meta


def build_entry_xml() -> str:
    """入口文件：材质 + 地面 + 生成的台阶。"""
    return (
        "<mujocoinclude>\n"
        "  <!-- 螃蟹步横移楼梯入口。台阶由 gen_crab_stairs.py 生成，勿手工编辑。 -->\n"
        '  <include file="assets.xml"/>\n'
        '  <include file="base.xml"/>\n'
        '  <include file="crab_stairs_obstacles.xml"/>\n'
        "</mujocoinclude>\n"
    )


ORIG_INCLUDE = (
    '<include file="terrains/universal_test/universal_test_terrain_compact.xml"/>'
)
CRAB_INCLUDE = '<include file="terrains/universal_test/crab_stairs.xml"/>'


def build_scene_xml() -> str:
    """elf3_crab.xml：复制 elf3.xml 只换地形 include，保证原正向爬楼链路不受影响。"""
    src = DATA / "elf3.xml"
    text = src.read_text(encoding="utf-8")
    if ORIG_INCLUDE not in text:
        raise SystemExit(f"[错误] {src} 里找不到原始地形 include，请检查 {ORIG_INCLUDE}")
    return text.replace(ORIG_INCLUDE, CRAB_INCLUDE)


def main() -> None:
    ap = argparse.ArgumentParser(description="螃蟹步横移楼梯地形生成器")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-up", type=int, default=6, help="上级数")
    ap.add_argument("--n-down", type=int, default=3, help="下级数（0=无下行段）")
    ap.add_argument("--no-landing", action="store_true", help="不要中间平台")
    ap.add_argument("--width", type=float, default=1.2, help="楼梯净宽 m（沿 X）")
    ap.add_argument("--rise-min", type=float, default=0.12)
    ap.add_argument("--rise-max", type=float, default=0.25)
    ap.add_argument("--tread-min", type=float, default=0.20)
    ap.add_argument("--tread-max", type=float, default=0.40)
    ap.add_argument("--landing-depth", type=float, default=0.30,
                    help="中间平台进深 m。太深会把 SPLIT 站姿撑到超 ankle_x 限位")
    ap.add_argument("--out-xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--write-entry", action="store_true",
                    help="同时写 crab_stairs.xml 入口文件")
    ap.add_argument("--write-scene", action="store_true",
                    help="同时写 elf3_crab.xml（复制 elf3.xml 只换地形 include）")
    args = ap.parse_args()

    # 相对路径一律按【地形目录】解释。否则 run_crab.py 传 "crab_stairs.json"
    # 会把文件写到当前工作目录（launch/），而 crab_climb.py 又去地形目录里找，
    # 两边各拿一份、静默读到旧的。
    if not args.out_xml.is_absolute():
        args.out_xml = TERRAIN_DIR / args.out_xml
    if not args.out_json.is_absolute():
        args.out_json = TERRAIN_DIR / args.out_json

    xml, meta = generate(
        seed=args.seed,
        n_up=args.n_up,
        n_down=args.n_down,
        landing=not args.no_landing,
        width=args.width,
        rise_min=args.rise_min,
        rise_max=args.rise_max,
        tread_min=args.tread_min,
        tread_max=args.tread_max,
        landing_depth=args.landing_depth,
    )

    args.out_xml.parent.mkdir(parents=True, exist_ok=True)
    args.out_xml.write_text(xml, encoding="utf-8")
    args.out_json.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.write_entry:
        (args.out_xml.parent / "crab_stairs.xml").write_text(
            build_entry_xml(), encoding="utf-8"
        )
        print(f"[OK] 入口 XML  -> {args.out_xml.parent / 'crab_stairs.xml'}")
    if args.write_scene:
        scene = DATA / "elf3_crab.xml"
        scene.write_text(build_scene_xml(), encoding="utf-8")
        print(f"[OK] 场景 XML  -> {scene}")

    print(f"[OK] 台阶 XML  -> {args.out_xml}")
    print(f"[OK] 真值表    -> {args.out_json}")
    print(f"     级数 {len(meta['treads'])} | y {meta['start_y']:.2f} -> {meta['end_y']:.2f}"
          f" | 最高 {meta['max_top']:.3f} m")
    print("     " + "-" * 60)
    print(f"     {'级':<14}{'y0':>7}{'y1':>7}{'top':>7}{'级高':>7}{'进深':>7}"
          f"{'侧倾deg':>9}{'落点off':>8}{'髋高窗口':>16}")
    for t in meta["treads"]:
        lo, hi = t["hip_h_window"]
        win = f"{lo:.3f} ~ {hi:.3f}"
        print(f"     {t['name']:<14}{t['y0']:>7.2f}{t['y1']:>7.2f}"
              f"{t['top']:>7.3f}{t['rise']:>7.3f}{t['depth']:>7.2f}"
              f"{np.degrees(t['tilt']):>9.1f}{t['off']:>8.3f}{win:>16}")


if __name__ == "__main__":
    main()
