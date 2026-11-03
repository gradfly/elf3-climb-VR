#!/usr/bin/env python3
"""检查「参考步态表」与「地形各难度档」是否匹配（纯离线，不需要 mjlab / GPU）。

三个环节
--------
1. **地形侧**：从 `train/mjlab_tasks/crab/terrain_cfg.py` 抽出四档 sub_terrain 参数，
   按 mjlab 源码（`primitive_terrains.py: BoxInvertedPyramidStairsTerrainCfg`）的构造
   规则算出每档的真实几何：

       num_steps  = int((size - 2*border - platform) / (2*step_width))
       平台半宽 p = (terrain_size - 2*num_steps*step_width) / 2     # 出生点到第一级踏面
       每行级高 h = lo + difficulty*(hi-lo),  difficulty = row/(num_rows-1)

   注意 `platform_width` **不**等于平台宽度，它只进 num_steps 的整数除法。按上式
   实算的四档平台半宽（2026-11-03 起）是
   flat 1.00 / easy 0.80 / mid 0.40 / hard 0.62 m —— 旧注的 0.60 / 0.52 有误：
   flat 的 (8-2*1.0-1.2)/(2*0.40)=5.999999999999999 被 int() 截成 5 -> 1.00 m；
   mid 改 step_width=0.32->0.30、platform_width=1.0->0.8 后 num_steps 9->10
   -> 半宽 0.52 -> 0.40 m。四条地形踏面 {0.24, 0.30, 0.40} 全部落在参考表
   tread 网格 {0.24, 0.30, 0.36, 0.40} 上，故 estimate() 无量化跳档。

2. **估计侧**：直接调用训练里那份 `ref_table.RefTable.estimate()`（同一个函数、同一套
   默认参数：nx=13 ny=21 res=0.1 y∈(0.15,1.00) edge_thr=0.03 dh_max=0.30），喂给它
   「机器人站在地形上、射线以 0.1 m 栅格向下打」时真正会看到的高度场。机器人沿 +Y
   从出生点(平台中心)一路走到第三级踏面，位置连续扫一遍 —— 因为栅格与台阶的相对
   相位是任意的，测出来的值不是一个常数，而是一个分布。

3. **匹配判据**：`select()` 用最近档选 tread（不插值），步态的 `dy_cycle` 恒等于该档
   的 tread —— 即一个周期前进一个踏面。所以

       步幅误差 = dy_cycle(选中档) - 地形真实 step_width

   就是「参考步态和地形匹不匹配」的可执行判据：误差 1 cm = 每个周期落脚点相对台沿
   漂 1 cm，且不收敛（相位误差随周期累加）。

用法
----
    python scripts/check_ref_vs_terrain.py              # 全表 + 分布
    python scripts/check_ref_vs_terrain.py --brief      # 只出结论
    python scripts/check_ref_vs_terrain.py --x-offset 0.4   # 机器人横向偏离中心
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train/mjlab_tasks/crab"))
from ref_table import RefTable  # noqa: E402

TERRAIN_CFG = REPO / "train/mjlab_tasks/crab/terrain_cfg.py"
TABLE = REPO / "train/assets/ref_tables/crab_ref_v1.npz"

SIZE = 8.0            # TerrainGeneratorCfg.size
NUM_ROWS = 10         # TerrainGeneratorCfg.num_rows
NX, NY = 13, 21       # SCAN_SIZE (1.2, 2.0) / 0.1 → +1
RES = 0.1
TORSO_ABOVE_GROUND = 1.03


def parse_sub_terrains(path: Path) -> dict[str, dict]:
    """从 terrain_cfg.py 抽四档参数（不 import，避免依赖 mjlab）。"""
    txt = re.sub(r"#[^\n]*", "", path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for m in re.finditer(r'"(?P<name>\w+)":\s*pyramid_stairs_inv\('
                         r'(?P<body>.*?)\n\s{8}\),', txt, re.S):
        body = m.group("body")
        d: dict = {"name": m.group("name")}
        for key in ("proportion", "step_width", "platform_width", "border_width"):
            mm = re.search(rf"{key}\s*=\s*([0-9.]+)", body)
            if mm:
                d[key] = float(mm.group(1))
        mm = re.search(r"step_height_range\s*=\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", body)
        if mm:
            d["step_height_range"] = (float(mm.group(1)), float(mm.group(2)))
        out[d["name"]] = d
    if len(out) != 4:
        raise SystemExit(f"[ERR] 只解析到 {len(out)} 档 sub_terrain：{list(out)}")
    return out


def geometry(sub: dict) -> dict:
    sw, bw, pw = sub["step_width"], sub["border_width"], sub["platform_width"]
    n = max(0, int(min(int((SIZE - 2 * bw - pw) / (2 * sw)),
                       int((SIZE - 2 * bw - pw) / (2 * sw)))))
    terrain_size = SIZE - 2 * bw
    return {"num_steps": n, "platform_half": (terrain_size - 2 * n * sw) / 2.0,
            "terrain_size": terrain_size, "step_width": sw}


def ground_rel(y_abs: np.ndarray, p: float, sw: float, h: float, n: int) -> np.ndarray:
    """相对「中心平台顶面」的地面高度（米），方形同心环 → 用切比雪夫距离。

    倒置金字塔：中心平台最低（0），往外每 sw 抬一级 h。最内环（j=0）紧邻平台。
    """
    d = np.abs(y_abs)
    j = np.clip(np.floor((d - p) / max(sw, 1e-6)), 0.0, float(n) - 1.0)
    rel = (j + 1.0) * h
    return np.where(d <= p, 0.0, rel)


def scan_at(tb: RefTable, geo: dict, h: float, y_robot: float, x0: float) -> torch.Tensor:
    """机器人在 (x0, y_robot) 处看到的 height_scan（躯干高 - 命中地面高）。"""
    xs = x0 + (np.arange(NX) - NX // 2) * RES
    ys = y_robot + (np.arange(NY) - NY // 2) * RES
    gx, gy = np.meshgrid(xs, ys)                          # [NY, NX]
    d = np.maximum(np.abs(gx), np.abs(gy))                # 方环：切比雪夫距离
    j = np.clip(np.floor((d - geo["platform_half"]) / geo["step_width"]),
                0.0, float(geo["num_steps"]) - 1.0)
    rel = np.where(d <= geo["platform_half"], 0.0, (j + 1.0) * h)
    rel0 = float(ground_rel(np.array([max(abs(y_robot), abs(x0))]),
                            geo["platform_half"], geo["step_width"], h,
                            geo["num_steps"])[0])
    hs = (TORSO_ABOVE_GROUND + rel0) - rel
    return torch.tensor(hs.reshape(1, -1), dtype=torch.float32)


def n_edges(tb: RefTable, geo: dict, h: float, y_robot: float, x0: float) -> int:
    scan = scan_at(tb, geo, h, y_robot, x0).reshape(NY, NX)
    mid = NX // 2
    prof = np.median(scan[:, mid - 1:mid + 2].numpy(), axis=-1)
    dh = prof[:-1] - prof[1:]
    y_mid = -NY * RES / 2.0 + (np.arange(NY - 1) + 0.5) * RES
    band = (np.abs(y_mid) > 0.15) & (np.abs(y_mid) < 1.00)
    mask = (dh > 0.03) & (dh < 0.30) & band & (y_mid > 0)
    return int(mask.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--x-offset", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=60, help="沿 +Y 的站位采样点数")
    args = ap.parse_args()

    subs = parse_sub_terrains(TERRAIN_CFG)
    tb = RefTable(TABLE)
    rise_grid = np.asarray(tb.rise.cpu())
    tread_grid = np.asarray(tb.tread.cpu())
    geo = {k: geometry(v) for k, v in subs.items()}

    print(f"[table] {TABLE.name}: rise {rise_grid[0]:.2f}..{rise_grid[-1]:.2f} "
          f"/{rise_grid[1]-rise_grid[0]:.2f} ({tb.R} 档)  "
          f"tread {[round(float(t), 2) for t in tread_grid]} ({tb.T} 档)")
    print(f"[scan ] nx={NX} ny={NY} res={RES}  y∈(0.15,1.00)  x-offset={args.x_offset:+.2f}")
    for k, g in geo.items():
        print(f"[geo  ] {k:<12} num_steps={g['num_steps']:<3} "
              f"平台半宽={g['platform_half']:.2f}m  踏面={g['step_width']:.2f}m  "
              f"级高 {subs[k]['step_height_range'][0]*100:.0f}~"
              f"{subs[k]['step_height_range'][1]*100:.0f}cm")
    max_rise = max(s["step_height_range"][1] for s in subs.values())
    print(f"[cover] 级高上限 {max_rise*100:.0f}cm vs 表上限 {rise_grid[-1]*100:.0f}cm -> "
          f"{'覆盖 OK' if max_rise <= rise_grid[-1] + 1e-9 else '超出表范围！'}")
    print(f"[cover] 踏面真值 {sorted({round(g['step_width'],2) for g in geo.values()})} vs "
          f"表网格 {[round(float(t),2) for t in tread_grid]}")
    print()

    hdr = (f"{'lvl':>3} {'col':<12} {'级高':>5} {'踏面':>5} | {'rise测':>6} {'tread测':>13} "
           f"{'选中':>5} {'步幅':>5} {'误差':>6} | {'边':>2} {'valid':>5}")
    if not args.brief:
        print(hdr)
        print("-" * len(hdr))

    rows_bad = []
    col_stat: dict[str, list] = {}
    for col, sub in subs.items():
        g = geo[col]
        lo, hi = sub["step_height_range"]
        y_max = min(g["platform_half"] + 3 * g["step_width"], 3.0)
        y_rs = np.linspace(0.0, y_max, args.samples)
        for row in range(NUM_ROWS):
            d = row / max(NUM_ROWS - 1, 1)
            h = lo + d * (hi - lo)
            rs, ts, vs, ne = [], [], [], []
            for yr in y_rs:
                r, t, v = tb.estimate(scan_at(tb, g, h, float(yr), args.x_offset))
                rs.append(float(r[0])); ts.append(float(t[0])); vs.append(bool(v[0]))
                ne.append(n_edges(tb, g, h, float(yr), args.x_offset))
            rs, ts = np.array(rs), np.array(ts)
            vfrac = float(np.mean(vs))

            t_mean = float(ts.mean())
            j = int(np.abs(t_mean - tread_grid).argmin())
            sel = float(tread_grid[j])
            stride = float(tb.v_ref[0, j] * tb.period * tb.dt)      # = dy_cycle
            err = stride - g["step_width"]
            rel = err / g["step_width"]
            col_stat.setdefault(col, []).append((row, h, rel, g["step_width"], sel,
                                                 ne, vfrac, rs.mean(), ts.mean()))
            if abs(rel) > 0.05:
                rows_bad.append((abs(rel), col, row, h, g["step_width"], sel, stride, err,
                                 int(np.median(ne)), vfrac))
            if not args.brief:
                flag = "" if abs(rel) <= 0.03 else ("  <== 失配" if abs(rel) > 0.05 else "  ~")
                print(f"{row:>3} {col:<12} {h*100:>4.1f}cm {g['step_width']*100:>4.0f}cm | "
                      f"{rs.mean()*100:>5.1f}cm "
                      f"{ts.mean()*100:>5.1f}±{ts.std()*100:>4.1f}cm {sel*100:>4.0f}cm "
                      f"{stride*100:>4.0f}cm {err*100:>+5.1f}cm | "
                      f"{int(np.median(ne)):>2} {vfrac:>5.0%}{flag}")
        if not args.brief:
            print("-" * len(hdr))

    print("\n[按列汇总]  踏面真值 vs 表网格；步幅相对误差；可见边数")
    for col, st in col_stat.items():
        errs = np.array([x[2] for x in st])
        ne = int(np.median([x[5] for x in st]))
        vf = float(np.mean([x[6] for x in st]))
        sels = sorted({x[4] for x in st})
        note = ""
        if ne < 2:
            note = "  ← tread 测不出来（<2 条边），一直用 mid_tread 0.36"
        print(f"  {col:<12} 踏面 {st[0][3]*100:.0f}cm  选中档 {[round(s*100) for s in sels]}  "
              f"误差 {errs.min()*100:+.0f}%~{errs.max()*100:+.0f}%  "
              f"边数中位 {ne}  valid {vf:.0%}{note}")

    print("\n[步幅误差 > 5% 的档位]")
    rows_bad.sort(reverse=True)
    if not rows_bad:
        print("  无")
    for b in rows_bad:
        print(f"  {b[0]*100:>4.0f}%  {b[1]:<12} row{b[2]:<2} 级高{b[3]*100:5.1f}cm "
              f"踏面{b[4]*100:4.0f}cm → 选中{b[5]*100:4.0f}cm 步幅{b[6]*100:4.0f}cm "
              f"({b[7]*100:+.0f}cm/周期)  边数{b[8]} valid{b[9]:.0%}")

    # ---- 结论 ----
    tier = {"匹配 (<=3%)": 0, "轻微 (3~10%)": 0, "明显 (>10%)": 0}
    for col, st in col_stat.items():
        for x in st:
            e = abs(x[2])
            tier["匹配 (<=3%)" if e <= 0.03 else
                 "轻微 (3~10%)" if e <= 0.10 else "明显 (>10%)"] += 1
    print("\n[结论] 40 个 (列,行) 档位按步幅相对误差分档：")
    for k, v in tier.items():
        print(f"  {k:<14} {v:>2} 个")
    print("  判据：一个周期前进一个踏面。误差 = 每周期落点相对台沿的漂移量，不收敛。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
