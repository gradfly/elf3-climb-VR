"""纯自驱竖直跳高实验：扫「摆臂幅度 / 摆臂集中度 / 伸腿幅度 / 蹬腿刚度 / 蹲深」，
量化各旋钮对【真实跳跃高度】的贡献。

与 long_jump_real.py 一致：零冲量注入，腾空/落地全真实物理。
本脚本只关心高度（竖直起跳，anchor_shift=0，不做前向蹬地），
看摆臂 + 蹬腿能否把质心顶得更高。

输出指标：
  vz            离地竖直速度 (m/s)，纯 PD 产出
  aerial_rise   vz^2/2g  —— 离地后的自由上升高度
  peak_com_z    质心全程峰值高度 (m)
  jump_height   peak_com_z - 站立起点 COM (m)  —— 直观“跳了多高”
  tau           峰值力矩占用
  land_ok       是否落地站稳（站稳保持≥2s 且 COM 裕度>2cm）

用法：
  python exp_jump_height.py                 # 跑默认网格
  python exp_jump_height.py --quick         # 只对比 摆臂/蹬腿 开关效应
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import long_jump as LJ
import long_jump_real as LR


def measure(arm_scale, arm_rate, rise, kp, drop, lean=0.0,
            anchor=0.0, tt=0.30, pf=0.40) -> dict:
    base = LJ.Params(drop_squat=drop, lean_squat=lean,
                     vx_takeoff=0.0, vz_takeoff=0.0,
                     pf_to=0.15, arm_up=-0.60, dxw_to=-0.06,
                     land_reach=0.22,
                     kp_scale=kp, kd_scale=8.0, sim_hz=1000)
    base.t_takeoff = tt
    cfg = dict(rise=rise, pf_end=pf, anchor_shift=anchor,
               arm_scale=arm_scale, arm_rate=arm_rate,
               kp=kp, drop=drop, lean=lean)
    sim = LR.LongJumpReal(base, cfg)
    sim.run()
    d = LR.evaluate(base, cfg)
    d["fail"] = sim.failed_reason
    return d


def fmt(d: dict) -> str:
    tag = "Y" if d["ok"] else ("-" if d["fail"] else "n")
    jh = d.get("jump_height", float("nan"))
    ar = d.get("aerial_rise", float("nan"))
    why = "" if d["ok"] else (f"  [{d.get('fail') or '?':.40s}]"[:42])
    return (f"{d['cfg'].get('arm_scale',1):4.1f} "
            f"{d['cfg'].get('arm_rate',2):4.1f} "
            f"{d['cfg'].get('rise',1):5.2f} "
            f"{d['cfg'].get('kp',8):3.0f} "
            f"{d['cfg'].get('drop',0.25):4.2f} | "
            f"{d['vz']:+6.2f} {ar*100:6.1f}cm "
            f"{d['peak_com_z']:.3f} {jh*100:6.1f}cm "
            f"{d['tau']*100:4.0f}% {tag}{why}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="只做 4 组对比：无摆臂/有摆臂 × 低刚度/高刚度")
    args = ap.parse_args()

    hdr = (f"{'arm':>4} {'rate':>4} {'rise':>5} {'kp':>3} {'drop':>4} | "
           f"{'vz':>7} {'aerial':>7} {'peakCOM':>7} {'jumpH':>7} {'tau':>5} ok")
    print(hdr)
    print("-" * len(hdr))

    if args.quick:
        # 直接对照：摆臂(arm_scale)与蹬腿刚度(kp)的开关效应
        grid = [
            dict(arm_scale=1.0, arm_rate=2.0, rise=1.04, kp=8,  drop=0.25),  # 基线
            dict(arm_scale=2.6, arm_rate=2.5, rise=1.04, kp=8,  drop=0.25),  # 只加摆臂
            dict(arm_scale=1.0, arm_rate=2.0, rise=1.04, kp=16, drop=0.25),  # 只加刚度
            dict(arm_scale=2.6, arm_rate=2.5, rise=1.15, kp=16, drop=0.35),  # 全加
        ]
    else:
        # 聚焦网格：在“能落地站稳”的约束下，找最高的纯自驱竖直跳。
        # 摆臂幅度 × 集中度 × 蹬腿刚度（rise/drop 取稳妥值）
        arm_scales = [1.0, 1.4, 1.8, 2.2]
        arm_rates = [1.5, 2.5]
        kps = [8, 12, 16]
        grid = []
        for a in arm_scales:
            for r in arm_rates:
                for k in kps:
                    grid.append(dict(arm_scale=a, arm_rate=r,
                                     rise=1.04, kp=k, drop=0.28, lean=0.0))

    best = None          # 落地站稳前提下的最高
    ceiling = None       # 不论是否落地的最高（物理上限，含翻滚）
    for g in grid:
        d = measure(**g)
        print(fmt(d))
        if d["ok"] and (best is None or d["jump_height"] > best["jump_height"]):
            best = d
        if ceiling is None or d["jump_height"] > ceiling["jump_height"]:
            ceiling = d

    print("-" * len(hdr))
    if ceiling:
        c = ceiling["cfg"]
        print(f"[物理上限(含翻滚)] arm_scale={c.get('arm_scale')} arm_rate={c.get('arm_rate')} "
              f"kp={c.get('kp')} → vz={ceiling['vz']:+.2f} m/s  "
              f"质心峰值 {ceiling['peak_com_z']:.3f} m  跳高 {ceiling['jump_height']*100:.1f} cm"
              f"  {'（能落地站稳）' if ceiling['ok'] else '（翻滚、落不稳）'}")
    if best:
        c = best["cfg"]
        print(f"[可落地站稳的最高跳] arm_scale={c.get('arm_scale')} arm_rate={c.get('arm_rate')} "
              f"rise={c.get('rise')} kp={c.get('kp')} drop={c.get('drop')}")
        print(f"  vz={best['vz']:+.2f} m/s   离地后自由上升 {best['aerial_rise']*100:.1f} cm")
        print(f"  质心峰值 {best['peak_com_z']:.3f} m   相对站立起点跳高 "
              f"{best['jump_height']*100:.1f} cm   峰值力矩 {best['tau']*100:.0f}%")


if __name__ == "__main__":
    main()
