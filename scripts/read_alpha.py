#!/usr/bin/env python
"""读某个 run 的参考步态幅度 α（训练日志里 `Metrics/ref_alpha` 的末值）。

为什么单独做一个脚本
--------------------
评估时必须把 α 钉成 checkpoint 训练时的值。钉错（尤其是没钉）会出现一种**静默失败**：

    train 配置的 α 默认走课程，`alpha_start=0` 起步、总长 1500 iter × 24 步，
    而动作项每次都是新建实例、`_step_i` 从 0 开始 —— 于是跑 1000 步时 α 只到 0.028。

α=0.028 时参考步态 ≈ 站姿，残差（限幅 ±0.55 rad）要独自憋出整个步态，实测表现为
「60% 维度顶死限幅、每 2.7 s 摔一次、20 s 自转 1.7 圈」。看着像策略烂，其实是参数错。

文档里 4 个地方要用这个数，内联 heredoc 抄 4 遍必然漏，所以做成脚本。

用法
----
    .venv/bin/python scripts/read_alpha.py                  # 最新的 run
    .venv/bin/python scripts/read_alpha.py <run_dir>        # 指定 run
    .venv/bin/python scripts/read_alpha.py --ckpt .../model_9000.pt

stdout 只输出数字（给 `ALPHA=$(...)` 用），说明信息走 stderr。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

TAG = "Metrics/ref_alpha"


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("run", nargs="?", default=None, help="run 目录（含 events.out.tfevents*）")
  ap.add_argument("--ckpt", default=None, help="从 checkpoint 路径取它的父目录当 run 目录")
  ap.add_argument("--tag", default=TAG, help=f"默认 {TAG}")
  ap.add_argument("--stat", action="store_true",
                  help="打印整条曲线的 min/mean/max/末值（判断某个指标是不是恒 0 时用）")
  a = ap.parse_args()

  run = a.run
  if run is None and a.ckpt:
    run = os.path.dirname(os.path.abspath(a.ckpt))
  if run is None:
    runs = sorted(glob.glob("logs/rsl_rl/elf3_crab/*"), key=os.path.getmtime)
    if not runs:
      print("[ERR] 没有 logs/rsl_rl/elf3_crab/*（是不是不在仓库根目录？）", file=sys.stderr)
      return 2
    run = runs[-1]

  if not glob.glob(os.path.join(run, "events.out.tfevents*")):
    print(f"[ERR] {run} 里没有 events.out.tfevents* 文件", file=sys.stderr)
    return 2

  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except Exception as e:      # noqa: BLE001
    print(f"[ERR] 读不了 tensorboard：{e}", file=sys.stderr)
    return 2

  # scalars=0 表示全量保留：默认的 10000 条是蓄水池采样，取 [-1] 未必真的是最后一条。
  ea = EventAccumulator(run, size_guidance={"scalars": 0})
  ea.Reload()
  tags = ea.Tags()["scalars"]
  if a.tag not in tags:
    print(f"[ERR] {run} 里没有 {a.tag}（ref_probe 没跑？现有 tag 样例：{tags[:4]}）",
          file=sys.stderr)
    return 2

  vals = ea.Scalars(a.tag)
  last = vals[-1]
  print(f"{last.value:.3f}")          # ← stdout 只给数字
  print(f"[info] run={run}", file=sys.stderr)
  print(f"[info] 采样点 {len(vals)} 个，末值 {last.value:.3f} @ step {last.step}",
        file=sys.stderr)
  if a.stat:
    arr = [e.value for e in vals]
    lo, hi = min(arr), max(arr)
    mean = sum(arr) / len(arr)
    print(f"[stat] {a.tag}: min {lo:.4f}  mean {mean:.4f}  max {hi:.4f}  last {last.value:.4f}",
          file=sys.stderr)
    print(f"[stat] 恒 0? {'是 —— 这条指标没在动，别拿它推理' if hi <= 1e-9 else '否'}"
          f"   恒定值? {'是（可能没被更新）' if hi - lo <= 1e-9 else '否'}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
