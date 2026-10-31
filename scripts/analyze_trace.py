#!/usr/bin/env python
"""读 `diagnose_policy.py --trace` 落盘的 npz，判定 X 向漂移属于 A/B/C 哪一类。

为什么单独一个脚本
------------------
漂移"看起来"只有一种，实际是三种完全不同的病，治法相反：

  A 偏航耦合    朝向慢慢歪了，机器人还以为自己在朝左走 -> 越走越斜。
                观测里没有 heading，只要歪了就无法自纠。**改奖励治不好，必须加观测。**
  B 步态不对称  朝向没问题，但左右腿出力不等，每一步都往同一个方向偏。
                这是奖励/限幅的问题，加 `lin_vel_x_zero` 就能压。
  C 扰动失能    平时走得直，被推一下之后回不来，偏差单向累积。
                这是鲁棒性问题，得先做扰动课程。

判据里最容易搞错的两件事（本脚本已修）：
  1) 漂移比必须是 |Δx| / |Δy|，且两者**同一基准**（都从本回合出生点算）。
     原方案写成 |dx|/vy，量纲是秒；而且 dx 每回合归零、dy 跨回合累加，除出来没有意义。
  2) 朝向必须用**相对本回合出生朝向**的偏角。`reset_root_state_uniform` 的 yaw 是
     ±0.15 rad（±8.6°）随机的，拿绝对值当"误差"会把初始偏移算成漂移。

用法
----
    python scripts/analyze_trace.py /tmp/trace_b0.npz
    python scripts/analyze_trace.py /tmp/trace_b4.npz --push
"""
from __future__ import annotations

import argparse
import numpy as np

TRAJ_KEYS = ("dx", "dy_ep", "yaw_rel", "vx", "vy", "alive")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
  """安全带常数检查的相关系数；样本太少或方差为 0 时返回 nan。"""
  if a.size < 10 or b.size != a.size:
    return float("nan")
  if np.std(a) < 1e-9 or np.std(b) < 1e-9:
    return float("nan")
  return float(np.corrcoef(a, b)[0, 1])


def _per_env_table(d, skip: int, yaw_thresh: float) -> dict:
  dx = d["dx"]
  dy = d["dy_ep"] if "dy_ep" in d else d["dy"]
  yaw = np.degrees(d["yaw_rel"] if "yaw_rel" in d else d["yaw"])
  vx, vy = d["vx"], d["vy"]
  alive = d["alive"].astype(bool)
  # 训练配置 episode 固定 20 s，正常跑完也会 dones=True。不把超时剔除，
  # 表里的"摔倒"就是把通关当成摔了 —— 尤其 1000 步正好等于一整段 episode。
  timeout = d["timeout"].astype(bool) if "timeout" in d else np.zeros_like(alive)
  n_steps, n_env = dx.shape

  idx = np.arange(n_steps)[:, None]
  # 冷启动屏蔽必须按**回合内步数**，不能按绝对步号：摔倒重开后，绝对步号继续走，
  # 用 `idx > skip` 会把新回合的前 60 步当成"已热启动"直接计入。
  if "ep_t" in d:
    warm = d["ep_t"] > skip
  else:
    warm = idx > skip
  valid = alive & warm

  # 每个 env 最后一个存活步 —— 不用 dx[-1]：那一步常常正好是 reset 步（dx 被清零）
  last = np.where(valid, np.broadcast_to(idx, dx.shape), -1).max(axis=0)
  ok = last >= 0
  lc = np.clip(last, 0, n_steps - 1)
  cols = np.arange(n_env)

  end_dx = np.where(ok, np.abs(dx[lc, cols]), np.nan)
  end_dy = np.where(ok, np.abs(dy[lc, cols]), np.nan)
  ratio = np.where(end_dy > 0.05, end_dx / np.maximum(end_dy, 1e-3), np.nan)

  quiet = valid & (np.abs(yaw) < yaw_thresh)
  active = valid & (np.abs(yaw) >= yaw_thresh)

  rows = []
  for e in range(n_env):
    # ⚠️ 必须先切列再掩码。直接写 yaw[m]（yaw 是 (N,B)）拿到的是 (N_sel, B) 二维块，
    # 后面传给 corrcoef 会和 1 维的 vx[m] 对不上 —— 静默返回 nan，不是报错。
    yw_e = yaw[:, e]
    vx_e = vx[:, e]
    m = valid[:, e]
    mm = m[1:] & m[:-1]

    q = quiet[:, e]
    a = active[:, e]
    q_n = int(q.sum())
    q_vx = float(vx_e[q].mean()) if q_n else float("nan")
    # 同号率：安静段 vx 有多"稳定"地指向同一侧。0.5 = 随机噪声，1.0 = 恒定偏置
    if q_n >= 5:
      p = float((vx_e[q] > 0).mean())
      consistency = max(p, 1.0 - p)
    else:
      consistency = float("nan")

    # 主判别量：横向速度与朝向的**水平相关**。
    # 偏航耦合时 vx ≈ -v·sin(yaw)，两者同涨同落 -> |corr| 接近 1；
    # 步态不对称时 vx 是常数、yaw 只是噪声 -> corr ≈ 0。
    # （原方案用的是 corr(Δdx, Δyaw)：差分会把这种慢变共模消掉，实测即便完美耦合
    #   也只有 -0.04，是选错了统计量，不是策略没问题。）
    c_vx_yaw = _corr(vx_e[m], yw_e[m])
    c_delta = _corr(np.diff(dx[:, e])[mm], np.diff(yw_e)[mm])

    # 速度方向 vs 朝向：若策略在 heading 系里追指令，速度矢量会跟着 yaw 一起转
    ang = np.degrees(np.arctan2(vx_e, np.maximum(vy[:, e], 1e-6)))
    c_head = _corr(ang[m], yw_e[m])

    rows.append(dict(
        env=e, dx=end_dx[e], dy=end_dy[e], ratio=ratio[e],
        yaw_end=yw_e[lc[e]] if ok[e] else np.nan,
        yaw_travel=float(np.abs(np.diff(yw_e))[mm].sum()) if mm.any() else np.nan,
        vx_q=q_vx, q_n=q_n, cons=consistency,
        corr=c_vx_yaw, corr_delta=c_delta, corr_head=c_head,
        vx_a=float(vx_e[a].mean()) if a.sum() else float("nan"),
        n_fall=int(((~alive[:, e]) & ~timeout[:, e]).sum()),
        n_to=int(timeout[:, e].sum()),
    ))
  return dict(rows=rows, valid=valid, end_dx=end_dx, end_dy=end_dy,
              ratio=ratio, quiet=quiet, n_env=n_env, dx=dx, dy=dy,
              yaw=yaw, vx=vx, vy=vy, alive=alive)


def _push_report(d, skip: int) -> None:
  """扰动恢复：每次被推之后，横向速度多久回到安静水平、X 偏移是否单向累积。"""
  if "push" not in d:
    print("[push] 这次 trace 没记推挤标记（要 --push 才有）")
    return
  push = d["push"].astype(bool)
  vx, dx, alive = d["vx"], d["dx"], d["alive"].astype(bool)
  ep_t = d.get("ep_t")
  n_env = vx.shape[1]
  quiet_thr = 0.02
  horizon = int(2.5 / 0.02)          # 2.5 s 内没恢复就算"没恢复"

  rec_t, rec_dx = [], []
  for e in range(n_env):
    for t in np.flatnonzero(push[:, e]):
      if (ep_t[t, e] if ep_t is not None else t) < skip:
        continue
      seg_alive = alive[t:t + horizon, e]
      seg_vx = np.abs(vx[t:t + horizon, e])
      back = np.flatnonzero(seg_alive & (seg_vx < quiet_thr))
      if back.size == 0:
        rec_t.append(np.nan)
        continue
      k = int(back[0])
      rec_t.append(k * 0.02)
      rec_dx.append(float(dx[t + k, e] - dx[t, e]))
  if not rec_t:
    print("[push] 有效推挤样本 0（--steps 太短，或 --skip 把推挤全跳过了）")
    return
  arr = np.array(rec_t, dtype=float)
  finite = np.isfinite(arr)
  print(f"[push] 推挤后回到 |vx|<{quiet_thr} 的耗时："
        f"中位 {np.nanmedian(arr):.2f} s  未恢复占比 {(~finite).mean() * 100:.0f}%  "
        f"(样本 {arr.size})")
  if rec_dx:
    dd = np.array(rec_dx)
    same = float((np.sign(dd) == np.sign(np.nanmedian(dd))).mean())
    print(f"[push] 恢复期间 Δx：mean {dd.mean():+.3f} m，"
          f"与中位同号占比 {same * 100:.0f}%  "
          f"({'单向累积 -> 没有横向纠偏能力' if same > 0.75 else '左右抵消，像噪声'})")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("trace", help="diagnose_policy.py --trace 落盘的 npz")
  ap.add_argument("--skip", type=int, default=60, help="跳过每回合前 N 步（默认 60 = 1.2 s）")
  ap.add_argument("--yaw-thresh", type=float, default=3.0,
                  help="「安静段」的朝向阈值（度，相对出生朝向）")
  ap.add_argument("--push", action="store_true", help="额外做扰动恢复分析")
  args = ap.parse_args(argv)

  d = dict(np.load(args.trace))
  missing = [k for k in ("dx", "alive", "vx", "vy") if k not in d]
  if missing:
    print(f"[ERR] {args.trace} 缺字段 {missing}；用新版 diagnose_policy.py 重跑")
    return 2
  if "yaw_rel" not in d:
    print("[warn] 没有 yaw_rel，退回绝对 yaw（会把 ±8.6° 的出生朝向当成漂移）")

  r = _per_env_table(d, args.skip, args.yaw_thresh)
  rows, n_env = r["rows"], r["n_env"]
  valid_rate = r["valid"].sum() / max(r["valid"].size, 1)
  print("=" * 78)
  print(f"trace      : {args.trace}")
  print(f"shape      : {r['dx'].shape[0]} 步 × {n_env} env   "
        f"有效样本 {valid_rate * 100:.0f}%")
  print("-" * 78)
  print(f"{'env':>3} {'末|dx|':>7} {'Δy':>7} {'漂移比':>7} {'yaw末':>7} "
        f"{'yaw总转':>7} {'安静vx':>8} {'同号':>5} {'c(vx,yaw)':>9} {'摔/超':>6}")
  for x in rows:
    print(f"{x['env']:>3} {x['dx']:>7.3f} {x['dy']:>7.3f} {x['ratio']:>7.3f} "
          f"{x['yaw_end']:>7.1f} {x['yaw_travel']:>7.1f} {x['vx_q']:>+8.4f} "
          f"{x['cons']:>5.2f} {x['corr']:>+9.2f} {x['n_fall']:>3}/{x['n_to']:<2}")
  print("-" * 78)

  ratio = np.array([x["ratio"] for x in rows], dtype=float)
  corr = np.array([x["corr"] for x in rows], dtype=float)
  c_delta = np.array([x["corr_delta"] for x in rows], dtype=float)
  vxq = np.array([x["vx_q"] for x in rows], dtype=float)
  cons = np.array([x["cons"] for x in rows], dtype=float)
  qn = np.array([x["q_n"] for x in rows], dtype=float)
  c_head = np.array([x["corr_head"] for x in rows], dtype=float)

  def _m(a):
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

  def _p(x, fmt="+.2f"):
    return "n/a" if not np.isfinite(x) else format(x, fmt)

  p90 = np.nanpercentile(ratio, 90) if np.isfinite(ratio).any() else float("nan")
  print(f"漂移比 |dx|/Δy  : mean {_p(_m(ratio), '.3f')}  p90 {_p(p90, '.3f')}  m/m")
  print(f"安静段 vx       : mean {_p(_m(vxq), '+.4f')}  |vx| {_p(_m(np.abs(vxq)), '.4f')} m/s "
        f"(安静样本 {int(np.nansum(qn))})")
  print(f"c(vx, yaw_rel)  : 每 env 均值 {_p(_m(corr))}  |c|>0.5 的 env "
        f"{int(np.nansum(np.abs(corr) > 0.5))}/{n_env}   <- 主判别量")
  print(f"c(Δdx, Δyaw)    : 每 env 均值 {_p(_m(c_delta))}  "
        f"(原方案用的这个；差分会消掉慢变共模，即便完美耦合也接近 0，别拿它下结论)")
  print(f"c(速度方向,朝向) : {_p(_m(c_head))}  "
        f"(|c|≈1 = 速度矢量跟着朝向转 = 策略在 heading 系里追指令)")
  n_fall = int(sum(x["n_fall"] for x in rows))
  n_to = int(sum(x["n_to"] for x in rows))
  print(f"摔倒 / 超时     : {n_fall} / {n_to} 次"
        + ("   ← 一次没摔（超时是正常跑完，不是失败）" if n_fall == 0 else ""))

  # ---- 判定 ----
  print("-" * 78)
  abs_vxq = _m(np.abs(vxq))
  c_abs = abs(_m(corr))
  if not np.isfinite(c_abs):
    verdict = "无法判定"
    why = ("朝向或速度读不到（t≈0 且 yaw 恒定 / 全静止），c 全是 nan。"
           "检查 _root_yaw 是否取到四元数、reset 后 yaw 是否落在 ±0.15 rad 内。")
  elif np.isfinite(ratio).all() and np.nanmax(ratio) < 0.08:
    verdict = "已达标"
    why = "漂移比 < 0.08，不需要改任何东西；把漂移比加进训练日志防回归即可。"
  elif _m(qn) < 40:
    verdict = "A（推定）"
    why = "几乎没有「安静段」样本（策略一开局就在偏），无法分离步态项 -> 按 A 处理。"
  elif c_abs > 0.5 and abs_vxq < 0.02:
    verdict = "A 偏航耦合"
    why = ("vx 与朝向强相关（|c|>0.5）而朝向没歪时 vx≈0 -> 横向偏移是偏航带出来的。"
           "观测里没有 heading，改奖励治不好，必须补 heading 观测。")
  elif c_abs < 0.3 and abs_vxq >= 0.02 and _m(cons) > 0.7:
    verdict = "B 步态/残差不对称"
    why = ("朝向稳定时仍以固定符号横向漂移、且与朝向无关 -> 左右腿出力不等。"
           "在奖励侧补 lin_vel_x_zero（lateral_drift 已有，但二次项在 0.2 m 处只有 -0.008/步，太弱）。")
  else:
    verdict = "混合（A+B）或 C"
    why = ("两个特征都不干净。先看 --push 那份报告：恢复慢 / 单向累积 -> C；"
           "否则按 A 先补观测（观测缺失无法用奖励补偿）。")

  print(f"判定: {verdict}")
  print(f"依据: {why}")

  if args.push:
    print("-" * 78)
    _push_report(d, args.skip)
  print("=" * 78)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
