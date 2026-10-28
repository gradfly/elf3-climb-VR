#!/usr/bin/env python
"""把「策略到底行不行」变成数字：无界面跑一段 play 环境并打印指标。

为什么需要它
------------
viser 里靠肉眼只能看到「站一会儿就倒」，无法区分下面三种情况：
  1) 加载的根本不是训练好的策略（没给 checkpoint / 给了很早的一轮）；
  2) 策略被加载了但参考步态与训练不一致（α、rise 估计、指令分布变了）；
  3) 策略确实没学会。
本脚本把 episode 长度、侧向位移、摔倒次数、rise/tread 估计、α、动作幅度
一次性打出来，三者立刻可分。

用法（服务器上，仓库根目录）
---------------------------
    .venv/bin/python scripts/diagnose_policy.py \
        --task Elf3-Crab-Residual-Play \
        --checkpoint logs/rsl_rl/elf3_crab_residual/<run>/model_6000.pt \
        --num-envs 8 --steps 1200

    # 什么都不加载，纯零动作（对照：这就是 viser 里 --agent zero 看到的）
    .venv/bin/python scripts/diagnose_policy.py --task Elf3-Crab-Residual-Play --zero

判读
----
* `obs_dim` 必须等于 actor 观测维度（残差任务 393），`action_dim` 29；
* `ep_len` 中位数 ~20 s、`falls` 少 -> 策略是好的，viser 里那次是加载错了；
* `ep_len` ~2 s、`v_y` ~0 -> 和 --zero 一样 -> 参考步态在开环播放，策略没生效；
* `ep_len` ~2 s 但 `v_y` 明显 >0 -> 策略真没学会平衡。
"""
from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import asdict

import torch


def _unwrap(env):
  seen = 0
  while hasattr(env, "unwrapped") and seen < 5:
    inner = env.unwrapped
    if inner is env:
      break
    env = inner
    seen += 1
  return env


def _root_pos(env):
  d = env.scene["robot"].data
  for name in ("root_link_pos_w", "root_pos_w"):
    v = getattr(d, name, None)
    if v is not None:
      return v
  raise AttributeError("找不到 root pos")


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Elf3-Crab-Residual-Play")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--zero", action="store_true", help="零动作（对照 viser --agent zero）")
  ap.add_argument("--num-envs", type=int, default=8)
  ap.add_argument("--steps", type=int, default=1200)
  ap.add_argument("--device", default=None)
  ap.add_argument("--log-root", default="logs/rsl_rl")
  args = ap.parse_args(argv)

  import mjlab.tasks  # noqa: F401  触发任务注册
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  base = _unwrap(env)
  B = base.num_envs
  dt = float(getattr(base, "step_dt", 0.02))

  if args.zero or not args.checkpoint:
    if not args.zero:
      print("[warn] 没给 --checkpoint，按零动作跑（等价 viser 的 --agent zero）")
    act_shape = env.action_space.shape

    class _Zero:
      def __call__(self, obs):
        del obs
        return torch.zeros(act_shape, device=device)

    policy = _Zero()
  else:
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)
    print(f"[info] 已加载 checkpoint: {args.checkpoint}")

  out = env.reset()
  obs = out[0] if isinstance(out, tuple) else out

  # ---- 指标累计 ----
  ep_len: list[int] = []
  ep_dy: list[float] = []
  ep_dx: list[float] = []
  vy_alive: list[float] = []
  step_in_ep = torch.zeros(B, dtype=torch.long, device=device)
  y0 = _root_pos(base)[:, 1].clone()
  x0 = _root_pos(base)[:, 0].clone()
  prev_y = y0.clone()
  act_abs, act_cnt = 0.0, 0

  try:
    term = base.action_manager.get_term("joint_pos")
  except Exception:
    term = None

  for i in range(args.steps):
    with torch.no_grad():
      actions = policy(obs)
    out = env.step(actions)
    if len(out) == 4:
      obs, _rew, dones, _info = out
    else:  # 老版本 rsl-rl
      obs, _rew, dones = out

    act_abs += float(actions.abs().mean())
    act_cnt += 1

    y = _root_pos(base)[:, 1]
    dy = (y - prev_y) / dt
    prev_y = y.clone()
    alive = ~dones.bool()
    if bool(alive.any()):
      vy_alive.extend(dy[alive].tolist())

    step_in_ep += 1
    if bool(dones.any()):
      d = dones.bool()
      for k in torch.nonzero(d, as_tuple=False).flatten().tolist():
        ep_len.append(int(step_in_ep[k].item()))
        ep_dy.append(float(y[k] - y0[k]))
        ep_dx.append(float(_root_pos(base)[k, 0] - x0[k]))
      step_in_ep[d] = 0
      y0 = torch.where(d, y, y0)
      x0 = torch.where(d, _root_pos(base)[:, 0], x0)

  # ---- 打印 ----
  steps_full = int(round(20.0 / dt))
  print("=" * 62)
  print(f"task           : {args.task}")
  print(f"num_envs/steps : {B} / {args.steps}  (dt={dt:.4f}s)")
  print(f"obs_dim        : {env.observation_space.shape}")
  print(f"action_dim     : {env.action_space.shape}")
  print(f"mean |action|  : {act_abs / max(act_cnt, 1):.3f}")

  if term is not None:
    try:
      print(f"ref_alpha      : {term.ref_alpha:.3f}")
      print(f"rise_est       : {float(term.rise_est.mean()):.3f} m "
            f"(min {float(term.rise_est.min()):.3f} / max {float(term.rise_est.max()):.3f})")
      print(f"tread_est      : {float(term.tread_est.mean()):.3f} m")
      print(f"ref_v          : {float(term.ref_v.mean()):.3f} m/s")
      print(f"phase          : {float(term.effective_phase.mean()):.3f}")
    except Exception as e:
      print(f"[warn] 读不到动作项内部量：{e}")

  if ep_len:
    med = statistics.median(ep_len)
    print("-" * 62)
    print(f"episodes       : {len(ep_len)}  (摔倒/超时合计)")
    print(f"ep_len  median : {med:.0f} steps = {med * dt:.2f} s")
    print(f"ep_len  mean   : {statistics.mean(ep_len):.0f} steps = "
          f"{statistics.mean(ep_len) * dt:.2f} s")
    print(f"ep_len  max    : {max(ep_len)} steps = {max(ep_len) * dt:.2f} s")
    full = sum(1 for n in ep_len if n >= steps_full - 10) / len(ep_len)
    print(f"活满 20s 比例  : {full * 100:.0f}%")
    print(f"每回合 Δy      : mean {statistics.mean(ep_dy):+.3f} m "
          f"(期望 = v_cmd × ep_len ≈ +{0.195 * med * dt:.2f})")
    print(f"每回合 Δx      : mean {statistics.mean(ep_dx):+.3f} m (应接近 0)")
  if vy_alive:
    print(f"存活期 v_y     : mean {statistics.mean(vy_alive):+.3f} m/s")
  print("=" * 62)
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
