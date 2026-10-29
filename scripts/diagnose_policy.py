#!/usr/bin/env python
"""把「策略到底行不行」变成数字：无界面跑一段 play 环境并打印指标。

为什么需要它
------------
viser 里靠肉眼只能看到「站一会儿就倒」，无法区分：
  1) 加载的根本不是训练好的策略（没给 checkpoint-file / 给了很早的一轮）；
  2) 策略能活但走不快（残差被限幅卡死、参考步态与地形不符）；
  3) 策略确实没学会。
本脚本把 episode 长度、净位移、侧向速度、动作饱和比例、rise/tread 估计、α
一次性打出来，三者立刻可分。

⚠️ 一个坑：mjlab 的 env.step() 内部会先 auto-reset 再返回，所以 `dones=True`
那一步读到的 `root_pos` 已经是**新地形块**的出生点（pyramid 地形每块原点相隔
好几米）。因此位移必须**逐步累加并排除重置那一步**，不能拿"末位置-初位置"。

用法（服务器上，仓库根目录）
---------------------------
    .venv/bin/python scripts/diagnose_policy.py \
        --task Elf3-Crab-Residual-Play \
        --checkpoint logs/rsl_rl/elf3_crab/<run>/model_XXXX.pt \
        --num-envs 8 --steps 1000

    # 对照 1：什么都不加载（等价 viser 里的 --agent zero）
    .venv/bin/python scripts/diagnose_policy.py --zero

    # 对照 2：强制站在第 6 级地形（有真台阶），看会不会爬
    .venv/bin/python scripts/diagnose_policy.py --terrain-level 6 ...

判读
----
* `净 Δy / 时长` 与 `v_y` 接近指令 0.15~0.24 -> 真在横着走；
* `v_y` 只有指令一半、`饱和比例` 接近 1 -> 残差被限幅顶死，控制退化成 bang-bang；
* `ep_len` ~2 s 且 `v_y` ~0 -> 和 --zero 一样 -> 参考步态在开环播放。
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


def _describe_obs(obs) -> str:
  """obs 可能是 TensorDict / Dict[str, Tensor] / 单个张量，按类型分别描述。"""
  if hasattr(obs, "keys"):
    try:
      parts = []
      for k in obs.keys():
        v = obs[k]
        parts.append(f"{k}{tuple(v.shape)}")
      return "dict{" + ", ".join(parts) + "}"
    except Exception:
      pass
  shape = getattr(obs, "shape", None)
  return str(tuple(shape)) if shape is not None else type(obs).__name__


def _force_terrain_level(base, level: int) -> bool:
  """把每个 env 的地形等级钉住（best-effort，字段名随版本变）。"""
  terr = getattr(getattr(base, "scene", None), "terrain", None)
  if terr is None:
    return False
  for attr in ("terrain_levels", "levels", "difficulty"):
    v = getattr(terr, attr, None)
    if v is not None and torch.is_tensor(v):
      v[:] = level
      return True
  return False


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Elf3-Crab-Residual-Play")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--zero", action="store_true", help="零动作（对照 viser --agent zero）")
  ap.add_argument("--num-envs", type=int, default=8)
  ap.add_argument("--steps", type=int, default=1000)
  ap.add_argument("--terrain-level", type=int, default=None,
                  help="钉住地形等级（0=平甲板；越大台阶越高）")
  ap.add_argument("--device", default=None)
  args = ap.parse_args(argv)

  import mjlab.tasks  # noqa: F401  触发任务注册
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  try:
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
  except KeyError:
    from mjlab.tasks.registry import list_tasks
    have = sorted(t for t in list_tasks() if "Crab" in t)
    print(f"[ERR] 任务 {args.task!r} 没注册。当前 Crab 任务：{have or '（一个都没有）'}")
    print("      原因通常是 venv 里那份任务包是旧的（install_task.sh 以前是 cp 复制）。")
    print("      先执行：bash train/install_task.sh")
    return 2

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

  if args.terrain_level is not None:
    if _force_terrain_level(base, args.terrain_level):
      out = env.reset()          # 让新等级在出生点上生效
      obs = out[0] if isinstance(out, tuple) else out
      print(f"[info] 地形等级钉在 {args.terrain_level}（已重新 reset）")
    else:
      print("[warn] 这个版本拿不到 terrain_levels，按默认等级 0 跑")

  # ---- 指标累计（位移必须逐步累加，reset 那一步跳过）----
  ep_len: list[int] = []
  step_in_ep = torch.zeros(B, dtype=torch.long, device=device)
  net_dy = torch.zeros(B, device=device)
  net_dx = torch.zeros(B, device=device)
  vy_alive: list[float] = []
  vx_alive: list[float] = []
  prev = _root_pos(base)[:, :2].clone()
  act_abs, act_cnt = 0.0, 0
  sat_sum, sat_cnt = 0.0, 0
  alive_steps = 0

  try:
    term = base.action_manager.get_term("joint_pos")
  except Exception:
    term = None

  for _ in range(args.steps):
    with torch.no_grad():
      actions = policy(obs)

    # 动作饱和比例：|a| 超过残差限幅折算到动作空间的比例
    if term is not None:
      try:
        clip_a = term._clip_action
        sat_sum += float((actions.abs() > clip_a).to(torch.float32).mean())
        sat_cnt += 1
      except Exception:
        pass
    act_abs += float(actions.abs().mean())
    act_cnt += 1

    out = env.step(actions)
    if len(out) == 4:
      obs, _rew, dones, _info = out
    else:
      obs, _rew, dones = out

    dones_b = dones.bool()
    alive = ~dones_b

    p = _root_pos(base)[:, :2]
    d = (p - prev) / dt
    prev = p.clone()          # dones 那一步 prev 已是新出生点，下一步差分仍正确
    if bool(alive.any()):
      vy_alive.extend(d[alive, 1].tolist())
      vx_alive.extend(d[alive, 0].tolist())
      alive_steps += int(alive.sum())

    # 位移累加用 d*dt（d 已排除重置那一步的跳变）
    net_dy += torch.where(alive, d[:, 1] * dt, torch.zeros_like(net_dy))
    net_dx += torch.where(alive, d[:, 0] * dt, torch.zeros_like(net_dx))

    step_in_ep += 1
    if bool(dones_b.any()):
      for k in torch.nonzero(dones_b, as_tuple=False).flatten().tolist():
        ep_len.append(int(step_in_ep[k].item()))
      step_in_ep[dones_b] = 0

  # ---- 打印 ----
  steps_full = int(round(20.0 / dt))
  print("=" * 62)
  print(f"task           : {args.task}")
  print(f"num_envs/steps : {B} / {args.steps}  (dt={dt:.4f}s)")
  print(f"obs            : {_describe_obs(obs)}")
  print(f"action_space   : {tuple(env.action_space.shape)}")
  print(f"mean |action|  : {act_abs / max(act_cnt, 1):.3f}")
  if sat_cnt:
    print(f"残差饱和比例   : {sat_sum / sat_cnt * 100:.0f}%  (|a| 超过限幅的维度占比)")

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

  print("-" * 62)
  if ep_len:
    med = statistics.median(ep_len)
    print(f"episodes       : {len(ep_len)} / {B}  (摔倒/超时合计)")
    print(f"ep_len  median : {med:.0f} steps = {med * dt:.2f} s")
    print(f"ep_len  max    : {max(ep_len)} steps = {max(ep_len) * dt:.2f} s")
  else:
    print(f"episodes       : 0 / {B}  (全部活到脚本结束 = 没摔)")
  print(f"存活步数合计   : {alive_steps}  (= {alive_steps} x {dt}s)")

  if vy_alive:
    print(f"v_y (存活期)   : mean {statistics.mean(vy_alive):+.3f} m/s "
          f"(中位 {statistics.median(vy_alive):+.3f})")
    print(f"v_x (存活期)   : mean {statistics.mean(vx_alive):+.3f} m/s (应接近 0)")
  t_run = args.steps * dt
  print(f"净 Δy 均值     : {float(net_dy.mean()):+.3f} m / {t_run:.1f}s "
        f"=> {float(net_dy.mean()) / t_run:+.3f} m/s")
  print(f"净 Δx 均值     : {float(net_dx.mean()):+.3f} m / {t_run:.1f}s "
        f"=> {float(net_dx.mean()) / t_run:+.3f} m/s")
  print(f"净 Δy 最差/最好: {float(net_dy.min()):+.2f} / {float(net_dy.max()):+.2f} m")
  print("=" * 62)
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
