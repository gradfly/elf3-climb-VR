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
import math
import statistics
import sys
from dataclasses import asdict

import mujoco
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


def _root_yaw(env):
  """基座朝向 yaw（rad）。取不到四元数就返回 None。"""
  d = env.scene["robot"].data
  q = None
  for name in ("root_link_quat_w", "root_quat_w", "root_link_quat"):
    v = getattr(d, name, None)
    if v is not None:
      q = v
      break
  if q is None:
    return None
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _root_yaw_rate(env):
  """世界系 yaw 角速度 ω_z（rad/s）—— 判断「螃蟹步 vs 转身走」最直接的量。"""
  d = env.scene["robot"].data
  for name in ("root_link_ang_vel_w", "root_ang_vel_w"):
    v = getattr(d, name, None)
    if v is not None:
      return v[:, 2]
  return None


def _pin_command(base, vec, name: str = "twist") -> bool:
  """把指令张量**就地**写死。

  为什么要这一步：`UniformVelocityCommand` 每 3~8 s 会 `_resample_command` 一次，
  而且 play 配置为了 viser 滑块把 `lin_vel_x` 放宽到 (0, 0.1)、`ang_vel_z` 放宽到
  (0, 0.1)。测 X 向漂移时这两条都必须在**结构上**关掉，不能靠"记得别开"。
  这里 cfg 层已经把范围钉成常数，函数内再每步覆写一次，防止别的版本仍在重采样。
  """
  try:
    term = base.command_manager.get_term(name)
  except Exception:
    return False
  for attr in ("command", "_command", "vel_command", "cmd"):
    t = getattr(term, attr, None)
    if torch.is_tensor(t) and t.dim() == 2 and t.shape[-1] >= vec.numel():
      try:
        # 只写前 3 列：有的版本指令是 (N,4)（多一个 heading），末尾那列留着不动。
        t[:, :vec.numel()] = vec
        return True
      except Exception:
        continue
  return False


def _disable_curriculum(base) -> None:
  """把课程计算哑掉。

  否则 `env.reset()` 内部会调 `curriculum_manager.compute()` 按上一回合表现
  重新分配地形等级，`--terrain-level` 刚设的值立刻被冲掉
  （实测：钉 6 级，跑完显示 min 4 / max 8）。
  """
  cm = getattr(base, "curriculum_manager", None)
  if cm is None:
    print("[warn] 找不到 curriculum_manager，无法禁用课程；地形等级可能被重置")
    return
  if getattr(cm, "_diag_disabled", False):
    return
  try:
    cm.compute = lambda *a, **k: {}      # type: ignore[method-assign]
    cm._diag_disabled = True
    print("[info] 已禁用 curriculum（否则 reset 会把地形等级重算回去）")
  except Exception as e:
    print(f"[warn] 禁用 curriculum 失败：{e}")


def _bind_camera_to_robot(base, env_i: int = 0, *, dist: float = 3.0,
                          elev: float = -45.0, azim: float = 90.0):
  """把离屏相机绑到机器人身上（跟拍，俯视 45°）。

  旧实现是「钉在世界系出生点」的固定机位，两个问题让录出来的画面里没有机器人：
    1) mjlab 的 OffscreenRenderer 只渲染主 env（`cfg.viewer.env_idx`，默认 0），
       旧代码却把 lookat 对准 `env_origins[env_i]` —— 用 `--record-env 1/2/3`
       时镜头对着 env2 的地形块，画面上却是 env0 的空地；
    2) 就算 env 对上了，机位也只设一次，机器人沿 +Y 爬 20 s 早就走出画面。
  这里改用 MuJoCo 的跟踪相机（mjCAMERA_TRACKING）：每帧 update_scene 时
  lookat 自动取 root body 的世界位置，机器人始终在画面中心。

  elevation 的符号（MuJoCo 约定，已实测）：-45 = 相机在斜上方 45° 俯视；
  +45 = 相机在下方 45° 仰视。

  返回 (cam, tracking_ok)。tracking_ok=False 时调用方要逐帧把 cam.lookat
  挪到机器人身上（这个版本的渲染器设不了跟踪相机时的退路）。
  """
  r = getattr(base, "_offline_renderer", None)
  cam = getattr(r, "_cam", None) if r is not None else None
  if cam is None:
    print("[warn] 拿不到离屏渲染器相机（需要 render_mode='rgb_array'），录像不可用")
    return None, False

  cam.distance = float(dist)
  cam.elevation = float(elev)
  cam.azimuth = float(azim)
  try:
    body_id = int(base.scene["robot"].indexing.root_body_id)
    cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING.value)
    cam.fixedcamid = -1
    cam.trackbodyid = body_id
  except Exception as e:
    print(f"[warn] 设跟踪相机失败（{e}），退化为逐帧 lookat 跟拍")
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE.value)
    cam.trackbodyid = -1
    return cam, False

  print(f"[info] 相机 -> 跟拍 env{env_i} root body {body_id}"
        f"（dist={cam.distance} elev={cam.elevation} azim={cam.azimuth}）")
  return cam, True


def _write_video(path: str, frames, fps: float) -> None:
  if not frames:
    print("[warn] 没收到任何帧，不写视频（是不是没开 render_mode='rgb_array'？）")
    return
  try:
    import mediapy as media
    media.write_video(path, frames, fps=int(round(fps)))
  except Exception:
    try:
      import imageio.v2 as iio
      iio.mimwrite(path, frames, fps=fps, macro_block_size=1)
    except Exception as e:
      print(f"[warn] 写 mp4 失败（{e}）；退而存 npz")
      import numpy as np
      np.savez_compressed(path + ".npz", frames=np.stack(frames))
      print(f"[info] 已保存 {path}.npz（{len(frames)} 帧）")
      return
  print(f"[info] 视频已保存: {path}  ({len(frames)} 帧 @ {int(round(fps))} fps)")


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


def _dump_scan(base, term, env_i: int = 0) -> None:
  """把 height_scan 原样打印出来，用来判断地形估计链路到底断在哪一环。

  只有三种可能，看这张表就能分开：
    1) 绝大多数格子是 5.0 -> 射线全 miss（geom group 被排除 / 传感器没接上）
    2) 剖面是一串台阶 -> 数据没问题，是估计器（检测带/阈值/广播）的问题
    3) 剖面完全平 -> 这个 env 真的站在平地上（地形等级 0 或平台太宽）
  """
  print("-" * 62)
  print(f"[scan] 取数：shape={tuple(term.scan_shape) or '—'} scan_ok={term.scan_ok}")
  if not term.scan_ok:
    print("[scan] ❌ 地形估计链路没跑通 —— 这就是 rise 恒为 0 的原因，"
          "不是地形问题。看上面有没有 [residual][ERR]/[warn] 那行。")

  h = term._read_heights()
  if h is None:
    print("[scan] _read_heights() 返回 None（传感器读不到，见上方 warning）")
    return
  nx = int(round(term.cfg.scan_size[0] / term.cfg.scan_resolution)) + 1
  ny = int(round(term.cfg.scan_size[1] / term.cfg.scan_resolution)) + 1
  if h.shape[-1] != nx * ny:
    print(f"[scan] ❌ 展平长度 {h.shape[-1]} != {nx}x{ny}={nx * ny}；h.shape={tuple(h.shape)}"
          " -> 典型 shape 广播 bug")
    return

  H = h[env_i].view(ny, nx)
  print(f"[scan] env{env_i} 高度场（单位 cm，行=y 从 -1.0 到 +1.0，列=x 从 -0.6 到 +0.6）")
  ys = [-1.0 + i * 0.1 for i in range(ny)]
  print("        x:" + "".join(f"{-0.6 + j * 0.1:+5.1f}" for j in range(nx)))
  for i in range(ny):
    row = "".join(f"{float(v) * 100:5.1f}" for v in H[i])
    print(f"  y={ys[i]:+.1f} {row}")

  miss = (H > 4.9).to(torch.float32).mean()
  print(f"[scan] 未命中比例 {float(miss) * 100:.0f}%（≈100% 说明射线没打到地形）")

  # 估计器内部量（与 ref_table.estimate 同一套算法，逐步打印便于定位）
  mid = nx // 2
  prof = H[:, max(0, mid - 1):mid + 2].median(dim=-1).values
  print("[scan] 中央 x 剖面的 y 方向高度（cm）与相邻差 dh（正=沿 +y 抬升）：")
  dh = prof[:-1] - prof[1:]
  for i in range(ny - 1):
    flag = "  <-- 上升边" if float(dh[i]) > 0.03 else ""
    print(f"   y={ys[i]:+.1f}->{ys[i + 1]:+.1f}  h={float(prof[i]) * 100:7.2f}"
          f"  dh={float(dh[i]) * 100:+6.2f}{flag}")

  raw = term.raw_rise
  val = term.rise_valid
  if raw is not None and val is not None:
    print(f"[scan] estimate -> raw_rise={float(raw[env_i]):.3f} m "
          f"valid={bool(val[env_i])}  (在检测带内找到上升边={bool(val[env_i])})")
  print(f"[scan] 机器人 root=({float(_root_pos(base)[env_i, 0]):+.2f}, "
        f"{float(_root_pos(base)[env_i, 1]):+.2f}, "
        f"{float(_root_pos(base)[env_i, 2]):+.2f})")
  terr = getattr(getattr(base, "scene", None), "terrain", None)
  for attr in ("terrain_levels", "levels"):
    v = getattr(terr, attr, None)
    if v is not None and torch.is_tensor(v):
      print(f"[scan] 地形等级 env{env_i} = {int(v[env_i])}")
      break
  print("-" * 62)


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
  ap.add_argument("--dump-scan", action="store_true",
                  help="打印 height_scan 原始高度场 + 估计器内部量，用于定位 rise 恒为 0")
  ap.add_argument("--dump-env", type=int, default=0, help="--dump-scan 打印哪个 env")
  ap.add_argument("--record", default=None, metavar="OUT.mp4",
                  help="录 mp4（相机跟拍机器人，俯视 45°）；需要 EGL")
  ap.add_argument("--record-env", type=int, default=0, help="录哪个 env")
  ap.add_argument("--record-every", type=int, default=1, help="每 N 步取一帧")
  ap.add_argument("--record-dist", type=float, default=3.0,
                  help="跟拍距离（越大机器人越小）")
  ap.add_argument("--record-elev", type=float, default=-45.0,
                  help="俯仰角，负值=从上往下看（-45 即俯视 45°）")
  ap.add_argument("--record-azim", type=float, default=90.0,
                  help="绕机器人的方位角")
  ap.add_argument("--trace", default=None, metavar="OUT.npz",
                  help="逐步轨迹落盘（dx/dy/yaw/vx/vy），用于判定漂移属于哪一类")
  ap.add_argument("--freeze-command", type=float, default=None, metavar="V_Y",
                  help="把速度指令钉死为 (0, V_Y, 0)：不重采样、不站立、不转向。"
                       "**测 X 向漂移必须开**——play 配置里 lin_vel_x=(0,0.1)、"
                       "ang_vel_z=(0,0.1) 会被随机采样，否则净 Δx 主要是指令驱动的")
  ap.add_argument("--push", action="store_true",
                  help="保留扰动（play 默认把 push_robot pop 掉了）")
  ap.add_argument("--train-env", action="store_true",
                  help="用训练配置（play=False）建环境，而不是 play 配置。"
                       "只有当 --terrain-level 在本版本上不改变地形高度时才需要它"
                       "（训练配置的 terrain_generator.curriculum=True，行=难度梯度）")
  args = ap.parse_args(argv)

  import mjlab.tasks  # noqa: F401  触发任务注册
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  try:
    env_cfg = load_env_cfg(args.task, play=not args.train_env)
    agent_cfg = load_rl_cfg(args.task)
  except KeyError:
    from mjlab.tasks.registry import list_tasks
    have = sorted(t for t in list_tasks() if "Crab" in t)
    print(f"[ERR] 任务 {args.task!r} 没注册。当前 Crab 任务：{have or '（一个都没有）'}")
    print("      原因通常是 venv 里那份任务包是旧的（install_task.sh 以前是 cp 复制）。")
    print("      先执行：bash train/install_task.sh")
    return 2

  env_cfg.scene.num_envs = args.num_envs
  if args.train_env:
    # 训练配置自带观测噪声和 push_robot（_retune_safety 已把它调成 3~6 s / ±0.2）。
    # 量化漂移时这两样都不是我们要测的变量：噪声关掉，push 按需保留。
    try:
      env_cfg.observations["actor"].enable_corruption = False
    except Exception as e:
      print(f"[warn] 关观测噪声失败（{e}），数字会带一点噪声抖动")
    if not args.push:
      try:
        env_cfg.events.pop("push_robot", None)
      except Exception:
        pass
    print("[info] 使用训练配置（play=False）：episode 20 s、无观测噪声、"
          f"{'保留' if args.push else '去掉'} push_robot")
  if args.record:
    # 离屏渲染器只画主 env（cfg.viewer.env_idx），跟踪相机跟的也是这一份数据。
    # 不同步的话 `--record-env 2` 会录成「镜头对着 env2、画面里是 env0 的空地」。
    viewer = getattr(env_cfg, "viewer", None)
    if viewer is not None and hasattr(viewer, "env_idx"):
      viewer.env_idx = int(args.record_env)

  if args.trace and args.freeze_command is None:
    print("[warn] 开了 --trace 但没冻结指令。play 配置里 lin_vel_x=(0,0.1)、"
          "ang_vel_z=(0,0.1)，20 s 内净 Δx 平均 ~+0.05 m/s、偏航 ~+0.05 rad/s，"
          "量到的全是指令，不是漂移。加 --freeze-command 0.17 再跑。")

  if args.freeze_command is not None:
    # 三层保险：范围钉成常数 / 不重采样 / 不抽站立 env。函数内每步还会再覆写一次。
    tw = env_cfg.commands["twist"]
    v = float(args.freeze_command)
    for k, val in (("lin_vel_x", (0.0, 0.0)),
                   ("lin_vel_y", (v, v)),
                   ("ang_vel_z", (0.0, 0.0))):
      try:
        setattr(tw.ranges, k, val)
      except Exception as e:
        print(f"[warn] 冻结指令：ranges.{k} 设置失败（{e}）")
    for k, val in (("resampling_time_range", (1e9, 1e9)),
                   ("rel_standing_envs", 0.0)):
      try:
        setattr(tw, k, val)
      except Exception as e:
        print(f"[warn] 冻结指令：{k} 设置失败（{e}）")
    print(f"[info] 指令已冻结为 (0, {v}, 0)：不重采样 / 不站立 / 不转向")

  push_state: dict[str, object] = {"buf": None}
  if args.push:
    # play 配置会 pop("push_robot")（env_cfg.py），评估抗扰漂移时在这里补回来。
    # 幅度与 env_cfg_residual._retune_safety 一致：3~6 s 一次，x/y ±0.2 m/s，yaw ±0.1 rad/s。
    #
    # 另外包一层记录：哪一步、哪些 env 被推了。没有这个时间戳，「恢复时间」就只能靠猜
    # （每 3~6 s 推一次、方向随机，事后无法把某段 dx 归因到某次推挤）。
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.tasks.velocity import mdp as vm
    _orig_push = vm.push_by_setting_velocity

    def _logged_push(*a, **k):
      buf = push_state["buf"]
      if buf is not None:
        ids = k.get("env_ids")
        if ids is None and len(a) > 1:
          ids = a[1]
        try:
          if ids is None:
            buf[:] = True
          else:
            buf[torch.as_tensor(ids, device=buf.device).long()] = True
        except Exception:
          pass
      return _orig_push(*a, **k)

    env_cfg.events["push_robot"] = EventTermCfg(
        func=_logged_push,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {
            "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
            "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15), "yaw": (-0.1, 0.1),
        }},
    )
    print("[info] 已恢复扰动 push_robot（3~6 s / ±0.2 m/s / yaw ±0.1），并记录推力时刻")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device,
                          render_mode="rgb_array" if args.record else None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  base = _unwrap(env)
  B = base.num_envs
  dt = float(getattr(base, "step_dt", 0.02))

  if push_state["buf"] is None and args.push:
    push_state["buf"] = torch.zeros(B, dtype=torch.bool, device=device)

  frozen_vec = None
  if args.freeze_command is not None:
    frozen_vec = torch.tensor([0.0, float(args.freeze_command), 0.0], device=device)

  def _pin():
    """在当前 obs 生成前把指令写死；返回是否成功。"""
    if frozen_vec is None:
      return True
    return _pin_command(base, frozen_vec)

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
    _disable_curriculum(base)
    if _force_terrain_level(base, args.terrain_level):
      out = env.reset()          # 让新等级在出生点上生效
      obs = out[0] if isinstance(out, tuple) else out
      # reset 内部按当前等级算好了 env_origins，但随后可能又把 terrain_levels
      # 改写回去（即使 curriculum 已哑掉）。再钉一次，让打印和实际地形块一致，
      # 否则会看到「钉 5 显示 min 1 / max 9」这种自相矛盾的读数。
      _force_terrain_level(base, args.terrain_level)
      print(f"[info] 地形等级钉在 {args.terrain_level}（已重新 reset）")
      try:
        tl = base.scene.terrain.terrain_levels
        print(f"[info] terrain_levels = min {int(tl.min())} / max {int(tl.max())}"
              f"  （若仍是 0，说明强制没生效）")
      except Exception:
        pass
    else:
      print("[warn] 这个版本拿不到 terrain_levels，按默认等级 0 跑")

  if not _pin():
    print("[warn] 拿不到 twist 指令项，--freeze-command 没生效（cmd 仍会被随机采样）")
  elif frozen_vec is not None:
    try:
      cm = base.command_manager.get_command("twist")
      print(f"[info] 冻结后实测 cmd（env0）= {[round(float(x), 4) for x in cm[0].tolist()]}"
            f" / 各 env 均值 {[round(float(x), 4) for x in cm.mean(0).tolist()]}")
    except Exception:
      pass

  cam = None
  cam_tracks = False
  if args.record:
    cam, cam_tracks = _bind_camera_to_robot(
        base, args.record_env, dist=args.record_dist,
        elev=args.record_elev, azim=args.record_azim)

  # ---- 指标累计（位移必须逐步累加，reset 那一步跳过）----
  ep_len: list[int] = []
  frames: list = []
  step_i = 0
  warned_render = False
  step_in_ep = torch.zeros(B, dtype=torch.long, device=device)
  net_dy = torch.zeros(B, device=device)
  net_dx = torch.zeros(B, device=device)
  vy_alive: list[float] = []
  vx_alive: list[float] = []
  cmd_list: list[float] = []
  z_list: list[float] = []
  prev = _root_pos(base)[:, :2].clone()
  x0 = _root_pos(base)[:, 0].clone()     # 本回合出生点 x（--trace 用，reset 那一步刷新）
  y0 = _root_pos(base)[:, 1].clone()     # 本回合出生点 y（漂移比的分母，必须同基准）
  prev_yaw = _root_yaw(base)          # None 表示这个版本读不到四元数
  yaw0 = prev_yaw.clone() if prev_yaw is not None else torch.zeros(B, device=device)
  yaw_ok = _root_yaw_rate(base) is not None
  yaw_rate: list[float] = []          # 存活期 |ω_z|
  yaw_travel = torch.zeros(B, device=device)   # 每 env 累计 |Δyaw|（rad）
  act_abs, act_cnt = 0.0, 0
  sat_sum, sat_cnt = 0.0, 0
  alive_steps = 0

  try:
    term = base.action_manager.get_term("joint_pos")
  except Exception:
    term = None

  if args.dump_scan:
    with torch.no_grad():
      obs = env.step(policy(obs))[0]      # 先走一步，让传感器的 hit_pos_w 是新的
    if term is None:
      print("[scan] 拿不到 joint_pos 动作项，--dump-scan 需要 ResidualGaitAction")
    else:
      term._update_terrain()              # 不等 terrain_refresh_steps，强制刷新
      _dump_scan(base, term, args.dump_env)

  # ---- 轨迹记录（--trace）----
  # 只落盘、不判断；判定逻辑在分析脚本里（见 docs 执行清单 §4）。
  rec: dict[str, list] = {}

  for _ in range(args.steps):
    _pin()                       # 每步写回：防某些版本的 compute() 仍在重采样
    if push_state["buf"] is not None:
      push_state["buf"].zero_()  # 清标记，step 内部若触发 interval 事件会置 True
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

    if args.record and (step_i % max(1, args.record_every) == 0):
      step_i += 1
      if cam is not None and not cam_tracks:
        # 退路：设不了跟踪相机时，逐帧把注视点挪到机器人身上
        try:
          cam.lookat[:] = _root_pos(base)[args.record_env].cpu().tolist()
        except Exception:
          pass
      try:
        fr = base.render()
        if fr is not None:
          frames.append(fr[args.record_env] if getattr(fr, "ndim", 0) == 4 else fr)
      except Exception as e:
        if not warned_render:
          print(f"[warn] 渲染失败，录像会中断：{e!r}")
          warned_render = True
        args.record = None
    else:
      step_i += 1

    dones_b = dones.bool()
    alive = ~dones_b

    push_b = torch.zeros(B, dtype=torch.bool, device=device)
    if push_state["buf"] is not None:
      push_b = push_state["buf"].clone() & alive    # 只认存活步：reset 那步别混进来

    p = _root_pos(base)[:, :2]
    d = (p - prev) / dt
    prev = p.clone()          # dones 那一步 prev 已是新出生点，下一步差分仍正确

    # 指令速度（每回合采样一次，8 个 env 抽样不同 -> 必须一起看，否则速度差会被误读）
    try:
      cmd_list.append(float(base.command_manager.get_command("twist")[:, 1].abs().mean()))
    except Exception:
      pass
    try:
      z_list.append(float(_root_pos(base)[0, 2]))    # 只看 env0：多 env 的地形块高度不同，均值无意义
    except Exception:
      pass
    if bool(alive.any()):
      vy_alive.extend(d[alive, 1].tolist())
      vx_alive.extend(d[alive, 0].tolist())
      alive_steps += int(alive.sum())

    # 朝向变化（螃蟹步的定义就是「朝向不变、侧向平移」，所以这个必须单独看）
    if yaw_ok:
      w = _root_yaw_rate(base)
      if bool(alive.any()):
        yaw_rate.extend(w[alive].abs().tolist())
      yaw_travel += torch.where(alive, w.abs() * dt, torch.zeros_like(w))
    elif prev_yaw is not None:          # 退路：用四元数差分
      yw = _root_yaw(base)
      dyaw = torch.atan2(torch.sin(yw - prev_yaw), torch.cos(yw - prev_yaw))
      if bool(alive.any()):
        yaw_rate.extend((dyaw[alive] / dt).abs().tolist())
      yaw_travel += torch.where(alive, dyaw.abs(), torch.zeros_like(dyaw))
      prev_yaw = yw

    # 位移累加用 d*dt（d 已排除重置那一步的跳变）
    net_dy += torch.where(alive, d[:, 1] * dt, torch.zeros_like(net_dy))
    net_dx += torch.where(alive, d[:, 0] * dt, torch.zeros_like(net_dx))

    if args.trace:
      # reset 那一步 p 已是新出生点：先刷新基线，于是该步 dx=dy=0（且 alive=False，分析时会屏蔽）
      yw_t = _root_yaw(base)
      yw_t = torch.zeros(B, device=device) if yw_t is None else yw_t
      wz_t = _root_yaw_rate(base)
      wz_t = torch.zeros(B, device=device) if wz_t is None else wz_t
      if bool(dones_b.any()):
        x0 = torch.where(dones_b, p[:, 0], x0)
        y0 = torch.where(dones_b, p[:, 1], y0)
        yaw0 = torch.where(dones_b, yw_t, yaw0)      # 初值 yaw 是 ±0.15 rad 随机的，必须扣除
      yaw_rel = torch.atan2(torch.sin(yw_t - yaw0), torch.cos(yw_t - yaw0))
      for key, val in (
          ("dx", p[:, 0] - x0),      # 本回合 X 偏移（世界系）
          ("dy_ep", p[:, 1] - y0),   # 本回合 Y 位移 —— 漂移比的分母，与 dx 同基准
          ("dy", net_dy),            # 跨回合累计 Y（老字段，保留兼容）
          ("yaw", yw_t),             # 世界系绝对 yaw
          ("yaw_rel", yaw_rel),      # 相对本回合出生朝向（判定用这个）
          ("ep_t", step_in_ep.to(torch.float32)),   # 本回合内第几步（跳过冷启动要按回合算）
          ("wz", wz_t), ("vx", d[:, 0]), ("vy", d[:, 1]),
          ("push", push_b),
          ("alive", alive)):
        rec.setdefault(key, []).append(val.detach().clone())

    step_in_ep += 1
    if bool(dones_b.any()):
      for k in torch.nonzero(dones_b, as_tuple=False).flatten().tolist():
        ep_len.append(int(step_in_ep[k].item()))
      step_in_ep[dones_b] = 0

  # ---- 打印 ----
  if args.record:
    _write_video(args.record, frames, 1.0 / dt)

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
      ok = getattr(term, "scan_ok", None)
      if ok is not None:
        print(f"地形估计链路   : {'OK' if ok else '❌ 断了 -> rise 永远 0，参考步态锁在平地档'}")
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
  if cmd_list:
    print(f"cmd_y (指令)   : mean {statistics.mean(cmd_list):.3f} m/s "
          f"(min {min(cmd_list):.3f} / max {max(cmd_list):.3f})")
    if vy_alive:
      print(f"跟踪比 v_y/cmd : {statistics.mean(vy_alive) / max(statistics.mean(cmd_list), 1e-6):.2f}"
            f"   <- 这才是真正的「走得动不动」，别只看 v_y 绝对值")
  if yaw_rate:
    yr = statistics.mean(yaw_rate)
    print(f"yaw 速率 |ω_z| : {yr:.3f} rad/s   <- 螃蟹步应 < 0.15；"
          f"{'✅ 朝向基本不变' if yr < 0.15 else '❌ 在自转，不是螃蟹步'}")
    deg = float(yaw_travel.mean()) * 180.0 / math.pi
    print(f"yaw 累计漂移   : {float(yaw_travel.mean()):.2f} rad ({deg:.0f}°) / {args.steps * dt:.0f}s")
  else:
    print("[warn] 读不到 root_link_ang_vel_w / root_link_quat_w，"
          "无法判断朝向是否变化（螃蟹步判不出来）")
  if z_list:
    print(f"基座 z         : {z_list[0]:.3f} -> {z_list[-1]:.3f} m "
          f"(Δ{z_list[-1] - z_list[0]:+.3f}；爬上台阶会升高)")
  t_run = args.steps * dt
  if frozen_vec is not None:
    try:
      cm = base.command_manager.get_command("twist").mean(0)
      ok_cmd = abs(float(cm[0])) < 1e-6 and abs(float(cm[2])) < 1e-6
      print(f"指令 cmd       : x {float(cm[0]):+.3f} / y {float(cm[1]):+.3f} / wz {float(cm[2]):+.3f}"
            f"  {'✅ 已冻结（Δx 不含指令成分）' if ok_cmd else '❌ 没冻住，下面的 Δx 不可用'}")
    except Exception:
      pass
  print(f"净 Δy 均值     : {float(net_dy.mean()):+.3f} m / {t_run:.1f}s "
        f"=> {float(net_dy.mean()) / t_run:+.3f} m/s")
  print(f"净 Δx 均值     : {float(net_dx.mean()):+.3f} m / {t_run:.1f}s "
        f"=> {float(net_dx.mean()) / t_run:+.3f} m/s")
  print(f"净 Δy 最差/最好: {float(net_dy.min()):+.2f} / {float(net_dy.max()):+.2f} m")

  if args.trace and rec:
    import os

    import numpy as np
    os.makedirs(os.path.dirname(os.path.abspath(args.trace)), exist_ok=True)
    arrays = {k: np.stack([t.cpu().numpy() for t in v]) for k, v in rec.items()}
    np.savez_compressed(args.trace, **arrays)

    dx, alive = arrays["dx"], arrays["alive"]
    dy_ep = arrays["dy_ep"] if "dy_ep" in arrays else arrays["dy"]
    n_steps, n_env = dx.shape
    idx = np.arange(n_steps)[:, None]
    mask = alive & (idx > 60)          # 跳过冷启动 1.2 s
    cnt = np.maximum(mask.sum(axis=0), 1)
    mean_abs = (np.abs(dx) * mask).sum(axis=0) / cnt

    # 「末值」取每个 env 最后一个存活步：直接取 dx[-1] 会撞上 reset 步（那一步 dx 被清零）
    last = np.where(mask, np.broadcast_to(idx, dx.shape), -1).max(axis=0)
    ok = last >= 0
    last_c = np.clip(last, 0, n_steps - 1)
    cols = np.arange(n_env)
    end_dx = np.where(ok, np.abs(dx[last_c, cols]), np.nan)
    end_dy = np.where(ok, np.abs(dy_ep[last_c, cols]), np.nan)
    ratio = np.where(end_dy > 0.05, end_dx / np.maximum(end_dy, 1e-3), np.nan)

    print(f"[trace] 已保存      : {args.trace}  ({n_steps} 步 × {n_env} env)")

    def _nm(a):
      """nanmean，但全 NaN 时不喷 RuntimeWarning。"""
      a = np.asarray(a, dtype=float)
      return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")

    fin = end_dx[np.isfinite(end_dx)]
    max_dx = float(fin.max()) if fin.size else float("nan")
    print(f"[trace] 末 |dx|      : mean {_nm(end_dx):.3f} m  max {max_dx:.3f} m")
    print(f"[trace] mean |dx|     : mean {mean_abs.mean():.3f} m")
    if np.isfinite(ratio).any():
      print(f"[trace] 漂移比 |dx|/Δy: mean {_nm(ratio):.3f} m/m "
            f"(p90 {np.nanpercentile(ratio, 90):.3f}，{int(np.isfinite(ratio).sum())}/{n_env} env 走过 >5cm)")
    else:
      print("[trace] 漂移比        : 分母全是 0（这些 env 根本没侧移，先看 track_ratio）")

    # 按地形类型分组：16 env 的列（flat/easy/mid/hard）是终身随机分配的，
    # 不分组的话 4 种台阶混在一个均值里，看起来像"漂移"，其实是"难度不同"。
    ttypes = None
    try:
      ttypes = np.asarray(base.scene.terrain.terrain_types.detach().cpu()).ravel()
    except Exception:
      try:
        ttypes = np.asarray(base.scene.terrain.terrain_types).ravel()
      except Exception:
        pass
    if ttypes is not None and ttypes.shape[0] == n_env:
      print(f"[trace] terrain_types : {ttypes.astype(int).tolist()}")
      for t in sorted(set(ttypes.astype(int).tolist())):
        sel = ttypes.astype(int) == t
        if sel.sum() == 0:
          continue
        print(f"[trace]   type {t} (n={int(sel.sum()):2d}) : "
              f"末|dx| {_nm(end_dx[sel]):.3f} m / "
              f"漂移比 {_nm(ratio[sel]):.3f} m/m / "
              f"Δy {_nm(end_dy[sel]):.3f} m")
    print(f"[trace] 深度分析请跑: python scripts/analyze_trace.py {args.trace}")
  print("=" * 62)
  env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
