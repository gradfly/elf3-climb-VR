"""残差任务专用的观测项与奖励项。

所有函数签名都是 mjlab 的 `func(env, **params)`。
读取 entity data 的字段名做了兼容（mjlab 版本间 root_link_* 与 root_* 有差异），
取不到时退化为零，避免整条训练因为一个字段名挂掉。
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------- 工具
def _term(env, action_name: str = "joint_pos"):
    return env.action_manager.get_term(action_name)


def _data(env):
    return env.scene["robot"].data


def _root_pos(env) -> torch.Tensor:
    d = _data(env)
    for name in ("root_link_pos_w", "root_pos_w"):
        v = getattr(d, name, None)
        if v is not None:
            return v
    raise AttributeError("entity.data 里找不到 root pos（root_link_pos_w / root_pos_w）")


def _root_quat(env) -> torch.Tensor:
    d = _data(env)
    for name in ("root_link_quat_w", "root_quat_w"):
        v = getattr(d, name, None)
        if v is not None:
            return v
    raise AttributeError("entity.data 里找不到 root quat")


def _zeros(env, n: int) -> torch.Tensor:
    return torch.zeros(env.num_envs, n, device=env.device)


# ---------------------------------------------------------------- 观测
def joint_pos_ref_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """12 条腿相对参考轨迹的偏差 [B,12]（按 target 顺序对齐，避开实体排序差异）。"""
    t = _term(env, action_name)
    ids = t.leg_pos
    q = _data(env).joint_pos[:, t.target_joint_ids]
    return q[:, ids] - t.ref_joint_pos[:, ids]


def phase_sincos(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _term(env, action_name).phase_sincos


def ref_meta(env, action_name: str = "joint_pos") -> torch.Tensor:
    """当前估计的 (rise, tread, v_ref)，做量纲归一化后给策略（都在 O(1) 量级）。"""
    t = _term(env, action_name)
    return torch.stack([t.rise_est * 5.0, t.tread_est * 2.5, t.ref_v * 5.0], dim=-1)


def ref_joint_pos(env, action_name: str = "joint_pos") -> torch.Tensor:
    """完整 29 维参考（只给 critic）。"""
    return _term(env, action_name).ref_joint_pos


# ---------------------------------------------------------------- 奖励
def forward_progress_y(env, command_name: str = "twist") -> torch.Tensor:
    """世界系 +Y 的推进速度（正数给分、倒退扣分），堵住「绕圈蹭分」。

    ⚠️ 2026-10-30 修：原来末尾是 `clamp(v, min=0.0)`，负向直接抹成 0。
    绕圈走的时候 dy 时正时负 —— 负的丢掉、正的照收，**净位移为 0 也拿满分**。
    实测（6000 轮 checkpoint，play 无扰动 20 s）：净位移只有 0.53 m（0.026 m/s），
    累计偏航 367°，而这条奖励报 0.113 m/s，差了 4 倍多。

    现在倒退按 30% 计负分：绕圈的期望收益归零，但也不至于把策略吓成不敢动
    （全额负分会让「原地站住」变成最优，那是另一个坑）。
    """
    y = _root_pos(env)[:, 1]
    prev = getattr(env, "_crab_prev_y", None)
    if prev is None or prev.shape != y.shape:
        env._crab_prev_y = y.clone()
        return torch.zeros_like(y)
    dy = y - prev
    env._crab_prev_y = y.clone()

    dt = float(getattr(env, "step_dt", 0.02))
    v = dy / max(dt, 1e-6)
    try:
        v_cmd = env.command_manager.get_command(command_name)[:, 1].abs()
        vmax = 1.5 * v_cmd
    except Exception:
        vmax = torch.full_like(v, 0.3)
    # clamp 要求 min/max 同为标量或同为张量；vmax 是逐环境张量，故用 minimum
    vmax = torch.as_tensor(vmax, dtype=v.dtype, device=v.device)
    v = torch.maximum(torch.minimum(v, vmax), -vmax)
    r = torch.where(v >= 0.0, v, 0.3 * v)      # 倒退只罚 30%
    # episode 刚 reset 的那一步 dy 跨了重置，丢弃
    if hasattr(env, "episode_length_buf"):
        r = torch.where(env.episode_length_buf <= 1, torch.zeros_like(r), r)
    return r


def ref_tracking(env, action_name: str = "joint_pos",
                 sigma: float = 0.35) -> torch.Tensor:
    """12 腿关节跟踪参考的 exp 核（用 exp 而非 L2，避免退化成行为克隆）。"""
    t = _term(env, action_name)
    ids = t.leg_pos
    q = _data(env).joint_pos[:, t.target_joint_ids]
    err = ((q[:, ids] - t.ref_joint_pos[:, ids]) ** 2).mean(dim=-1)
    return torch.exp(-err / (2.0 * sigma * sigma))


def yaw_drift(env) -> torch.Tensor:
    """帧间偏航角速度的平方 (rad/s)²：转圈要走不动。

    ⚠️ 2026-10-30 换实现。原来是 `sin(yaw)**2`，它对 yaw 和 yaw+2π 取值相同 ——
    **原地绕一整圈回到原点就「无罪」**。实测（6000 轮 checkpoint，play 20 s）
    累计偏航 367°，而这条只报 sin²=0.018（等价于净偏航 7.5°），等于完全没罚。

    现在看「每步的 yaw 变化」：先用 atan2(sinΔ, cosΔ) 把帧间差 wrap 到 [-π, π]
    （否则 ±π 处会跳变），再除 dt 得 rad/s，最后取平方 —— 平方让小抖动几乎不罚、
    大转动重罚，比绝对值更不容易把正常步态的轻微摆动也一起压死。
    """
    try:
        q = _root_quat(env)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)

    prev = getattr(env, "_crab_prev_yaw", None)
    if prev is None or prev.shape != yaw.shape:
        env._crab_prev_yaw = yaw.clone()
        return torch.zeros_like(yaw)
    d = torch.atan2(torch.sin(yaw - prev), torch.cos(yaw - prev))
    env._crab_prev_yaw = yaw.clone()
    # reset 那一步跨了 episode，丢弃
    if hasattr(env, "episode_length_buf"):
        d = torch.where(env.episode_length_buf <= 1, torch.zeros_like(d), d)
    dt = float(getattr(env, "step_dt", 0.02))
    return (d / max(dt, 1e-6)) ** 2


def lateral_drift(env) -> torch.Tensor:
    """(x - x0)^2：世界 +X 方向不该有位移。"""
    x = _root_pos(env)[:, 0]
    x0 = getattr(env, "_crab_x0", None)
    if x0 is None or x0.shape != x.shape:
        env._crab_x0 = x.clone()
        return torch.zeros_like(x)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        if bool(reset.any()):
            env._crab_x0 = torch.where(reset, x, env._crab_x0)
    return (x - env._crab_x0) ** 2


def is_terminated(env) -> torch.Tensor:
    """终止标志（一次性惩罚用）。字段名随版本不同，取不到就返回 0。"""
    for name in ("reset_terminated", "reset_buf", "terminated"):
        v = getattr(env, name, None)
        if v is not None:
            return v.to(torch.float32)
    return torch.zeros(env.num_envs, device=env.device)


def ref_probe(env, action_name: str = "joint_pos") -> torch.Tensor:
    """把参考步态/地形估计的内部量写进 `Metrics/*`，本身不参与学习。

    为什么需要它
    ------------
    `rise_est` 是「height_scan 到底有没有读到台阶」的唯一证据 —— 它恒为 0 时
    参考步态锁死在平地档，训练再久也是平地策略（2026-10-29 就这么白训了 12000 轮）。
    没有这一路指标就只能靠猜。

    为什么权重是 1e-9 而不是 0
    -------------------------
    mjlab 的 RewardManager 对 `weight == 0.0` 的项是 `continue` —— **连 func 都不调用**，
    所以写不进任何日志。用 1e-9：函数照跑、对 reward 的污染量级 1e-9 × 0.02，可忽略。
    """
    dev = _root_pos(env).device
    log = getattr(env, "extras", None)
    log = log.get("log") if isinstance(log, dict) else None
    try:
        term = _term(env, action_name)
    except Exception:
        return torch.zeros(env.num_envs, device=dev)

    def _put(key, t):
        if log is not None:
            log[key] = torch.as_tensor(t, device=dev).float().mean()

    try:
        _put("Metrics/scan_ok", torch.tensor(1.0 if term.scan_ok else 0.0, device=dev))
        _put("Metrics/rise_est", term.rise_est)
        _put("Metrics/tread_est", term.tread_est)
        if term.rise_valid is not None:
            _put("Metrics/rise_valid_frac", term.rise_valid.to(torch.float32))
        if term.raw_rise is not None:
            _put("Metrics/rise_raw", term.raw_rise)
        _put("Metrics/ref_alpha", torch.tensor(float(term.ref_alpha), device=dev))
    except Exception:
        pass
    return torch.zeros(env.num_envs, device=dev)


def terrain_levels_time(env, env_ids, promote_frac: float = 0.25,
                        demote_frac: float = 0.10, min_dist: float = 0.30,
                        **kwargs) -> dict:
    """按「活够久 + 真的挪了」升降地形，取代内置的「走够远」。

    为什么必须换掉内置 terrain_levels_vel
    ------------------------------------
    内置的判据是：

        time_left = max_episode_length_s - ep_len
        commanded_distance = |cmd| * time_left + 已走距离   # 约等于 |cmd| * 20 s
        move_up   = distance > 0.8 * commanded_distance
        move_down = distance < 0.5 * commanded_distance

    那段 `time_left` 的设计意图是「早摔不该减轻要求」，但后果是
    **摔得越早、要求反而越高**。实测 200 轮 ep_len = 4.2 s：

        要求走 0.8 × 3.8 m = 3.04 m，实际只走 0.42 m（|v_y| ≈ 0.1 m/s）

    于是 `move_up` 永远 False、`move_down` 永远 True -> 全部 env 钉死在
    level 0（flat，台阶 0~2 cm），策略一辈子见不到真台阶，只能学会在平地
    上踉跄。2026-10-29 之前那 12000 轮无效训练是同一个机理。

    现在的判据
    ---------
        升级：活到 max_episode_length 的 25%（5 s）**且**离开出生点 >= 0.30 m
        降级：活不到 10%（2 s）
        其余：保持不变

    升降不对称（25% vs 10%），避免偶尔摔一次就被打回 level 0。
    `min_dist` 那一条专门防「原地罚站也能升级」——站着不动拿不到难地形。

    ⚠️ 副作用：这条判据只保证「站得住」，不保证「走得远」。推进靠
    `forward_progress_y` 奖励项拉动；如果训练后 v_y 一直上不去，说明
    该加重它，而不是把这条课程判据改回距离。
    """
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    if terrain is None or not hasattr(terrain, "update_env_origins"):
        return {}

    step_dt = float(getattr(env, "step_dt", 0.02))
    t_max = float(getattr(env, "max_episode_length_s", 20.0))

    # 注意：本函数在 `_reset_idx` 里被调用，此时 episode_length_buf 还是
    # 「刚结束那一回合的长度」，尚未清零 —— 内置实现也是这么用的。
    t_alive = env.episode_length_buf[env_ids].to(torch.float32) * step_dt

    # 离开出生点的水平距离（env_origins 此刻还是本回合的出生点）
    try:
        pos = _root_pos(env)[env_ids, :2]
        origins = getattr(env.scene, "env_origins", None)
        if origins is not None:
            dist = torch.norm(pos - origins[env_ids, :2].to(pos.dtype), dim=1)
        else:
            dist = torch.zeros_like(t_alive)
    except Exception:
        dist = torch.zeros_like(t_alive)

    move_up = (t_alive >= t_max * float(promote_frac)) & (dist >= float(min_dist))
    move_down = (~move_up) & (t_alive < t_max * float(demote_frac))

    # 首次 reset（还没走过一步）时 distance 无意义，冻结，否则会把所有 env 平白升级
    if int(getattr(env, "common_step_counter", 1) or 0) == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    try:
        terrain.update_env_origins(env_ids, move_up, move_down)
    except TypeError as e:
        print(f"[curriculum][ERR] update_env_origins 调用失败（版本差异？）：{e}")
        return {}

    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}
