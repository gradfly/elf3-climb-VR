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
    """世界系 +Y 的推进速度（正数才给分），堵住「扭身横漂也拿分」。"""
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
    r = torch.minimum(torch.clamp(v, min=0.0), vmax)
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
    """sin(yaw)^2：转身就罚。当前「步态怪异」的主要嫌疑之一。"""
    try:
        q = _root_quat(env)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.sin(yaw) ** 2
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)


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
