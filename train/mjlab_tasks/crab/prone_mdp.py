"""趴姿动作（站立->趴 / 爬行 / 趴->站立）专用的观测、奖励、终止项。

和 `squat_mdp.py` 的分工
------------------------
关节跟踪、高度跟踪、躯干朝向跟踪、对称、存活这些函数与任务无关，直接复用
`squat_mdp`；这里只补**趴姿动作才有**的两件事：

* **水平位置跟踪**：爬行会从 x=0.14 走到 x=0.62，`base_anchor`（钉在原点）
  在这里等于惩罚正确行为，必须改成「和参考的 xy 比」；
* **趴姿的失败判据**：蹲起用 `base_too_low`（z<0.28）和 `bad_orientation`
  （倾角>55°），但趴姿的 z 本来就只有 0.10 m、躯干本来就是躺平的 ——
  这两条会直接把正常回合判死，所以改成「和参考的 z / xy 偏离过大」。
"""
from __future__ import annotations

import torch

from .mdp_extra import _root_pos
from .squat_mdp import _term, _zeros, traj_pitch_err, traj_ref_err


# ---------------------------------------------------------------- 水平位置
def base_xy_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """基座水平位置 - 参考水平位置 [B,2]（米）。"""
    t = _term(env, action_name)
    xy = t.ref_base_xy
    if xy is None:
        return torch.zeros(env.num_envs, 2, device=env.device)
    return _root_pos(env)[:, :2] - xy


def xy_track(env, action_name: str = "joint_pos", sigma: float = 0.10) -> torch.Tensor:
    """水平位置跟踪 exp 核。σ=0.10 m：偏 10 cm 还有 0.61 的梯度。

    爬行段参考位移 0.48 m（normal 档），如果只靠关节跟踪，策略完全可以
    「原地做爬行动作」—— 关节对得上但一步没走。这条是「真的走过去了」的唯一约束。
    """
    e = base_xy_err(env, action_name)
    mse = (e * e).mean(dim=-1)
    return torch.exp(-mse / (2.0 * sigma * sigma))


# ---------------------------------------------------------------- 终止
def traj_z_far(env, action_name: str = "joint_pos", max_err: float = 0.35) -> torch.Tensor:
    """基座高度偏离参考超过 max_err [B]。趴姿正常 z≈0.10，站立≈1.03。"""
    t = _term(env, action_name)
    e = _root_pos(env)[:, 2] - t.ref_base_z
    return e.abs() > float(max_err)


def traj_xy_far(env, action_name: str = "joint_pos", max_dist: float = 1.0) -> torch.Tensor:
    """水平位置偏离参考超过 max_dist [B]（爬行段 0.48 m，1.0 m 留足余量）。"""
    e = base_xy_err(env, action_name)
    return (e * e).sum(dim=-1) > float(max_dist) ** 2


# ---------------------------------------------------------------- 探针
def prone_probe(env, action_name: str = "joint_pos") -> torch.Tensor:
    """把跟踪质量写进 Metrics/*，不参与学习（权重 1e-9）。"""
    dev = _root_pos(env).device
    log = getattr(env, "extras", None)
    log = log.get("log") if isinstance(log, dict) else None
    try:
        term = _term(env, action_name)
    except Exception:
        return _zeros(env)

    def _put(key, t):
        if log is not None:
            log[key] = torch.as_tensor(t, device=dev).float().mean()

    try:
        err = traj_ref_err(env, action_name)
        leg = err[:, term.leg_pos]
        _put("Metrics/prone_progress", term.progress)
        _put("Metrics/prone_frames", term.frame_index.float())
        _put("Metrics/prone_z", _root_pos(env)[:, 2])
        _put("Metrics/prone_z_ref", term.ref_base_z)
        _put("Metrics/prone_xy_err", base_xy_err(env, action_name).norm(dim=-1))
        _put("Metrics/prone_err_rms_all", (err * err).mean(-1).sqrt())
        _put("Metrics/prone_err_rms_leg", (leg * leg).mean(-1).sqrt())
        _put("Metrics/prone_pitch_deg", torch.rad2deg(traj_pitch_err(env, action_name).squeeze(-1)))
    except Exception:
        pass
    return _zeros(env)
