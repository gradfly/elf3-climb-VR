"""蹲下 / 起立跟踪任务专用的观测项、奖励项、终止项。

全部是 mjlab 的 `func(env, **params)` 形式。取数据的写法刻意与
`mdp_extra.py` 保持一致（同一套 `_term` / `_data` / `_root_*` 兼容层），
因为那些字段名在本版 mjlab 上已经实测可用。

跟踪任务的奖励设计（为什么不是「关节 L2 越小越好」）
--------------------------------------------------
轨迹是开环 IK 解，物理上不完全可行（PD 跟踪误差、足底接触抖动都在里面）。
所以：

* 关节项用 **exp 核**（`exp(-mean(e²)/2σ²)`），误差 0.3 rad 时仍有 ~0.6 的梯度，
  误差 1.5 rad 时自然饱和 —— 不会像 -‖e‖² 那样在早期把梯度全用在「别动」上；
* 腿和臂分开：腿负责站住，臂只是配重。臂跟不准不该拖累腿的分；
* **基座高度单独一项**：蹲多深是这个任务的真正目标，关节角跟对了但没蹲下去
  （或者塌下去）必须能被区分出来；
* **躯干朝向单独一项**：极限蹲是 18.5° 前倾，用 `upright`（要求竖直）会直接
  和任务打架，所以比的是「本体系重力反方向」与参考的差。
"""
from __future__ import annotations

import torch

from .mdp_extra import _data, _root_pos, _root_quat, is_terminated


# ---------------------------------------------------------------- 工具
def _term(env, action_name: str = "joint_pos"):
    return env.action_manager.get_term(action_name)


def _zeros(env, n: int = 1) -> torch.Tensor:
    return torch.zeros(env.num_envs, n, device=env.device)


def _up_body(env) -> torch.Tensor:
    """躯干「重力反方向」在本体系下的单位向量 [B,3]。

    与烘焙脚本里的 `up = R^T e_z` 同一公式（四元数 wxyz，R 为 world<-body）。
    比直接读欧拉角稳：不依赖 rx/ry/rz 的字段名与旋转顺序约定。
    """
    q = _root_quat(env)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        dim=-1,
    )


# ---------------------------------------------------------------- 观测
def traj_ref_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """29 个关节相对参考的偏差（弧度，target 顺序）。"""
    t = _term(env, action_name)
    q = _data(env).joint_pos[:, t.target_joint_ids]
    return q - t.ref_joint_pos


def traj_leg_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """12 条腿的偏差（弧度）。"""
    t = _term(env, action_name)
    return traj_ref_err(env, action_name)[:, t.leg_pos]


def traj_ref_joint_pos(env, action_name: str = "joint_pos") -> torch.Tensor:
    """29 维参考关节角（critic 用；actor 只看误差就够）。"""
    return _term(env, action_name).ref_joint_pos


def traj_progress(env, action_name: str = "joint_pos") -> torch.Tensor:
    """归一化进度 [B,1]，0 = 起点，1 = 末帧。"""
    return _term(env, action_name).progress.unsqueeze(-1)


def traj_progress_sincos(env, action_name: str = "joint_pos") -> torch.Tensor:
    """进度的 sin/cos（π 周期）：比线性量更平滑，且首尾同值便于策略对齐。"""
    p = _term(env, action_name).progress
    ang = p * 3.141592653589793
    return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)


def traj_height_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """基座高度 - 参考高度 [B,1]（米）。正 = 站得比参考高。"""
    t = _term(env, action_name)
    z = _root_pos(env)[:, 2]
    return (z - t.ref_base_z).unsqueeze(-1)


def traj_pitch_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """躯干朝向与参考的夹角（弧度）[B,1]。"""
    t = _term(env, action_name)
    up = _up_body(env)
    cos = (up * t.ref_up).sum(-1).clamp(-1.0, 1.0)
    return torch.acos(cos).unsqueeze(-1)


# ---------------------------------------------------------------- 奖励
def traj_track(env, action_name: str = "joint_pos", sigma: float = 0.35,
               legs_only: bool = False) -> torch.Tensor:
    """exp 核关节跟踪（默认 29 个关节，legs_only=True 只看 12 腿）。"""
    t = _term(env, action_name)
    err = traj_ref_err(env, action_name)
    if legs_only:
        err = err[:, t.leg_pos]
    mse = (err * err).mean(dim=-1)
    return torch.exp(-mse / (2.0 * sigma * sigma))


def height_track(env, action_name: str = "joint_pos", sigma: float = 0.05) -> torch.Tensor:
    """基座高度跟踪：蹲深了/没蹲下去都要罚。sigma 5 cm。"""
    e = traj_height_err(env, action_name).squeeze(-1)
    return torch.exp(-(e * e) / (2.0 * sigma * sigma))


def up_track(env, action_name: str = "joint_pos", sigma: float = 0.25) -> torch.Tensor:
    """躯干朝向跟踪：向量差的平方（等价 2-2cosθ），σ 0.25 约对应 14°。"""
    t = _term(env, action_name)
    diff = _up_body(env) - t.ref_up
    mse = (diff * diff).mean(dim=-1)
    return torch.exp(-mse / (2.0 * sigma * sigma))


def alive(env) -> torch.Tensor:
    """活着的每一步给 1（一次性终止惩罚的补集，防止「早摔早结束」）。"""
    return 1.0 - is_terminated(env)


def squat_probe(env, action_name: str = "joint_pos") -> torch.Tensor:
    """把跟踪质量写进 `Metrics/*`，不参与学习（权重 1e-9，理由见 mdp_extra.ref_probe）。"""
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
        _put("Metrics/traj_progress", term.progress)
        _put("Metrics/traj_done_frac", term.finished.float())
        _put("Metrics/traj_z", _root_pos(env)[:, 2])
        _put("Metrics/traj_z_ref", term.ref_base_z)
        _put("Metrics/traj_err_rms_all", (err * err).mean(-1).sqrt())
        _put("Metrics/traj_err_rms_leg", (leg * leg).mean(-1).sqrt())
        _put("Metrics/traj_pitch_deg", torch.rad2deg(traj_pitch_err(env, action_name).squeeze(-1)))
        _put("Metrics/traj_frames", term.frame_index.float())
    except Exception:
        pass
    return _zeros(env)


# ---------------------------------------------------------------- 终止
def traj_finished(env, action_name: str = "joint_pos") -> torch.Tensor:
    """参考播完 -> 本回合成功结束（time_out=True，不算摔倒）。"""
    try:
        return _term(env, action_name).finished
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def base_too_low(env, min_z: float = 0.28) -> torch.Tensor:
    """躯干高度跌破阈值 = 已经塌了（极限蹲的蹲姿是 0.523 m，0.28 m 留足余量）。"""
    return _root_pos(env)[:, 2] < float(min_z)


def base_drifted(env, max_xy: float = 1.2) -> torch.Tensor:
    """水平漂出直径 1.2 m（说明在「走」而不是在原地蹲）。"""
    p = _root_pos(env)
    return (p[:, :2] * p[:, :2]).sum(-1) > float(max_xy) ** 2
