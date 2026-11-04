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
    # ⚠️ RewardManager 要求奖励项返回一维 [B]，返回 [B,1] 会直接
    #    `_check_term_shape` 报错 —— 观测项可以带特征维，奖励项不行。
    return torch.zeros(env.num_envs, device=env.device)


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


# ---------------------------------------------------------------- 「别动」（保持任务）
def joint_speed(env, action_name: str = "joint_pos") -> torch.Tensor:
    """29 个关节速度的 RMS（rad/s）[B]。

    跟踪任务里关节速度天然在变（参考本身在动），罚它没有意义；
    但**保持任务**的目标就是「不动」，抖动会通过脚底传到地面、再反作用回机身，
    是长时保持最主要的失稳源。所以这一项只在 hold 任务里用。
    """
    t = _term(env, action_name)
    try:
        v = _data(env).joint_vel[:, t.target_joint_ids]
    except Exception:
        return _zeros(env)
    return (v * v).mean(-1).sqrt()


def quiet(env, action_name: str = "joint_pos", sigma: float = 0.8) -> torch.Tensor:
    """静止奖励 exp(-mean(v²)/2σ²)，σ 单位 rad/s。

    为什么用 exp 核而不是 -L2：早期策略全身都在抖（v_rms 常到 2~3 rad/s），
    纯 L2 会给一个很大的常数负偏置，把「活着」的收益淹掉；exp 核在 2.4 rad/s
    （3σ）之后基本饱和，梯度只落在「已经很安静、还能更安静」的区间。
    标定（σ=0.8）：0.2 -> 0.97、0.5 -> 0.82、0.8 -> 0.61、1.5 -> 0.17。
    """
    s = joint_speed(env, action_name)
    return torch.exp(-(s * s) / (2.0 * sigma * sigma))


# ---------------------------------------------------------------- 左右对称
# 参考轨迹是**严格左右对称**的（用 down/q 逐列验证过：同号关节 |l-r| 最大 0.0000，
# 反号关节 |l+r| 最大 0.0000）。跟踪奖励按维度平均，天然允许「左腿外撇 20°、
# 右腿内收 20°」这种等价解 —— 实测第一版 4000 轮就收敛到不对称站姿（骨盆扭转、
# 一只手臂压在腿上），看着别扭，而且单侧受力会慢慢把身体推倒。
# 这条奖励直接把左右差额罚掉，让解空间里只剩对称解。
#
# 符号规则来自实测：绕 y（屈伸）两侧同号，绕 x/z（外展/扭转）两侧反号。
_AXIS_SIGN_SAME = ("y",)


def _mirror_groups(term):
    """返回 (iL, iR, sgn)：满足 q_L ≈ sgn * q_R 的左右关节下标（target 顺序）。

    与 `TrajTrackAction` 里 `_target_joint_names()` 同一套名字来源，
    所以下标一定和 `_data(env).joint_pos[:, target_joint_ids]` 对齐。
    """
    cached = getattr(term, "_sym_groups", None)
    if cached is not None:
        return cached
    try:
        names = term._target_joint_names()
    except Exception:
        names = None
    if not names:
        return None
    idx = {n: i for i, n in enumerate(names)}
    iL, iR, sg = [], [], []
    for i, n in enumerate(names):
        if not n.startswith("l_"):
            continue
        r = "r_" + n[2:]
        if r not in idx:
            continue
        parts = n.split("_")                      # l_hip_y_joint -> ['l','hip','y','joint']
        axis = parts[2] if len(parts) > 2 else "y"
        iL.append(i)
        iR.append(idx[r])
        sg.append(1.0 if axis in _AXIS_SIGN_SAME else -1.0)
    if not iL:
        return None
    dev = term.target_joint_ids.device
    groups = (
        torch.tensor(iL, dtype=torch.long, device=dev),
        torch.tensor(iR, dtype=torch.long, device=dev),
        torch.tensor(sg, dtype=torch.float32, device=dev),
    )
    try:
        term._sym_groups = groups
    except Exception:
        pass
    return groups


def sym_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """左右关节差额的 RMS（弧度）[B]。对称时 = 0。"""
    t = _term(env, action_name)
    g = _mirror_groups(t)
    if g is None:
        return _zeros(env)
    iL, iR, sgn = g
    q = _data(env).joint_pos[:, t.target_joint_ids]
    d = q[:, iL] - sgn * q[:, iR]
    return (d * d).mean(-1).sqrt()


def symmetry(env, action_name: str = "joint_pos", sigma: float = 0.12) -> torch.Tensor:
    """左右对称奖励：exp(-mean(Δ²)/2σ²)，σ 0.12 rad（7°）。"""
    e = sym_err(env, action_name)
    return torch.exp(-(e * e) / (2.0 * sigma * sigma))


# ---------------------------------------------------------------- 位置锚定
# 参考轨迹全程基座 x=y=0（`down/up` 两份 npz 实测 max|x| = max|y| = 0.0000），
# 所以「水平别跑、别转」是纯粹的任务约束，不是额外假设。
#
# 为什么必须显式罚：回合原本只有 2.46 s，策略从没学过「长时间站住」，
# 回放里让它保持几十秒就会慢慢飘、最后倒（实测就是这样倒的）。
# 光加长回合不够——没有位置的绝对参考，慢漂本身在奖励里是中性的。
def base_anchor(env) -> torch.Tensor:
    """基座水平位移的平方 (m²) [B]。原点 = 出生点（reset 只随机 ±1 cm）。"""
    p = _root_pos(env)
    return p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1]


def yaw_anchor(env) -> torch.Tensor:
    """基座偏航角的平方 (rad²) [B]。参考全程朝 +X，转一点就是错的。"""
    q = _root_quat(env)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw * yaw


# ---------------------------------------------------------------- 手臂-腿接触
# 参考蹲姿里肘-膝最小间隙 18.6 cm（`squat_extreme.py` 的约束），但残差限幅
# ±0.35 rad 足以让大腿或手臂怼上去；而且接触在奖励里是**免费**的
# （除了躯干触地，没有任何接触惩罚），于是策略学会了「把手臂搁在大腿上」省力。
_ARM_GUARD_WARNED = [False]


def arm_leg_contact(env, sensor_name: str = "arm_leg_contact") -> torch.Tensor:
    """手臂-腿接触标志 [B]，0 = 没碰，1 = 碰上了（用传感器 found 字段）。"""
    from mjlab.tasks.velocity import mdp as _m

    c = None
    err = None
    # ① 本项目已在服务器上实测过的入口（足底接触就是这么读的）
    try:
        c = _m.foot_contact(env, sensor_name=sensor_name)
    except Exception as e:                        # noqa: BLE001
        err = e
    # ② 直接读传感器 data.found / data.force（foot_contact 依赖 air-time 字段时用这条）
    if c is None:
        try:
            sens = env.scene[sensor_name]
            d = getattr(sens, "data", None)
            v = getattr(d, "found", None)
            if v is None:
                v = getattr(d, "force", None)
            if v is not None:
                c = v if torch.is_tensor(v) else torch.as_tensor(v, device=env.device)
        except Exception as e:                    # noqa: BLE001
            err = err or e
    if c is None:
        if not _ARM_GUARD_WARNED[0]:
            _ARM_GUARD_WARNED[0] = True
            print(f"[squat][warn] 读不到接触传感器 {sensor_name}（{err!r}），"
                  f"手臂-腿接触惩罚按 0 处理；需要的话 ARM_GUARD=0 关掉这一项")
        return _zeros(env)
    c = c.to(torch.float32)
    if c.dim() > 1:
        c = c.reshape(c.shape[0], -1).amax(-1)
    return c.clamp(0.0, 1.0)


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
        # 对称 / 漂移 / 接触：判断「不对称、慢慢倒、手臂压腿」这三件事靠这几条
        _put("Metrics/sym_err_rad", sym_err(env, action_name))
        _put("Metrics/base_xy", base_anchor(env).sqrt())
        _put("Metrics/base_yaw_deg", torch.rad2deg(yaw_anchor(env).sqrt()))
        _put("Metrics/arm_leg_touch", arm_leg_contact(env))
        # 保持任务的主指标：关节速度 RMS。跟踪任务里它只作参考
        _put("Metrics/joint_speed", joint_speed(env, action_name))
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
