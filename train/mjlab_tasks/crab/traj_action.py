"""参考轨迹跟踪动作项：q_target = q_ref(t) + scale * clip(Δq)

和 `residual_action.py` 的关系
------------------------------
那个是「周期步态表 + 相位」的残差；这个是「一次性时间序列 + 帧号」的残差。
骨架完全一样（继承 mjlab 的 `JointPositionAction`、每步改 `_offset`），
区别只有参考来源与推进方式：

    residual_action : phase ∈ [0,1)，按指令速度调速率，查 90 帧的步态表
    traj_action     : 帧号 0..T-1，每个 env step 前进 1 帧（= 50 Hz，与轨迹 dt 一致）

为什么用「时间轴 + 残差」而不是纯行为克隆
----------------------------------------
轨迹是开环 IK 解出来的，物理上并不完全可行（PD 跟踪误差、接触抖动都在里面）。
纯 q_target = q_ref 等于要求策略精确复现一个不可行的目标；给 ±clip 的残差，
策略才能在「按参考动」和「别摔」之间自己找折中——这也是蹲姿能站住的前提。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .traj_ref import load_traj

# 残差限幅（弧度）：腿要能救平衡，腰/臂只微调
# 腿 0.35 rad（20°）足够把重心拉回来，又不至于让策略无视参考自己乱蹲；
# 腰 0.15、臂 0.25：臂在极限蹲里承担配重，允许比腰大一点。
_DEFAULT_CLIP = {"legs": 0.35, "waist": 0.15, "arms": 0.25}

LEG_JOINTS = (
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint",
    "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint",
    "r_ankle_y_joint", "r_ankle_x_joint",
)
WAIST_JOINTS = ("waist_y_joint", "waist_x_joint", "waist_z_joint")


def _clip_for(name: str, legs: float, waist: float, arms: float) -> float:
    if name in LEG_JOINTS:
        return legs
    if name in WAIST_JOINTS:
        return waist
    return arms


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[traj][warn] {name}={raw!r} 不是数字，用默认 {default}")
        return default


@dataclass(kw_only=True)
class TrajTrackActionCfg(JointPositionActionCfg):
    """参考轨迹跟踪动作配置。"""

    traj_path: str = ""
    traj_key: str = "down"                  # down / up
    advance: int = 1                        # 每个 env step 前进几帧
    leg_clip_rad: float = _DEFAULT_CLIP["legs"]
    waist_clip_rad: float = _DEFAULT_CLIP["waist"]
    arm_clip_rad: float = _DEFAULT_CLIP["arms"]
    # 残差总开关：0 = 纯参考开环（诊断「轨迹本身能不能站点」用）
    residual_scale: float = 1.0
    # 参考起始淡入：前 N 步把参考从「机器人出生姿态」渐变到 q_ref[0]。
    # 默认 0（出生姿态本来就取自 q_ref[0]，不需要）。
    fade_in_steps: int = 0
    # 冻结在第几帧（负值 = 从末帧数，-1 就是最后一帧）。非 None 时帧号不推进、
    # `finished` 恒 False —— 参考寄存器变成一个**静态位形**，于是所有以
    # `ref_joint_pos / ref_base_z / ref_up` 为基础的奖励项自动退化成「保持这个位形」。
    # 「蹲姿保持」独立任务就是这么复用跟踪代码的，不需要额外的参考表。
    freeze_frame: int | None = None
    debug: bool = False

    def build(self, env) -> "TrajTrackAction":
        return TrajTrackAction(self, env)


class TrajTrackAction(JointPositionAction):
    cfg: TrajTrackActionCfg

    def __init__(self, cfg: TrajTrackActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        dev = self._raw_actions.device
        self._B, self._D = self._raw_actions.shape

        if not cfg.traj_path:
            raise ValueError("TrajTrackActionCfg.traj_path 不能为空")
        self._traj = load_traj(cfg.traj_path, cfg.traj_key, device=dev)
        T = self._traj.T
        if T < 2:
            raise ValueError(f"轨迹 {cfg.traj_key} 只有 {T} 帧，无法跟踪")
        # 一个 env step 走一帧 = 控制周期必须等于轨迹 dt，否则「演得比参考快/慢」
        ctrl_dt = float(getattr(env, "step_dt", 0.02) or 0.02)
        if abs(ctrl_dt * max(1, int(cfg.advance)) - self._traj.dt) > 1e-3:
            print(f"[traj][warn] 控制周期 {ctrl_dt:.4f}s × advance "
                  f"{cfg.advance} != 轨迹 dt {self._traj.dt:.4f}s，"
                  f"参考会被播快/播慢（不是错，但和脚本轨迹不一致）")

        # ---- 目标关节顺序 -> 表列顺序 的置换 ----
        names = self._target_joint_names()
        table_names = self._traj.names
        if names is not None and set(names) == set(table_names):
            perm = torch.tensor([table_names.index(n) for n in names],
                                dtype=torch.long, device=dev)
        else:
            print(f"[traj][warn] 关节名对不上，按默认顺序假设（target={names}）")
            perm = torch.arange(self._D, device=dev)
        self._perm = perm

        raw_ids = getattr(self, "_target_ids", None)
        self._target_ids_t = torch.as_tensor(
            list(raw_ids) if raw_ids is not None else range(self._D),
            dtype=torch.long, device=dev,
        )

        # 12 条腿在 target 顺序里的位置
        if names:
            self._leg_pos = torch.tensor(
                [names.index(n) for n in LEG_JOINTS if n in names],
                dtype=torch.long, device=dev,
            )
        else:
            self._leg_pos = torch.arange(12, dtype=torch.long, device=dev)

        # ---- 残差限幅（弧度 -> action 单位）----
        clip_rad = torch.tensor(
            [_clip_for(n, cfg.leg_clip_rad, cfg.waist_clip_rad, cfg.arm_clip_rad)
             for n in (names or table_names)],
            dtype=torch.float32, device=dev,
        )
        scale = self._scale
        if not torch.is_tensor(scale):
            scale = torch.full((self._B, self._D), float(scale), device=dev)
        self._clip_action = clip_rad.unsqueeze(0) / scale.clamp_min(1e-6)

        self._res_scale = max(0.0, float(cfg.residual_scale))
        _rs = os.environ.get("ELF3_RESIDUAL_SCALE")
        if _rs is not None and _rs.strip() != "":
            try:
                self._res_scale = max(0.0, float(_rs))
                print(f"[traj] 残差缩放 ELF3_RESIDUAL_SCALE -> {self._res_scale}")
            except ValueError:
                print(f"[traj][warn] ELF3_RESIDUAL_SCALE={_rs!r} 不是数字，忽略")
        if self._res_scale == 0.0:
            print("[traj] 残差已关闭：纯参考开环（诊断模式）")

        # ---- per-env 状态 ----
        self._freeze: int | None = None
        if cfg.freeze_frame is not None:
            f = int(cfg.freeze_frame)
            if f < 0:
                f += T
            if not (0 <= f < T):
                raise ValueError(
                    f"freeze_frame={cfg.freeze_frame} 越界（轨迹 {cfg.traj_key} 只有 {T} 帧）"
                )
            self._freeze = f

        self._idx = torch.zeros(self._B, dtype=torch.long, device=dev)
        if self._freeze is not None:
            self._idx.fill_(self._freeze)
        self._ref_q = torch.zeros(self._B, self._D, device=dev)
        self._start_q = torch.zeros(self._B, self._D, device=dev)   # fade_in 用
        self._fade = max(0, int(cfg.fade_in_steps))

        self._offset = torch.zeros(self._B, self._D, device=dev)
        self._ref_q.copy_(self._traj.q[self._idx][:, self._perm])
        self._start_q.copy_(self._ref_q)
        self._offset.copy_(self._ref_q)

        if cfg.debug or self._freeze is not None:
            extra = ""
            if self._freeze is not None:
                extra = (f" | 冻结在第 {self._freeze} 帧（不推进，参考=静态位形，"
                         f"z={float(self._traj.z[self._freeze]):.3f} m）")
            print(f"[traj] {cfg.traj_key}: T={T} 帧 / {self._traj.duration:.2f} s，"
                  f"z {float(self._traj.z[0]):.3f} -> {float(self._traj.z[-1]):.3f} m，"
                  f"限幅(leg/waist/arm)="
                  f"{cfg.leg_clip_rad}/{cfg.waist_clip_rad}/{cfg.arm_clip_rad} rad{extra}")

    # ------------------------------------------------------------ 辅助
    def _target_joint_names(self):
        tn = getattr(self, "_target_names", None)
        if tn:
            return list(tn)
        try:
            jn = getattr(self._entity, "joint_names", None)
            ids = None
            for attr in ("_target_ids", "_joint_ids", "_actuator_ids"):
                ids = getattr(self, attr, None)
                if ids is not None:
                    break
            if jn is None or ids is None:
                return None
            idx = ids.tolist() if torch.is_tensor(ids) else list(ids)
            return [jn[i] for i in idx]
        except Exception:
            return None

    # ------------------------------------------------------------ 生命周期
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._B, device=self._raw_actions.device)
        env_ids = torch.as_tensor(env_ids, device=self._raw_actions.device).long()
        self._idx[env_ids] = self._freeze if self._freeze is not None else 0
        r = getattr(super(), "reset", None)
        if callable(r):
            r(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        idx = self._idx.clamp_(max=self._traj.T - 1)
        q_ref = self._traj.q[idx][:, self._perm]                 # [B, J] target 顺序

        if self._fade > 0:
            step = self._env.episode_length_buf.to(q_ref.device).float().unsqueeze(-1)
            w = (step / float(self._fade)).clamp(0.0, 1.0)
            q_ref = self._start_q + w * (q_ref - self._start_q)

        self._ref_q.copy_(q_ref)

        raw = actions.clamp(-self._clip_action, self._clip_action) * self._res_scale
        self._offset.copy_(self._ref_q)
        super().process_actions(raw)

        if self._freeze is None:
            self._idx.add_(max(1, int(self.cfg.advance))).clamp_(max=self._traj.T - 1)

    # ------------------------------------------------------------ 只读属性
    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self._ref_q

    @property
    def target_joint_ids(self) -> torch.Tensor:
        return self._target_ids_t

    @property
    def leg_pos(self) -> torch.Tensor:
        return self._leg_pos

    @property
    def frame_index(self) -> torch.Tensor:
        return self._idx

    @property
    def progress(self) -> torch.Tensor:
        """0 -> 1 的归一化进度 [B]。"""
        return self._idx.float() / float(max(1, self._traj.T - 1))

    @property
    def ref_base_z(self) -> torch.Tensor:
        return self._traj.z[self._idx.clamp(max=self._traj.T - 1)]

    @property
    def ref_up(self) -> torch.Tensor:
        return self._traj.up[self._idx.clamp(max=self._traj.T - 1)]

    @property
    def ref_pitch_deg(self) -> torch.Tensor:
        return self._traj.pitch[self._idx.clamp(max=self._traj.T - 1)]

    @property
    def finished(self) -> torch.Tensor:
        if self._freeze is not None:
            # 冻结模式永不算「播完」：保持任务的回合该由时长/摔倒决定，不是参考走完
            return torch.zeros_like(self._idx, dtype=torch.bool)
        return self._idx >= (self._traj.T - 1)

    @property
    def traj_duration(self) -> float:
        return self._traj.duration

    @property
    def traj_frames(self) -> int:
        return self._traj.T

    @property
    def ref_meta(self) -> dict:
        return self._traj.meta
