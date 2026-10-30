"""残差步态动作项：q_target = q_ref(phase) + scale * clip(Δq)

设计要点
--------
* 继承 mjlab 的 JointPositionAction，只重写 `process_actions` / `reset`。
* 父类的 `_offset` 是静态 default_joint_pos；这里每步换成参考表取出的 q_ref，
  于是 `processed = raw * scale + q_ref`，策略输出天然变成「残差」。
* **相位是 per-env 状态量**：指令速度不同 -> 相位速率不同，不能只按时间积分。
* cfg 里只存 `table_path` 与网格元信息。1 MB 的表直接塞 cfg 会让 train.py
  的 env.yaml dump / deepcopy 爆掉；实表在 __init__ 里懒加载并做模块级缓存。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .ref_table import RefTable, scan_to_heights

# 模块级缓存：同一进程内多个 env 复用同一张表
_TABLE_CACHE: dict[tuple[str, str], RefTable] = {}

# 默认残差限幅（弧度）
# 2026-10-29：实测 |action| 均值 3.33，而旧限幅 0.30 rad / scale 0.2 ≈ 1.5 动作单位，
# 76% 的输出顶死在限幅上 -> 梯度在 clamp 外恒为 0，学不到细调。放宽到 0.40。
_DEFAULT_CLIP = {
    "legs": 0.40,
    "waist": 0.20,
    "arms": 0.20,
}

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


@dataclass(kw_only=True)
class ResidualGaitActionCfg(JointPositionActionCfg):
    """残差步态动作配置。"""

    table_path: str = ""
    period_frames: int = 90
    hold_steps: int = 25                    # 0.5 s，对应脚本 HOLD
    leg_clip_rad: float = 0.40
    waist_clip_rad: float = 0.20
    arm_clip_rad: float = 0.20
    scan_size: tuple[float, float] = (1.2, 2.0)
    scan_resolution: float = 0.1
    terrain_refresh_steps: int = 5
    rise_ema_alpha: float = 0.5
    tread_ema_alpha: float = 0.3
    v_rate_range: tuple[float, float] = (0.6, 1.6)
    phase_offset_choices: tuple[float, ...] = (0.0, 0.5)
    # 表第 33 帧 / 第 78 帧是双支撑起点；相位平移后 phase=0 落在双支撑上，
    # phase=0.5 正好落在另一个双支撑（0.5+0.3667 -> 帧 78）。
    phase_shift: float = 0.3667
    # 站立基准（表顺序的 29 个关节角）。α=0 时参考退化成这个姿势。
    ref_stand: tuple[float, ...] = ()
    # 参考幅度课程：α 从 0 线性升到 1 所需的 env step 数；0 表示不做课程。
    ramp_steps: int = 36000
    alpha_fixed: float = -1.0               # >=0 则固定 α（play 用 1.0）
    # 课程**起点**：alpha_fixed<0 时 α 从该值线性升到 1.0。
    # 续训一个 α 已涨到 0.7 的 checkpoint 时必须用它接上 —— 否则 ramp 从 0 重来，
    # 参考步态被打回站姿，而策略是按 70% 幅度长出来的，两者错位当场崩。
    # alpha_fixed>=0 时本字段被忽略。
    alpha_start: float = 0.0
    debug: bool = False

    def build(self, env) -> "ResidualGaitAction":
        return ResidualGaitAction(self, env)


class ResidualGaitAction(JointPositionAction):
    cfg: ResidualGaitActionCfg

    def __init__(self, cfg: ResidualGaitActionCfg, env):
        # 参考表存的是绝对关节角，父类的 default offset 不再使用
        super().__init__(cfg, env)
        self._env = env
        dev = self._raw_actions.device
        self._B, self._D = self._raw_actions.shape

        if not cfg.table_path:
            raise ValueError("ResidualGaitActionCfg.table_path 不能为空")
        key = (str(cfg.table_path), str(dev))
        if key not in _TABLE_CACHE:
            _TABLE_CACHE[key] = RefTable(cfg.table_path, device=dev)
        self._table = _TABLE_CACHE[key]

        # ---- 目标关节顺序 -> 表列顺序 的置换 ----
        names = self._target_joint_names()
        table_names = self._table.joint_names
        if names is not None and set(names) == set(table_names):
            perm = torch.tensor([table_names.index(n) for n in names], dtype=torch.long, device=dev)
        else:
            if cfg.debug:
                print(f"[residual] 关节顺序按默认假设（target={names}）")
            perm = torch.arange(self._D, device=dev)
        self._perm = perm

        # target 顺序对应的 entity 关节 id（观测/奖励按它取 q，避免顺序错位）
        raw_ids = getattr(self, "_target_ids", None)
        self._target_ids_t = torch.as_tensor(
            list(raw_ids) if raw_ids is not None else range(self._D),
            dtype=torch.long, device=dev,
        )

        # ---- 站立基准（表顺序 -> target 顺序）----
        # α=0 时 q_ref 退化成这个姿势：与基线任务「动作全零 = 默认站姿」等价，
        # 于是残差任务在训练早期和已验证可训练的基线完全同难度。
        if cfg.ref_stand and len(cfg.ref_stand) == len(table_names):
            stand = torch.tensor(list(cfg.ref_stand), dtype=torch.float32, device=dev)
            self._stand = stand[self._perm].unsqueeze(0)
        else:
            self._stand = None
        if cfg.alpha_fixed >= 0.0:
            self._alpha = max(0.0, float(cfg.alpha_fixed))
        else:
            # 课程模式：初值取 alpha_start（续训时不为 0），避免第一帧参考跳变
            self._alpha = min(1.0, max(0.0, float(getattr(cfg, "alpha_start", 0.0))))

        # 12 条腿在 target 顺序里的位置
        self._leg_pos = torch.tensor(
            [names.index(n) for n in LEG_JOINTS if n in names], dtype=torch.long, device=dev
        ) if names else torch.arange(12, dtype=torch.long, device=dev)

        try:
            jn_all = getattr(self._entity, "joint_names", None) or []
            self._leg_joint_ids = [jn_all.index(n) for n in LEG_JOINTS if n in jn_all]
        except Exception:
            self._leg_joint_ids = []

        # ---- 动作空间限幅（残差弧度 -> action 空间） ----
        clip_rad = torch.tensor(
            [_clip_for(n, cfg.leg_clip_rad, cfg.waist_clip_rad, cfg.arm_clip_rad)
             for n in (names or table_names)],
            dtype=torch.float32, device=dev,
        )
        scale = self._scale
        if not torch.is_tensor(scale):
            scale = torch.full((self._B, self._D), float(scale), device=dev)
        self._clip_action = clip_rad.unsqueeze(0) / scale.clamp_min(1e-6)

        # ---- per-env 状态 ----
        self._phase = torch.zeros(self._B, device=dev)
        self._rise = torch.zeros(self._B, device=dev)
        self._tread = torch.full((self._B,), float(self._table.mid_tread), device=dev)
        self._ref_q = torch.zeros(self._B, self._D, device=dev)
        self._ref_v = torch.zeros(self._B, device=dev)
        self._step_i = 0
        self._warned_no_scan = False
        # 诊断用：地形估计链路是否真的跑通了（见 _read_heights 的 bug 说明）
        self._scan_ok = False
        self._scan_shape: tuple[int, ...] = ()
        self._last_valid: torch.Tensor | None = None
        self._last_raw_rise: torch.Tensor | None = None

        # offset 必须是张量，父类在 use_default_offset=False 时会留成 float
        self._offset = torch.zeros(self._B, self._D, device=dev)

        # 初始参考 = 站立基准（α=0）；alpha_fixed>=0 时直接给最终参考
        self._ref_q.copy_(self._compose_ref(self._alpha))
        self._offset.copy_(self._ref_q)
        self._ref_v.fill_(float(self._table.v_ref.mean()))

        if cfg.debug:
            print(f"[residual] target_joints={names}")
            print(f"[residual] table={self._table.R}x{self._table.T} "
                  f"period={self._table.period}")

    # ------------------------------------------------------------ 辅助
    def _target_joint_names(self):
        # 父类 BaseAction 在 __init__ 里给出了 _target_names（target 顺序的关节名），
        # 这是最权威的来源，避免用 _target_ids 去索引 entity.joint_names 时的排序歧义。
        tn = getattr(self, "_target_names", None)
        if tn:
            return list(tn)
        try:
            ent = self._entity
            jn = getattr(ent, "joint_names", None)
            if jn is None:
                return None
            ids = None
            for attr in ("_target_ids", "_joint_ids", "_actuator_ids"):
                ids = getattr(self, attr, None)
                if ids is not None:
                    break
            if ids is None:
                return None
            idx = ids.tolist() if torch.is_tensor(ids) else list(ids)
            return [jn[i] for i in idx]
        except Exception:
            return None

    def _alpha_now(self) -> float:
        """参考幅度系数 α ∈ [0,1]：线性课程，防止一开始就被步态拽倒。"""
        if self.cfg.alpha_fixed >= 0.0:
            return float(self.cfg.alpha_fixed)
        n = int(self.cfg.ramp_steps)
        start = min(1.0, max(0.0, float(getattr(self.cfg, "alpha_start", 0.0))))
        if n <= 0:
            return 1.0
        frac = min(1.0, max(0.0, self._step_i / float(n)))
        return start + (1.0 - start) * frac

    def _compose_ref(self, alpha: float) -> torch.Tensor:
        """q_ref = q_stand + α · (q_gait(phase) − q_stand)，返回 target 顺序 [B,J]。"""
        q_gait, _ = self._table.select(self._rise, self._tread, self.effective_phase)
        q_gait = q_gait[:, self._perm]
        if self._stand is None:
            return q_gait
        return self._stand + alpha * (q_gait - self._stand)

    def _read_heights(self) -> torch.Tensor | None:
        """从 terrain_scan 传感器取「躯干高出地形」的高度（米），[B, F*N]。

        ⚠️ 2026-10-29 致命 bug（rise_est 恒为 0.000 的真正原因）
        ------------------------------------------------------
        原来写的是 `z = frame_pos_w[:, :, 2:3]`（形状 [B,1,1]）减
        `hit_pos_w[..., 2]`（形状 [B,N]）。二者广播成 **[B,B,N]**，
        `reshape(B, -1)` 于是得到 [B, B*N]（B=4 时 1092 列而不是 273），
        紧接着被 `_update_terrain` 的维度守卫判为不符 -> **每帧都直接 return，
        estimate() 一次都没执行过**。于是 `_rise` 永远停在 0、`_tread` 永远
        停在 mid_tread(0.36)，参考步态锁死在平地档：
          * `rise_est = 0.000 (min 0.000 / max 0.000)`
          * `tread_est = 0.360`  <- 恰好等于 RefTable.mid_tread 初值
        B=1 时 (1,1,1)-(1,N) 广播恰好是 (1,1,N)，reshape 后正确，
        所以单环境回放看不出来，1024 env 的训练与 4/8 env 的诊断里全程失效。
        正确写法与 mjlab 自己的 `envs_mdp.height_scan` 保持一致：
        先把两个操作数都 view 成 [B,F,N] 再相减。
        """
        try:
            sensor = self._env.scene["terrain_scan"]
            data = sensor.data
            self._scan_ok = True
            return scan_to_heights(
                data.frame_pos_w[:, :, 2:3],      # [B,F,1]
                data.hit_pos_w[..., 2],           # [B,F*N]
                data.distances,                   # [B,F*N]，-1 = miss
                float(sensor.cfg.max_distance),
            )
        except Exception as e:
            if not self._warned_no_scan:
                print(f"[residual][warn] 读不到 terrain_scan，rise 恒为 0（{e!r}）")
                self._warned_no_scan = True
            return None

    def _update_terrain(self) -> None:
        h = self._read_heights()
        if h is None:
            return
        nx = int(round(self.cfg.scan_size[0] / self.cfg.scan_resolution)) + 1
        ny = int(round(self.cfg.scan_size[1] / self.cfg.scan_resolution)) + 1
        self._scan_shape = tuple(h.shape)
        if h.shape[-1] != nx * ny:
            # 以前这里是静默 return，导致 rise 永远是 0 却没人发现。
            if not self._warned_no_scan:
                print(f"[residual][ERR] height_scan 展平长度 {int(h.shape[-1])} != {nx}x{ny}="
                      f"{nx * ny}（h.shape={tuple(h.shape)}），参考步态将停在平地档。"
                      f" 这是 shape 广播 bug 的典型症状，别忽略这条。")
                self._warned_no_scan = True
            self._scan_ok = False
            return
        self._scan_ok = True
        rise, tread, valid = self._table.estimate(h, nx=nx, ny=ny,
                                                  size_y=float(self.cfg.scan_size[1]),
                                                  res=float(self.cfg.scan_resolution))
        self._last_valid = valid
        self._last_raw_rise = rise
        a = float(self.cfg.rise_ema_alpha)
        new_rise = torch.where(valid, a * rise + (1 - a) * self._rise, self._rise)
        # 长期无效（平地）-> 缓慢回落到 0
        new_rise = torch.where(valid, new_rise, 0.9 * self._rise)
        self._rise.copy_(new_rise.clamp(0.0, float(self._table.rise[-1])))
        # tread 原来直接取当帧值，一条假边就能把它顶到 clamp 上限 0.40（实测如此）。
        # 同样做 EMA（比 rise 快一些，踏面变化更局部）。
        at = float(self.cfg.tread_ema_alpha)
        self._tread.copy_(torch.where(valid, at * tread + (1 - at) * self._tread,
                                      self._tread))

    def _v_cmd(self) -> torch.Tensor:
        try:
            cmd = self._env.command_manager.get_command("twist")
            return cmd[:, 1].to(self._raw_actions.device).float()
        except Exception:
            return torch.zeros(self._B, device=self._raw_actions.device)

    # ------------------------------------------------------------ 生命周期
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._B, device=self._raw_actions.device)
        env_ids = torch.as_tensor(env_ids, device=self._raw_actions.device).long()
        n = env_ids.numel()
        choices = torch.tensor(self.cfg.phase_offset_choices,
                               device=self._raw_actions.device, dtype=torch.float32)
        pick = torch.randint(0, choices.numel(), (n,), device=self._raw_actions.device)
        self._phase[env_ids] = choices[pick]
        self._rise[env_ids] = 0.0
        self._tread[env_ids] = self._table.mid_tread
        r = getattr(super(), "reset", None)
        if callable(r):
            r(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        dev = self._raw_actions.device

        # ① 地形估计（每 N 步刷新）
        if self._step_i % max(1, int(self.cfg.terrain_refresh_steps)) == 0:
            self._update_terrain()

        # ② 取当前相位的参考（先取，再推进）；α 课程把参考从站姿渐变到步态
        alpha = self._alpha_now()
        self._alpha = alpha
        q_ref, v_ref = self._table.select(self._rise, self._tread, self.effective_phase)
        q_ref = q_ref[:, self._perm]
        if self._stand is not None:
            q_ref = self._stand + alpha * (q_ref - self._stand)
        self._ref_q.copy_(q_ref)
        self._ref_v.copy_(v_ref)

        # ③ 相位速率：指令速度 / 参考步速；起步 HOLD 段冻结
        v_cmd = self._v_cmd()
        rate = (v_cmd / self._ref_v.clamp_min(1e-3)).clamp(*self.cfg.v_rate_range)
        try:
            hold = self._env.episode_length_buf < int(self.cfg.hold_steps)
        except Exception:
            hold = torch.zeros(self._B, dtype=torch.bool, device=dev)
        rate = torch.where(hold, torch.zeros_like(rate), rate)

        # ④ 残差限幅 + 时变 offset
        raw = actions.clamp(-self._clip_action, self._clip_action)
        self._offset.copy_(self._ref_q)
        super().process_actions(raw)

        # ⑤ 推进相位与计数
        self._phase.add_(rate / float(self.cfg.period_frames)).remainder_(1.0)
        self._step_i += 1

    # ------------------------------------------------------------ 只读属性（给观测/奖励）
    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self._ref_q

    @property
    def leg_joint_ids(self) -> list[int]:
        return getattr(self, "_leg_joint_ids", [])

    @property
    def leg_pos(self) -> torch.Tensor:
        return self._leg_pos

    @property
    def ref_v(self) -> torch.Tensor:
        return self._ref_v

    @property
    def phase(self) -> torch.Tensor:
        return self._phase

    @property
    def effective_phase(self) -> torch.Tensor:
        """实际用于查表的相位（含双支撑平移）。观测与查表必须用同一个。"""
        return torch.remainder(self._phase + float(self.cfg.phase_shift), 1.0)

    @property
    def target_joint_ids(self) -> torch.Tensor:
        """target 顺序的 entity 关节 id，供观测/奖励按同一顺序取 q。"""
        return self._target_ids_t

    @property
    def ref_alpha(self) -> float:
        return float(self._alpha)

    @property
    def rise_est(self) -> torch.Tensor:
        return self._rise

    @property
    def scan_ok(self) -> bool:
        """地形估计链路是否跑通（False = height_scan 取数/维度有问题）。"""
        return self._scan_ok

    @property
    def scan_shape(self) -> tuple[int, ...]:
        return self._scan_shape

    @property
    def raw_rise(self) -> torch.Tensor | None:
        """未过 EMA 的原始 rise 估计（最近一次刷新）。"""
        return self._last_raw_rise

    @property
    def rise_valid(self) -> torch.Tensor | None:
        """最近一次刷新时是否在扫描带内找到上升边缘。"""
        return self._last_valid

    @property
    def tread_est(self) -> torch.Tensor:
        return self._tread

    @property
    def phase_sincos(self) -> torch.Tensor:
        ang = self.effective_phase * (2.0 * 3.141592653589793)
        return torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
