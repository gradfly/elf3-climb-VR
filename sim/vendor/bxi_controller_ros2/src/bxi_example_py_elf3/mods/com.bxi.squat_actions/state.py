"""蹲起三态：squat_down / squat_hold / squat_up。

- down/up 继承 MotionReplayState：policy.finished() 时自动 request_state 到下一态。
- hold 手写且**永不主动切换**：唯一出口是 mod.yaml 的 squat_up route；
  btn_2 零力矩与 IMU 保护（daemon force=True）作为安全例外保留。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bxi_example_py_elf3.framework.mod_api import (
    MotionReplayState,
    ResourceHandle,
    RobotControlState,
)
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)

from .squat_residual_policy import SquatResidualPolicy

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import (
        RobotControlContext,
        StateBehavior,
    )

ZERO_TORQUE = "com.bxi.basic_actions/zero_torque"


class SquatDownState(MotionReplayState[SquatResidualPolicy]):
    """下蹲：24…124 帧，完成后自动切 squat_hold。"""

    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[SquatResidualPolicy]
    ) -> None:
        super().__init__(
            name,
            state_id,
            policy,
            finish_state="com.bxi.squat_actions/squat_hold",
            finish_trigger="squat_down_finished",
            end_frame_trim=0,
            # down→hold 必须平滑过渡：训练时 down 的 5 s 末帧保持段由**同一个
            # down 网络**跑 freeze 帧（squat_env_cfg.py:351-357），部署却切到
            # 独立 hold 网络——两网络残差输出不连续，soft_switch（0.02 s）会
            # 把目标角突变直接打进电机，机器人切换瞬间突沉（2026-11-19 仿真
            # 实测）。dual_running_blend 0.3 s 把两侧目标做 smootherstep 插值。
            # 契约：hold.sample_running_frame(advance=False) 无副作用（已 AST
            # 校验），down 侧已 clamp 末帧、advance_from 无所谓。
            end_transition={
                "profile": "dual_running_blend",
                "duration": 0.3,
                "curve": "smootherstep",
                "sample_from": True,
                "sample_to": True,
                "advance_from": False,
                "advance_to": False,
            },
        )
        self._elapsed = 0.0

    def on_enter(self, ctx: "RobotControlContext") -> None:
        super().on_enter(ctx)
        self._elapsed = 0.0

    def on_update(self, ctx: "RobotControlContext", dt: float) -> None:
        self._elapsed += dt
        # 超时保护：2.46 s + 2 s 缓冲；卡帧时强制零力矩（安全，不算动作类出口）
        if self._elapsed > self.policy.frames * 0.02 + 2.0 and not self.policy.finished():
            ctx.request_state(ZERO_TORQUE, trigger="squat_down_stuck", force=True)
            return
        super().on_update(ctx, dt)


class SquatUpState(MotionReplayState[SquatResidualPolicy]):
    """起立：102 帧，完成后自动切回走路（normal，不是 normal_run）。"""

    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[SquatResidualPolicy]
    ) -> None:
        super().__init__(
            name,
            state_id,
            policy,
            finish_state="com.bxi.basic_actions/normal",
            finish_trigger="squat_up_finished",
            # 提前 8 帧（0.16 s）切走：up 末段存在参考轨迹的失稳流形（从源
            # 蹲姿起步 k93 后仰摔倒，2026-11-19 实测）；trim 点实测姿态很稳
            # （frame 94: z 1.027 / tilt 6.3° / 裕度 +10 cm），从开环 hold 末
            # 态起步则能完整跑满 102 帧——trim=8 是两者通用的安全网。
            end_frame_trim=8,
            end_transition={
                "profile": "dual_running_blend",
                "duration": 0.3,
                "curve": "smootherstep",
                "sample_from": True,
                "sample_to": True,
                # blend 期间不推进任一侧时间线，避免 up 末帧被推过 / AMP 相位乱跑
                "advance_from": False,
                "advance_to": False,
            },
        )
        self._elapsed = 0.0

    def on_enter(self, ctx: "RobotControlContext") -> None:
        super().on_enter(ctx)
        self._elapsed = 0.0

    def on_update(self, ctx: "RobotControlContext", dt: float) -> None:
        self._elapsed += dt
        if self._elapsed > self.policy.frames * 0.02 + 2.0 and not self.policy.finished():
            ctx.request_state(ZERO_TORQUE, trigger="squat_up_stuck", force=True)
            return
        super().on_update(ctx, dt)


class SquatHoldState(RobotControlState, EntryFrameProvider, RunningFrameProvider):
    """蹲姿保持：策略时间轴锁在 down 末帧（123），永不主动切状态。

    两种保持模式（hold_mode）：
      - "open_loop"（默认）：不跑 ONNX，固定下发参考蹲姿 target + kp/kd=训练×8。
        实测依据（2026-11-19 本地 MuJoCo）：训练柔顺 kp 下参考蹲姿物理撑不住
        （开环跟踪 q_ref 也会滑到前倾 42°），squat_hold.onnx 学到的是顺应物理的
        慢发散——每秒约 +2.2° 前倾，hold 5-10 s 必倒（12 s 裕度 -5 cm），无法
        满足长期保持；开环 ×8 实测 60 s 零漂移（z 0.513 / tilt 24.5° / 裕度
        +7.3 cm），且省电 11%（Στ² 0.89x），up 从其末态直接完成（97%）。
      - "onnx"：跑 squat_hold.onnx 锁帧推理（现状逻辑）。仅适用于短过渡
        （≤3 s）；hold >3 s 后开环纠姿也难拉回。重训 hold 完成并本地复验
        长期稳定后，可切回此模式（改 plugin.py 的 hold_mode 参数）。

    唯一动作类出口是 mod.yaml 里 from=squat_hold 的 squat_up route；
    on_update 内绝不调用 request_state（安全看门狗除外）；on_action 恒 False。
    安全例外：卡帧/非法输出/姿态漂移时 force 切零力矩；IMU 保护由 daemon 直接 force 切换。
    """

    # 开环保持的增益倍率。×8 实测长期零漂移；×4 不足（50 帧滑到 49°）。
    _OPEN_LOOP_GAIN = 8.0
    # 开环看门狗：实际关节偏离固定目标超过此值（被推/踩空/打滑）→ 零力矩。
    _DRIFT_LIMIT = 0.5  # rad

    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[SquatResidualPolicy],
        *, hold_mode: str = "open_loop",
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self._elapsed = 0.0
        if hold_mode not in ("open_loop", "onnx"):
            raise ValueError(f"hold_mode must be open_loop/onnx, got {hold_mode!r}")
        self._hold_mode = hold_mode
        self._fixed_target: np.ndarray | None = None
        self._fixed_kp: np.ndarray | None = None
        self._fixed_kd: np.ndarray | None = None

    @property
    def policy(self) -> SquatResidualPolicy:
        return self._policy.get()

    def on_prepare(
        self, ctx: "RobotControlContext", from_state: "StateBehavior"
    ) -> None:
        ctx.preheat_model(self.policy)

    def on_enter(self, ctx: "RobotControlContext") -> None:
        self._elapsed = 0.0
        if self._hold_mode == "open_loop":
            pol = self.policy
            # 固定目标 = 参考表蹲姿（down 末帧 123）；增益 = 训练 kp/kd × 8
            self._fixed_target = np.asarray(
                pol._q_ref[pol.frames - 1], dtype=np.float32).copy()
            self._fixed_kp = pol._parameters.kp * self._OPEN_LOOP_GAIN
            self._fixed_kd = pol._parameters.kd * self._OPEN_LOOP_GAIN

    def _open_loop_output(self):
        """借用 policy 的输出缓冲发布固定目标（MotorFrame 构造路径复用）。"""
        self.policy.publish_output(
            self._fixed_target, self._fixed_kp, self._fixed_kd, completed=False
        )
        return self.policy.output

    def get_entry_frame(self, ctx: "RobotControlContext") -> MotorFrame:
        if self._hold_mode == "open_loop":
            return self._motor_frame_from_target(ctx, self._open_loop_output().joints)
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self, ctx: "RobotControlContext", dt: float, *, advance: bool
    ) -> MotorFrame:
        # preview 无副作用契约：onnx 模式给策略传 advance=False（帧锁死、
        # previous_action 不变）；open_loop 模式天然无副作用（固定输出）。
        if self._hold_mode == "open_loop":
            del dt, advance
            return self._motor_frame_from_target(ctx, self._open_loop_output().joints)
        del advance
        output = self.policy.step(ctx.inference_frame, dt, advance=False)
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: "RobotControlContext", dt: float) -> None:
        self._elapsed += dt
        policy = self.policy

        if self._hold_mode == "open_loop":
            target = self._fixed_target
            # 看门狗：固定目标未初始化 / 输出非有限 / 实际关节偏离超限 → 零力矩
            if target is None:
                ctx.request_state(ZERO_TORQUE, trigger="squat_hold_no_target", force=True)
                return
            if not np.all(np.isfinite(target)):
                ctx.request_state(ZERO_TORQUE, trigger="squat_hold_bad_output", force=True)
                return
            q_meas = ctx.inference_frame.joints.position
            if float(np.max(np.abs(q_meas - target))) > self._DRIFT_LIMIT:
                ctx.request_state(ZERO_TORQUE, trigger="squat_hold_drift", force=True)
                return
            self._apply_frame(ctx, self._motor_frame_from_target(
                ctx, self._open_loop_output().joints))
            # 注意：这里刻意没有 ctx.request_state —— hold 只能被外部 route 切走。
            return

        # ---- onnx 模式（原逻辑）----
        # 安全看门狗：帧号被意外推进或输出非有限值 → 强制零力矩
        if policy.frame != policy.frames - 1:
            ctx.request_state(ZERO_TORQUE, trigger="squat_hold_bad_frame", force=True)
            return
        if not np.all(np.isfinite(policy.output.joints.position)):
            ctx.request_state(ZERO_TORQUE, trigger="squat_hold_bad_output", force=True)
            return

        self._apply_frame(ctx, self.sample_running_frame(ctx, dt, advance=True))
        # 注意：这里刻意没有 ctx.request_state —— hold 只能被外部 route 切走。

    def on_action(self, ctx: "RobotControlContext", action_name: str) -> bool:
        return False  # 屏蔽 toggle_pause 等所有 action 类事件


__all__ = ["SquatDownState", "SquatHoldState", "SquatUpState"]
