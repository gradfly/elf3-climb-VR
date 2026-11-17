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
            end_transition="soft_switch",
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
            end_frame_trim=0,
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

    唯一动作类出口是 mod.yaml 里 from=squat_hold 的 squat_up route；
    on_update 内绝不调用 request_state；on_action 恒 False（屏蔽 toggle_pause 等）。
    安全例外：卡帧/非法输出时 force 切零力矩；IMU 保护由 daemon 直接 force 切换。
    """

    def __init__(
        self, name: str, state_id: int, policy: ResourceHandle[SquatResidualPolicy]
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        self._policy = policy
        self._elapsed = 0.0

    @property
    def policy(self) -> SquatResidualPolicy:
        return self._policy.get()

    def on_prepare(
        self, ctx: "RobotControlContext", from_state: "StateBehavior"
    ) -> None:
        ctx.preheat_model(self.policy)

    def on_enter(self, ctx: "RobotControlContext") -> None:
        self._elapsed = 0.0

    def get_entry_frame(self, ctx: "RobotControlContext") -> MotorFrame:
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self, ctx: "RobotControlContext", dt: float, *, advance: bool
    ) -> MotorFrame:
        # hold 冻结帧：无论 advance 取何值，都给策略传 False —— 帧号锁在
        # down 末帧、previous_action 不变。这样正常 update 与 blend preview
        # 行为一致，且严格满足 RunningFrameProvider「preview 无副作用」契约。
        del advance
        output = self.policy.step(ctx.inference_frame, dt, advance=False)
        return self._motor_frame_from_target(ctx, output.joints)

    def on_update(self, ctx: "RobotControlContext", dt: float) -> None:
        self._elapsed += dt
        policy = self.policy

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
