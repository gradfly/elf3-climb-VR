"""com.bxi.squat_actions —— 自训蹲起策略的 Mod 入口。

三份 ONNX（down/up/hold）各一个 Policy 资源，按需加载；
state_factories 的三个键必须与 mod.yaml 的 states 完全一致。
"""

from __future__ import annotations

from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)

from .squat_residual_policy import SquatResidualPolicy
from .state import SquatDownState, SquatHoldState, SquatUpState

POLICY_DOWN = ResourceKey[SquatResidualPolicy]("com.bxi.squat_actions/down_policy")
POLICY_HOLD = ResourceKey[SquatResidualPolicy]("com.bxi.squat_actions/hold_policy")
POLICY_UP = ResourceKey[SquatResidualPolicy]("com.bxi.squat_actions/up_policy")


def _make_loader(mode: str):
    def _load(context: ResourceLoadContext) -> SquatResidualPolicy:
        return SquatResidualPolicy(
            str(context.asset(f"assets/squat_{mode}.onnx")),
            str(context.asset("assets/squat_ref_v1.npz")),
            mode=mode,
        )

    return _load


def create_mod(context: ModLoadContext) -> ModDefinition:
    for mode, key in (
        ("down", POLICY_DOWN),
        ("hold", POLICY_HOLD),
        ("up", POLICY_UP),
    ):
        context.register_resource(key, _make_loader(mode), policy="on_demand")

    return ModDefinition(
        state_factories={
            "squat_down": lambda state: SquatDownState(
                state.name, state.state_id, context.resource(POLICY_DOWN)
            ),
            "squat_hold": lambda state: SquatHoldState(
                state.name, state.state_id, context.resource(POLICY_HOLD),
                # open_loop：固定参考蹲姿 + kp×8，长期保持 60 s 零漂移（默认）。
                # 重训 hold 完成并复验长期稳定后可改回 "onnx"（短过渡省电模式）。
                hold_mode="open_loop",
            ),
            "squat_up": lambda state: SquatUpState(
                state.name, state.state_id, context.resource(POLICY_UP)
            ),
        }
    )
