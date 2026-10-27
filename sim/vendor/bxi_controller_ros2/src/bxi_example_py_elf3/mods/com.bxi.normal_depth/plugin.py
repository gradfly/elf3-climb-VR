from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
    StateBuildContext,
)

from .depth import (
    HumanoidGaitDepthPolicyIsaaclab,
    HumanoidGaitOriginCameraPolicyIsaaclab,
)
from .depth_projection import ProjectionSpec
from .state import NormalDepthState


LEGACY_POLICY = ResourceKey[HumanoidGaitDepthPolicyIsaaclab](
    "com.bxi.normal_depth/legacy_policy"
)
ORIGIN_POLICY = ResourceKey[HumanoidGaitOriginCameraPolicyIsaaclab](
    "com.bxi.normal_depth/origin_policy"
)


def _load_legacy_policy(
    context: ResourceLoadContext,
) -> HumanoidGaitDepthPolicyIsaaclab:
    return HumanoidGaitDepthPolicyIsaaclab(
        str(context.asset("assets/normal_depth.onnx"))
    )


def _load_origin_policy(
    context: ResourceLoadContext,
) -> HumanoidGaitOriginCameraPolicyIsaaclab:
    return HumanoidGaitOriginCameraPolicyIsaaclab(
        str(context.asset("assets/dagger2.onnx"))
    )


def _build_normal_depth_state(
    state: StateBuildContext,
    legacy_policy,
    origin_policy,
) -> NormalDepthState:
    mode = state.string_param("mode", "origin_camera")
    if mode == "origin_camera":
        policy = origin_policy
        projection = ProjectionSpec(
            output_width=36,
            output_height=48,
            horizontal_fov_deg=state.float_param("horizontal_fov", 45.2),
            vertical_fov_deg=state.float_param("vertical_fov", 58.0616969),
            minimum_m=state.float_param("min_dist", 0.2),
            maximum_m=state.float_param("max_dist", 3.0),
        )
    elif mode == "depth_walk":
        policy = legacy_policy
        projection = ProjectionSpec(
            output_width=64,
            output_height=36,
            horizontal_fov_deg=state.float_param("horizontal_fov", 89.24),
            vertical_fov_deg=state.float_param("vertical_fov", 58.06),
            minimum_m=state.float_param("min_dist", 0.2),
            maximum_m=state.float_param("max_dist", 2.5),
        )
    else:
        raise ValueError(
            f"state '{state.name}' param 'mode' must be "
            "'origin_camera' or 'depth_walk'"
        )

    return NormalDepthState(
        state.name,
        state.state_id,
        policy,
        mode=mode,
        camera_name=state.string_param("camera_name", "head_depth_camera"),
        depth_image_topic=state.string_param("topic", ""),
        camera_info_topic=state.string_param("camera_info_topic", ""),
        depth_uint16_scale=state.float_param("depth_uint16_scale", 0.001),
        depth_timeout_sec=state.float_param("depth_timeout_sec", 1.0),
        projection=projection,
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(LEGACY_POLICY, _load_legacy_policy, policy="on_demand")
    context.register_resource(ORIGIN_POLICY, _load_origin_policy, policy="on_demand")
    legacy_policy = context.resource(LEGACY_POLICY)
    origin_policy = context.resource(ORIGIN_POLICY)

    return ModDefinition(
        state_factories={
            "normal_depth": lambda state: _build_normal_depth_state(
                state,
                legacy_policy,
                origin_policy,
            )
        }
    )
