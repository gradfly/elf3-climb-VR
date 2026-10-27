from __future__ import annotations

import time
from threading import Lock
from typing import TYPE_CHECKING, Optional, Protocol

import numpy as np
from numpy.typing import NDArray
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from bxi_example_py_elf3.framework.mod_api import ResourceHandle
from bxi_example_py_elf3.framework.mod_api import RobotControlState
from bxi_example_py_elf3.framework.mod_api import StateBehavior
from bxi_example_py_elf3.framework.mod_api.transition import (
    EntryFrameProvider,
    MotorFrame,
    RunningFrameProvider,
)
from bxi_example_py_elf3.framework.inference import InferenceFrame, PolicyOutput

from .camera_topics import resolve_camera_topics, validate_camera_name
from .depth_projection import ProjectionSpec, project_depth

if TYPE_CHECKING:
    from bxi_example_py_elf3.framework.mod_api import RobotControlContext


class DepthPolicy(Protocol):
    output: PolicyOutput
    depth_update_period: float

    def reset(self, frame: InferenceFrame) -> None:
        ...

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        ...


class NormalDepthState(
    RobotControlState,
    EntryFrameProvider,
    RunningFrameProvider,
):
    """Run a locomotion policy using depth images published by another node."""

    _EXPECTED_SHAPES = {
        "origin_camera": (36, 48),
        "depth_walk": (64, 36),
    }

    def __init__(
        self,
        name: str,
        state_id: int,
        policy: ResourceHandle[DepthPolicy],
        *,
        mode: str,
        camera_name: str,
        depth_image_topic: str,
        camera_info_topic: str,
        depth_uint16_scale: float,
        depth_timeout_sec: float,
        projection: ProjectionSpec,
    ) -> None:
        super().__init__(name, state_id, resources=(policy,))
        if mode not in self._EXPECTED_SHAPES:
            raise ValueError(f"unsupported depth mode: {mode}")
        if bool(depth_image_topic) != bool(camera_info_topic):
            raise ValueError(
                "depth image topic and camera info topic must be configured together"
            )
        if depth_uint16_scale <= 0.0:
            raise ValueError("depth_uint16_scale must be greater than zero")
        if depth_timeout_sec <= 0.0:
            raise ValueError("depth_timeout_sec must be greater than zero")

        self._policy = policy
        self.mode = mode
        self.camera_name = camera_name.strip()
        if self.camera_name:
            self.camera_name = validate_camera_name(self.camera_name)
        self.depth_image_topic = depth_image_topic.strip()
        self.camera_info_topic = camera_info_topic.strip()
        self.depth_configured = bool(self.camera_name or self.depth_image_topic)
        self.expected_depth_shape = self._EXPECTED_SHAPES[mode]
        self.depth_uint16_scale = depth_uint16_scale
        self.depth_timeout_sec = depth_timeout_sec
        self.projection = projection

        self._depth_lock = Lock()
        self._camera_info: Optional[CameraInfo] = None
        self._depth_rotated: Optional[NDArray[np.float32]] = None
        self._latest_depth_frame_id = 0
        self._last_depth_time: Optional[float] = None
        self._policy_depth_rotated: Optional[NDArray[np.float32]] = None
        self._policy_depth_frame_id: Optional[int] = None
        self._last_policy_depth_time: Optional[float] = None
        self._last_running_frame: Optional[MotorFrame] = None
        self._depth_enter_time = time.monotonic()
        self._missing_depth_warned = False
        self._bad_depth_warned = False
        self._depth_timeout_warned = False
        self._depth_subscription = None
        self._camera_info_subscription = None

    @property
    def policy(self) -> DepthPolicy:
        return self._policy.get()

    def on_bind(self, ctx: RobotControlContext) -> None:
        if not self.depth_configured:
            self.logger.warning(
                "depth camera is not configured; set camera_name or both "
                "topic and camera_info_topic"
            )
            return
        node = ctx.ros_node
        if not self.depth_image_topic:
            topic_prefix = str(node.get_parameter("/topic_prefix").value).strip("/")
            self.depth_image_topic, self.camera_info_topic = resolve_camera_topics(
                topic_prefix,
                self.camera_name,
            )
        qos = QoSProfile(
            depth=1,
            durability=qos_profile_sensor_data.durability,
            reliability=qos_profile_sensor_data.reliability,
        )
        self._depth_subscription = node.create_subscription(
            Image,
            self.depth_image_topic,
            self.depth_image_callback,
            qos,
        )
        self._camera_info_subscription = node.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos,
        )
        self.logger.info(
            f"depth state mode={self.mode}, image={self.depth_image_topic}, "
            f"camera_info={self.camera_info_topic}, "
            f"post-rotation shape={self.expected_depth_shape}"
        )

    def on_unbind(self, ctx: RobotControlContext) -> None:
        subscription = self._depth_subscription
        self._depth_subscription = None
        if subscription is not None:
            ctx.ros_node.destroy_subscription(subscription)
        info_subscription = self._camera_info_subscription
        self._camera_info_subscription = None
        if info_subscription is not None:
            ctx.ros_node.destroy_subscription(info_subscription)

    def is_available(self, ctx: RobotControlContext) -> bool:
        """Allow entry only while the continuously subscribed depth feed is fresh."""
        with self._depth_lock:
            depth_rotated = self._depth_rotated
            last_depth_time = self._last_depth_time
            camera_info = self._camera_info
        return (
            self.depth_configured
            and camera_info is not None
            and depth_rotated is not None
            and last_depth_time is not None
            and time.monotonic() - last_depth_time <= self.depth_timeout_sec
        )

    def depth_image_callback(self, msg: Image) -> None:
        depth_meters = self._depth_msg_to_meters(msg)
        if depth_meters is None:
            return

        with self._depth_lock:
            camera_info = self._camera_info
        if camera_info is None:
            self._warn_bad_depth_once(
                f"waiting for CameraInfo: {self.camera_info_topic}"
            )
            return
        try:
            projected, _crop = project_depth(
                depth_meters,
                camera_info,
                self.projection,
            )
        except ValueError as exc:
            self._warn_bad_depth_once(f"cannot project depth image: {exc}")
            return

        depth_rotated = np.ascontiguousarray(
            np.rot90(projected, k=-1).astype(np.float32)
        )
        if depth_rotated.shape != self.expected_depth_shape:
            self._warn_bad_depth_once(
                "unexpected post-rotation depth shape from "
                f"{self.depth_image_topic}: got {depth_rotated.shape}, "
                f"expected {self.expected_depth_shape}"
            )
            return

        now = time.monotonic()
        with self._depth_lock:
            self._depth_rotated = depth_rotated
            self._latest_depth_frame_id += 1
            self._last_depth_time = now
        self._missing_depth_warned = False
        self._bad_depth_warned = False
        self._depth_timeout_warned = False

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if msg.width <= 0 or msg.height <= 0 or len(msg.k) != 9:
            self._warn_bad_depth_once(
                f"invalid CameraInfo from {self.camera_info_topic}"
            )
            return
        if not np.isfinite(np.asarray(msg.k, dtype=np.float64)).all():
            self._warn_bad_depth_once(
                f"non-finite CameraInfo from {self.camera_info_topic}"
            )
            return
        if msg.k[0] <= 0.0 or msg.k[4] <= 0.0:
            self._warn_bad_depth_once(
                f"invalid focal lengths from {self.camera_info_topic}"
            )
            return
        with self._depth_lock:
            self._camera_info = msg

    def _depth_msg_to_meters(
        self,
        msg: Image,
    ) -> Optional[NDArray[np.float32]]:
        encoding = msg.encoding.lower()
        if encoding in ("16uc1", "mono16"):
            dtype = np.dtype(np.uint16).newbyteorder(">" if msg.is_bigendian else "<")
            scale = self.depth_uint16_scale
        elif encoding == "32fc1":
            dtype = np.dtype(np.float32).newbyteorder(">" if msg.is_bigendian else "<")
            scale = 1.0
        else:
            self._warn_bad_depth_once(
                f"unsupported depth image encoding '{msg.encoding}' "
                f"from {self.depth_image_topic}"
            )
            return None

        itemsize = dtype.itemsize
        row_values = int(msg.step) // itemsize
        if msg.width <= 0 or msg.height <= 0 or row_values < msg.width:
            self._warn_bad_depth_once(
                f"invalid depth image layout from {self.depth_image_topic}: "
                f"width={msg.width}, height={msg.height}, step={msg.step}"
            )
            return None

        expected_values = row_values * int(msg.height)
        expected_bytes = expected_values * itemsize
        if len(msg.data) < expected_bytes:
            self._warn_bad_depth_once(
                f"incomplete depth image from {self.depth_image_topic}: "
                f"got {len(msg.data)} bytes, expected {expected_bytes} bytes"
            )
            return None

        data = np.frombuffer(msg.data, dtype=dtype, count=expected_values)
        depth = data.reshape(int(msg.height), row_values)[:, : int(msg.width)]
        return (depth.astype(np.float32) * scale).copy()

    def _warn_bad_depth_once(self, message: str) -> None:
        if self._bad_depth_warned:
            return
        self.logger.warning(message)
        self._bad_depth_warned = True

    def on_prepare(
        self,
        ctx: RobotControlContext,
        from_state: StateBehavior[RobotControlContext],
    ) -> None:
        self.policy.reset(ctx.inference_frame)
        self._depth_enter_time = time.monotonic()
        self._depth_timeout_warned = False
        self._policy_depth_rotated = None
        self._policy_depth_frame_id = None
        self._last_policy_depth_time = None
        self._last_running_frame = None

    def get_entry_frame(self, ctx: RobotControlContext) -> MotorFrame:
        return self._motor_frame_from_target(ctx, self.policy.output.joints)

    def sample_running_frame(
        self,
        ctx: RobotControlContext,
        dt: float,
        *,
        advance: bool,
    ) -> Optional[MotorFrame]:
        if not advance:
            return self._last_running_frame or self.get_entry_frame(ctx)

        depth_for_inference = self._get_depth_for_inference()
        if depth_for_inference is None:
            if not self._missing_depth_warned:
                self.logger.warning(
                    f"waiting for depth image: {self.depth_image_topic}"
                )
                self._missing_depth_warned = True
            return None

        depth_image, depth_frame_id = depth_for_inference
        self.get_cmd_vel(ctx)
        ctx.inference_frame.depth = depth_image
        ctx.inference_frame.depth_frame_id = depth_frame_id
        output = self.policy.step(
            ctx.inference_frame,
            dt,
            advance=True,
        )
        frame = self._motor_frame_from_target(ctx, output.joints)
        self._last_running_frame = frame
        return frame

    def _get_depth_for_inference(
        self,
    ) -> Optional[tuple[NDArray[np.float32], int]]:
        with self._depth_lock:
            latest_depth = self._depth_rotated
            latest_frame_id = self._latest_depth_frame_id
        if latest_depth is None:
            return None

        now = time.monotonic()
        min_interval = float(self.policy.depth_update_period)
        if (
            self._policy_depth_rotated is None
            or self._policy_depth_frame_id is None
            or self._last_policy_depth_time is None
            or now - self._last_policy_depth_time >= min_interval
        ):
            self._policy_depth_rotated = latest_depth
            self._policy_depth_frame_id = latest_frame_id
            self._last_policy_depth_time = now
        return self._policy_depth_rotated, self._policy_depth_frame_id

    def _is_depth_timed_out(self) -> bool:
        with self._depth_lock:
            last_depth_time = self._last_depth_time
        now = time.monotonic()
        reference = (
            self._depth_enter_time if last_depth_time is None else last_depth_time
        )
        return now - reference > self.depth_timeout_sec

    def on_update(self, ctx: RobotControlContext, dt: float) -> None:
        if ctx.is_orientation_unsafe(ctx.current_quat_xyzw):
            ctx.request_state("com.bxi.basic_actions/zero_torque", trigger="safety")
            return

        if self._is_depth_timed_out():
            if not self._depth_timeout_warned:
                self.logger.warning(
                    "depth image timeout, switching to normal: "
                    f"{self.depth_image_topic}"
                )
                self._depth_timeout_warned = True
            ctx.request_state("com.bxi.basic_actions/normal", trigger="no_depth")
            return

        frame = self.sample_running_frame(ctx, dt, advance=True)
        if frame is not None:
            self._apply_frame(ctx, frame)
