from pathlib import Path
import contextlib
import io

import numpy as np

from bxi_example_py_elf3.framework.inference import (
    HistoryBuffer,
    InferenceFrame,
    InferenceRuntime,
    JointPolicy,
    ModelSpec,
    PolicyJointContract,
    PolicyOutput,
    default_runtime,
)
from bxi_example_py_elf3.framework.mod_api.geometry import get_gravity_orientation
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS

_CV2 = None
_CV2_IMPORT_TRIED = False


def _get_cv2():
    global _CV2, _CV2_IMPORT_TRIED
    if _CV2_IMPORT_TRIED:
        return _CV2
    _CV2_IMPORT_TRIED = True
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            import cv2 as cv2_module
        _CV2 = cv2_module
    except Exception:
        _CV2 = None
    return _CV2


class HumanoidGaitDepthPolicyIsaaclab(JointPolicy):
    """带深度相机输入的 ELF3 / BX 29DoF 行走策略部署类。"""

    joint_contract = PolicyJointContract(
        observation=ELF3_ISAAC_PARAMETERS.layout,
        action=ELF3_ISAAC_PARAMETERS.layout,
    )

    def __init__(
        self,
        model: str | ModelSpec,
        cmd_is_joystick_ratio: bool = False,
        depth_profile: str = "default",
        *,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ):
        """
        Args:
            model: 模型路径或包含多个后端模型的 ModelSpec。
            cmd_is_joystick_ratio: True 时按 deploy_mujoco_bx.py 的摇杆比例命令处理；
                False 时把 cmd_vel 当作实际速度命令，适配 bxi_sim.py 的调用方式。
            depth_profile: 深度相机预处理配置，"default" 对应旧 8 帧 WAQ-depth，
                "origin_camera" 对应 walk_bx_waq_origin_camera_29 的单帧窄 FOV 相机。
        """
        super().__init__()
        model_source = (
            model if isinstance(model, ModelSpec) else self._resolve_policy_path(model)
        )
        self._runtime = runtime or default_runtime()
        self._policy_name = "depth"
        self.cmd_is_joystick_ratio = cmd_is_joystick_ratio
        self.depth_profile = depth_profile
        self.num_actions = self.joint_contract.action.dof_num
        self.num_obs = 96
        self.history_length = 10
        self.control_dt = 0.02
        self.depth_update_period = 0.02
        self.if_use_stand = True
        self.force_phase_active = False
        self.clip_action_limit = 100.0
        self._parameters = ELF3_ISAAC_PARAMETERS

        self.range_velx = np.array([0.0, 1.0], dtype=np.float32)
        self.range_vely = np.array([0.0, 0.0], dtype=np.float32)
        self.range_velz = np.array([-1.57, 1.57], dtype=np.float32)
        self.cmd_scale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 1.0
        self.ang_vel_scale = 1.0
        self.cmd = np.zeros(3, dtype=np.float32)

        self.camera_name = "body_depth_camera"
        self.output_resolution = [64, 36]
        self.width = 36
        self.height = 64
        self.crop_region = [6, 9, 16, 16]
        self.gaussian_kernel_size = (3, 3)
        self.gaussian_sigma = 1.0
        self.depth_range = [0.2, 2.5]
        self.depth_normalize = True
        self.depth_output_range = [0.0, 1.0]
        self.depth_history_len = 8
        self.depth_h = 21
        self.depth_w = 32
        self.depth_input_rank = 4
        self.depth_rotate_transpose = True
        self.depth_crop_rows = None
        self.depth_crop_rows_after_noise = False
        self.depth_hidden_body_names = ()
        # self.depth_obs_indices = np.array(
        #     [-15, -13, -11, -9, -7, -5, -3, -1], dtype=np.int32
        # )
        # self.depth_obs_indices = np.array(
        #     [-29, -25, -21, -17, -13, -9, -5, -1], dtype=np.int32
        # )
        self.depth_obs_indices = np.array(
            [-36, -31, -26, -21, -16, -11, -6, -1], dtype=np.int32
        )
        self.depth_buffer_length = 37
        self._apply_depth_profile(depth_profile)

        self._single_obs = np.empty(self.num_obs, dtype=np.float32)
        self._action = np.zeros(self.num_actions, dtype=np.float32)
        self._previous_action = np.zeros(self.num_actions, dtype=np.float32)
        self._target = self._target_buffer.position
        np.copyto(self._target, self._parameters.default_position)
        self.actor_obs_buffer = np.zeros(
            self.history_length * self.num_obs, dtype=np.float32
        )
        self._obs_history = HistoryBuffer(
            self.history_length,
            (self.num_obs,),
            dtype=np.float32,
        )
        self.counter = 0
        self.last_depth_frame_id = None
        self._depth_previewed = False

        self._initialize_model(model_source, backend)
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    def _apply_depth_profile(self, depth_profile):
        if depth_profile in ("default", None):
            return
        if depth_profile != "origin_camera":
            raise ValueError(f"Unsupported depth profile: {depth_profile}")

        self.camera_name = "origin_body_depth_camera"
        self.width = 48
        self.height = 36
        self.crop_region = None
        self.depth_rotate_transpose = True
        # self.depth_crop_rows = (6, 42)
        self.depth_crop_rows = (2, 38)
        # self.depth_crop_rows = (0, 36)
        self.depth_crop_rows_after_noise = True
        self.depth_range = [0.2, 3.0]
        self.depth_output_range = [-0.5, 0.5]
        self.depth_update_period = 0.05
        self.depth_history_len = 1
        self.depth_h = 36
        self.depth_w = 36
        self.depth_obs_indices = np.array([-1], dtype=np.int32)
        self.depth_buffer_length = 1
        self.depth_hidden_body_names = ("torso_link",)
        self.range_velx = np.array([0.0, 0.8], dtype=np.float32)
        self.range_vely = np.array([-0.5, 0.5], dtype=np.float32)
        self.range_velz = np.array([-1.57, 1.57], dtype=np.float32)

    def _initialize_model(self, model, backend):
        spec = (
            model
            if isinstance(model, ModelSpec)
            else ModelSpec.portable_onnx(
                model,
                input_names=(),
                output_names=(),
            )
        )
        self._backend = self._runtime.open_backend(
            spec,
            backend=backend,
        )
        self._configure_model_io()
        self._allocate_model_buffers()
        self._warmup_depth_preprocessing()
        self._clear_state()
        self._warmup_policy()

    def _clear_state(self) -> None:
        required_size = self.history_length * self.num_obs
        if self.actor_obs_buffer.size != required_size:
            self.actor_obs_buffer = np.empty(required_size, dtype=np.float32)
            self._obs_history = HistoryBuffer(
                self.history_length,
                (self.num_obs,),
                dtype=np.float32,
            )
        self.actor_obs_buffer.fill(0.0)
        self._obs_history.clear()
        self._depth_history.clear()
        self._depth_storage.fill(0.0)
        if hasattr(self, "_selected_depth"):
            self._selected_depth.fill(0.0)
        if hasattr(self, "_single_input_buffer"):
            self._single_input_buffer.fill(0.0)
        if hasattr(self, "_recurrent_state"):
            self._recurrent_state.fill(0.0)
        self._previous_action.fill(0.0)
        self._action.fill(0.0)
        np.copyto(self._target, self._parameters.default_position)
        self.cmd.fill(0.0)
        self.counter = 0
        self.last_depth_frame_id = None
        self._depth_previewed = False

    def reset(self, frame: InferenceFrame) -> None:
        self.bind_joints(frame)
        self._clear_state()
        self.publish_output(
            self._target,
            self._parameters.kp,
            self._parameters.kd,
        )

    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        del dt
        joints = self.bind_joints(frame)
        cmd_vel = frame.command
        if cmd_vel is None:
            raise ValueError("HumanoidGaitDepthPolicyIsaaclab requires frame.command")
        obs = self._build_observation(
            joints.position,
            joints.velocity,
            frame.quat_wxyz,
            frame.angular_velocity,
            cmd_vel,
            advance=advance,
        )
        self._prepare_policy_inputs(
            obs,
            frame.depth,
            depth_frame_id=frame.depth_frame_id,
            advance=advance,
        )
        ort_outputs = self._backend.run(self._policy_inputs)
        if advance and self.recurrent_output_name is not None:
            np.copyto(
                self._recurrent_state,
                ort_outputs[self.recurrent_output_name],
            )
        action_out = np.asarray(ort_outputs[self.action_output_name]).reshape(-1)
        np.clip(
            action_out[: self.num_actions],
            -self.clip_action_limit,
            self.clip_action_limit,
            out=self._action,
        )
        if advance:
            np.copyto(self._previous_action, self._action)

        np.multiply(
            self._action,
            self._parameters.action_scale,
            out=self._target,
        )
        self._target += self._parameters.default_position
        return self.output

    def _build_observation(self, qj, dqj, quat, omega, cmd_vel, *, advance: bool):
        self._update_command(cmd_vel)

        vel_norm = np.linalg.norm(self.cmd)
        if advance:
            if self.if_use_stand:
                if vel_norm > 0.1 or self.force_phase_active:
                    self.counter += 1
                else:
                    self.counter = 0
            else:
                self.counter += 1

        qj = np.asarray(qj, dtype=np.float32)[: self.num_actions]
        dqj = np.asarray(dqj, dtype=np.float32)[: self.num_actions]

        obs = self._single_obs
        np.multiply(omega, self.ang_vel_scale, out=obs[0:3])
        obs[3:6] = get_gravity_orientation(quat)
        np.multiply(self.cmd, self.cmd_scale, out=obs[6:9])
        np.subtract(qj, self._parameters.default_position, out=obs[9:38])
        obs[9:38] *= self.dof_pos_scale
        np.multiply(dqj, self.dof_vel_scale, out=obs[38:67])
        obs[67:96] = self._previous_action

        if self.num_obs == 98:
            phase = (self.counter * self.control_dt) % 1.0
            obs[96] = np.sin(2 * np.pi * phase)
            obs[97] = np.cos(2 * np.pi * phase)
        if obs.shape[0] != self.num_obs:
            raise ValueError(
                f"AMP depth obs dim mismatch: got {obs.shape[0]}, "
                f"expected {self.num_obs}"
            )
        np.clip(obs, -100, 100, out=obs)
        return obs

    def _preprocess_depth(self, depth_image):
        if depth_image is None:
            return self._zero_depth

        depth_image = np.asarray(depth_image, dtype=np.float32)
        if depth_image.shape == (self.depth_h, self.depth_w):
            return depth_image

        if self.depth_rotate_transpose:
            depth_image = np.flip(np.transpose(depth_image, (1, 0)), axis=0)
        if self.crop_region is not None:
            top, _bottom, left, _right = self.crop_region
            target_h = self.depth_w
            target_w = self.depth_h
            start_h = left
            start_w = top
            depth_image = depth_image[
                start_w : start_w + target_w, start_h : start_h + target_h
            ]
        if self.depth_crop_rows is not None and not self.depth_crop_rows_after_noise:
            start, end = self.depth_crop_rows
            depth_image = depth_image[start:end, :]

        cv2_module = _get_cv2()
        if self.gaussian_kernel_size is not None and cv2_module is not None:
            depth_image = cv2_module.GaussianBlur(
                depth_image.astype(np.float32),
                self.gaussian_kernel_size,
                self.gaussian_sigma,
            )

        d_min, d_max = self.depth_range
        depth_image = np.clip(depth_image, d_min, d_max)
        if self.depth_normalize:
            depth_image = (depth_image - d_min) / max(d_max - d_min, 1e-6)
            out_min, out_max = self.depth_output_range
            depth_image = depth_image * (out_max - out_min) + out_min
        if self.depth_crop_rows is not None and self.depth_crop_rows_after_noise:
            start, end = self.depth_crop_rows
            depth_image = depth_image[start:end, :]

        if depth_image.shape != (self.depth_h, self.depth_w):
            raise ValueError(
                f"AMP depth image shape mismatch: got {depth_image.shape}, "
                f"expected {(self.depth_h, self.depth_w)}"
            )
        return np.asarray(depth_image, dtype=np.float32)

    def _prepare_policy_inputs(
        self, obs, depth_image, depth_frame_id=None, *, advance: bool
    ):
        if self.model_input_mode == "history_only":
            self._update_obs_history(obs, advance=advance)
            return

        depth_buffer = self._get_depth_image_downsample_obs(
            depth_image,
            depth_frame_id=depth_frame_id,
            advance=advance,
        )
        if self.model_input_mode == "single_obs_depth":
            flat = self._single_input_buffer[0]
            flat[: self.num_obs] = obs
            flat[self.num_obs :] = depth_buffer.reshape(-1)
            return

        self._update_obs_history(obs, advance=advance)

    def _update_obs_history(self, obs, *, advance: bool):
        if not self._obs_history.initialized:
            if advance:
                self._obs_history.fill(obs)
                self._obs_history.write_into(self.actor_obs_buffer)
            else:
                self.actor_obs_buffer.reshape(self.history_length, self.num_obs)[
                    ...
                ] = obs
        elif advance:
            self._obs_history.append(obs)
            self._obs_history.write_into(self.actor_obs_buffer)
        else:
            self._obs_history.preview_append_into(obs, self.actor_obs_buffer)
        return self.actor_obs_buffer.reshape(1, -1)

    def _get_depth_image_downsample_obs(
        self,
        depth_image,
        depth_frame_id=None,
        *,
        advance: bool,
    ):
        if advance and depth_frame_id is not None:
            if (
                self.last_depth_frame_id is not None
                and depth_frame_id == self.last_depth_frame_id
                and self._depth_history.initialized
                and not self._depth_previewed
            ):
                return self._selected_depth

            self.last_depth_frame_id = depth_frame_id

        depth = self._preprocess_depth(depth_image)
        if not advance:
            if self._depth_history.initialized:
                self._depth_history.preview_append_into(depth, self._depth_storage)
            else:
                self._depth_storage[...] = depth
            self._depth_previewed = True
        else:
            if self._depth_history.initialized:
                self._depth_history.append(depth)
            else:
                self._depth_history.fill(depth)
            self._depth_history.write_into(self._depth_storage)
            self._depth_previewed = False
        np.take(
            self._depth_storage,
            self.depth_obs_indices,
            axis=0,
            out=self._selected_depth,
        )
        return self._selected_depth

    def _update_command(self, cmd_vel):
        cmd_vel = np.asarray(cmd_vel, dtype=np.float32)
        if self.cmd_is_joystick_ratio:
            self.cmd[0] = np.clip(
                cmd_vel[0] * self.range_velx[1],
                self.range_velx[0],
                self.range_velx[1],
            )
            self.cmd[1] = np.clip(
                cmd_vel[1] * self.range_vely[1],
                self.range_vely[0],
                self.range_vely[1],
            )
            self.cmd[2] = np.clip(
                cmd_vel[2] * self.range_velz[1],
                self.range_velz[0],
                self.range_velz[1],
            )
        else:
            self.cmd[0] = np.clip(cmd_vel[0], self.range_velx[0], self.range_velx[1])
            self.cmd[1] = np.clip(cmd_vel[1], self.range_vely[0], self.range_vely[1])
            self.cmd[2] = np.clip(cmd_vel[2], self.range_velz[0], self.range_velz[1])

    def _configure_model_io(self):
        inputs = self._backend.input_names
        outputs = self._backend.output_names
        if not inputs or not outputs:
            raise ValueError("Depth policy model must declare inputs and outputs")
        self.input_name = inputs[0]
        action_output_index = 0
        for i, output_name in enumerate(outputs):
            out_name = output_name.lower()
            if any(token in out_name for token in ("action", "actions", "policy")):
                action_output_index = i
        self.action_output_name = outputs[action_output_index]
        self.recurrent_input_name = None
        self.recurrent_output_name = None
        self.recurrent_state_shape = None

        self.model_input_mode = "history_only"
        self.obs_input_name = self.input_name
        self.depth_input_name = None
        if len(inputs) == 1:
            flat_dim = self._shape_dim(self._backend.input_shape(inputs[0])[1])
            if flat_dim == self.history_length * self.num_obs:
                return
            self.model_input_mode = "single_obs_depth"
            self.history_length = 1
            actor_proprio_dim = (
                flat_dim - self.depth_history_len * self.depth_h * self.depth_w
            )
            if actor_proprio_dim != self.num_obs:
                raise ValueError(
                    "Single-input ONNX proprio dim mismatch: "
                    f"got {actor_proprio_dim}, "
                    f"expected {self.num_obs}"
                )
            return

        self.model_input_mode = "multi_input_history"
        self.obs_input_name = inputs[0]
        self.depth_input_name = inputs[1]
        recurrent_inputs = inputs[2:]
        if len(recurrent_inputs) > 1:
            raise ValueError(
                "Depth policy supports at most one recurrent input, got "
                f"{recurrent_inputs}"
            )
        if recurrent_inputs:
            recurrent_outputs = tuple(
                name for name in outputs if name != self.action_output_name
            )
            if len(recurrent_outputs) != 1:
                raise ValueError(
                    "Recurrent depth policy must have one state output in "
                    f"addition to actions, got {recurrent_outputs}"
                )
            self.recurrent_input_name = recurrent_inputs[0]
            self.recurrent_output_name = recurrent_outputs[0]
            input_state_shape = self._resolve_recurrent_shape(
                self._backend.input_shape(self.recurrent_input_name)
            )
            output_state_shape = self._resolve_recurrent_shape(
                self._backend.output_shape(self.recurrent_output_name)
            )
            if input_state_shape != output_state_shape:
                raise ValueError(
                    "Recurrent depth policy state shape mismatch: "
                    f"input={input_state_shape}, output={output_state_shape}"
                )
            self.recurrent_state_shape = input_state_shape
        self.history_length = int(
            self._shape_dim(self._backend.input_shape(inputs[0])[1]) / self.num_obs
        )
        depth_shape = self._backend.input_shape(inputs[1])
        self.depth_input_rank = len(depth_shape)
        if self.depth_input_rank == 3:
            self.depth_history_len = 1
            self.depth_h = self._shape_dim(depth_shape[1])
            self.depth_w = self._shape_dim(depth_shape[2])
            self.depth_obs_indices = np.array([-1], dtype=np.int32)
            self.depth_buffer_length = 1
        elif self.depth_input_rank == 4:
            self.depth_history_len = self._shape_dim(depth_shape[1])
            self.depth_h = self._shape_dim(depth_shape[2])
            self.depth_w = self._shape_dim(depth_shape[3])
        else:
            raise ValueError(
                f"Unsupported depth input rank: {self.depth_input_rank}, "
                f"shape={depth_shape}"
            )

    def _allocate_model_buffers(self):
        required_obs_size = self.history_length * self.num_obs
        if self.actor_obs_buffer.size != required_obs_size:
            self.actor_obs_buffer = np.empty(required_obs_size, dtype=np.float32)
            self._obs_history = HistoryBuffer(
                self.history_length,
                (self.num_obs,),
                dtype=np.float32,
            )
        self._selected_depth = np.empty(
            (self.depth_obs_indices.size, self.depth_h, self.depth_w),
            dtype=np.float32,
        )
        self._selected_depth.fill(0.0)
        self._depth_history = HistoryBuffer(
            self.depth_buffer_length,
            (self.depth_h, self.depth_w),
            dtype=np.float32,
        )
        self._depth_storage = np.zeros(
            (self.depth_buffer_length, self.depth_h, self.depth_w),
            dtype=np.float32,
        )
        self._zero_depth = np.zeros((self.depth_h, self.depth_w), dtype=np.float32)
        if self.model_input_mode == "history_only":
            self._policy_inputs = {
                self.input_name: self.actor_obs_buffer.reshape(1, -1),
            }
        elif self.model_input_mode == "single_obs_depth":
            self._single_input_buffer = np.empty(
                (1, self.num_obs + self._selected_depth.size),
                dtype=np.float32,
            )
            self._single_input_buffer.fill(0.0)
            self._policy_inputs = {self.input_name: self._single_input_buffer}
        else:
            depth_view = (
                self._selected_depth[-1].reshape(1, self.depth_h, self.depth_w)
                if self.depth_input_rank == 3
                else self._selected_depth.reshape(
                    1, self.depth_obs_indices.size, self.depth_h, self.depth_w
                )
            )
            self._policy_inputs = {
                self.obs_input_name: self.actor_obs_buffer.reshape(1, -1),
                self.depth_input_name: depth_view,
            }
        if self.recurrent_input_name is not None:
            self._recurrent_state = np.zeros(
                self.recurrent_state_shape,
                dtype=np.float32,
            )
            self._policy_inputs[self.recurrent_input_name] = self._recurrent_state

    def _warmup_depth_preprocessing(self) -> None:
        """Move lazy OpenCV initialization out of the first control cycle."""
        d_min, d_max = self.depth_range
        sample = np.full(
            (self.height, self.width),
            (d_min + d_max) * 0.5,
            dtype=np.float32,
        )
        self._preprocess_depth(sample)

    def _warmup_policy(self):
        for _ in range(5):
            self._backend.run(self._policy_inputs)

    def close(self):
        self._backend.close()

    @staticmethod
    def _shape_dim(dim):
        if isinstance(dim, int):
            return dim
        if isinstance(dim, str):
            raise ValueError(f"Dynamic ONNX dimension is unsupported here: {dim}")
        return int(dim)

    @staticmethod
    def _resolve_recurrent_shape(shape):
        resolved = []
        for dimension in shape:
            if isinstance(dimension, int):
                if dimension <= 0:
                    raise ValueError(
                        "Recurrent depth policy dimensions must be positive, "
                        f"got {shape}"
                    )
                resolved.append(dimension)
            elif isinstance(dimension, str) and "batch" in dimension.lower():
                resolved.append(1)
            else:
                raise ValueError(
                    "Unsupported recurrent depth policy dimension "
                    f"{dimension!r} in {shape}"
                )
        return tuple(resolved)

    @staticmethod
    def _resolve_policy_path(path):
        project_root = Path(__file__).resolve().parents[3]
        expanded = Path(path).expanduser()
        if expanded.is_absolute() and expanded.exists():
            return str(expanded)

        candidates = [
            Path.cwd() / expanded,
            project_root / expanded,
            project_root / "deploy_bx/deploy/policy/loco29_depth/model" / expanded,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(project_root / expanded)


class HumanoidGaitOriginCameraPolicyIsaaclab(HumanoidGaitDepthPolicyIsaaclab):
    """walk_bx_waq_origin_camera_29 单帧 origin-camera 部署类。"""

    def __init__(
        self,
        model: str | ModelSpec,
        cmd_is_joystick_ratio: bool = False,
        *,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ):
        super().__init__(
            model,
            cmd_is_joystick_ratio=cmd_is_joystick_ratio,
            depth_profile="origin_camera",
            runtime=runtime,
            backend=backend,
        )
