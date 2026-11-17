"""自训蹲起 ONNX 的适配层（obs 拼装 + 残差后处理 + 参考帧计数）。

三份 ONNX（squat_down / squat_up / squat_hold）的契约：
  - INPUT  obs  [1, 132]（down/up）/ [1, 130]（hold，无 traj_progress）
  - OUTPUT actions [1, 29] —— 是**残差**，不是关节目标
  - 12 键 CSV metadata；归一化已烘焙进图，部署侧不能再归一化
  - joint_names 有 31 个（末两个是被动头关节 head_z/head_y），action 是 29 维

obs 分块（严格按 metadata 的 observation_names 顺序，见 docs/squat_rl_training.md §7）：
  0:3   base_lin_vel       真机无 odom → 填 0
  3:6   base_ang_vel       IMU 角速度
  6:9   projected_gravity  IMU 四元数反投影
  9:40  joint_pos          29 关节 + 头关节默认角(2)
  40:71 joint_vel          29 关节速度 + 头关节 0(2)
  71:100 actions           上一步**截幅前 raw** 输出
  100:129 traj_joint_err   q_meas - q_ref[frame]
  129:130 traj_height_err  z_meas - z_ref[frame]（真机无基座高度 → 填 0）
  130:132 traj_progress    [sin(pi*p), cos(pi*p)]（hold 无此块）

动作后处理：q_target = q_ref[frame] + scale * clip(a, -clip_rad/scale, clip_rad/scale)
clip_rad = 腿 0.35 / 腰 0.15 / 臂 0.12 rad。
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from bxi_example_py_elf3.framework.inference.api import InferenceFrame, PolicyOutput
from bxi_example_py_elf3.framework.inference.contract import PolicyJointContract
from bxi_example_py_elf3.framework.inference.model import ModelSpec
from bxi_example_py_elf3.framework.inference.policy import JointPolicy
from bxi_example_py_elf3.framework.inference.runtime import (
    InferenceRuntime,
    default_runtime,
)
from bxi_example_py_elf3.framework.joints import JointParameterSet
from bxi_example_py_elf3.framework.mod_api.geometry import get_gravity_orientation
from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS

# 控制周期 / 参考表推进（20 ms 一帧，见 docs/squat_rl_training.md §6.3）
DT = 0.02

# 期望的观测项顺序（down/up 9 项；hold 去掉末尾 traj_progress）
_OBS_TERMS_MOTION = (
    "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
    "actions", "traj_joint_err", "traj_height_err", "traj_progress",
)
_OBS_TERMS_HOLD = _OBS_TERMS_MOTION[:-1]

# 残差限幅（rad）：腰 3 / 腿 12 / 臂 14，顺序与 ELF3_POLICY_JOINTS 一致
_CLIP_WAIST = 0.15
_CLIP_LEG = 0.35
_CLIP_ARM = 0.12


def _csv_floats(value: str) -> np.ndarray:
    """解析裸 CSV 字符串（metadata 里的数组都是这种格式）。"""
    return np.fromstring(value, sep=",", dtype=np.float32)


def _make_clip_rad(dof: int) -> np.ndarray:
    clip = np.full(dof, _CLIP_ARM, dtype=np.float32)
    clip[0:3] = _CLIP_WAIST          # waist_y / waist_x / waist_z
    clip[3:15] = _CLIP_LEG           # 左右腿各 6 关节
    clip[15:dof] = _CLIP_ARM         # 左右臂各 7 关节
    return clip


class SquatResidualPolicy(JointPolicy):
    """一份蹲起 ONNX + 一条参考时间轴 = 一个可回放的残差策略。"""

    joint_contract = PolicyJointContract(
        observation=ELF3_POLICY_JOINTS,
        action=ELF3_POLICY_JOINTS,
    )

    def __init__(
        self,
        onnx_path: str,
        ref_table_path: str,
        *,
        mode: Literal["down", "up", "hold"],
        gain_scale: float = 1.0,
        runtime: InferenceRuntime | None = None,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if mode not in ("down", "up", "hold"):
            raise ValueError(f"mode must be down/up/hold, got {mode!r}")
        self.mode = mode
        self.gain_scale = float(gain_scale)

        spec = ModelSpec.portable_onnx(
            onnx_path, input_names=("obs",), output_names=("actions",)
        )
        self._runtime = runtime or default_runtime()
        self._backend = self._runtime.open_backend(spec, backend=backend)
        metadata = dict(self._backend.metadata)

        self._validate_metadata(metadata)

        # 参数：metadata 的 default_pos 是 31 维（含两个头关节），
        # JointParameterSet 契约是 29 维 → 只取前 29。
        self._parameters = JointParameterSet.from_arrays(
            self.joint_contract.action,
            default_position=_csv_floats(metadata["default_joint_pos"])[:29],
            kp=_csv_floats(metadata["joint_stiffness"]),
            kd=_csv_floats(metadata["joint_damping"]),
            action_scale=_csv_floats(metadata["action_scale"]),
        )
        self._head_default = _csv_floats(metadata["default_joint_pos"])[29:31]
        if self._head_default.shape != (2,):
            raise ValueError("default_joint_pos metadata must carry 31 values")

        self._scale = self._parameters.action_scale
        self._clip_rad = _make_clip_rad(self.joint_contract.action.dof_num)

        # 参考表
        ref = np.load(ref_table_path, allow_pickle=False)
        traj_key = "down" if mode == "hold" else mode  # hold 冻结在 down 末帧
        self._q_ref = np.asarray(ref[f"{traj_key}/q"], dtype=np.float32)
        self._z_ref = np.asarray(ref[f"{traj_key}/z"], dtype=np.float32)
        self._frames = int(self._q_ref.shape[0])
        if self._frames < 2 or self._z_ref.shape[0] != self._frames:
            raise ValueError("bad squat reference table shape")
        self._last_frame = self._frames - 1
        # hold 冻结帧 = down 表末帧 123（export_onnx.py SQUAT_MODES hold-squat freeze=-1）
        self._freeze_frame = self._last_frame if mode == "hold" else None

        # 观测缓冲与状态
        self._obs_dim = 130 if mode == "hold" else 132
        self._obs = np.zeros((1, self._obs_dim), dtype=np.float32)
        self._inputs = {"obs": self._obs}
        self._previous_action = np.zeros(
            self.joint_contract.action.dof_num, dtype=np.float32
        )
        self._action_checkpoint = np.empty_like(self._previous_action)
        self._frame = 0.0
        self._target = self._target_buffer.position

        self._backend.warmup(self._inputs, self._runtime.options.warmup_runs)
        self.reset_to_start()

    # ------------------------------------------------------------------ metadata
    def _validate_metadata(self, metadata: dict[str, str]) -> None:
        required = (
            "joint_names", "default_joint_pos", "joint_stiffness", "joint_damping",
            "action_scale", "observation_names", "observation_terms_scale",
            "observation_terms_history_length",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"squat ONNX metadata missing keys: {missing}")

        names = tuple(n.strip() for n in metadata["joint_names"].split(","))
        if names[:29] != self.joint_contract.observation.names:
            raise ValueError(
                "squat ONNX joint_names[:29] does not match ELF3_POLICY_JOINTS"
            )
        if len(names) != 31:
            raise ValueError(f"expected 31 joint_names in metadata, got {len(names)}")

        expected_terms = _OBS_TERMS_HOLD if self.mode == "hold" else _OBS_TERMS_MOTION
        terms = tuple(t.strip() for t in metadata["observation_names"].split(","))
        if terms != expected_terms:
            raise ValueError(
                f"observation_names mismatch for mode {self.mode!r}: {terms}"
            )

        # 无历史帧、观测 scale 全 1（归一化已烘焙进图）
        hist = _csv_floats(metadata["observation_terms_history_length"])
        if np.any(hist != 0.0):
            raise ValueError("squat ONNX unexpectedly uses observation history")
        scale = _csv_floats(metadata["observation_terms_scale"])
        if scale.shape != (len(expected_terms),) or np.any(scale != 1.0):
            raise ValueError("unexpected observation_terms_scale")

        shape = self._backend.input_shape("obs")
        if len(shape) != 2 or shape[1] != (130 if self.mode == "hold" else 132):
            raise ValueError(f"unexpected obs input shape: {shape}")

    # ------------------------------------------------------------------ 时间轴
    @property
    def frame(self) -> int:
        return int(round(self._frame))

    @property
    def frames(self) -> int:
        return self._frames

    def reset_to_start(self) -> None:
        self._frame = 0.0 if self.mode != "hold" else float(self._freeze_frame)
        self._previous_action.fill(0.0)
        np.copyto(self._target, self._q_ref[self.frame])
        self.publish_output(
            self._target,
            self._parameters.kp * self.gain_scale,
            self._parameters.kd * self.gain_scale,
            completed=self.finished(),
        )

    def reset(self, frame: InferenceFrame | None = None) -> None:
        """兼容 preheat_model(model.reset(inference_frame)) 的调用约定。"""
        self.reset_to_start()

    def _advance(self, dt: float) -> None:
        if self.mode == "hold":
            return  # 锁帧
        self._frame = min(self._frame + dt / DT, self._last_frame)

    def finished(self, trim: int = 0) -> bool:
        if self.mode == "hold":
            return False
        return self._frame >= (self._last_frame - trim)

    # ------------------------------------------------------------------ obs
    def _build_obs(self, frame: InferenceFrame, joints, q_ref, z_ref) -> np.ndarray:
        obs = self._obs[0]
        off = 0

        # 0:3 base_lin_vel —— 真机无 odom，填 0
        obs[off:off + 3] = 0.0
        off += 3

        # 3:6 base_ang_vel
        obs[off:off + 3] = frame.angular_velocity
        off += 3

        # 6:9 projected_gravity
        obs[off:off + 3] = get_gravity_orientation(frame.quat_wxyz)
        off += 3

        # 9:40 joint_pos：训练里 joint_pos_rel(biased=True) = q - default_joint_pos，
        # 即**相对默认姿态的偏差**，不是绝对角（见 .workbuddy/mjlab_ref/observations.py:51
        # 与已跑通的 crab_replay.py:656）。真机读到的 joints.position 是绝对角，必须减掉
        # metadata 里的 default_joint_pos 前 29 维。两个被动头关节真机无编码器，而训练时
        # 它们的观测也恒为 0 偏差，故直接填 0（不要填 head_default 本身）。
        obs[off:off + 29] = joints.position - self._parameters.default_position
        obs[off + 29:off + 31] = 0.0
        off += 31

        # 40:71 joint_vel（29 可控 + 2 头关节 0）
        obs[off:off + 29] = joints.velocity
        obs[off + 29:off + 31] = 0.0
        off += 31

        # 71:100 actions —— 截幅前 raw
        obs[off:off + 29] = self._previous_action
        off += 29

        # 100:129 traj_joint_err
        obs[off:off + 29] = joints.position - q_ref
        off += 29

        # 129:130 traj_height_err —— 真机无基座高度反馈，z_meas 取 0
        obs[off] = -z_ref
        off += 1

        # 130:132 traj_progress（hold 不拼）
        if self.mode != "hold":
            p = self._frame / self._last_frame if self._last_frame > 0 else 0.0
            obs[off] = np.sin(np.pi * p)
            obs[off + 1] = np.cos(np.pi * p)
            off += 2

        assert off == self._obs_dim, (off, self._obs_dim)
        return self._obs

    # ------------------------------------------------------------------ step
    def step(
        self,
        frame: InferenceFrame,
        dt: float,
        *,
        advance: bool = True,
    ) -> PolicyOutput:
        if not advance:
            np.copyto(self._action_checkpoint, self._previous_action)
            frame_checkpoint = self._frame

        joints = self.bind_joints(frame)
        idx = self.frame
        q_ref = self._q_ref[idx]
        z_ref = self._z_ref[idx]
        self._build_obs(frame, joints, q_ref, z_ref)

        outputs = self._backend.run(self._inputs)
        raw = np.asarray(outputs["actions"]).reshape(-1)
        if raw.shape != (self.joint_contract.action.dof_num,):
            raise ValueError(f"unexpected actions shape: {raw.shape}")

        # 残差后处理：先按 clip_rad/scale 截幅，再乘 scale；q_target = q_ref + delta
        a = np.clip(raw, -self._clip_rad / self._scale, self._clip_rad / self._scale)
        delta = self._scale * a
        np.add(q_ref, delta, out=self._target)

        if advance:
            np.copyto(self._previous_action, raw)  # 关键：回填截幅前 raw
            self._advance(dt)
        else:
            np.copyto(self._previous_action, self._action_checkpoint)
            self._frame = frame_checkpoint

        self.publish_output(
            self._target,
            self._parameters.kp * self.gain_scale,
            self._parameters.kd * self.gain_scale,
            completed=self.finished(),
        )
        return self.output

    def decode_into(self, outputs) -> None:
        """JointPolicy 要求实现；本类在 step() 内直接 decode，保留空实现。"""
        return None

    def close(self) -> None:
        self._backend.close()


__all__ = ["SquatResidualPolicy", "DT"]
