"""自训蹲起 ONNX 的适配层（obs 拼装 + 残差后处理 + 参考帧计数）。

三份 ONNX（squat_down / squat_up / squat_hold）的契约：
  - INPUT  obs  [1, 132]（down/up）/ [1, 126 或 130]（hold）
  - OUTPUT actions [1, 29] —— 是**残差**，不是关节目标
  - 12 键 CSV metadata；归一化已烘焙进图，部署侧不能再归一化
  - joint_names 有 31 个（末两个是被动头关节 head_z/head_y），action 是 29 维

hold 有两个契约版本（2026-11-19 重训起为「真机对齐版」，旧 ONNX 仍可加载回归）：
  - 126 维（新，真机对齐）：base_ang_vel / projected_gravity / joint_pos /
    joint_vel / actions / traj_joint_err —— 真机传感器能给全的六项，
    训练时就拿掉了 base_lin_vel 与 traj_height_err，部署侧不再需要"填 0"近似；
  - 130 维（旧，legacy）：在上面基础上多 base_lin_vel(填 0) 与
    traj_height_err(填 0)。

obs 分块由 metadata 的 observation_names **驱动**（逐项查 _TERM_SIZES 算偏移），
不按固定下标硬拼 —— 换新版 ONNX 时改文件即可，适配层自动跟随。

down/up 的 obs 分块（132 维）：
  0:3   base_lin_vel       真机无 odom → 填 0
  3:6   base_ang_vel       IMU 角速度
  6:9   projected_gravity  IMU 四元数反投影
  9:40  joint_pos          29 关节 + 头关节 0(2)（相对 default 的偏差）
  40:71 joint_vel          29 关节速度 + 头关节 0(2)
  71:100 actions           上一步**截幅前 raw** 输出
  100:129 traj_joint_err   q_meas - q_ref[frame]
  129:130 traj_height_err  真机无基座高度 → 填 0
  130:132 traj_progress    [sin(pi*p), cos(pi*p)]

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

# 每个观测项的维数（与训练侧 _build_observations 一一对应）
_TERM_SIZES = {
    "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,
    "joint_pos": 31, "joint_vel": 31, "actions": 29,
    "traj_joint_err": 29, "traj_height_err": 1, "traj_progress": 2,
}
# 期望的观测项顺序（严格按训练侧字典构造序）
# down/up：9 项 132 维
_OBS_TERMS_MOTION = (
    "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
    "actions", "traj_joint_err", "traj_height_err", "traj_progress",
)
# hold（legacy 130 维）：base_lin_vel / traj_height_err 真机没有，部署填 0
_OBS_TERMS_HOLD_LEGACY = _OBS_TERMS_MOTION[:-1]
# hold（真机对齐 126 维，2026-11-19 重训）：真机传感器能给全的六项，
# 训练时就拿掉了 base_lin_vel 与 traj_height_err（squat_hold_cfg.py）
_OBS_TERMS_HOLD_ALIGNED = (
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
    "actions", "traj_joint_err",
)

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

        # 观测缓冲与状态（obs_dim / 布局由 _validate_metadata 按 metadata 决定）
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

        # 观测项顺序校验 + 布局解析（偏移表）。hold 兼容两个契约版本：
        #   130 维 legacy（base_lin_vel / traj_height_err 由部署侧填 0）
        #   126 维 真机对齐（训练已去掉这两项，部署侧无近似）
        terms = tuple(t.strip() for t in metadata["observation_names"].split(","))
        if self.mode == "hold":
            if terms not in (_OBS_TERMS_HOLD_ALIGNED, _OBS_TERMS_HOLD_LEGACY):
                raise ValueError(
                    f"observation_names mismatch for hold: {terms} "
                    f"(expected 126-dim aligned {_OBS_TERMS_HOLD_ALIGNED} "
                    f"or 130-dim legacy {_OBS_TERMS_HOLD_LEGACY})"
                )
        elif terms != _OBS_TERMS_MOTION:
            raise ValueError(
                f"observation_names mismatch for mode {self.mode!r}: {terms}"
            )
        self._terms = terms
        offsets: dict[str, tuple[int, int]] = {}
        off = 0
        for t in terms:
            size = _TERM_SIZES.get(t)
            if size is None:
                raise ValueError(f"unknown observation term {t!r} in metadata")
            offsets[t] = (off, size)
            off += size
        self._obs_offsets = offsets
        self._obs_dim = off

        # 无历史帧、观测 scale 全 1（归一化已烘焙进图）
        hist = _csv_floats(metadata["observation_terms_history_length"])
        if np.any(hist != 0.0):
            raise ValueError("squat ONNX unexpectedly uses observation history")
        scale = _csv_floats(metadata["observation_terms_scale"])
        if scale.shape != (len(terms),) or np.any(scale != 1.0):
            raise ValueError("unexpected observation_terms_scale")

        shape = self._backend.input_shape("obs")
        if len(shape) != 2 or shape[1] != self._obs_dim:
            raise ValueError(
                f"obs input shape {shape} != metadata-derived dim {self._obs_dim} "
                f"(terms: {terms})"
            )

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
    def _build_obs(self, frame: InferenceFrame, joints, q_ref) -> np.ndarray:
        """按 metadata 声明的观测项逐块拼装（偏移由 _obs_offsets 决定）。

        各项语义与训练侧 _build_observations 严格一致：
          - joint_pos = joint_pos_rel(biased=True) = 绝对角 - default_joint_pos
            （见 .workbuddy/mjlab_ref/observations.py:51 与已跑通的 crab_replay.py:656）；
          - 两个被动头关节真机无编码器，训练时其观测恒为 0 偏差，直接填 0；
          - base_lin_vel / traj_height_err 仅出现在 130 维 legacy hold 与
            down/up 里：真机没有这两个量源，填 0（训练噪声 ±0.3 / ±0.005 内的
            近似；126 维新 hold 已在训练侧拿掉，无需近似）。
        """
        obs = self._obs[0]
        for name in self._terms:
            start, size = self._obs_offsets[name]
            if name == "base_lin_vel":                # 真机无 odom
                obs[start:start + size] = 0.0
            elif name == "base_ang_vel":
                obs[start:start + size] = frame.angular_velocity
            elif name == "projected_gravity":
                obs[start:start + size] = get_gravity_orientation(frame.quat_wxyz)
            elif name == "joint_pos":                 # 29 可控 + 2 被动头关节
                obs[start:start + 29] = joints.position - self._parameters.default_position
                obs[start + 29:start + 31] = 0.0
            elif name == "joint_vel":
                obs[start:start + 29] = joints.velocity
                obs[start + 29:start + 31] = 0.0
            elif name == "actions":                   # 截幅前 raw
                obs[start:start + size] = self._previous_action
            elif name == "traj_joint_err":
                obs[start:start + size] = joints.position - q_ref
            elif name == "traj_height_err":           # 真机无基座高度
                # 近似为"贴合参考"（err=0）：蹲起时真实 err 仅几 cm，落训练噪声
                # Unoise(±0.005) 附近。千万别填 -z_ref（=假设 z_meas=0，err 达 -1.0，
                # 深度分布外且落入训练 base_too_low 终止区，策略会误判"已摔倒"）。
                obs[start] = 0.0
            elif name == "traj_progress":
                p = self._frame / self._last_frame if self._last_frame > 0 else 0.0
                obs[start] = np.sin(np.pi * p)
                obs[start + 1] = np.cos(np.pi * p)
            else:                                     # pragma: no cover
                raise ValueError(f"unhandled observation term {name!r}")
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
        self._build_obs(frame, joints, q_ref)

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
