"""本地自检：不依赖 ROS2 / colcon，直接验证三份蹲起 ONNX 的适配层。

用法（在 FW 目录下）：
    python mods/com.bxi.squat_actions/verify_policy.py

检查：
  1. 三份 ONNX 能被 SquatResidualPolicy 加载（含 metadata 12 键校验）
  2. obs 维度 132/130 与各块布局正确
  3. 残差后处理 q_target = q_ref + scale*clip(a) 与直接 ORT 推理逐帧一致
  4. down/up 帧计数推进到 T-1、finished 正确；hold 帧锁死且 finished 恒 False
  5. last_action 回填的是截幅前 raw
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

import numpy as np

_FW = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_FW))


def _load_mod():
    def ensure(name):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
        return sys.modules[name]

    for n in ("com", "com.bxi"):
        ensure(n)
    pkg = ensure("com.bxi.squat_actions")
    pkg.__path__ = [str(Path(__file__).resolve().parent)]
    spec = importlib.util.spec_from_file_location(
        "com.bxi.squat_actions.squat_residual_policy",
        str(Path(__file__).resolve().parent / "squat_residual_policy.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sp = _load_mod()
    from bxi_example_py_elf3.framework.inference.api import InferenceFrame
    from bxi_example_py_elf3.framework.joints import JointStateView
    from bxi_example_py_elf3.policies.joints import ELF3_POLICY_JOINTS
    import onnxruntime as ort

    assets = Path(__file__).resolve().parent / "assets"
    ref_path = str(assets / "squat_ref_v1.npz")
    layout = ELF3_POLICY_JOINTS  # 本身就是 JointLayout
    dof = layout.dof_num

    failures = []
    for mode, onnx_name, expect_dim, expect_frames in (
        ("down", "squat_down.onnx", 132, 124),
        ("up", "squat_up.onnx", 132, 102),
        ("hold", "squat_hold.onnx", 130, 124),
    ):
        policy = sp.SquatResidualPolicy(str(assets / onnx_name), ref_path, mode=mode)
        print(f"\n=== {mode}: obs_dim={policy._obs_dim}, frames={policy.frames} ===")
        if policy._obs_dim != expect_dim:
            failures.append(f"{mode}: obs_dim {policy._obs_dim} != {expect_dim}")
        if policy.frames != expect_frames:
            failures.append(f"{mode}: frames {policy.frames} != {expect_frames}")

        # metadata 校验（加载时不抛异常即通过），核对 action_scale 长度
        if policy._scale.shape != (dof,):
            failures.append(f"{mode}: action_scale shape {policy._scale.shape}")

        # 造几帧观测（站姿附近的合理值）
        sess = ort.InferenceSession(str(assets / onnx_name), providers=["CPUExecutionProvider"])
        for k in range(6):
            q = np.zeros(dof, dtype=np.float32)
            q[3] = -0.1 - 0.02 * k
            q[6] = 0.3 + 0.05 * k
            dq = np.zeros(dof, dtype=np.float32)
            frame = InferenceFrame(
                joints=JointStateView(layout, q, dq, 0),
                quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                command=np.zeros(3, dtype=np.float32),
            )
            out = policy.step(frame, 0.02, advance=True)
            if out.joints.position.shape != (dof,):
                failures.append(f"{mode}: output shape {out.joints.position.shape}")
                break
            # 用 policy 内部已构造的 obs 直接喂 ORT，核对输出 -> 后处理一致
            raw = np.asarray(sess.run(None, {"obs": policy._obs})[0]).reshape(-1)
            a = np.clip(raw, -policy._clip_rad / policy._scale, policy._clip_rad / policy._scale)
            expected = policy._q_ref[policy.frame - 1 if k > 0 else 0] + policy._scale * a
            # 注意 step 内先算 q_target 再 advance，故这里比较用 advance 前的帧
            # 用上一帧 ref 复算（k=0 时 frame 仍 0）
            ref_idx = max(0, policy.frame - 1)
            expected = policy._q_ref[ref_idx] + policy._scale * a
            err = float(np.max(np.abs(out.joints.position - expected)))
            if err > 1e-3:
                failures.append(f"{mode} k={k}: q_target mismatch {err:.2e}")
            # last_action 回填 raw（截幅前）
            if not np.allclose(policy._previous_action, raw, atol=1e-6):
                failures.append(f"{mode} k={k}: previous_action != raw")

        if mode == "hold":
            f0 = policy.frame
            for _ in range(10):
                policy.step(frame, 0.02, advance=True)
            if policy.frame != f0:
                failures.append(f"hold: frame moved {f0} -> {policy.frame}")
            if policy.finished():
                failures.append("hold: finished() should always be False")
        else:
            # 推进到末帧
            for _ in range(expect_frames + 5):
                policy.step(frame, 0.02, advance=True)
            if not policy.finished():
                failures.append(f"{mode}: not finished after {expect_frames} steps")
            if policy.frame != expect_frames - 1:
                failures.append(f"{mode}: frame {policy.frame} != {expect_frames - 1}")
        print(f"  末帧 frame={policy.frame}, finished={policy.finished()}, "
              f"q_target[0:3]={np.round(out.joints.position[:3], 4)}")

    if failures:
        print("\n[FAIL] 问题:")
        for f in failures:
            print("  -", f)
        return 1
    print("\n[OK] 三份 ONNX 适配层全部自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
