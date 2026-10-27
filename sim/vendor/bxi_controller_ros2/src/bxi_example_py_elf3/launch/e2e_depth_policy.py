"""端到端验证：MuJoCo 渲染深度图 -> bxi 官方框架 -> 深度策略 -> 29 维关节目标。

不依赖 ROS：框架的 framework.* / policies.* 均可在纯 Python 下导入。
state.py 那层（ROS 订阅）用等价的几步内联复刻。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WS = Path(r"E:/mujoco/climb stairs")
FW = WS / "vendor" / "bxi_controller_ros2" / "src" / "bxi_example_py_elf3"
# depth.py / assets/*.onnx 现已归入 mod 目录（2026-10-13 搬迁后）
MOD = FW / "mods" / "com.bxi.normal_depth"
sys.path.insert(0, str(FW))
sys.path.insert(0, str(MOD))

import mujoco  # noqa: E402

from bxi_example_py_elf3.framework.inference import (  # noqa: E402
    InferenceFrame,
    ModelSpec,
)
from bxi_example_py_elf3.framework.joints import JointStateBuffer  # noqa: E402
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS  # noqa: E402
import depth  # noqa: E402

XML = FW / "data" / "mujoco_simulation" / "elf3.xml"


def render_depth(model, data, cam_name="head_depth_camera", hw=(480, 848)):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        raise KeyError(f"模型里没有相机 {cam_name}")
    model.vis.global_.offwidth = max(int(hw[1]), model.vis.global_.offwidth)
    model.vis.global_.offheight = max(int(hw[0]), model.vis.global_.offheight)
    r = mujoco.Renderer(model, height=hw[0], width=hw[1])
    r.enable_depth_rendering()
    r.update_scene(data, camera=cid)
    img = r.render().astype(np.float32)
    r.close()
    return img


def project_like_state_py(raw_hw_depth, out_w=36, out_h=48):
    """复刻 state.py:project_depth 的等价形状变换（最邻近缩放 + rot90）。

    真实 state.py 会先用内参按目标 FOV 做中心裁剪再缩放；这里为纯 CPU 冒烟测试
    直接用最近邻缩放到 (out_h, out_w) 后 rot90，得到策略侧的 (36, 48) 中间形状。
    """
    h, w = raw_hw_depth.shape
    xi = np.minimum((np.arange(out_w) * (w / out_w)).astype(int), w - 1)
    yi = np.minimum((np.arange(out_h) * (h / out_h)).astype(int), h - 1)
    small = raw_hw_depth[np.ix_(yi, xi)]
    return np.ascontiguousarray(np.rot90(small, k=-1)).astype(np.float32)


def main():
    print(f"[1] 加载 MuJoCo 模型 {XML.name}")
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    cam_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(model.ncam)
    ]
    print(f"    nq={model.nq} nv={model.nv} nu={model.nu} ncam={model.ncam}")
    print(f"    相机: {cam_names}")
    print(f"    传感器数: {model.nsensor}  nuser_sensor: {model.nuser_sensor}")

    print("\n[2] 渲染 head_depth_camera 深度图")
    raw = render_depth(model, data)
    print(f"    原始深度 {raw.shape}  有效范围 [{raw[raw > 0].min():.3f}, {raw.max():.3f}] m")

    # 每个模型对应的 ProjectionSpec（来自 plugin.py / mod.yaml 的两种 profile）
    PROJECTION = {
        "dagger2.onnx": (36, 48),            # origin_camera: 旋转前 36x48
        "dagger_new_position.onnx": (36, 48),  # origin_camera
        "normal_depth.onnx": (64, 36),       # depth_walk:  旋转前 64x36
    }

    results = {}
    for name in ["dagger2.onnx", "dagger_new_position.onnx", "normal_depth.onnx"]:
        print(f"\n[3] {name}")
        ow, oh = PROJECTION[name]
        proj = project_like_state_py(raw, out_w=ow, out_h=oh)
        print(f"    投影配置 {ow}x{oh} -> 中间形状 {proj.shape}")
        spec = ModelSpec.portable_onnx(str(MOD / "assets" / name), input_names=(), output_names=())
        cls = (
            depth.HumanoidGaitDepthPolicyIsaaclab
            if name == "normal_depth.onnx"
            else depth.HumanoidGaitOriginCameraPolicyIsaaclab
        )
        pol = cls(spec)

        layout = pol.joint_contract.observation
        jbuf = JointStateBuffer(layout, dtype=np.float32)
        # 初始关节 = 默认位（policy 自带的 default_position，按 action 布局）
        jbuf.update(
            position=ELF3_ISAAC_PARAMETERS.default_position,
            velocity=np.zeros(layout.dof_num, dtype=np.float32),
        )

        frame = InferenceFrame(
            joints=jbuf.view,
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            command=np.array([0.5, 0.0, 0.0], dtype=np.float32),  # 前进 0.5 m/s
            depth=proj,
            depth_frame_id=None,
        )
        pol.reset(frame)
        tgt = np.asarray(pol.output.joints.position).copy()
        for i in range(5):
            frame.depth_frame_id = i + 1
            out = pol.step(frame, 0.02, advance=True)
            tgt = np.asarray(out.joints.position).copy()
        results[name] = tgt
        print(f"    step OK  关节目标维度={tgt.shape}  |q|max={np.abs(tgt).max():.4f} rad")
        names = layout.names
        pick = [n for n in names if "hip_y" in n or "knee_y" in n or "ankle_y" in n]
        for n in pick:
            i = names.index(n)
            print(f"      {n:<18} {tgt[i]:+.4f}")
        pol.close()

    print("\n[4] dagger2 与 dagger_new_position 目标差异")
    if "dagger2.onnx" in results and "dagger_new_position.onnx" in results:
        d = np.abs(results["dagger2.onnx"] - results["dagger_new_position.onnx"])
        print(f"    max|Δq| = {d.max():.5f} rad   平均 {d.mean():.5f}")
    print("\n[完成] MuJoCo -> 官方框架 -> 深度策略 全链路可运行（无需 ROS）")


if __name__ == "__main__":
    main()
