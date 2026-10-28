"""elf3 机器人资产配置（mjlab EntityCfg）。

数据来源全部是项目里的真值：
  - XML: elf3_robot_only.xml（由 train/assets/make_robot_xml.py 生成：去地形、去 motor）
  - 增益/默认位/action_scale/effort: launch/elf3_play.py 的常量表（与训练真值一致）
  - 初始基座高度 1.0339: launch/crab_climb.py:244（qpos[2] = 1.0339）
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# --------------------------------------------------------------------------
# 路径：默认 ~/elf3，可用 ELF3_REPO 覆盖
# --------------------------------------------------------------------------
REPO = Path(os.environ.get("ELF3_REPO", str(Path.home() / "elf3"))).expanduser()
XML = (
    REPO
    / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
    / "elf3_robot_only.xml"
)

# --------------------------------------------------------------------------
# 29 个可控关节（顺序与策略输出、ONNX 元数据一致）
# --------------------------------------------------------------------------
JOINT_NAMES = [
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint",
    "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint",
    "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
]

DEFAULT_JOINT_POS = [
    0.0, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]

STIFFNESS = [
    108.448, 162.672, 176.421,
    176.421, 176.421, 54.224, 176.421, 33.493, 10.000,
    176.421, 176.421, 54.224, 176.421, 33.493, 10.000,
    54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
    54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
]

DAMPING = [
    6.904, 10.356, 11.231,
    11.231, 11.231, 3.452, 11.231, 2.132, 1.000,
    11.231, 11.231, 3.452, 11.231, 2.132, 1.000,
    3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
    3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
]

ACTION_SCALE = [
    0.207, 0.154, 0.213,
    0.213, 0.213, 0.207, 0.213, 0.299, 0.375,
    0.213, 0.213, 0.207, 0.213, 0.299, 0.375,
    0.207, 0.207, 0.313, 0.207, 0.313, 0.313, 0.313,
    0.207, 0.207, 0.313, 0.207, 0.313, 0.313, 0.313,
]

EFFORT_LIMIT = [
    100, 100, 100,
    100, 100, 50, 150, 50, 20,
    100, 100, 50, 150, 50, 20,
    50, 50, 25, 50, 25, 25, 25,
    50, 50, 25, 50, 25, 25, 25,
]

NATURAL_FREQ = 10.0 * 2.0 * 3.141592653589793
ARMATURE = [s / (NATURAL_FREQ ** 2) for s in STIFFNESS]

INIT_BASE_Z = 1.0339


def get_spec() -> mujoco.MjSpec:
    if not XML.exists():
        raise FileNotFoundError(
            f"找不到 {XML}。请先运行: python train/assets/make_robot_xml.py"
        )
    return mujoco.MjSpec.from_file(str(XML))


# 每个关节一个位置执行器，增益与仿真真值一致
ACTUATORS = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=STIFFNESS[i],
        damping=DAMPING[i],
        effort_limit=float(EFFORT_LIMIT[i]),
        armature=ARMATURE[i],
    )
    for i, name in enumerate(JOINT_NAMES)
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

# 只保留足底碰撞（与仿真一致）
FOOT_COLLISION = CollisionCfg(
    geom_names_expr=(r"^[lr]_foot[1-7]_collision$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, INIT_BASE_Z),
    joint_pos={n: float(q) for n, q in zip(JOINT_NAMES, DEFAULT_JOINT_POS)},
    joint_vel={".*": 0.0},
)


def get_elf3_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=KEYFRAME,
        collisions=(FOOT_COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# JointPositionActionCfg 的 scale：正则 -> 缩放
ACTION_SCALE_MAP = {name: float(s) for name, s in zip(JOINT_NAMES, ACTION_SCALE)}
