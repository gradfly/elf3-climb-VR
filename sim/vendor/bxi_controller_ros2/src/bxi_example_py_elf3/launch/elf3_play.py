"""elf3 人形机器人 · MuJoCo 策略仿真播放器（纯 CPU，不依赖 mjlab / MuJoCo-Warp）

为什么是独立脚本
----------------
`play.py` 依赖完整的 mjlab 训练框架（mjlab.envs / rsl_rl / MuJoCo-Warp），需要有 GPU
并且要注册好 elf3 的自定义任务才能跑。本脚本把那条链路**在纯 MuJoCo 上等价复刻**：

  * 模型：elf3.xml（+ 可选平面/楼梯场景），执行器改成 mjlab 训练时用的
    MuJoCo 内置 `<position>` 执行器（kp / kd / 力矩上限 / armature 全部来自真值）。
  * 观测：严格按照 mjlab tracking 任务的 160 维布局拼装。
  * 动作：`ctrl = default_joint_pos + action_scale * policy_output`
    （等价 mjlab `JointPositionActionCfg(use_default_offset=True)`）。
  * 控制频率：timestep 0.005 s × decimation 4 = 50 Hz，与动作数据 50 fps 对齐。
  * 策略：policy.onnx（内含 obs 归一化），输入 obs[1,160] + time_step[1,1]。

参数真值来源
------------
policy.onnx 的 ONNX metadata 里保存了训练时的全部关键参数
（joint_stiffness / joint_damping / default_joint_pos / action_scale /
observation_names / anchor_body_name / body_names），本脚本直接读取，无需猜测。
所以“先跑通普通任务（基础链路自检），再跑 motion-tracking”两种用法：

  # 1) 基础链路自检：零动作，PD 保持默认站姿（确认模型/物理/渲染都通）
  python elf3_play.py --policy zero --viewer native

  # 2) 动作跟踪：策略跟踪 fight.npz 的“打拳”动作
  python elf3_play.py --policy onnx --viewer native

  # 无窗口跑评测，输出跟踪误差指标
  python elf3_play.py --policy onnx --viewer headless --steps 1000
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mujoco

# 目录布局（2026-10-13 文件搬迁后）：
#   <vendor>/bxi_controller_ros2/src/bxi_example_py_elf3/launch/elf3_play.py   <- 本文件
#   <vendor>/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml
#   <vendor>/policy.onnx, <vendor>/fight.npz
HERE = Path(__file__).resolve().parent                     # .../bxi_example_py_elf3/launch
FW = HERE.parent                                           # .../bxi_example_py_elf3
MODEL_DIR = FW / "data" / "mujoco_simulation"              # elf3.xml + meshes/ + terrains/
ELF3_XML = MODEL_DIR / "elf3.xml"
RUN_DIR = HERE.parents[3]                                  # 工作区 vendor/（policy.onnx / fight.npz）


def _first_existing(*candidates: Path) -> Path:
    """返回第一个存在的候选路径，都不存在时返回第一个（便于报错时显示）。"""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_ONNX = _first_existing(RUN_DIR / "policy.onnx", HERE / "policy.onnx")
DEFAULT_MOTION = _first_existing(RUN_DIR / "fight.npz", HERE / "fight.npz")

# ---------------------------------------------------------------------------
# 1. 训练时参数的兜底常量（正常情况下从 policy.onnx metadata 读取并覆盖）
# ---------------------------------------------------------------------------

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
BODY_NAMES = [
    "torso_link",
    "l_hip_x_link", "l_knee_y_link", "l_ankle_x_link",
    "r_hip_x_link", "r_knee_y_link", "r_ankle_x_link",
    "l_shoulder_x_link", "l_elbow_y_link", "l_wrist_z_link",
    "r_shoulder_x_link", "r_elbow_y_link", "r_wrist_z_link",
]
ANCHOR_BODY = "torso_link"
EE_BODY_NAMES = ("l_ankle_x_link", "r_ankle_x_link", "l_wrist_z_link", "r_wrist_z_link")

_DEFAULT_JOINT_POS = np.array([
    0.0, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)

_JOINT_STIFFNESS = np.array([
    108.448, 162.672, 176.421,
    176.421, 176.421, 54.224, 176.421, 33.493, 10.000,
    176.421, 176.421, 54.224, 176.421, 33.493, 10.000,
    54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
    54.224, 54.224, 16.747, 54.224, 16.747, 16.747, 16.747,
], dtype=np.float64)

_JOINT_DAMPING = np.array([
    6.904, 10.356, 11.231,
    11.231, 11.231, 3.452, 11.231, 2.132, 1.000,
    11.231, 11.231, 3.452, 11.231, 2.132, 1.000,
    3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
    3.452, 3.452, 1.066, 3.452, 1.066, 1.066, 1.066,
], dtype=np.float64)

_ACTION_SCALE = np.array([
    0.207, 0.154, 0.213,
    0.213, 0.213, 0.207, 0.213, 0.299, 0.375,
    0.213, 0.213, 0.207, 0.213, 0.299, 0.375,
    0.207, 0.207, 0.313, 0.207, 0.313, 0.313, 0.313,
    0.207, 0.207, 0.313, 0.207, 0.313, 0.313, 0.313,
], dtype=np.float64)

# mjlab: kp = armature * (2*pi*10)^2, kd = 2*2*armature*(2*pi*10)
NATURAL_FREQ = 10.0 * 2.0 * np.pi
OMEGA2 = NATURAL_FREQ ** 2                       # 3947.84
KD_OVER_KP = 4.0 / NATURAL_FREQ                  # 0.06366

# elf3.xml 里 <motor ctrlrange> 给出的关节力矩上限（消融用）
_XML_EFFORT = np.array([
    100, 100, 100,
    100, 100, 50, 150, 50, 20,
    100, 100, 50, 150, 50, 20,
    50, 50, 25, 50, 25, 25, 25,
    50, 50, 25, 50, 25, 25, 25,
], dtype=np.float64)


@dataclass
class TrainParams:
    """训练时的控制器/模型参数（默认值 = 兜底常量，随后由 ONNX metadata 覆盖）。"""
    joint_names: list[str] = field(default_factory=lambda: list(JOINT_NAMES))
    body_names: list[str] = field(default_factory=lambda: list(BODY_NAMES))
    anchor_body: str = ANCHOR_BODY
    default_joint_pos: np.ndarray = field(default_factory=lambda: _DEFAULT_JOINT_POS.copy())
    stiffness: np.ndarray = field(default_factory=lambda: _JOINT_STIFFNESS.copy())
    damping: np.ndarray = field(default_factory=lambda: _JOINT_DAMPING.copy())
    action_scale: np.ndarray = field(default_factory=lambda: _ACTION_SCALE.copy())
    obs_terms: list[str] = field(default_factory=lambda: [
        "command", "motion_anchor_pos_b", "motion_anchor_ori_b",
        "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions",
    ])
    source: str = "builtin constants"

    @property
    def effort_limit(self) -> np.ndarray:
        """mjlab: action_scale = 0.25 * effort_limit / stiffness。"""
        return 4.0 * self.action_scale * self.stiffness

    @property
    def armature(self) -> np.ndarray:
        """mjlab: stiffness = armature * (2*pi*10)^2。"""
        return self.stiffness / OMEGA2


def load_params_from_onnx(onnx_path: Path) -> TrainParams:
    """从 policy.onnx 的 metadata 读取训练参数（读不到就抛异常，调用方回退）。"""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map

    def arr(key: str) -> np.ndarray:
        return np.array([float(x) for x in meta[key].split(",")], dtype=np.float64)

    p = TrainParams()
    p.joint_names = [s for s in meta["joint_names"].split(",")]
    p.body_names = [s for s in meta["body_names"].split(",")]
    p.anchor_body = meta["anchor_body_name"]
    p.default_joint_pos = arr("default_joint_pos")
    p.stiffness = arr("joint_stiffness")
    p.damping = arr("joint_damping")
    p.action_scale = arr("action_scale")
    p.obs_terms = [s for s in meta["observation_names"].split(",")]
    p.source = f"{onnx_path.name} metadata (run={meta.get('run_path', '?')})"
    return p


# ---------------------------------------------------------------------------
# 2. 四元数 / 矩阵工具（wxyz，与 mjlab & MuJoCo 一致）
# ---------------------------------------------------------------------------

def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / max(float(np.dot(q, q)), 1e-12)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[0] * t + np.cross(xyz, t)


def quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_inv(q), v)


def quat2mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def subtract_frame_transforms(t01, q01, t02, q02):
    """T12 = T01^-1 * T02 —— 与 mjlab.utils.lab_api.math 完全一致。"""
    q10 = quat_inv(q01)
    return quat_apply(q10, np.asarray(t02) - np.asarray(t01)), quat_mul(q10, q02)


# ---------------------------------------------------------------------------
# 3. 动作数据（fight.npz）
# ---------------------------------------------------------------------------

class Motion:
    """RSI 参考动作。字段与 mjlab MotionLoader 对齐。"""

    def __init__(self, path: Path, joint_names: list[str], body_names: list[str]):
        data = np.load(path, allow_pickle=True)
        self.fps = float(np.ravel(data["fps"])[0])
        mj = [str(x) for x in data["joint_names"]]
        mb = [str(x) for x in data["body_names"]]
        if mj != list(joint_names):
            raise ValueError(f"动作文件关节顺序与模型不一致:\n {mj}\n {list(joint_names)}")
        idx = [mb.index(b) for b in body_names]      # 只保留被跟踪的 13 个 body
        self.joint_pos = data["joint_pos"].astype(np.float64)
        self.joint_vel = data["joint_vel"].astype(np.float64)
        self.body_pos_w = data["body_pos_w"].astype(np.float64)[:, idx]
        self.body_quat_w = data["body_quat_w"].astype(np.float64)[:, idx]
        self.body_lin_vel_w = data["body_lin_vel_w"].astype(np.float64)[:, idx]
        self.body_ang_vel_w = data["body_ang_vel_w"].astype(np.float64)[:, idx]
        self.num_frames = self.joint_pos.shape[0]
        # anchor = body_names[0] (torso_link)，与 mjlab MotionCommand 一致
        self.anchor_pos_w = self.body_pos_w[:, 0]
        self.anchor_quat_w = self.body_quat_w[:, 0]

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps


# ---------------------------------------------------------------------------
# 4. 建模：elf3.xml -> mjlab 等价的 MuJoCo 模型
# ---------------------------------------------------------------------------

_PLANE_SCENE_XML = """
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" group="0" rgba="0.45 0.5 0.55 1"/>"""


def build_model(params: TrainParams, scene: str = "plane",
                armature: str = "metadata", joint_damping: str = "xml",
                effort: str = "derived", xml_path: Path = ELF3_XML,
                imu_site: tuple[float, float, float] = (0.0, 0.0, 0.0),
                friction_scale: float = 1.0) -> mujoco.MjModel:
    """生成 mjlab 等价的 elf3 模型。"""
    xml_path = Path(xml_path)
    txt = xml_path.read_text(encoding="utf-8")

    # 去掉 XML 注释：官方 elf3.xml 里有一段被注释掉的 <actuator>（mjlab affine 增益参考），
    # 会干扰下面「替换 <actuator>」的正则（re.subn 会匹配到注释里那一份，导致 n_sub=2）。
    txt = re.sub(r"<!--.*?-->", "", txt, flags=re.S)

    # --- 场景：平面 or 楼梯 ---
    if scene == "plane":
        # 去掉「地面 + 台阶/地形」的 include，改用一块平面（对应 mjlab 的 TerrainCfg(plane)）。
        # 兼容两种 elf3.xml：老版根目录版 include scene_bx.xml；
        # 官方版（data/mujoco_simulation）include terrains/universal_test/... 地形。
        stripped = txt.replace('  <include file="scene_bx.xml"/>', '')
        stripped = re.sub(r'\n[ \t]*<include file="terrains/[^"]*"/>', '', stripped)
        if stripped == txt:
            raise RuntimeError("elf3.xml 中找不到可移除的地形 include（scene_bx.xml 或 terrains/*）")
        txt = stripped.replace("<worldbody>", "<worldbody>" + _PLANE_SCENE_XML, 1)
    elif scene != "stairs":
        raise ValueError(f"未知场景: {scene}")

    # --- 物理步长 / 积分器：与 mjlab tracking 任务一致 ---
    opts = ('<option timestep="0.005" integrator="implicitfast" cone="pyramidal" '
            'solver="Newton" tolerance="1e-8" iterations="10" ls_iterations="20" '
            'impratio="1" gravity="0 0 -9.81"/>')
    txt = re.sub(r'<compiler([^>]*)/>', lambda m: f'<compiler{m.group(1)}/>\n  {opts}', txt, count=1)

    # --- 执行器：把 <motor>(力矩) 换成 mjlab 用的内置 <position>(PD 位置) ---
    eff = params.effort_limit if effort == "derived" else _XML_EFFORT
    acts = "\n".join(
        f'    <position name="{n}" joint="{n}" kp="{kp:.6f}" kv="{kd:.6f}" '
        f'ctrllimited="false" inheritrange="0" '
        f'forcelimited="true" forcerange="{-e:.6f} {e:.6f}"/>'
        for n, kp, kd, e in zip(params.joint_names, params.stiffness, params.damping, eff)
    )
    txt, n_sub = re.subn(r"<actuator>.*?</actuator>", f"<actuator>\n{acts}\n  </actuator>",
                         txt, flags=re.S)
    if n_sub != 1:
        raise RuntimeError("替换 <actuator> 失败")

    # --- IMU 传感器：训练用到 robot/imu_ang_vel、robot/imu_lin_vel ---
    # 两种 elf3.xml 的命名不同，这里都补一份标准名，后续逻辑即可通用：
    #   老版（根目录版）      : site imu_in_pelvis + gyro Body_Gyro
    #   官方版（data/mujoco_simulation）: site imu_in_torso + gyro/velocimeter torso_*
    txt = txt.replace(
        '<gyro name="Body_Gyro" site="imu_in_pelvis"/>',
        '<gyro name="Body_Gyro" site="imu_in_pelvis"/>\n'
        '    <gyro name="imu_ang_vel" site="imu_in_pelvis"/>',
    )
    txt = txt.replace(
        '<gyro name="torso_ang_vel" site="imu_in_torso"/>',
        '<gyro name="torso_ang_vel" site="imu_in_torso"/>\n'
        '    <gyro name="imu_ang_vel" site="imu_in_torso"/>',
    )
    txt = txt.replace(
        '<velocimeter name="torso_lin_vel" site="imu_in_torso"/>',
        '<velocimeter name="torso_lin_vel" site="imu_in_torso"/>\n'
        '    <velocimeter name="imu_lin_vel" site="imu_in_torso"/>',
    )

    # --- 初始 keyframe（default_joint_pos 的来源） ---
    key_qpos = [0.0, 0.0, 1.1, 1.0, 0.0, 0.0, 0.0] + list(params.default_joint_pos)
    txt = txt.replace(
        "</mujoco>",
        "  <keyframe>\n    <key name=\"init_state\" qpos=\""
        + " ".join(f"{v:.6f}" for v in key_qpos) + "\"/>\n  </keyframe>\n</mujoco>",
    )

    # MjSpec.from_string 以「当前工作目录」为基准解析相对路径（include / meshdir），
    # 而 elf3.xml 与其 meshes/、sensors/、terrains/ 同级，故编译期间临时切到该目录。
    with contextlib.chdir(xml_path.parent):
        spec = mujoco.MjSpec.from_string(txt)
        model = spec.compile()

    # --- 编译后再显式写一遍 opt（等价 mjlab MujocoCfg.apply） ---
    model.opt.timestep = 0.005
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.tolerance = 1e-8
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    model.opt.impratio = 1.0
    model.opt.gravity[:] = (0.0, 0.0, -9.81)

    # --- 关节级覆盖：armature / 粘性阻尼 ---
    jids = [_jid(model, n) for n in params.joint_names]
    dofs = np.array([model.jnt_dofadr[j] for j in jids])
    if armature == "metadata":
        model.dof_armature[dofs] = params.armature
    elif armature != "xml":
        raise ValueError(f"未知 armature 模式: {armature}")
    if joint_damping == "zero":
        model.dof_damping[dofs] = 0.0
    elif joint_damping != "xml":
        raise ValueError(f"未知 joint_damping 模式: {joint_damping}")

    # --- IMU 站点位置（velocity 传感器在机体系下的线性速度与站点偏移有关） ---
    if tuple(np.round(imu_site, 8)) != (0.0, 0.0, 0.0):
        sid = _imu_site_id(model)
        if sid < 0:
            raise KeyError("模型里没有 IMU 站点（imu_in_pelvis / imu_in_torso）")
        model.site_pos[sid] = np.asarray(imu_site, dtype=np.float64)

    # --- 接触摩擦缩放 ---
    if friction_scale != 1.0:
        model.geom_friction[:, :3] *= friction_scale
        model.pair_friction[:, :2] *= friction_scale
    return model


def _jid(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"模型里没有关节 {name}")
    return jid


# ---------------------------------------------------------------------------
# 5. 播放器
# ---------------------------------------------------------------------------

class Elf3Player:
    """把 elf3 + 策略 + 参考动作串成 mjlab 等价的闭环。"""

    def __init__(self, model: mujoco.MjModel, params: TrainParams, motion: Motion,
                 policy_path: Path | None = None, policy: str = "onnx",
                 gain_scale: float = 1.0, scale_mult: float = 1.0,
                 seed: int = 0):
        self.model = model
        self.params = params
        self.motion = motion
        self.policy_kind = policy
        self.rng = np.random.default_rng(seed)

        self.data = mujoco.MjData(model)
        self.decimation = 4

        self.jids = [_jid(model, n) for n in params.joint_names]
        self.jqadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.jvadr = np.array([model.jnt_dofadr[j] for j in self.jids])

        self.default = params.default_joint_pos.copy()
        self.scale = params.action_scale * scale_mult
        # kp / kd 运行时缩放（调试用；1.0 = 训练真值）
        if gain_scale != 1.0:
            act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                       for n in params.joint_names]
            for a in act_ids:
                model.actuator_gainprm[a, 0] *= gain_scale
                model.actuator_biasprm[a, 1] *= gain_scale
                model.actuator_biasprm[a, 2] *= gain_scale

        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                         for b in params.body_names]
        self.anchor_id = self.body_ids[0]
        self.ee_ids = [self.body_ids[params.body_names.index(b)] for b in EE_BODY_NAMES]

        self.sens_v = _sensor_slice(model, "imu_lin_vel", "torso_lin_vel")
        self.sens_w = _sensor_slice(model, "imu_ang_vel", "torso_ang_vel")

        self.last_action = np.zeros(len(params.joint_names))
        self.sess = None
        self.in_names = None
        if policy == "onnx":
            if policy_path is None:
                raise ValueError("policy=onnx 需要 policy.onnx")
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(policy_path),
                                             providers=["CPUExecutionProvider"])
            self.in_names = [i.name for i in self.sess.get_inputs()]
            self.out_names = [o.name for o in self.sess.get_outputs()]
        self.camera = None  # 由 viewer 设置

    # ---- 状态读写 ----

    def reset_to_frame(self, frame: int) -> None:
        """RSI：把机器人写到参考动作第 frame 帧（等价 mjlab MotionCommand.reset_to_frame）。"""
        m = self.motion
        f = int(np.clip(frame, 0, m.num_frames - 1))
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = m.body_pos_w[f, 0]
        d.qpos[3:7] = m.body_quat_w[f, 0]
        d.qpos[self.jqadr] = m.joint_pos[f]
        d.qvel[self.jvadr] = m.joint_vel[f]
        d.qvel[0:3] = m.body_lin_vel_w[f, 0]
        d.qvel[3:6] = m.body_ang_vel_w[f, 0]
        d.ctrl[:] = self.default
        self.last_action[:] = 0.0
        mujoco.mj_forward(self.model, d)

    # ---- 观测 ----

    def build_obs(self, frame: int) -> np.ndarray:
        """按 mjlab tracking 任务的 160 维布局拼装观测。"""
        m, d = self.motion, self.data
        f = int(np.clip(frame, 0, m.num_frames - 1))

        anchor_pos = d.xpos[self.anchor_id]
        anchor_quat = d.xquat[self.anchor_id]
        pos_b, ori_b = subtract_frame_transforms(
            anchor_pos, anchor_quat, m.anchor_pos_w[f], m.anchor_quat_w[f])
        ori_b_flat = quat2mat(ori_b)[:, :2].reshape(-1)   # 与 mjlab mat[..., :2].reshape 等价

        base_lin_vel = d.sensordata[self.sens_v[0]:self.sens_v[0] + self.sens_v[1]]
        base_ang_vel = d.sensordata[self.sens_w[0]:self.sens_w[0] + self.sens_w[1]]
        joint_pos_rel = d.qpos[self.jqadr] - self.default
        joint_vel_rel = d.qvel[self.jvadr]

        return np.concatenate([
            m.joint_pos[f], m.joint_vel[f],      # command
            pos_b,                               # motion_anchor_pos_b
            ori_b_flat,                          # motion_anchor_ori_b
            base_lin_vel, base_ang_vel,          # base_lin_vel / base_ang_vel
            joint_pos_rel, joint_vel_rel,        # joint_pos / joint_vel
            self.last_action,                    # actions
        ]).astype(np.float32)

    # ---- 策略 ----

    def act(self, obs: np.ndarray, frame: int) -> np.ndarray:
        if self.policy_kind == "onnx":
            feed = {"obs": obs[None, :], "time_step": np.array([[frame]], dtype=np.float32)}
            feed = {k: v for k, v in feed.items() if k in self.in_names}
            out = self.sess.run([self.out_names[0]], feed)[0]
            return out[0].astype(np.float64)
        if self.policy_kind == "zero":
            return np.zeros(len(self.params.joint_names))
        if self.policy_kind == "random":
            return self.rng.uniform(-1.0, 1.0, len(self.params.joint_names))
        raise ValueError(f"未知策略: {self.policy_kind}")

    # ---- 单步 ----

    def step(self, frame: int) -> tuple[np.ndarray, int]:
        """执行一个控制步（20 ms），返回 (obs_after, next_frame)。"""
        obs = self.build_obs(frame)
        action = self.act(obs, frame)
        self.data.ctrl[:] = self.default + self.scale * action
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)   # 与 mjlab 一样：obs 前刷新派生量
        self.last_action = action
        return obs, frame + 1

    # ---- 评测指标 ----

    def tracking_error(self, frame: int) -> dict[str, float]:
        m, d = self.motion, self.data
        f = int(np.clip(frame, 0, m.num_frames - 1))
        robot_anchor = d.xpos[self.anchor_id]
        ref_anchor = m.anchor_pos_w[f]
        robot_bodies = d.xpos[self.body_ids]
        ref_bodies = m.body_pos_w[f]
        return {
            "anchor_pos": float(np.linalg.norm(robot_anchor - ref_anchor)),
            "anchor_height": float(robot_anchor[2] - ref_anchor[2]),
            "body_pos": float(np.linalg.norm(robot_bodies - ref_bodies, axis=-1).mean()),
            "joint_pos": float(np.linalg.norm(d.qpos[self.jqadr] - m.joint_pos[f])),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(
                quat2mat(d.xquat[self.anchor_id])[2, 2], -1.0, 1.0)))),
        }

    def failed(self) -> bool:
        """mjlab tracking 的终止条件（anchor_pos_z_only / anchor_ori / ee_body_pos）。"""
        m, d = self.motion, self.data
        f = int(np.clip(getattr(self, "_frame", 0), 0, m.num_frames - 1))
        if abs(m.anchor_pos_w[f][2] - d.xpos[self.anchor_id][2]) > 0.25:
            return True
        g = np.array([0.0, 0.0, -1.0])
        mg = quat_apply_inv(m.anchor_quat_w[f], g)[2]
        rg = quat_apply_inv(d.xquat[self.anchor_id], g)[2]
        if abs(mg - rg) > 0.8:
            return True
        for bid in self.ee_ids:
            i = self.body_ids.index(bid)
            if abs(m.body_pos_w[f][i][2] - d.xpos[bid][2]) > 0.25:
                return True
        return False

    def close(self) -> None:
        self.sess = None


def _sensor_slice(model: mujoco.MjModel, *names: str) -> tuple[int, int]:
    """按顺序返回第一个存在的传感器的 (adr, dim)。"""
    for name in names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid >= 0:
            return int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
    raise KeyError("模型里没有传感器 " + " / ".join(names))


# 两种 elf3.xml 的 IMU 命名：老版 imu_in_pelvis，官方版 data/mujoco_simulation 是 imu_in_torso
_IMU_SITE_NAMES = ("imu_in_pelvis", "imu_in_torso")


def _imu_site_id(model: mujoco.MjModel) -> int:
    """返回 IMU 站点 id（兼容 imu_in_pelvis / imu_in_torso 两种命名）。"""
    for name in _IMU_SITE_NAMES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return int(sid)
    return -1


# ---------------------------------------------------------------------------
# 6. 运行循环
# ---------------------------------------------------------------------------

def run(player: Elf3Player, viewer_kind: str, steps: int, start_frame: int,
        loop: bool, terminate: bool, record_dir: Path | None, record_every: int,
        max_record: int, print_interval: int, record_camera: str = "tracking") -> dict[str, float]:
    model = player.model
    motion = player.motion

    ts = 0.005 * player.decimation
    print(f"[信息] 控制频率 {1.0 / ts:.0f} Hz | 动作 {motion.num_frames} 帧 "
          f"({motion.duration:.1f} s) | 策略 {player.policy_kind}")

    # 复位时把命令索引设为 s（mjlab reset 之后 time_steps 会 +1）
    frame = start_frame
    player.reset_to_frame(frame)
    frame = frame + 1

    stats: list[dict[str, float]] = []
    stats.append(player.tracking_error(frame))

    viewer = None
    renderer = None
    if viewer_kind == "native":
        import mujoco.viewer as mjv
        viewer = mjv.launch_passive(model, player.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = player.anchor_id
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -10.0
    elif viewer_kind == "headless":
        if record_dir is not None:
            renderer = mujoco.Renderer(model, height=480, width=640)

    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)

    step_i = 0
    recorded = 0
    t_wall = time.perf_counter()
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if steps > 0 and step_i >= steps:
                break

            player._frame = frame % motion.num_frames
            obs, frame = player.step(frame)
            step_i += 1

            err = player.tracking_error(frame)
            stats.append(err)
            if terminate and player.failed():
                print(f"[终止] 第 {step_i} 步跟踪失败，RSI 复位到第 {start_frame} 帧")
                player.reset_to_frame(start_frame)
                frame = start_frame + 1

            if frame >= motion.num_frames:
                if loop:
                    player.reset_to_frame(0)
                    frame = 1
                else:
                    break

            if renderer is not None and step_i % record_every == 0 and recorded < max_record:
                if record_camera:
                    renderer.update_scene(player.data, camera=record_camera)
                else:
                    renderer.update_scene(player.data)
                _save_frame(renderer.render(), record_dir / f"frame_{recorded:04d}.png")
                recorded += 1

            if print_interval and step_i % print_interval == 0:
                recent = stats[-min(len(stats), print_interval):]
                print(f"[{step_i:6d}] frame={frame:6d} "
                      f"anchor_err={np.mean([r['anchor_pos'] for r in recent]):.3f} "
                      f"body_err={np.mean([r['body_pos'] for r in recent]):.3f} "
                      f"tilt={recent[-1]['tilt_deg']:5.1f}° "
                      f"height={player.data.xpos[player.anchor_id][2]:.3f}")

            if viewer is not None:
                viewer.sync()
                dt = time.perf_counter() - t_wall
                wait = ts - dt
                if wait > 0:
                    time.sleep(wait)
                t_wall = time.perf_counter()
    finally:
        if renderer is not None:
            renderer.close()
        if viewer is not None:
            viewer.close()

    if terminate:
        pass
    summary = {
        "steps": step_i,
        "anchor_pos_mean": float(np.mean([s["anchor_pos"] for s in stats])),
        "anchor_pos_max": float(np.max([s["anchor_pos"] for s in stats])),
        "body_pos_mean": float(np.mean([s["body_pos"] for s in stats])),
        "joint_pos_mean": float(np.mean([s["joint_pos"] for s in stats])),
        "height_final": float(player.data.xpos[player.anchor_id][2]),
        "tilt_final_deg": float(stats[-1]["tilt_deg"]),
        "recorded": recorded,
    }
    print("\n[汇总] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                  for k, v in summary.items()))
    return summary


def _save_frame(rgb: np.ndarray, path: Path) -> None:
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except Exception:
        import imageio.v2 as imageio
        imageio.imwrite(path, rgb)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="elf3 机器人 MuJoCo 策略仿真播放器（play.py 的纯 MuJoCo 等价实现）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--policy", choices=["onnx", "zero", "random"], default="onnx",
                    help="onnx=训练策略(动作跟踪)；zero=零动作PD保持(基础自检)")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--motion", default=str(DEFAULT_MOTION))
    ap.add_argument("--viewer", choices=["native", "headless"], default="native")
    ap.add_argument("--scene", choices=["plane", "stairs"], default="plane")
    ap.add_argument("--steps", type=int, default=0, help="0 = 跑到动作结束/关闭窗口")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--no-loop", dest="loop", action="store_false")
    ap.add_argument("--terminate", action="store_true", help="启用 mjlab 终止条件+RSI 复位")
    ap.add_argument("--print-interval", type=int, default=50)
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--record-every", type=int, default=10)
    ap.add_argument("--max-record", type=int, default=240)
    ap.add_argument("--record-camera", default="tracking",
                    help="离屏渲染用相机名（elf3.xml 里的 tracking），空串=自由相机")
    # 调试/消融
    ap.add_argument("--gain-scale", type=float, default=1.0)
    ap.add_argument("--scale-mult", type=float, default=1.0)
    ap.add_argument("--armature", choices=["metadata", "xml"], default="metadata")
    ap.add_argument("--joint-damping", choices=["xml", "zero"], default="xml")
    ap.add_argument("--effort", choices=["derived", "xml"], default="derived")
    ap.add_argument("--imu-site", default="0 0 0",
                    help="imu_in_pelvis 站点位置，例如 '0.04525 0 -0.08339'")
    ap.add_argument("--friction-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # --- 参数真值 ---
    try:
        params = load_params_from_onnx(Path(args.onnx))
    except Exception as e:                              # pragma: no cover
        print(f"[警告] 无法从 ONNX 读取训练参数（{e}），使用内置常量")
        params = TrainParams()
    print(f"[信息] 参数来源: {params.source}")
    print(f"[信息] 关节 {len(params.joint_names)} 个，跟踪 body {len(params.body_names)} 个，"
          f"动作缩放 {np.round(params.action_scale, 3).tolist()[:4]}...")

    model = build_model(params, scene=args.scene, armature=args.armature,
                        joint_damping=args.joint_damping, effort=args.effort,
                        imu_site=tuple(float(v) for v in args.imu_site.split()),
                        friction_scale=args.friction_scale)
    motion = Motion(Path(args.motion), params.joint_names, params.body_names)

    player = Elf3Player(model, params, motion, policy_path=Path(args.onnx),
                        policy=args.policy, gain_scale=args.gain_scale,
                        scale_mult=args.scale_mult, seed=args.seed)

    print(f"[信息] 模型: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} ngeom={model.ngeom} nsensor={model.nsensor}")

    run(player, args.viewer, args.steps, args.start_frame, args.loop,
        args.terminate, Path(args.record_dir) if args.record_dir else None,
        args.record_every, args.max_record, args.print_interval, args.record_camera)
    player.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
