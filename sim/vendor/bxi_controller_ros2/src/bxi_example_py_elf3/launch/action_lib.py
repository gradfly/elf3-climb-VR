"""ELF3 脚本动作（走正步 / 敬礼）共用工具。

这里只依赖 numpy + mujoco，不 import 项目里那些带场景副作用的生成器。提供：

* 29 关节常量与**立正站姿**：脚掌水平放平的 ``ankle = -(hip+knee)``。
  （训练的 nominal pose 里 ankle=+0.3 是踮脚，基座要抬到 1.1077 m 才算落地，
  拿来当站姿会让整段动作悬空。）
* 基于 ``mj_jacBody`` 的阻尼最小二乘 IK：腿 2 自由度（hip_y / knee_y，
  脚掌朝向解析补偿）、手臂 4 自由度（肩 3 + 肘 1，可同时约束肘和腕的位置）。
* 38 维 qpos 轨迹的落盘格式，与 ``squat_extreme.py`` / ``prone_crawl.py`` 一致
  （``time / qpos / dt / joint_names / action / meta``），现有回放器可直接播。

用法见 ``goose_step.py`` / ``salute.py``。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_squat.xml"                 # nq=38 的平地场景
REPO = HERE.parents[5]                          # 仓库根
WEB_XML = REPO / "web" / "assets" / "elf3.xml"  # nq=36，烘焙时按关节名重映射

DT = 0.02                                       # 50 Hz，与训练控制频率一致
STAND_Z = 1.033915353040388                     # 脚掌水平站姿的基座高
TORSO_TO_HIP = 0.3825                           # 基座 -> 髋（0.2265 + 0.156）
FOOT_SOLE = 0.04                                # 踝 -> 脚底
L_THIGH = L_SHIN = 0.32

JOINT_NAMES = [
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint",
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint",
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
]
IDX = {n.replace("_joint", ""): i for i, n in enumerate(JOINT_NAMES)}

HIP0, KNEE0 = -0.3, 0.6
ANK0 = -(HIP0 + KNEE0)                          # -0.3，脚掌真正水平


def stand_q() -> np.ndarray:
    """立正站姿 29 关节角（脚掌水平，基座 z = STAND_Z）。"""
    q = np.zeros(29, dtype=np.float64)
    for s in (IDX["l_hip_y"], IDX["r_hip_y"]):
        q[s] = HIP0
        q[s + 3] = KNEE0
        q[s + 4] = ANK0
    for s, sgn in ((IDX["l_shoulder_y"], 1.0), (IDX["r_shoulder_y"], -1.0)):
        q[s] = 0.2
        q[s + 1] = sgn * 0.2                   # shoulder_x 两侧反号
        q[s + 3] = 0.6
    return q


def smootherstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def leg_ik_2d(dx: float, dz: float, L: float = L_THIGH) -> tuple[float, float]:
    """解析两连杆腿 IK，返回 (hip_y, knee_y)，**恒定选膝向前弯的分支**。

    dx = 踝相对髋的世界 x（腿局部系），dz = 髋到踝的向下垂直距离。
    由和差化积解 sin(hip)+sin(hip+knee) / cos(hip)+cos(hip+knee) 得：

        mid  = atan2(-dx, dz)
        half = acos(dist / 2L)
        hip  = mid - half,  knee = 2*half

    站姿自检：dx=0, dz=0.6114 -> hip=-0.3, knee=+0.6，与 nominal 完全一致。

    为什么不用 DLS：正步的圆弧半径只比腿全长 2L=0.64 m 小 5 mm，Jacobian
    接近奇异，数值解在支撑/摆动切换处会翻到「膝向后弯」的另一分支
    （实测单帧跳 0.40 rad）。解析解没有分支歧义，也更快。
    """
    dist = math.hypot(dx, dz)
    half = math.acos(min(1.0, max(-1.0, dist / (2.0 * L))))
    mid = math.atan2(-dx, dz)
    return mid - half, 2.0 * half


def foot_level(q: np.ndarray, side: str, pitch: float = 0.0,
               toe: float = 0.0) -> None:
    """令脚掌在世界系水平；toe<0 时额外下压脚尖（正步绷脚面）。

    推导同 ``squat_stand.squat_joints``：脚掌世界俯仰 =
    ``pitch + hip_y + knee_y + ankle_y``，置零即解出 ankle_y。
    ankle_y 不参与脚踝位置，所以可以放在位置 IK 之后补，不破坏接触。
    """
    i = IDX[f"{side}_hip_y"]
    q[i + 4] = -(pitch + q[i] + q[i + 3]) + toe


class IK:
    """轻量 DLS 逆解器：一次实例化，逐帧 warm start 复用。"""

    def __init__(self, scene: Path | str = SCENE):
        self.m = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        if self.m.nq != 38:
            raise RuntimeError(f"需要 nq=38 的仿真模型，实际 {self.m.nq}")
        self.d = mujoco.MjData(self.m)
        self.jacp = np.zeros((3, self.m.nv))
        self._jid: dict[str, int] = {}
        self._bid: dict[str, int] = {}

    def jid(self, name: str) -> int:
        if name not in self._jid:
            self._jid[name] = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self._jid[name]

    def bid(self, name: str) -> int:
        if name not in self._bid:
            self._bid[name] = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        return self._bid[name]

    def qadr(self, name: str) -> int:
        return int(self.m.jnt_qposadr[self.jid(name)])

    def dadr(self, name: str) -> int:
        return int(self.m.jnt_dofadr[self.jid(name)])

    def limits(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for n in names:
            j = self.jid(n)
            if self.m.jnt_limited[j]:
                lo.append(float(self.m.jnt_range[j][0]))
                hi.append(float(self.m.jnt_range[j][1]))
            else:
                lo.append(-10.0)
                hi.append(10.0)
        return np.asarray(lo), np.asarray(hi)

    def set_base(self, x: float, z: float, pitch: float = 0.0) -> None:
        """自由基座：[x, y, z, qw, qx, qy, qz]，绕 y 轴俯仰。"""
        d = self.d.qpos
        d[0], d[1], d[2] = x, 0.0, z
        d[3], d[4], d[5], d[6] = math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0

    def solve(
        self,
        q: np.ndarray,
        targets: list[tuple[str, np.ndarray, float]],
        names: list[str],
        *,
        iters: int = 60,
        lam: float = 2e-6,
        step: float = 0.15,
        rows: tuple[int, ...] = (0, 1, 2),
        prior: np.ndarray | None = None,
        prior_w: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """阻尼最小二乘，返回 (更新后的 29 维关节角, 最大末端残差 m)。

        targets: [(body_name, 目标 xyz, 权重)]。多个 body 可以按权重同时约束，
        敬礼时用「腕 + 肘」双约束把肘固定在外侧，避免 IK 收敛到反关节分支。
        prior / prior_w: 向设计姿态靠拢的零空间正则（长度与 names 相同）。
        """
        cols = [self.dadr(n) for n in names]
        qadrs = [self.qadr(n) for n in names]
        lo, hi = self.limits(names)
        n = len(names)
        ri = list(rows)
        self.d.qpos[7:36] = q

        for _ in range(iters):
            mujoco.mj_forward(self.m, self.d)
            A = lam * np.eye(n)
            b = np.zeros(n)
            for bname, tgt, w in targets:
                bid = self.bid(bname)
                e = np.asarray(tgt, dtype=np.float64) - self.d.xpos[bid]
                mujoco.mj_jacBody(self.m, self.d, self.jacp, None, bid)
                J = self.jacp[np.ix_(ri, cols)]
                A += w * (J.T @ J)
                b += w * (J.T @ e[ri])
            if prior is not None and prior_w > 0.0:
                A += prior_w * np.eye(n)
                b += prior_w * (np.asarray(prior) - self.d.qpos[qadrs])
            dq = np.linalg.solve(A, b)
            dq = np.clip(dq, -step, step)
            self.d.qpos[qadrs] = np.clip(
                self.d.qpos[qadrs] + dq, lo + 1e-3, hi - 1e-3)

        mujoco.mj_forward(self.m, self.d)
        resid = 0.0
        for bname, tgt, w in targets:
            if w <= 0.0:
                continue
            e = np.asarray(tgt, dtype=np.float64) - self.d.xpos[self.bid(bname)]
            resid = max(resid, float(np.linalg.norm(e[ri])))
        return self.d.qpos[7:36].copy(), resid

    def body_pos(self, name: str) -> np.ndarray:
        return self.d.xpos[self.bid(name)].copy()


def make_qpos_row(q: np.ndarray, x: float, z: float,
                  pitch: float = 0.0, head: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """29 关节 + 基座位姿 -> 38 维 qpos（head_z/head_y 恒 0 或指定）。"""
    row = np.zeros(38, dtype=np.float64)
    row[0], row[1], row[2] = x, 0.0, z
    row[3], row[5] = math.cos(pitch / 2), math.sin(pitch / 2)
    row[7:36] = q
    row[36:38] = head
    return row


def save_traj(path: Path, qpos: np.ndarray, action: str,
              meta: dict, dt: float = DT) -> None:
    """按项目既有格式落盘（time / qpos / dt / joint_names / action / meta）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(qpos)
    np.savez(path,
             time=np.arange(n, dtype=np.float64) * dt,
             qpos=np.asarray(qpos, dtype=np.float64),
             dt=float(dt),
             joint_names=np.array(JOINT_NAMES),
             action=action,
             meta=json.dumps(meta, ensure_ascii=False))
    print(f"[OK] {action}: {n} 帧 @ {1/dt:.0f} Hz = {n*dt:.2f} s -> "
          f"{path}  ({path.stat().st_size/1024:.0f} KB)")


def limit_report(qpos: np.ndarray, scene: Path | str = SCENE) -> list[str]:
    """返回关节越界描述（空列表 = 全部在限位内）。"""
    m = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
    q = qpos[:, 7:36]
    out = []
    for i, name in enumerate(JOINT_NAMES):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not m.jnt_limited[j]:
            continue
        lo, hi = m.jnt_range[j]
        c = q[:, i]
        if c.min() < lo - 1e-6 or c.max() > hi + 1e-6:
            out.append(f"{name}: [{np.degrees(c.min()):.1f}, "
                       f"{np.degrees(c.max()):.1f}]° 超出 "
                       f"[{np.degrees(lo):.1f}, {np.degrees(hi):.1f}]°")
    return out
