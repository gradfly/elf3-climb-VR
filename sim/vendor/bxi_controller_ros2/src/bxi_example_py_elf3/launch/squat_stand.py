"""ELF3 站立 → 蹲姿 动作序列生成器 + 静态稳定平衡校核。

输出与 crab_climb.py / prone_crawl.py 完全同格式的 npz（time / qpos / dt），
现有回放器可直接播放；额外带 joint_names / meta / 稳定性指标供校验脚本用。

==================== 姿态定义 ====================
站姿取训练里的 nominal pose，但把踝关节补成 −(hip+knee)，让**脚掌真正水平放平**。
原 default 的 ankle=0，下肢累计角是 hip+knee=−0.3+0.6=+0.3 rad，脚掌前倾 17°，
实际是踮着脚尖站（脚底最低点在脚尖、脚跟离地）。放平后基座高 1.0339 m，
与训练初始高度完全一致，说明这才是这组关节角的几何自洽解。

腿用两连杆（0.32 + 0.32）在矢状面解 IK：脚踝的世界目标锁在 (0, ·, 0.04)，
脚掌世界俯仰锁在 0（ankle = −(θ+hip+knee)）。两个约束一上，下蹲时脚既不前后滑
（实测全程 < 1e-15 m）也不翘脚跟。

**躯干前倾必须用基座俯仰，不能用 waist_y**：这台机器的腰部三个关节挂在
torso_link 下方、腿挂在腰下面（肩直接挂 torso），所以 waist_y 转的是"骨盆
相对躯干"，转它等于把腿整体往前甩（实测 −0.14 rad 时脚前移 8.8 cm），躯干纹丝不动。
蹲姿的"撅屁股 + 上身前倾"因此要写成：基座绕 +y 前倾 θ，腿在倾斜后的坐标系
里重新做 IK。附带好处：前倾时臀部自然后移 0.3825·sinθ，正好把重心往脚心拉。

蹲深受两条边界夹住：腿长（drop > ~0.36 直接不可达）和踝关节限位 −50°。
θ 越大，脚掌保持水平所需的背屈越小，所以带前倾的蹲法反而能蹲得更深。

==================== 静态稳定判据 ====================
1. **ZMP/COM 投影落在支撑多边形内**且留裕度。静态时 ZMP 就是 COM 的地面投影；
   支撑多边形取 14 个脚底胶囊球心的 xy 凸包（脚掌水平时即接触点的真实投影）。
2. **关节力矩不超限**：取 q̇=0 时的 qfrc_bias（重力广义力）作保守上界，
   与 XML 的 actuatorfrcrange 比对。
3. 静态水平力为零，摩擦锥不会越界；报告给的是保持该姿态的力矩裕度。

==================== 用法 ====================
    python squat_stand.py                        # 生成 + 打印校核摘要
    python squat_stand.py --drop 0.19 --lean 10   # 改蹲深/前倾
    python squat_stand.py --return                # 生成 站→蹲→站 完整循环
    python squat_stand.py --check-only            # 只算不落盘
    python squat_stand.py --write-scene           # 生成 elf3_squat.xml 平地场景
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
FLAT_XML = DATA / "terrains" / "universal_test" / "squat_flat.xml"
SCENE = DATA / "elf3_squat.xml"
DEFAULT_OUT = DATA / "squat_stand_traj.npz"

ORIG_INCLUDE = (
    '<include file="terrains/universal_test/universal_test_terrain_compact.xml"/>'
)
SQUAT_INCLUDE = '<include file="terrains/universal_test/squat_flat.xml"/>'

# 腿的几何（elf3.xml）：torso→hip 0.2265+0.156，hip→knee 0.32，knee→ankle 0.32
L_THIGH = 0.32
L_SHIN = 0.32
TORSO_TO_HIP = 0.2265 + 0.156
FOOT_SOLE = 0.04                        # 脚踝到脚底（capsule pos z=−0.03 减半径 0.01）
STANCE_HIP = -0.3
STANCE_Z = FOOT_SOLE + 2 * L_THIGH * math.cos(0.3) + TORSO_TO_HIP   # 1.0339
ANKLE_LIMIT = -0.87266                  # ankle_y 下限（rad），蹲深的硬约束

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
IDX = {n.replace("_joint", ""): i for i, n in enumerate(JOINT_NAMES)}

STANCE = np.array([
    0.0, 0.0, 0.0,
    STANCE_HIP, 0.0, 0.0, -2.0 * STANCE_HIP, -STANCE_HIP, 0.0,
    STANCE_HIP, 0.0, 0.0, -2.0 * STANCE_HIP, -STANCE_HIP, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)

# 场景外观：站立演示要能看清腿的折叠，地面用浅色棋盘格，主光从斜前上方来
FLOOR_RGB1 = "0.62 0.63 0.62"
FLOOR_RGB2 = "0.50 0.51 0.50"
FLOOR_TEXREPEAT = 24
FLOOR_HALF = 20.0
FLOOR_HALF_T = 0.05


# ---------------------------------------------------------------- 参数
@dataclass
class Params:
    drop: float = 0.25          # 蹲姿基座下降量（m）
    ank_x: float = 0.0          # 蹲姿脚踝世界 x（m，0=脚不离原位）
    lean: float = 13.0          # 基座前倾（deg）
    arm: float = -0.75          # shoulder_y 前摆（rad，负=向前）
    elbow: float = 0.80         # elbow_y 屈肘
    arm_out: float = 0.28       # shoulder_x
    head_pitch: float = 10.0    # 抬头补偿（deg，正=抬）
    pre_frac: float = 0.25      # 预蹲深度占 drop 的比例
    t_pre: float = 0.45         # 站姿 → 预蹲（s）
    t_desc: float = 1.15        # 预蹲 → 蹲姿（s）
    t_hold: float = 0.60        # 蹲姿静止保持（s）
    do_return: bool = False     # 是否加 蹲 → 站 回程
    t_rise: float = 1.25        # 蹲 → 站（s）
    t_settle: float = 0.35      # 回站后静止（s）
    dt: float = 0.02            # 采样周期（50 Hz，与训练控制频率一致）
    scene: str = str(SCENE)


# ---------------------------------------------------------------- 几何 / 求解
def leg_ik(lx: float, lz: float) -> tuple[float, float]:
    """两连杆 IK（腿自身坐标系）。lx=脚踝相对髋的 x，lz=向下为正的垂直距离。

    返回 (hip, knee)。knee>0 是正常屈膝方向；不可达抛 ValueError。
    """
    if not (0.0 < lz <= L_THIGH + L_SHIN + 1e-9):
        raise ValueError(f"腿长不可达: lz={lz:.3f}")

    def lz_of(q1: float) -> float:
        s = -(lx + L_THIGH * math.sin(q1)) / L_SHIN
        if abs(s) > 1.0:
            return -1e9
        return L_THIGH * math.cos(q1) + L_SHIN * math.cos(math.asin(s))

    lo, hi = -1.75, 0.02       # −100°~+1°：深蹲时大腿要能抬到接近水平以上
    if not (lz_of(lo) <= lz <= lz_of(hi)):
        raise ValueError(f"腿长不可达: lx={lx:.3f} lz={lz:.3f} "
                         f"(可达 {lz_of(lo):.3f}~{lz_of(hi):.3f})")
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if lz_of(mid) > lz:
            hi = mid            # 腿还太长 → 继续弯（q1 更负）
        else:
            lo = mid
    q1 = 0.5 * (lo + hi)
    s = -(lx + L_THIGH * math.sin(q1)) / L_SHIN
    q2 = math.asin(max(-1.0, min(1.0, s))) - q1
    return q1, q2


def squat_joints(drop: float, lean_deg: float, ank_x: float = 0.0):
    """给定基座下降量 / 前倾角 / 脚踝世界 x，返回 (hip, knee, ankle, 基座高)。

    基座绕 +y 转 θ 后髋的世界位置是 (−0.3825 sinθ, ·, h−0.3825 cosθ)；
    脚踝要求落在世界 (ank_x, ·, FOOT_SOLE) 且脚掌水平（世界俯仰 = 0）。
    """
    th = math.radians(lean_deg)
    h = STANCE_Z - drop
    hip_x = -TORSO_TO_HIP * math.sin(th)
    hip_z = h - TORSO_TO_HIP * math.cos(th)
    dxw = ank_x - hip_x
    dzw = hip_z - FOOT_SOLE
    # 世界位移转到腿坐标系（随基座俯仰转 −θ；dzw 的定义是「向下为正」，
    # 所以世界 z 分量是 −dzw，代入 R_y(−θ) 后两项符号都跟 dzw 同号）
    lx = dxw * math.cos(th) + dzw * math.sin(th)
    lz = -dxw * math.sin(th) + dzw * math.cos(th)
    q1, q2 = leg_ik(lx, lz)
    return q1, q2, -(th + q1 + q2), h


def make_pose(p: Params, tail: float = 0.0) -> tuple[np.ndarray, float, np.ndarray]:
    """返回 (29 维关节, 基座俯仰角 rad, head 两维)。tail=0 站姿，1 目标蹲姿。"""
    q = STANCE.copy()
    q1, q2, qa, _ = squat_joints(p.drop * tail, p.lean * tail, p.ank_x * tail)
    for s in (IDX["l_hip_y"], IDX["r_hip_y"]):
        q[s] = q1
        q[s + 3] = q2
        q[s + 4] = qa
    # 绕 y 的前摆/屈肘两侧同号；绕 x 的外张（shoulder_x）两侧反号 ——
    # 写同号会让右臂内收 0.48 rad 直接插进躯干 2.8 cm
    for s, sgn in ((IDX["l_shoulder_y"], 1.0), (IDX["r_shoulder_y"], -1.0)):
        q[s] = 0.2 + (p.arm - 0.2) * tail          # shoulder_y 前摆
        q[s + 1] = sgn * (0.2 + (p.arm_out - 0.2) * tail)  # shoulder_x 外张
        q[s + 3] = 0.6 + (p.elbow - 0.6) * tail    # elbow_y
    head = np.array([0.0, math.radians(p.head_pitch) * tail])
    return q, math.radians(p.lean) * tail, head


def smootherstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


# ---------------------------------------------------------------- 运动学 + 指标
class Kin:
    def __init__(self, scene: Path):
        self.m = mujoco.MjModel.from_xml_path(str(scene))
        self.d = mujoco.MjData(self.m)
        self.GE = mujoco.mjtObj.mjOBJ_GEOM
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._foot_geoms = [
            g for g in range(self.m.ngeom)
            if "_foot" in (mujoco.mj_id2name(self.m, self.GE, g) or "")
            and (mujoco.mj_id2name(self.m, self.GE, g) or "").endswith("_collision")
            and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
        ]

    def foot_low(self) -> float:
        d, m = self.d, self.m
        zs = []
        for g in self._foot_geoms:
            half = float(m.geom_size[g][1])
            r = float(m.geom_size[g][0])
            R = d.geom_xmat[g].reshape(3, 3)
            zs.append(d.geom_xpos[g][2] - (half * abs(R[2, 2]) + r))
        return float(min(zs))

    def foot_points(self) -> np.ndarray:
        """脚底胶囊两个球心的世界 xy（脚掌水平时即为接触点投影）。"""
        d, m = self.d, self.m
        pts = []
        for g in self._foot_geoms:
            half = float(m.geom_size[g][1])
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g]
            pts.append((c + half * R[:, 2])[:2])
            pts.append((c - half * R[:, 2])[:2])
        return np.array(pts)

    def place(self, q29: np.ndarray, head: np.ndarray, pitch: float,
              base_z: float | None = None) -> float:
        """写 qpos，并把脚底落到 z=0。返回实际基座高。"""
        self.d.qpos[:] = 0.0
        self.d.qpos[3:7] = [math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]
        self.d.qpos[7:36] = q29
        self.d.qpos[36:38] = head
        self.d.qpos[2] = 1.0 if base_z is None else base_z
        mujoco.mj_forward(self.m, self.d)
        if base_z is None:
            self.d.qpos[2] -= self.foot_low()
            mujoco.mj_forward(self.m, self.d)
        return float(self.d.qpos[2])

    def com(self) -> np.ndarray:
        return self.d.subtree_com[self.torso].copy()

    def self_contacts(self) -> list[str]:
        out = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            n1 = mujoco.mj_id2name(self.m, self.GE, c.geom1) or ""
            n2 = mujoco.mj_id2name(self.m, self.GE, c.geom2) or ""
            if "ground" in n1 + n2 or "floor" in n1 + n2:
                continue
            out.append(f"{n1}<->{n2}")
        return out

    def joint_torque(self) -> np.ndarray:
        """q̇=0 时的重力广义力（保守的关节力矩上界）。"""
        self.d.qvel[:] = 0.0
        self.d.qacc[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return np.array([self.d.qfrc_bias[self.m.jnt_dofadr[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)]] for n in JOINT_NAMES])


def limit_violations(kin: Kin, qpos: np.ndarray) -> list[tuple]:
    """全程越出机械限位的关节（MuJoCo 会把 qpos 夹住，脚掌就不水平了）。"""
    bad = []
    for i, n in enumerate(JOINT_NAMES):
        j = mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if not kin.m.jnt_limited[j]:
            continue
        lo, hi = kin.m.jnt_range[j]
        v = qpos[:, 7 + i]
        if float(v.min()) < lo - 1e-9 or float(v.max()) > hi + 1e-9:
            bad.append((n, float(v.min()), float(v.max()), float(lo), float(hi)))
    return bad


def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(pts, 9), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo: list = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    up: list = []
    for p in pts[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def margin_of(pt: np.ndarray, hull: np.ndarray) -> tuple[float, np.ndarray]:
    """点到凸包边界的带符号距离（内部为正）与最近的边外法线。"""
    best, nrm = 1e9, np.zeros(2)
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-12:
            continue
        nout = np.array([e[1], -e[0]]) / L      # 逆时针凸包的外法线
        sd = float(np.dot(a - pt, nout))        # 内部为正
        if sd < best:
            best, nrm = sd, nout
    return best, nrm


# ---------------------------------------------------------------- 序列构建
def build(p: Params, kin: Kin) -> dict:
    key_specs = [(0.0, 0.0)]
    t = 0.0
    t += p.t_pre
    key_specs.append((t, p.pre_frac))
    t += p.t_desc
    key_specs.append((t, 1.0))
    t += p.t_hold
    key_specs.append((t, 1.0))
    if p.do_return:
        t += p.t_rise
        key_specs.append((t, 0.0))
        t += p.t_settle
        key_specs.append((t, 0.0))

    n = int(round(t / p.dt)) + 1
    times = np.arange(n) * p.dt

    qpos = np.zeros((n, 38))
    qpos[:, 3] = 1.0
    pitches = np.zeros(n)
    # 关键点之间插值的是**任务参数 tail**，不是关节角：每帧重跑一次腿 IK，
    # 脚踝世界 x 和脚掌水平才能逐帧严格成立（插关节角会让脚在地上蹭几厘米）
    for i, tt in enumerate(times):
        k = 0
        while k < len(key_specs) - 2 and tt >= key_specs[k + 1][0]:
            k += 1
        t0 = key_specs[k][0]
        t1 = key_specs[k + 1][0]
        s = smootherstep(0.0 if t1 <= t0 else (tt - t0) / (t1 - t0))
        tail = key_specs[k][1] + s * (key_specs[k + 1][1] - key_specs[k][1])
        q, pitch, head = make_pose(p, tail)
        qpos[i, 7:36] = q
        qpos[i, 36:38] = head
        pitches[i] = pitch

    coms = np.zeros((n, 3))
    margins = np.zeros(n)
    torso_z = np.zeros(n)
    hulls = []
    torques = np.zeros((n, len(JOINT_NAMES)))
    self_col: list[str] = []
    for i in range(n):
        torso_z[i] = kin.place(qpos[i, 7:36], qpos[i, 36:38], pitches[i])
        qpos[i, 2] = torso_z[i]
        qpos[i, 3] = math.cos(pitches[i] / 2)      # 基座绕 +y 俯仰的四元数
        qpos[i, 5] = math.sin(pitches[i] / 2)
        coms[i] = kin.com()
        hull = convex_hull(kin.foot_points())
        hulls.append(hull)
        margins[i], _ = margin_of(coms[i, :2], hull)
        torques[i] = kin.joint_torque()
        for c in kin.self_contacts():
            if c not in self_col:
                self_col.append(c)

    qd = np.gradient(qpos[:, 7:36], p.dt, axis=0)

    return dict(time=times, qpos=qpos, com=coms, margin=margins, pitch=pitches,
                torso_z=torso_z, hull=hulls[-1], torques=torques, qvel=qd,
                self_contacts=self_col, n=n)


def report(res: dict, p: Params, kin: Kin) -> None:
    times, qpos = res["time"], res["qpos"]
    margins, coms, tz = res["margin"], res["com"], res["torso_z"]
    n = res["n"]
    i_sq = min(int(round((p.t_pre + p.t_desc) / p.dt)), n - 1)

    hull = res["hull"]
    print("\n" + "=" * 74)
    print("ELF3 站立 → 蹲姿   动作序列 + 静态稳定校核")
    print("=" * 74)

    print(f"\n[时序] {n} 帧 @ {1/p.dt:.0f} Hz = {times[-1]:.2f} s")
    segs = [("站姿 → 预蹲", 0.0, p.t_pre),
            ("预蹲 → 蹲姿", p.t_pre, p.t_pre + p.t_desc),
            ("蹲姿保持", p.t_pre + p.t_desc, p.t_pre + p.t_desc + p.t_hold)]
    if p.do_return:
        a = p.t_pre + p.t_desc + p.t_hold
        segs.append(("蹲姿 → 站起", a, a + p.t_rise))
        segs.append(("站姿静止", a + p.t_rise, times[-1]))
    for name, a, b in segs:
        print(f"       {a:5.2f} – {b:5.2f} s   {name}")

    print(f"\n[站姿] 躯干 z = {tz[0]:.4f} m   脚底 = 0   前倾 0.0°")
    print(f"[蹲姿] 躯干 z = {tz[i_sq]:.4f} m   (下降 {tz[0]-tz[i_sq]:.4f} m, "
          f"{100*(1-tz[i_sq]/tz[0]):.1f}%)   前倾 {math.degrees(res['pitch'][i_sq]):.1f}°")
    q1 = qpos[i_sq, 7 + IDX["l_hip_y"]]
    q2 = qpos[i_sq, 7 + IDX["l_knee_y"]]
    qa = qpos[i_sq, 7 + IDX["l_ankle_y"]]
    print(f"       髋 {math.degrees(q1):+.1f}°  膝 {math.degrees(q2):+.1f}°  "
          f"踝 {math.degrees(qa):+.1f}°  (踝限位 {math.degrees(ANKLE_LIMIT):.1f}°)")
    print(f"       肩前摆 {math.degrees(qpos[i_sq, 7 + IDX['l_shoulder_y']]):+.1f}°  "
          f"屈肘 {math.degrees(qpos[i_sq, 7 + IDX['l_elbow_y']]):+.1f}°  "
          f"抬头 {math.degrees(qpos[i_sq, 37]):+.1f}°")

    print(f"\n[支撑多边形] {len(hull)} 顶点   "
          f"x∈[{hull[:,0].min():+.3f}, {hull[:,0].max():+.3f}]  "
          f"y∈[{hull[:,1].min():+.3f}, {hull[:,1].max():+.3f}]  (m)")
    area = abs(np.dot(hull[:, 0], np.roll(hull[:, 1], 1))
               - np.dot(hull[:, 1], np.roll(hull[:, 0], 1))) / 2
    print(f"             面积 {area:.4f} m²   前后长 {hull[:,0].max()-hull[:,0].min():.3f} m")

    print(f"\n[静态稳定] 蹲姿 COM 投影 = ({coms[i_sq,0]:+.4f}, {coms[i_sq,1]:+.4f}) m")
    print(f"           稳定裕度（到支撑边界）= {margins[i_sq]*100:.2f} cm "
          f"= 脚长的 {100*margins[i_sq]/0.22:.0f}%")
    print(f"           全程最小裕度 = {margins.min()*100:.2f} cm @ t = "
          f"{times[int(margins.argmin())]:.2f} s")
    print(f"           ZMP 判据：{'满足 (裕度 > 2 cm)' if margins.min() > 0.02 else '不满足 (裕度 < 2 cm)'}")

    tau = np.abs(res["torques"][i_sq])
    limits = np.array([abs(kin.m.jnt_actfrcrange[
        mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_JOINT, nm)])[1] for nm in JOINT_NAMES])
    ratio = tau / np.maximum(limits, 1e-9)
    print(f"\n[关节力矩] 蹲姿最大占用 {100*ratio.max():.1f}%  ({JOINT_NAMES[int(ratio.argmax())]})")
    for i in np.argsort(-ratio)[:3]:
        print(f"           {JOINT_NAMES[i]:20s} {tau[i]:7.2f} N·m  "
              f"{100*ratio[i]:5.1f}% of {limits[i]:.0f}")

    qd = res["qvel"]
    peak = np.abs(qd).max(axis=0)
    i_pk = int(peak.argmax())
    print(f"\n[关节速度] 峰值 {math.degrees(peak[i_pk]):.0f} °/s ({JOINT_NAMES[i_pk]})")

    lim = limit_violations(kin, qpos)
    if lim:
        print(f"\n[关节限位] {len(lim)} 个关节越界（MuJoCo 会夹住，脚掌将不水平）：")
        for jname, vmin, vmax, lo, hi in lim:
            print(f"           {jname:20s} [{math.degrees(vmin):+.1f}, {math.degrees(vmax):+.1f}]"
                  f"  限位 [{math.degrees(lo):+.1f}, {math.degrees(hi):+.1f}]")
    else:
        print("\n[关节限位] 全程在机械范围内")

    print(f"\n[自碰撞] {'无' if not res['self_contacts'] else res['self_contacts']}")
    print("=" * 74)


# ---------------------------------------------------------------- 场景
def write_scene(force: bool = False) -> None:
    """生成 elf3_squat.xml + squat_flat.xml（幂等）。"""
    FLAT_XML.parent.mkdir(parents=True, exist_ok=True)
    flat = (
        "<mujocoinclude>\n"
        "  <!-- 蹲姿演示用的纯平地面，由 squat_stand.py 生成（勿手改）。 -->\n"
        '  <include file="assets.xml"/>\n'
        "  <asset>\n"
        '    <texture name="squat_floor_tex" type="2d" builtin="checker" '
        f'rgb1="{FLOOR_RGB1}" rgb2="{FLOOR_RGB2}" width="512" height="512"/>\n'
        f'    <material name="squat_floor" texture="squat_floor_tex" '
        f'texrepeat="{FLOOR_TEXREPEAT} {FLOOR_TEXREPEAT}" '
        'specular="0.15" shininess="0.10"/>\n'
        "  </asset>\n"
        "  <worldbody>\n"
        '    <light name="squat_world_light" pos="0 0 6" dir="0 0 -1" directional="true"\n'
        '           diffuse="0.55 0.55 0.55" ambient="0.32 0.32 0.32"\n'
        '           specular="0.05 0.05 0.05" castshadow="false"/>\n'
        '    <geom name="squat_ground" type="box" '
        f'pos="0 0 -{FLOOR_HALF_T}" size="{FLOOR_HALF} {FLOOR_HALF} {FLOOR_HALF_T}" '
        'material="squat_floor" friction="1.0 0.01 0.001" condim="3"/>\n'
        "  </worldbody>\n"
        "</mujocoinclude>\n"
    )
    if force or not FLAT_XML.exists() or FLAT_XML.read_text(encoding="utf-8") != flat:
        FLAT_XML.write_text(flat, encoding="utf-8")
        print(f"[场景] 写入 {FLAT_XML.relative_to(DATA)}")

    src = (DATA / "elf3.xml").read_text(encoding="utf-8")
    if ORIG_INCLUDE not in src:
        raise SystemExit(f"[错误] elf3.xml 里找不到 {ORIG_INCLUDE}")
    text = src.replace(ORIG_INCLUDE, SQUAT_INCLUDE)
    # zfar=2 会把 2 m 外的相机画面整个裁掉（离屏渲染时表现为"机器人凭空消失"）
    text = text.replace('<map znear="0.0005" zfar="2"/>',
                        '<map znear="0.0005" zfar="100"/>')
    text = text.replace('<visual>', '<visual><global offwidth="1280" offheight="720"/>')
    key = ('    <light name="squat_key" pos="1.6 -1.9 2.4" dir="-0.55 0.65 -1.0" '
           'directional="false" diffuse="1.0 1.0 1.0" ambient="0.25 0.25 0.25" '
           'specular="0.08 0.08 0.08" castshadow="true"/>\n')
    if "squat_key" not in text:
        text = text.replace("<worldbody>\n", "<worldbody>\n" + key, 1)
    # 头顶那盏 trackcom 会在地上压出一块纯黑圆斑，直接删
    text = text.replace('      <light pos="0 0 2" dir="0 0 -1" mode="trackcom"/>\n', "")
    if force or not SCENE.exists() or SCENE.read_text(encoding="utf-8") != text:
        SCENE.write_text(text, encoding="utf-8")
        print(f"[场景] 写入 {SCENE.name}")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="ELF3 站立→蹲姿 动作序列生成器（含静态稳定校核）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drop", type=float, default=0.25, help="基座下降量 m")
    ap.add_argument("--lean", type=float, default=13.0, help="基座前倾 deg")
    ap.add_argument("--ank-x", type=float, default=0.0, help="蹲姿脚踝世界 x（m）")
    ap.add_argument("--arm", type=float, default=-0.75, help="肩前摆 rad")
    ap.add_argument("--elbow", type=float, default=0.80, help="屈肘 rad")
    ap.add_argument("--arm-out", type=float, default=0.28, help="肩外张 rad")
    ap.add_argument("--head", type=float, default=10.0, help="抬头 deg")
    ap.add_argument("--dt", type=float, default=0.02, help="采样周期 s")
    ap.add_argument("--return", dest="do_return", action="store_true",
                    help="加 蹲→站 回程")
    ap.add_argument("--out", default=None, help="输出 npz")
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--write-scene", action="store_true")
    ap.add_argument("--force-scene", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--json", default=None, help="把指标额外写一份 JSON")
    args = ap.parse_args()

    p = Params(drop=args.drop, lean=args.lean, ank_x=args.ank_x, arm=args.arm,
               elbow=args.elbow, arm_out=args.arm_out, head_pitch=args.head,
               dt=args.dt, do_return=args.do_return, scene=args.scene)

    if args.write_scene or not Path(p.scene).exists() or args.force_scene:
        write_scene(force=args.force_scene)

    try:
        squat_joints(p.drop, p.lean, p.ank_x)
    except ValueError as e:
        raise SystemExit(f"[错误] {e}；把 --drop 调小或 --lean 调大")

    kin = Kin(Path(p.scene))
    res = build(p, kin)
    report(res, p, kin)

    if args.check_only:
        return

    out = Path(args.out) if args.out else DEFAULT_OUT
    i_sq = min(int(round((p.t_pre + p.t_desc) / p.dt)), res["n"] - 1)
    meta = dict(
        drop=p.drop, ank_x=p.ank_x, lean_deg=p.lean, arm_rad=p.arm,
        elbow_rad=p.elbow, head_deg=p.head_pitch,
        duration=float(res["time"][-1]),
        t_pre=p.t_pre, t_desc=p.t_desc, t_hold=p.t_hold, t_rise=p.t_rise,
        stance_z=float(res["torso_z"][0]),
        squat_z=float(res["torso_z"][i_sq]),
        squat_margin=float(res["margin"][i_sq]),
        min_margin=float(res["margin"].min()),
        squat_com=[float(x) for x in res["com"][i_sq]],
        squat_hip_deg=float(math.degrees(res["qpos"][i_sq, 7 + IDX["l_hip_y"]])),
        squat_knee_deg=float(math.degrees(res["qpos"][i_sq, 7 + IDX["l_knee_y"]])),
        squat_ankle_deg=float(math.degrees(res["qpos"][i_sq, 7 + IDX["l_ankle_y"]])),
    )
    np.savez(out, time=res["time"], qpos=res["qpos"], dt=p.dt,
             joint_names=JOINT_NAMES, meta=meta)
    print(f"\n[OK] 轨迹 -> {out}  ({res['n']} 帧, {out.stat().st_size/1024:.0f} KB)")
    print(f"     回放: python squat_replay.py --traj {out.name} --mode kin")

    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(meta=meta, params=asdict(p),
                            margin=[float(x) for x in res["margin"]]), indent=2),
            encoding="utf-8")
        print(f"     指标 -> {args.json}")


if __name__ == "__main__":
    main()
