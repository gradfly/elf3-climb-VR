"""螃蟹步横移爬楼梯 —— 闭式 IK + DLS 精修。

只用 numpy + MuJoCo C API，不需要 scipy / pinocchio。

==================== 运动学模型 ====================
坐标：机器人朝 +X，左腿在 +Y 侧，z 向上。
腿链：hip_y -> hip_x -> hip_z -> knee_y -> ankle_y -> ankle_x
大腿 = 小腿 = 0.32 m，故单腿可达半径 rho <= 0.64 m。

运动学分解（精确，已用数值 IK 交叉验证到 0.04 deg）：
    腿是一条平面二连杆。给定足踝相对髋的三维偏移 (X, Y, D)（D = 髋高 - 踝高）：
      1) 先在"矢状面"内解出 (X, D_s)，其中 D_s = hypot(Y, D)
      2) 再绕 X 轴整体滚转 theta = atan2(Y, D)
    于是：
        rho    = |(X, Y, D)| = 0.64 cos(k/2)
        t0     = acos(rho / 0.64)
        s      = arctan2(-X, D_s)
        hip_y  = s - t0 ;  knee = 2*t0 ;  ankle_y = -(s + t0)
        hip_x  = +theta ;  ankle_x = -theta        （足掌保持水平必须这么补）

    注意一个【常见错解】：侧向增益不是 0.32(1+cos k)，而是 rho。
        Y = rho sin(theta)，即 theta = asin(Y / rho) = atan2(Y, D)
    用错公式在 knee=0.6 时偏 4.5%，knee=1.7 时偏 34%，会把可行性算飞。

==================== 硬约束（elf3.xml 实测限位）====================
    knee   in [-5, 150] deg   -> 取 >= 8.6 deg 防奇异 -> rho <= 0.638
    ankle_y in [-50, +45] deg -> 平脚时 |s + t0| 受限 -> rho >= 0.411 (X=0 时)
    ankle_x in [-20, +20] deg -> 脚掌水平时 hip_x = -ankle_x -> |hip_x| <= 20 deg
                              -> 单脚侧向偏移 |Y| <= rho * sin20deg ≈ 0.206 m
    hip_x  L in [-28, 175] deg / R in [-175, 28] deg（不是瓶颈）
    ankle_x 执行器只有 ±20 Nm —— 横向受力主要靠 hip_x，不靠踝

==================== MuJoCo API 陷阱 ====================
    mj_jacSite / mj_jac / mj_jacBody 在只调 mj_kinematics 时返回【全零矩阵】！
    必须先 mj_kinematics + mj_comPos + mj_crb（或直接 mj_forward）。
"""
from __future__ import annotations

import numpy as np
import mujoco

THIGH = 0.32
SHANK = 0.32
LEG_REACH = THIGH + SHANK          # 0.64
RHO_MAX = 0.6399                   # 留一点数值余量
RHO_MIN = 0.411                    # 平脚 + ankle_y 限位的下限（X=0 时）
KNEE_MIN = 0.15                    # 防膝奇异 (rad)
# 膝角下限 k>=KNEE_MIN 等价于 rho <= 0.64*cos(KNEE_MIN/2)，比 RHO_MAX 更紧
RHO_KNEE = LEG_REACH * np.cos(KNEE_MIN / 2.0)
ANKLE_Y_LO = -0.87266              # -50 deg
ANKLE_Y_HI = 0.7854                # +45 deg
ANKLE_X_LIMIT = 0.34907            # ±20 deg —— 脚掌水平时它直接限制 hip_x

ANKLE_SOLE_DROP = 0.04             # 足底比 ankle_x_link 原点低 4 cm（胶囊 z=-0.03, r=0.01）

# ---- 骨盆侧倾 waist_x（关键自由度）----
# waist_x 在 torso_link 下游：转它只动双腿，不动躯干、不动相机。
# 实测标定（elf3_crab.xml，ISAAC 默认位形）：
#   waist_x=+15deg -> +Y 髋(L) 抬高 4.02 cm，-Y 髋(R) 降低 4.02 cm，髋高差 8.0 cm
# 正向侧倾让"上坡侧髋抬高、下坡侧髋降低"，同时放松两条腿的 rho 约束，
# 效果远好于把双脚前后错开（25 cm 级高：侧倾只需 10.8deg，前后错位要 31 cm/侧）。
HIP_LATERAL = 0.136     # 髋关节在 ±Y 的标称位置
HIP_DROP = 0.156        # 髋关节在 waist_z 关节原点下方
WAIST_X_LIMIT = 0.2618  # ±15 deg（elf3.xml: waist_x_joint range）


def tilt_delta(phi: float) -> float:
    """waist_x = phi 时，+Y 侧髋相对中位【抬高】的量（米）。

    phi > 0 抬高 +Y 髋（L 腿）—— SPLIT 相位里 L 脚在上层踏面，正需要这个方向。
    """
    return HIP_LATERAL * np.sin(phi) + HIP_DROP * (1.0 - np.cos(phi))

LEG_JOINTS = (
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint",
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint",
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
)
FOOT_BODIES = ("l_ankle_x_link", "r_ankle_x_link")


# ---------------------------------------------------------------- 闭式解
def sagittal_closed_form(X: float, D: float):
    """矢状面：足踝相对髋的前后偏移 X、垂直落差 D -> (hip_y, knee_y, ankle_y)。

    保留只是为了自检复现 ISAAC 默认位形。三维情形请用 leg_solution()。
    """
    rho = float(np.hypot(X, D))
    if rho > RHO_MAX:
        sc = RHO_MAX / rho
        X *= sc
        D *= sc
        rho = RHO_MAX
    if rho < 1e-9:
        rho = 1e-9
    t0 = float(np.arccos(np.clip(rho / LEG_REACH, -1.0, 1.0)))
    s = float(np.arctan2(-X, D))
    return s - t0, 2.0 * t0, -(s + t0)


def leg_solution(X: float, Y: float, D: float) -> dict:
    """三维精确解：足踝相对髋的偏移 (X, Y, D) -> 六个关节角。

    D = 髋高 - 踝高（恒为正）。超出可达半径时沿半径方向整体回缩。
    返回 dict，含各关节角、rho 与逐项限位判定。
    """
    v = np.array([float(X), float(Y), float(D)])
    rho = float(np.linalg.norm(v))
    if rho > RHO_MAX:
        v *= RHO_MAX / rho
        rho = RHO_MAX
    X, Y, D = (float(t) for t in v)

    D_s = float(np.hypot(Y, D))            # 矢状面内的等效垂直落差
    theta = float(np.arctan2(Y, D))        # 绕 X 轴的整体滚转 = hip_x
    rho = float(np.clip(rho, 1e-9, LEG_REACH))
    t0 = float(np.arccos(np.clip(rho / LEG_REACH, -1.0, 1.0)))
    s = float(np.arctan2(-X, D_s))
    h = s - t0
    k = 2.0 * t0
    a_y = -(s + t0)
    return {
        "X": X, "Y": Y, "D": D, "rho": rho,
        "hip_y": h, "hip_x": theta, "hip_z": 0.0,
        "knee": k, "ankle_y": a_y, "ankle_x": -theta,
        "ok_knee": k >= KNEE_MIN - 1e-9,
        "ok_ankle_y": ANKLE_Y_LO - 1e-9 <= a_y <= ANKLE_Y_HI + 1e-9,
        "ok_ankle_x": abs(theta) <= ANKLE_X_LIMIT - 1e-9,
        "ok_rho": rho >= RHO_MIN - 1e-9,
        "ok": (k >= KNEE_MIN - 1e-9
               and ANKLE_Y_LO - 1e-9 <= a_y <= ANKLE_Y_HI + 1e-9
               and abs(theta) <= ANKLE_X_LIMIT - 1e-9),
    }


def lateral_angle(Y: float, rho: float) -> float:
    """侧向偏移 Y 所需的 hip_x（= -ankle_x）。rho = |(X,Y,D)|。

    注意：增益是 rho，不是 0.32(1+cos k)。
    """
    if rho < 1e-9:
        return 0.0
    return float(np.arcsin(np.clip(Y / rho, -0.999, 0.999)))


def foot_flat_quat():
    """足掌平贴、朝向机器人正前方(+X) 的目标四元数（世界系 = 单位四元数）。"""
    return np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------- 可行性
def check_pose(X: float, Y: float, D: float) -> dict:
    """检查单个 (X, Y, D) 足踝目标是否可行（leg_solution 的同义入口）。"""
    return leg_solution(X, Y, D)


# ---------------------------------------------------------------- 级高/错位 可行性
def split_window(r: float, sep: float, tilt: float = 0.0, n: int = 241):
    """SPLIT 相位：扫描髋【平均】高 H（相对下坡踏面）的可行区间。

    r    : 级高
    sep  : 两脚在 Y 上的间距（上坡脚在 +Y）
    tilt : waist_x 侧倾，正 = +Y 髋抬高

    侧倾 delta 让上坡腿的 D 变大（少弯膝）、下坡腿的 D 变小（别伸太直），
    两个约束同时放松 —— 这就是侧倾比前后错位更划算的原因。
    返回 (lo, hi) 或 None。
    """
    c = ANKLE_SOLE_DROP
    dl = tilt_delta(tilt)
    # 侧倾后髋的半间距缩小为 HIP_LATERAL*cos(tilt)；两脚相对各自髋的侧移对称
    A = sep / 2.0 - HIP_LATERAL * np.cos(tilt)
    ok = []
    for H in np.linspace(0.32, 0.78, n):
        D_u = H + dl - r - c
        D_d = H - dl - c
        if D_u <= 0.02 or D_d <= 0.02:      # 髋不能低于脚
            continue
        if leg_solution(0.0, +A, D_u)["ok"] and leg_solution(0.0, -A, D_d)["ok"]:
            ok.append(H)
    if not ok:
        return None
    return float(min(ok)), float(max(ok))


def feasible_H_range(r: float, sx: float = 0.0, tilt: float = 0.0, n: int = 241):
    """兼容旧签名：默认按"双脚正好在髋正下方"(sep = 2*HIP_LATERAL) 估算。"""
    return split_window(r, 2.0 * HIP_LATERAL + 2.0 * sx, tilt, n)


def tilt_for_rise(r: float, sep: float = 2.0 * HIP_LATERAL,
                  min_window: float = 0.030, step: float = 0.002):
    """求覆盖级高 r 所需的【最小】骨盆侧倾（带符号）。

    r > 0（上楼，L 脚更高）：需要抬高 L 髋 -> tilt > 0
    r < 0（下楼，L 脚更低）：需要抬高 R 髋 -> tilt < 0
    两者镜像对称，所以只在 sign(r) 方向搜索。

    侧倾越小越好（躯干负荷更对称），所以只要髋高窗口达到 min_window 就停。
    返回 (tilt_rad, 窗口宽, (lo_H, hi_H)) 或 None。
    """
    sgn = 1.0 if r >= 0.0 else -1.0
    best = None
    for k in np.arange(0.0, WAIST_X_LIMIT + 1e-9, step):
        rng = split_window(r, sep, sgn * float(k))
        if rng is None:
            continue
        w = rng[1] - rng[0]
        if best is None or w > best[1]:
            best = (sgn * float(k), float(w), rng)
        if w >= min_window:
            break
    return best


def foot_placement(tread_depth: float, foot_half_y: float = 0.04,
                   gap: float = 0.02, edge_margin: float = 0.02):
    """双脚落在【同一级】踏面时，各自相对踏面中心的 Y 偏移 off。

    脚掌 Y 向宽 8 cm。约束：
        两脚不重叠 : 2*off >= 2*foot_half_y + gap
        都在踏面内 : 2*off + 2*foot_half_y <= depth - 2*edge_margin
    """
    off = (tread_depth - 2.0 * foot_half_y - 2.0 * edge_margin) / 2.0
    return float(max(off, foot_half_y + gap / 2.0))


# ---------------------------------------------------------------- IK 求解器
class CrabIK:
    """双腿 IK：闭式解析种子 + DLS 精修。"""

    def __init__(self, model, data, use_waist_x: bool = False):
        self.model = model
        self.data = data
        self.joint_names = list(LEG_JOINTS)
        if use_waist_x:
            self.joint_names.append("waist_x_joint")

        self.jnt_id = []
        for n in self.joint_names:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j < 0:
                raise SystemExit(f"[错误] 模型里没有关节 {n}")
            self.jnt_id.append(j)
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jnt_id])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.jnt_id])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jnt_id])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jnt_id])

        self.body_id = []
        for n in FOOT_BODIES:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                raise SystemExit(f"[错误] 模型里没有 body {n}")
            self.body_id.append(b)

        self.ndof = len(self.joint_names)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ---- 解析种子 ----
    def hip_positions(self):
        """读取当前姿态下左右髋关节的世界坐标。

        髋关节位置只由躯干姿态 + 腰关节决定，与腿关节无关，
        所以在 IK 求解过程中保持不变，可以直接读出来用。
        """
        m, d = self.model, self.data
        out = []
        for n in ("l_hip_y_link", "r_hip_y_link"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            out.append(d.xpos[b].copy())
        return out[0], out[1]

    def seed(self, hip_l, hip_r, target_l, target_r):
        """由左右【髋关节世界坐标】与两足踝目标位置给出解析种子。

        注意：必须传髋关节位置而不是骨盆/躯干位置 —— 髋在 waist_z_link 下方
        0.156 m，用骨盆位置算会让 D 差 15.6 cm，种子直接失准。
        """
        q = np.zeros(self.ndof)
        for side, tgt, hip in ((0, target_l, hip_l), (1, target_r, hip_r)):
            sol = leg_solution(
                float(tgt[0] - hip[0]),
                float(tgt[1] - hip[1]),
                float(hip[2] - tgt[2]),
            )
            h, k, a_y, theta = sol["hip_y"], sol["knee"], sol["ankle_y"], sol["hip_x"]
            base = side * 6
            q[base + 0] = h
            q[base + 1] = theta
            q[base + 2] = 0.0          # hip_z：残余偏航，交给 DLS 修
            q[base + 3] = k
            q[base + 4] = a_y
            q[base + 5] = -theta       # ankle_x 补偿，保持足掌水平
        if self.ndof > 12:
            q[12] = 0.0
        return np.clip(q, self.lo, self.hi)

    # ---- 单次误差 ----
    def _residual(self, targets):
        """返回 (e, J)。e 为 12 维误差（每脚 3 位置 + 3 旋转）。"""
        e = np.zeros(12)
        J = np.zeros((12, self.ndof))
        d = self.data
        m = self.model
        for i, b in enumerate(self.body_id):
            mujoco.mj_jacBody(m, d, self._jacp, self._jacr, b)
            dp = targets[i][0] - d.xpos[b]
            # 旋转误差：q_err = q_target * conj(q_cur)，再转成旋转向量
            qtgt = targets[i][1]
            qcur = d.xquat[b]
            qneg = np.array([qcur[0], -qcur[1], -qcur[2], -qcur[3]])
            qerr = np.zeros(4)
            mujoco.mju_mulQuat(qerr, qtgt, qneg)
            if qerr[0] < 0:            # 取短弧
                qerr = -qerr
            dr = np.zeros(3)
            mujoco.mju_quat2Vel(dr, qerr, 1.0)
            e[i * 6: i * 6 + 3] = dp
            e[i * 6 + 3: i * 6 + 6] = dr
            J[i * 6: i * 6 + 3, :] = self._jacp[:, self.vadr]
            J[i * 6 + 3: i * 6 + 6, :] = self._jacr[:, self.vadr]
        return e, J

    def _apply(self, q):
        self.data.qpos[self.qadr] = q
        self.data.qvel[:] = 0.0
        # 关键：雅可比需要 kinematics + comPos + crb，缺一则全零
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_crb(self.model, self.data)

    def solve(self, q_init, targets, iters=40, lam=0.03, pos_w=1.0, rot_w=0.6):
        """DLS 精修。targets = [(pos3, quat4), (pos3, quat4)]（世界系）。

        rot_w 可以是标量（双脚同权重）或 (w_L, w_R)。
        摆动脚离地后【不必保持水平】，把它的 rot_w 降到 0 能解锁 ankle_x
        的 ±20deg 硬限位 —— 这是宽站姿下完成重心转移的关键。
        """
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.lo, self.hi)
        self._apply(q)
        rw = np.atleast_1d(np.asarray(rot_w, dtype=float))
        if rw.size == 1:
            rw = np.repeat(rw, 2)
        W = np.array([pos_w] * 3 + [rw[0]] * 3 + [pos_w] * 3 + [rw[1]] * 3)
        for _ in range(iters):
            e, J = self._residual(targets)
            Jw = J * W[:, None]
            ew = e * W
            A = Jw @ Jw.T + (lam ** 2) * np.eye(12)
            dq = Jw.T @ np.linalg.solve(A, ew)
            q = np.clip(q + dq, self.lo, self.hi)
            # 膝角下限保护：腿伸直是奇异点
            q[3] = max(q[3], KNEE_MIN)
            q[9] = max(q[9], KNEE_MIN)
            self._apply(q)
        e, _ = self._residual(targets)
        return q, float(np.abs(e[:3]).max()), float(np.abs(e[6:9]).max()), \
            float(np.abs(e[3:6]).max()), float(np.abs(e[9:12]).max())


# ---------------------------------------------------------------- 自检
def _selftest(scene: str):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS as P

    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)

    print("=== 1) 闭式解能否反解出 ISAAC 默认位形 ===")
    h, k, a = sagittal_closed_form(0.0, 0.6114)
    print(f"  输入 X=0, D=0.6114 -> hip_y={h:.4f} knee={k:.4f} ankle_y={a:.4f}")
    print(f"  期望            -> hip_y=-0.3000 knee=0.6000 ankle_y=-0.3000")
    ok = abs(h + 0.3) < 1e-3 and abs(k - 0.6) < 1e-3 and abs(a + 0.3) < 1e-3
    print("  " + ("PASS" if ok else "FAIL"))

    print()
    print("=== 2) 侧向增益 ===")
    print(f"  默认站姿(knee=0.6): dY/d(hip_x) = {THIGH*(1+np.cos(0.6)):.4f} m/rad")

    print()
    print("=== 3) DLS 精修精度（平地站立）===")
    ik = CrabIK(m, d)
    for n, q0 in zip(P.layout.names, P.default_position):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q0
    d.qpos[0:3] = [0.0, 0.0, 1.0339]
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    hip_l, hip_r = ik.hip_positions()
    print(f"  髋关节世界坐标: L={np.round(hip_l,3)}  R={np.round(hip_r,3)}")
    tgt_l = np.array([hip_l[0], hip_l[1], ANKLE_SOLE_DROP])
    tgt_r = np.array([hip_r[0], hip_r[1], ANKLE_SOLE_DROP])
    q0 = ik.seed(hip_l, hip_r, tgt_l, tgt_r)
    q, el, er, rl, rr = ik.solve(q0, [(tgt_l, foot_flat_quat()), (tgt_r, foot_flat_quat())])
    print(f"  位置误差 L={el*1000:.2f} mm  R={er*1000:.2f} mm")
    print(f"  姿态误差 L={rl:.4f} rad  R={rr:.4f} rad")
    print(f"  解出 knee L={q[3]:.3f} R={q[9]:.3f} (下限 {KNEE_MIN})")


def _verify(scene: str):
    """端到端复核：用真实数值 IK 在最坏工况上验证闭式模型给出的结论。

    闭式模型只用于"快速扫描出可行域"，真正落地必须过这一关 ——
    因为关节轴是逐级传递的（hip_y 会改变 hip_x 的轴方向），
    闭式分解里的"角度相加"只是近似，数值 IK 才是判据。
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS as P

    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG_JOINTS}
    jid["waist_x_joint"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "waist_x_joint")
    lim = {n: (m.jnt_range[jid[n]][0], m.jnt_range[jid[n]][1]) for n in jid}
    for n, q0 in zip(P.layout.names, P.default_position):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            d.qpos[m.jnt_qposadr[j]] = q0
    d.qpos[3:7] = [1, 0, 0, 0]
    ik = CrabIK(m, d)

    def pose(tilt, fL, fR, H_world):
        """fL/fR = (y, 踏面顶 z)。返回 IK 解与误差。"""
        d.qpos[m.jnt_qposadr[jid["waist_x_joint"]]] = tilt
        d.qpos[1] = 0.5 * (fL[0] + fR[0]) - HIP_DROP * np.sin(tilt)
        d.qpos[2] = 1.0339
        for _ in range(6):                       # 迭代对齐髋的平均高度
            mujoco.mj_kinematics(m, d)
            hl, hr = ik.hip_positions()
            d.qpos[2] += H_world - 0.5 * (hl[2] + hr[2])
        mujoco.mj_kinematics(m, d)
        hl, hr = ik.hip_positions()
        tl = np.array([0.0, fL[0], fL[1] + ANKLE_SOLE_DROP])
        tr = np.array([0.0, fR[0], fR[1] + ANKLE_SOLE_DROP])
        q0 = ik.seed(hl, hr, tl, tr)
        return ik.solve(q0, [(tl, foot_flat_quat()), (tr, foot_flat_quat())], iters=150)

    def report(tag, tilt, fL, fR, H_rel):
        zlo = min(fL[1], fR[1])                  # 髋高窗口相对【下级】踏面
        q, el, er, rl, rr = pose(tilt, fL, fR, zlo + H_rel)
        bad = [f"{n}={np.degrees(q[i]):.1f}d" for i, n in enumerate(LEG_JOINTS)
               if q[i] < lim[n][0] - 1e-6 or q[i] > lim[n][1] + 1e-6]
        ok = (not bad) and max(el, er) < 1e-3 and max(rl, rr) < 5e-3
        print(f"{tag:<34} {'OK ' if ok else 'FAIL'} {max(el, er)*1000:5.2f}mm "
              f"{max(rl, rr):.4f}rad  "
              f"L[k{np.degrees(q[3]):5.1f} ay{np.degrees(q[4]):5.1f} "
              f"anx{np.degrees(q[5]):5.1f}] "
              f"R[k{np.degrees(q[9]):5.1f} ay{np.degrees(q[10]):5.1f} "
              f"anx{np.degrees(q[11]):5.1f}]" + ("  !! " + ", ".join(bad) if bad else ""))
        return ok

    print("=== 真实数值 IK 复核（足掌严格水平 + 全部关节限位）===")
    print(f"{'工况':<34} {'判定':<4} {'位置误差':>8} {'姿态误差':>8}   膝/踝y/踝x（度）")
    print("-" * 116)
    ok = True
    ok &= report("SPLIT  r=.25 d=.20/.20 sep=.20", 0.1820, (0.20, 0.25), (0.00, 0.00), 0.690)
    ok &= report("SPLIT  r=.25 d=.40/.40 sep=.40", 0.1820, (0.40, 0.25), (0.00, 0.00), 0.690)
    ok &= report("SPLIT  r=.25 d=.20/.40 sep=.30", 0.1820, (0.30, 0.25), (0.00, 0.00), 0.690)
    ok &= report("SPLIT  r=.12 d=.20/.20 sep=.20", 0.0000, (0.20, 0.12), (0.00, 0.00), 0.625)
    ok &= report("SPLIT  r=.18 d=.40/.40 sep=.40", 0.0000, (0.40, 0.18), (0.00, 0.00), 0.655)
    ok &= report("GATHER d=.20 sep=.10  H=.62", 0.0, (0.05, 0.25), (-0.05, 0.25), 0.620)
    ok &= report("GATHER d=.20 sep=.10  H=.52", 0.0, (0.05, 0.25), (-0.05, 0.25), 0.520)
    ok &= report("GATHER d=.40 sep=.30  H=.62", 0.0, (0.15, 0.25), (-0.15, 0.25), 0.620)
    ok &= report("GATHER d=.40 sep=.30  H=.45", 0.0, (0.15, 0.25), (-0.15, 0.25), 0.450)
    print()
    print("全部通过" if ok else "存在失败用例")
    return ok


def _grid():
    """纯运动学可行性回归。

    关键拆分：两个相位的骨盆高度是【独立】的，不能混用同一个 H。
      SPLIT  : 一脚在下级、一脚在上级。髋高 H_s 相对【下级】踏面。
               下坡腿 D = H_s - 0.04       需 rho <= RHO_KNEE
               上坡腿 D = H_s - r - 0.04   需 rho >= RHO_MIN
               => 唯一受级高约束的相位
      GATHER : 双脚同踏面，髋高 H_g 相对【该踏面】。两腿 D 相同 = H_g - 0.04。
               => 与级高 r 无关，只要 H_g 选对就恒成立
    由此得到级高上限：
        r <= sqrt(RHO_KNEE^2 - sx^2) - sqrt(RHO_MIN^2 - sx^2)
    sx=0 时 = 0.6382 - 0.411 = 0.227 m；sx 越大上限越高。
    """
    cap = min(RHO_MAX, RHO_KNEE)
    print("=== 螃蟹步可行性网格（SPLIT 相位为约束瓶颈）===")
    print(f"约束: {RHO_MIN:.3f} <= rho <= {cap:.3f}（后者由膝角 >= {KNEE_MIN} rad 推出）")
    print(f"解法: 骨盆侧倾 waist_x（限位 ±{np.degrees(WAIST_X_LIMIT):.0f}deg），"
          f"不做前后错位。侧倾 15deg 可让髋高差达 {2*tilt_delta(WAIST_X_LIMIT)*100:.1f} cm")
    print(f"骨盆水平时的级高硬上限 = {cap - RHO_MIN:.3f} m")
    print()
    print(f"{'级高':>6} | {'所需侧倾':>9} {'髋高窗口':>16} {'窗口宽':>8} {'上坡膝角':>9} {'下坡踝角':>9}")
    print("-" * 70)
    for r in (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28):
        s = tilt_for_rise(r)
        if s is None:
            print(f"{r:>6.2f} | {'--':>9} {'--':>16} {'--':>8} {'--':>9} {'--':>9}  不可行")
            continue
        phi, w, (lo, hi) = s
        H = 0.5 * (lo + hi)
        dl = tilt_delta(phi)
        pu_ = check_pose(0.0, 0.0, H + dl - r - ANKLE_SOLE_DROP)
        pd_ = check_pose(0.0, 0.0, H - dl - ANKLE_SOLE_DROP)
        print(f"{r:>6.2f} | {np.degrees(phi):>8.1f}d {f'{lo:.3f} ~ {hi:.3f}':>16} {w*1000:>7.0f}mm "
              f"{np.degrees(pu_['knee']):>8.1f}d {np.degrees(pd_['ankle_y']):>8.1f}d")

    print()
    print("=== 若改用【满侧倾 15deg】，全量程余量 ===")
    print(f"{'级高':>6} | {'髋高窗口':>16} {'窗口宽':>8}")
    print("-" * 36)
    for r in np.arange(0.12, 0.2801, 0.02):
        rng = feasible_H_range(r, 0.0, WAIST_X_LIMIT)
        if rng is None:
            print(f"{r:>6.2f} | {'--':>16} {'--':>8}  不可行")
        else:
            print(f"{r:>6.2f} | {f'{rng[0]:.3f} ~ {rng[1]:.3f}':>16} "
                  f"{1000*(rng[1]-rng[0]):>7.0f}mm")

    print()
    print("=== GATHER 相位（双脚同踏面）所需髋内收角与侧向余量 ===")
    print("脚宽 8 cm，踏面进深 d；双脚对称放在踏面中心两侧 ±off")
    print(f"{'进深':>6} | {'off':>7} {'脚Y跨度':>8} {'髋内收':>8} {'|ankle_x|':>10} "
          f"{'髋高窗口':>16}")
    print("-" * 66)
    for d_ in (0.20, 0.25, 0.30, 0.35, 0.40):
        off = foot_placement(d_)
        sep = 2.0 * off
        A = off - HIP_LATERAL                  # 脚相对髋的侧移（负=内收）
        # 取髋高 0.62 估算 ankle_x
        sol = leg_solution(0.0, A, 0.62 - ANKLE_SOLE_DROP)
        rng = split_window(0.0, sep, 0.0)
        win = f"{rng[0]:.3f} ~ {rng[1]:.3f}" if rng else "--"
        print(f"{d_:>6.2f} | {off:>7.3f} {sep:>8.3f} "
              f"{np.degrees(sol['hip_x']):>7.1f}d {np.degrees(sol['ankle_x']):>9.1f}d "
              f"{win:>16}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="螃蟹步 IK 自检 / 可行性网格")
    ap.add_argument("--scene", default=str(
        Path(__file__).resolve().parent.parent
        / "data" / "mujoco_simulation" / "elf3_crab.xml"))
    ap.add_argument("--grid", action="store_true", help="跑可行性网格（不需要加载模型）")
    ap.add_argument("--tilt", type=float, default=None, metavar="RISE",
                    help="查询某个级高该用多大骨盆侧倾")
    ap.add_argument("--verify", action="store_true",
                    help="用真实数值 IK 复核最坏工况（需要加载模型）")
    args = ap.parse_args()

    if args.verify:
        _verify(args.scene)
    elif args.tilt is not None:
        s = tilt_for_rise(args.tilt)
        if s is None:
            print(f"级高 {args.tilt:.3f} m 不可行（超出侧倾限位能覆盖的范围）")
        else:
            phi, w, (lo, hi) = s
            print(f"级高 {args.tilt:.3f} m -> 骨盆侧倾 waist_x = {phi:.4f} rad "
                  f"({np.degrees(phi):.2f} deg)")
            print(f"  髋高窗口 {lo:.4f} ~ {hi:.4f} m（宽 {w*1000:.0f} mm），"
                  f"建议取中值 H={0.5*(lo+hi):.4f}")
    elif args.grid:
        _grid()
    else:
        _selftest(args.scene)
