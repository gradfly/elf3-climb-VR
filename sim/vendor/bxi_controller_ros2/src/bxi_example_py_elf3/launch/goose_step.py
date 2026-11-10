"""ELF3 走正步（goose step / 阅兵步）开环动作序列生成器。

设计要点
--------
1. **两条腿始终绷直**：这是与「普通走路」最本质的区别。腿全长 2L=0.64 m，
   这里让髋→踝距离恒等于 ``leg_r=0.639``（膝屈角恒为 6.4°，肉眼就是一条直线）。
   但「腿直 + 脚贴地 + 大步幅 + 基座高度固定」四者不可兼得：脚贴地且腿直时，
   髋高必须 = 脚底 + √(R² − 水平偏移²)，所以**基座会随支撑脚前后位置起伏**
   （约 4.5 cm，正步本来就有这个铿锵的上下节奏）。原地版支撑脚不动，基座不起伏。
2. **摆动腿走圆轨迹**：脚的目标高度不再用「先抬高再前伸」的独立包络，而是
   直接取同一个圆的下半支 ``tz = hip_z − √(R²−dx²)``，于是整条摆动腿也全程
   绷直，脚贴着地面划出去、再抬到身前 23 cm 高。只有落地/起脚处因为前倾的
   质心补偿让两脚不对称，用一个足够软的 ``softplus`` 地板把脚托住（局部
   膝屈 35°，仅 2~3 帧）。早期显式抬腿包络会让摆动腿中段屈膝 65°。
3. **单脚支撑（duty=0.5）**：左右腿严格差半周期，任一时刻恰好一条支撑腿，
   髋高才有唯一解。支撑腿切换瞬间两腿的水平偏移不同，髋高会跳 ~6 cm
   （支撑脚后移越多跳得越多）；先算出整段髋高序列再做高斯平滑（σ=1.8 帧），
   把跳变摊开，单帧关节跳变从 0.35 rad 降到 0.18 rad（代价是切换处膝屈 ~28°，
   仅 2~3 帧）。
4. **支撑脚不滑、且大腿微微后倾**：支撑腿踝的世界 x 由构造直接锁死，再用解析
   两连杆 IK 反解髋/膝。前进时基座匀速前移，踝相对髋的 x 从 +step 线性退到
   −step，二者正好抵消。**原地版不能只把 v 设 0**，那样脚会相对髋滑 2×step=39 cm；
   原地版把支撑脚钉在「髋后 dx_sup」处。
   支撑脚整体放在**质心后方** ``stance_back = shift − dx_sup``（默认 13 cm），
   于是支撑腿大腿相对竖直后倾 ``stance_back_deg``（默认 5°，膝在髋后方 5 cm），
   这正是行进中身体在前的观感。代价：重心投影跑到脚掌前缘（踝前 13 cm，
   脚掌 −9~+15 cm），所以这是「动态平衡」的姿势，不是静立姿势。
5. **脚掌**：位置 IK 只解 hip_y / knee_y（2 自由度，x-z 平面），踝用解析式
   ``ankle = −(pitch+hip+knee)`` 补成脚掌水平；摆动相再叠负 toe 绷脚面，
   且 toe 幅度随离地高度淡入，擦地那几帧不下压、不会戳进地面。
6. **大臂前摆到最高点时小臂水平横在胸前**：前摆峰值用解析反解一次四关节：
   大臂前摆 55°、外展 6°，再靠肩内旋（``shoulder_z`` ≈ −78°）把折了 90° 的
   小臂转到**横向**，于是「肘→腕」在世界系里 z 分量恒为 0（水平）、方向指向
   身体中线（横向 73°，腕落在胸前中线旁）。后摆是自然下垂（肘角 0.85）。
   中间按前摆量在关节空间插值，不做逐帧 IK —— 4 自由度腕肘双约束有等价分支，
   逐帧解会让肩角翻 180°（敬礼那一轮踩过）。
   **交会相必须抹掉一部分肩内旋**（``arm_swing_open``，默认 0.5）：手是 2026-11-10
   才加到腕上的纯视觉 mesh，从腕再往外伸 0.119 m。内旋按前摆量线性插值时，
   两臂在 front≈0.5 那一刻**同时**处于「小臂横指对侧、腕已跨过中线」的半程姿态，
   两只拳正面相遇，实测只差 16 mm —— 浏览器里就是撞在一起。
   现在交会相只保留一半内旋，两条小臂都指向前方、拳各在己侧，最小间距 234 mm；
   **前后摆峰值姿态一字未动**（峰值处 mid=0）。``arm_swing_abd`` / ``arm_swing_fold``
   是另外两种备选修法（外展 / 折肘），默认关；改任一项都要重跑
   ``scripts/check_hand_clearance.py``。
7. 上体**略微前倾**（默认 5°）、腰部锁零、头随躯干；用 FK 实测「质心 x − 髋 x」
   得到 ``shift``，再把支撑脚放到 ``shift − dx_sup`` 处，让腿的后倾角可控。

产物（与 squat_extreme.py / prone_crawl.py 同格式，可被 squat_replay.py 直接回放）
--------------------------------------------------------------------------------
    data/mujoco_simulation/goose_step_traj.npz          前进 4 个周期 + 立正起收势
    data/mujoco_simulation/goose_step_inplace_traj.npz  原地正步，首尾同相位可无缝循环

    python goose_step.py                       # 两个都生成（默认前倾 5° + 质心补偿）
    python goose_step.py --cycles 6 --cycle 1.1
    python goose_step.py --lean 8                # 前倾改 8°，补偿量自动重算
    python goose_step.py --no-balance            # 关掉质心补偿，用于对比
    python goose_step.py --check-only           # 只算不落盘
    python squat_replay.py --traj goose_step_traj.npz --mode kin --loop
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import mujoco

from action_lib import (DATA, DT, FOOT_SOLE, IDX, L_THIGH, SCENE, STAND_Z,
                        TORSO_TO_HIP, foot_level, leg_ik_2d, limit_report,
                        make_qpos_row, save_traj, smootherstep, stand_q)

# 左/右臂在 shoulder_y 上的符号（前摆时肩角为负）
ARM_SGN = {"l": 1.0, "r": -1.0}
# shoulder_x 外展左右反号
SHOULDER_X_SGN = {"l": 1.0, "r": -1.0}


@dataclass
class Params:
    cycle: float = 1.20          # 一个完整左右周期 (s)，即 0.6 s/步 = 100 步/分
    cycles: int = 4              # 行进周期数
    duty: float = 0.50           # 支撑相占周期比例（0.5 = 严格单脚支撑）
    lean_deg: float = 5.0        # 上体前倾角 (deg)，正 = 头向 +x 前倾
    pitch: float = field(init=False, default=0.0)   # = radians(lean_deg)，勿手填
    leg_r: float = 0.639         # 髋 -> 踝 恒定距离（2L=0.64，膝屈恒 6.4°）
    step: float = 0.197         # 半步幅：支撑脚相对质心的前后极限 (m)
    kick_lift: float = 0.230     # 踢腿峰值脚底离地 (m)
    stance_back_deg: float = 5.0  # 支撑腿大腿后倾角（相对竖直，正 = 向后）
    arm_back_deg: float = 30.0   # 后摆峰值大臂后摆角（相对竖直）
    arm_fwd_deg: float = 55.0    # 前摆峰值大臂前摆角（相对竖直，越大肘越高）
    arm_abd_deg: float = 6.0     # 前摆峰值大臂外展角
    arm_medial: float = 0.96     # 前臂方向里「横向」分量占比（1 = 纯横向）
    arm_out: float = 0.24        # 后摆时的肩外展 (rad)，贴身但不蹭大腿
    elbow_back: float = 0.85     # 后摆肘关节角（0 = 折 90°，0.85 ≈ 折 41°）
    arm_swing_abd: float = 0.0   # 交会相额外肩外展 (rad)：让两臂从体侧绕开而不是对穿
    arm_swing_fold: float = 0.0  # 交会相额外折肘 (rad)：缩短前臂伸出的水平投影
    arm_swing_open: float = 0.50 # 交会相抹掉的肩内旋比例 0~1：小臂改为指向前方
    arm_front: tuple = field(init=False, default=())   # 左臂前摆峰值 (sy,sx,sz,e)
    toe: float = -0.18           # 摆动腿脚尖下压峰值 (rad)
    n_hold: int = 25             # 立正保持帧数 (0.5 s)
    n_blend: int = 25            # 立正 <-> 步态 过渡帧数 (0.5 s)
    blend_lift: float = 0.009    # 过渡段抬基座，抵消关节插值造成的脚掌倾斜穿地
    hip_smooth: float = 1.8      # 髋高序列高斯平滑 σ（帧），消除支撑腿切换跳变
    floor_soft: float = 0.010    # 摆动脚地板的 softplus 软化量 (m)
    balance: bool = True         # False = 关闭质心补偿（用于对比/排查）
    dt: float = DT

    def __post_init__(self) -> None:
        self.pitch = math.radians(self.lean_deg)
        self.arm_front = solve_arm_front(self)


def solve_arm_front(p: Params) -> tuple[float, float, float, float]:
    """解析反解「大臂前摆到峰值、前臂水平横在胸前」的左臂四关节 (sy,sx,sz,e)。

    链路 torso -> shoulder_y(+y) -> shoulder_x(+x) -> shoulder_z(+z) -> elbow_y(+y)，
    大臂沿 shoulder_z 的 −z、小臂沿 elbow 的 +x（所以 e=0 就是折 90°，
    e=90° 才伸直）。两段长度都是 0.256 m。于是：

    * 大臂方向 u 在基座系里已知（前摆 ``arm_fwd_deg``、外展 ``arm_abd_deg``），
      由 ``u = (sinθcosα, sinα, −cosθcosα)`` 直接得
      ``sy = atan2(−ux, −uz)``、``sx = asin(uy)``；
    * 小臂方向 v（世界系水平、指向身体中线）转到基座系后，令
      ``A = Rx(sx)ᵀ Ry(sy)ᵀ v``，则 ``A = (cos e cos sz, cos e sin sz, −sin e)``，
      得 ``e = asin(−A_z)``、``sz = atan2(A_y, A_x)``。

    只解一次端点姿态（右臂镜像 = (sy, −sx, −sz, e)），整段动画在关节空间
    插值，不做逐帧 IK —— 4 自由度腕肘双约束有等价分支，逐帧解会让肩角翻
    180°（敬礼那一轮踩过）。
    """
    i = IDX["l_shoulder_y"]
    th = math.radians(p.arm_fwd_deg)
    ab = math.radians(p.arm_abd_deg)
    u = np.array([math.sin(th) * math.cos(ab), math.sin(ab),
                  -math.cos(th) * math.cos(ab)])
    sy = math.atan2(-u[0], -u[2])
    sx = math.asin(max(-1.0, min(1.0, u[1])))
    # 世界系前臂方向：水平、横向（+x 为前，−y 为左臂向中线）
    m = min(1.0, max(0.0, p.arm_medial))
    v_world = np.array([math.sqrt(1.0 - m * m), -m, 0.0])
    cy, sy_ = math.cos(p.pitch), math.sin(p.pitch)
    # 转基座系（基座整体前倾 pitch）
    v = np.array([cy * v_world[0] - sy_ * v_world[2],
                  v_world[1],
                  sy_ * v_world[0] + cy * v_world[2]])
    Ry = np.array([[math.cos(sy), 0, math.sin(sy)],
                   [0, 1, 0], [-math.sin(sy), 0, math.cos(sy)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(sx), -math.sin(sx)],
                   [0, math.sin(sx), math.cos(sx)]])
    A = Rx.T @ Ry.T @ v
    e = math.asin(max(-1.0, min(1.0, -A[2])))
    sz = math.atan2(A[1], A[0])
    return sy, sx, sz, e


def lerp_qpos(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    """两帧 38 维 qpos 线性插值，四元数归一化（俯仰很小，线性插值足够）。"""
    r = a * (1.0 - s) + b * s
    r[3:7] /= np.linalg.norm(r[3:7])
    return r


def com_hip_offset(m, d, rows: np.ndarray) -> float:
    """FK 实测「整机质心 x − 髋 x」，步态段抽样取均值。

    质量用 ``body_mass`` 加权 ``xpos``（含双臂、双腿、头），参考点取左髋
    body 的世界 x。抽样步长 7 与周期 60 帧互质，摆臂造成的前后波动会
    在整数个周期上互相抵消，剩下的是前倾带来的**系统性**前移量。
    """
    mass = m.body_mass
    hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_hip_y_link")
    vals = []
    for row in rows[::7]:
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        cx = float((mass[:, None] * d.xpos).sum(axis=0)[0] / mass.sum())
        vals.append(cx - float(d.xpos[hid, 0]))
    return float(np.mean(vals))


def smooth_series(a: list[float], sigma: float) -> np.ndarray:
    """高斯平滑，边缘用 edge 填充（前进版首尾是立正过渡，原地版本身恒定）。"""
    v = np.asarray(a, dtype=np.float64)
    k = max(1, int(math.ceil(3.0 * sigma)))
    xs = np.arange(-k, k + 1)
    w = np.exp(-0.5 * (xs / sigma) ** 2)
    w /= w.sum()
    return np.convolve(np.pad(v, (k, k), mode="edge"), w, mode="valid")


def soft_floor(v: float, floor: float, delta: float) -> float:
    """softplus 版 max(v, floor)：x→floor+δ/2，远离地板时 ≈ v，无硬折角。"""
    x = v - floor
    return floor + 0.5 * (x + math.sqrt(x * x + delta * delta))


def build_gait(p: Params, in_place: bool) -> tuple[np.ndarray, dict]:
    """生成纯步态段：整数个周期 + 末帧（末帧与首帧同相位）。

    两阶段：先按「支撑腿锁脚」算整段髋高并平滑，再逐帧解两条腿的 IK
    （髋高已是已知量，摆动腿脚高由圆轨迹给出）。
    """
    hip_dx = -TORSO_TO_HIP * math.sin(p.pitch)   # 前倾让髋相对基座后移
    zoff = TORSO_TO_HIP * math.cos(p.pitch)      # base_z = hip_z + zoff
    n = int(round(p.cycles * p.cycle / p.dt)) + 1
    kick_phase = p.duty + (1.0 - p.duty) * 0.5  # 左腿踢到峰值的相位

    def build_rows(shift: float) -> tuple[np.ndarray, dict]:
        R = p.leg_r
        half = math.acos(min(1.0, R / (2.0 * L_THIGH)))     # 直腿时的半膝角
        # 支撑腿大腿后倾 stance_back_deg：thigh = half − mid，mid = atan2(−dx, dz)
        mid = half + math.radians(p.stance_back_deg)
        dx_sup = 0.0
        for _ in range(12):                                  # dz 与 dx 互相耦合，迭代
            dx_sup = -math.sqrt(max(0.0, R * R - dx_sup * dx_sup)) * math.tan(mid)
        # 支撑脚在世界系里落在质心后方 stance_back 处 -> 髋在大腿前、腿向后
        stance_back = shift - dx_sup
        # 踢腿峰值出现在支撑相中点（s=0.5，脚正好在 dx_sup），此时髋最高
        hip_z_peak = FOOT_SOLE + math.sqrt(max(0.0, R * R - dx_sup * dx_sup))
        reach = math.sqrt(max(0.0, R * R -
                              (hip_z_peak - FOOT_SOLE - p.kick_lift) ** 2))
        # 支撑/摆动共用基准 base_x = 髋 + dx_sup；摆动峰值从 base_x 前伸 kick_off
        kick_off = reach - dx_sup
        if abs(kick_off) > 0.95 * R or abs(dx_sup) + p.step >= 0.95 * R:
            raise SystemExit(
                f"[错误] 后倾 {p.stance_back_deg}° / 前倾 {p.lean_deg}° 的补偿 "
                f"{shift*100:.1f} cm 与步幅 {p.step:.3f} 超出腿长 R={R:.3f} 的可达范围；"
                f"请减小 --lean / --step / --stance-back")
        v = 0.0 if in_place else 2.0 * p.step / (p.duty * p.cycle)

        # ---- 阶段 1：脚目标 + 髋高 ----
        frames: list[dict] = []
        for k in range(n):
            t = k * p.dt
            x = v * t
            hip_x = x + hip_dx
            base_x = hip_x + dx_sup
            feet: dict[str, tuple] = {}
            for side, off in (("l", 0.0), ("r", 0.5)):
                phase = (t / p.cycle + off) % 1.0
                if phase < p.duty:                    # ---- 支撑相：踝锁在世界系
                    s = phase / p.duty
                    if in_place:
                        # 原地版不是「把 v 设 0」就完事：支撑相 ±step 的平移
                        # 在 v=0 时会让脚滑 2*step（实测 39 cm）。原地版把脚
                        # 钉在 hip+dx_sup（质心后方约 14 cm），大腿微微后倾。
                        tx = base_x
                    else:
                        tx = base_x + p.step * (1.0 - 2.0 * s)
                    feet[side] = (tx, True, 0.0, 0.0)
                else:                                 # ---- 摆动相：圆弧前踢
                    u = (phase - p.duty) / (1.0 - p.duty)
                    env = math.sin(math.pi * u) ** 2
                    if in_place:
                        tx = base_x + kick_off * env
                    else:
                        # 端点 ±step（与支撑相首尾严格连续），峰值 kick_off；
                        # (1-env) 压掉前后平移项，避免与前踢项叠加后超出腿长。
                        tx = base_x + (p.step * (2.0 * smootherstep(u) - 1.0)
                                       * (1.0 - env) + kick_off * env)
                    feet[side] = (tx, False, u, env)
            sup = next(s for s in ("l", "r") if feet[s][1])
            dx_s = feet[sup][0] - hip_x
            hip_z = FOOT_SOLE + math.sqrt(max(0.0, R * R - dx_s * dx_s))
            frames.append(dict(x=x, hip_x=hip_x, feet=feet, sup=sup, hip_z=hip_z))

        hip_zs = smooth_series([f["hip_z"] for f in frames], p.hip_smooth)

        # ---- 阶段 2：逐帧解 IK ----
        rows: list[np.ndarray] = []
        q = stand_q()
        knee_sup: list[float] = []
        knee_swing: list[float] = []
        max_unreach = 0.0
        for k, f in enumerate(frames):
            hip_z = float(hip_zs[k])
            base_z = hip_z + zoff
            t = k * p.dt
            swing = math.cos(2.0 * math.pi * (t / p.cycle - kick_phase))
            for side in ("l", "r"):
                tx, support, u, env = f["feet"][side]
                toe = 0.0
                if support:
                    tz = FOOT_SOLE
                else:
                    dxw = tx - f["hip_x"]
                    # 同一半径 R 的圆下半支 -> 腿恒等长、恒绷直；
                    # softplus 地板只在起脚/落脚的 2~3 帧把脚托住。
                    zc = hip_z - math.sqrt(max(0.0, R * R - dxw * dxw))
                    tz = soft_floor(zc, FOOT_SOLE, p.floor_soft)
                    # 擦地段不下压脚尖，离地 >10 cm 后才给足峰值 toe
                    toe = (p.toe * min(1.0, max(0.0, (tz - FOOT_SOLE) / 0.10))
                           * env)
                # 世界位移 -> 腿局部系（随基座俯仰转 -pitch），与 squat_stand 同式
                dxw, dzw = tx - f["hip_x"], hip_z - tz
                dxl = dxw * math.cos(p.pitch) + dzw * math.sin(p.pitch)
                dzl = -dxw * math.sin(p.pitch) + dzw * math.cos(p.pitch)
                hip, knee = leg_ik_2d(dxl, dzl)
                # 只统计支撑腿：它与「真实腿全长 2L」比较才是有意义的
                # 落地可达性。摆动腿的脚本来就在空中，偶尔被 softplus 地板
                # 托高后距离略超 2L 只会让膝更直（knee=0），不影响接触。
                if side == f["sup"]:
                    max_unreach = max(max_unreach,
                                     math.hypot(dxl, dzl) - 2.0 * L_THIGH)
                (knee_sup if side == f["sup"] else knee_swing).append(
                    math.degrees(knee))
                i = IDX[f"{side}_hip_y"]
                q[i], q[i + 3] = hip, knee
                foot_level(q, side, p.pitch, toe)
                # 踝限位 [-50°, 45°]：踢腿峰值时脚掌水平补偿会顶到上界，
                # 轻微 clip 比让关节越界安全（脚掌歪几度肉眼看不出）。
                q[i + 4] = float(np.clip(q[i + 4], -0.862, 0.775))

            # 摆臂：与腿反相（左腿踢出峰值 -> 右臂前摆）。
            # 前摆峰值用解析解出的四关节：(大臂前摆 50° + 内旋) 让「肘→腕」在
            # 世界系里水平、指向身体中线横在胸前；后摆是自然下垂。
            sy_f, sx_f, sz_f, e_f = p.arm_front
            sy_b = math.radians(p.arm_back_deg)
            for side in ("l", "r"):
                i = IDX[f"{side}_shoulder_y"]
                sgn = SHOULDER_X_SGN[side]            # 左 +1 / 右 −1
                front = (1.0 - ARM_SGN[side] * swing) / 2.0   # 0=后摆, 1=前摆峰值
                # 交会相（front≈0.5，两臂都在体前中段）的额外外展 / 折肘权重。
                # 前后摆两个端点权重为 0，所以峰值姿态一字不动，只改中间路径：
                # 手加上之后（腕外再伸 0.119 m 的拳），原来的路径会让两只拳头在
                # 身体中线上对穿相遇（实测最小间距 16 mm），这里让手臂从体侧绕开。
                mid = 4.0 * front * (1.0 - front)
                q[i] = sy_b + (sy_f - sy_b) * front
                q[i + 1] = sgn * (p.arm_out + (sx_f - p.arm_out) * front
                                  + p.arm_swing_abd * mid)
                q[i + 2] = sgn * sz_f * front * (1.0 - p.arm_swing_open * mid)
                q[i + 3] = (p.elbow_back + (e_f - p.elbow_back) * front
                            + p.arm_swing_fold * mid)
                q[i + 4:i + 7] = 0.0
            rows.append(make_qpos_row(q, f["x"], base_z, p.pitch))

        hip_zs_arr = np.asarray(rows)[:, 2]
        return np.stack(rows), dict(
            step=round(p.step, 5), speed=round(v, 5),
            travel=round(v * (n - 1) * p.dt, 4),
            kick_reach=round(kick_off, 4),
            stance_back_m=round(stance_back, 4),
            stance_dx_m=round(dx_sup, 4),
            max_unreachable_mm=round(max(0.0, max_unreach) * 1000, 3),
            knee_support_deg=round(float(np.mean(knee_sup)), 2),
            knee_support_max_deg=round(float(np.max(knee_sup)), 2),
            knee_swing_deg=round(float(np.mean(knee_swing)), 2),
            knee_swing_max_deg=round(float(np.max(knee_swing)), 2),
            hip_rise_cm=round(float(hip_zs_arr.max() - hip_zs_arr.min()) * 100, 2),
            frames=n)

    # 质心补偿量与腿姿态互相耦合（脚前移会带动小腿质量前移），迭代收敛。
    shift = 0.0
    m = d = None
    if p.balance:
        m = mujoco.MjModel.from_xml_path(str(SCENE))
        d = mujoco.MjData(m)
        for _ in range(6):
            probe, _ = build_rows(shift)
            new = com_hip_offset(m, d, probe)
            converged = abs(new - shift) < 2e-4
            shift = new
            if converged:
                break
    rows, info = build_rows(shift)
    residual = com_hip_offset(m, d, rows) if m is not None else None
    info.update(lean_deg=p.lean_deg, pitch_rad=round(p.pitch, 6),
                balance_shift=round(shift, 6),
                com_hip_residual=round(residual - shift, 6)
                if residual is not None else 0.0,
                arm_fwd_deg=p.arm_fwd_deg, arm_back_deg=p.arm_back_deg,
                arm_front_deg=[round(math.degrees(v), 2) for v in p.arm_front],
                elbow_front_deg=round(math.degrees(p.arm_front[3]), 2),
                kick_peak_foot_z=round(p.kick_lift, 4))
    return rows, info


def with_attention(p: Params, gait: np.ndarray) -> np.ndarray:
    """前进版：立正保持 -> 起步过渡 -> 步态 -> 收步过渡 -> 立正保持。

    过渡段是关节角线性插值，脚掌会短暂倾斜；按两端为 0 的正弦包络把基座
    抬高 ``blend_lift`` 抵消，步态段本身不动。
    """
    q0 = stand_q()
    stand0 = make_qpos_row(q0, 0.0, STAND_Z)
    stand1 = make_qpos_row(q0, gait[-1, 0], STAND_Z)

    def blend(a: np.ndarray, b: np.ndarray, rev: bool = False) -> np.ndarray:
        out = []
        for i in range(p.n_blend):
            s = smootherstep((i + 1 if rev else i) / p.n_blend)
            r = lerp_qpos(a, b, s)
            r[2] += p.blend_lift * math.sin(math.pi * smootherstep(i / p.n_blend))
            out.append(r)
        return np.stack(out)

    parts = [np.repeat(stand0[None, :], p.n_hold, axis=0),
             blend(stand0, gait[0]),
             gait,
             blend(gait[-1], stand1, rev=True),
             np.repeat(stand1[None, :], p.n_hold, axis=0)]
    return np.concatenate(parts, axis=0)


def report(name: str, qpos: np.ndarray, info: dict, p: Params) -> None:
    print(f"\n=== {name} ===")
    print(f"  帧数 {len(qpos)}  时长 {len(qpos)*p.dt:.2f} s  "
          f"基座 x: {qpos[0,0]:.3f} -> {qpos[-1,0]:.3f} m")
    print(f"  半步幅 {info['step']:.3f} m  速度 {info['speed']:.3f} m/s  "
          f"踢腿峰值前伸 {info['kick_reach']:.3f} m")
    print(f"  上体前倾 {p.lean_deg:+.1f}°  质心补偿 {info['balance_shift']*100:+.2f} cm  "
          f"收敛残差 {info['com_hip_residual']*1000:+.2f} mm")
    print(f"  支撑腿大腿后倾 {p.stance_back_deg:.1f}°（支撑脚在质心后 "
          f"{info['stance_back_m']*100:.1f} cm）")
    print(f"  双腿膝屈：支撑 {info['knee_support_deg']:.1f}° (max {info['knee_support_max_deg']:.1f}°)  "
          f"摆动 {info['knee_swing_deg']:.1f}° (max {info['knee_swing_max_deg']:.1f}°)")
    print(f"  基座起伏 {info['hip_rise_cm']:.2f} cm")
    print(f"  摆臂：大臂后摆 {p.arm_back_deg:.0f}° / 前摆 {p.arm_fwd_deg:.0f}°，"
          f"前摆峰值关节 {info['arm_front_deg']}（肘角 {info['elbow_front_deg']:.1f}°），"
          f"小臂水平横在胸前；交会相抹掉 {p.arm_swing_open*100:.0f}% 肩内旋避让双手")
    print(f"  踢腿峰值脚底离地 {info['kick_peak_foot_z']*100:.1f} cm  "
          f"最大腿长不可达 {info['max_unreachable_mm']:.3f} mm")
    jump = float(np.abs(np.diff(qpos[:, 7:36], axis=0)).max())
    print(f"  单帧最大关节跳变 {jump:.4f} rad ({math.degrees(jump):.2f}°)")
    viol = limit_report(qpos)
    print(f"  关节限位：{'全部通过' if not viol else '越界 -> ' + '; '.join(viol)}")
    if jump >= 0.30:
        print("  [警告] 单帧跳变超过 0.30 rad，回放会顿一下")


def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 走正步动作生成器")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--cycle", type=float, default=1.20)
    ap.add_argument("--lean", type=float, default=5.0,
                    help="上体前倾角 deg（正=前倾，0=竖直，负=后仰）")
    ap.add_argument("--no-balance", action="store_true",
                    help="关闭质心补偿，用于对比/排查")
    ap.add_argument("--leg-r", type=float, default=0.639,
                    help="髋->踝恒定距离（2L=0.64，越接近越直）")
    ap.add_argument("--step", type=float, default=0.197, help="半步幅 m")
    ap.add_argument("--kick", type=float, default=0.23, help="踢腿峰值脚底离地 m")
    ap.add_argument("--stance-back", type=float, default=5.0,
                    help="支撑腿大腿后倾角 deg（正=膝盖在髋后方）")
    ap.add_argument("--arm-fwd", type=float, default=55.0,
                    help="前摆峰值大臂前摆角 deg")
    ap.add_argument("--arm-back", type=float, default=30.0,
                    help="后摆峰值大臂后摆角 deg")
    ap.add_argument("--arm-medial", type=float, default=0.96,
                    help="前臂横向占比 0~1（1=完全横在胸前，0=指向前方）")
    ap.add_argument("--arm-out", type=float, default=0.24,
                    help="后摆肩外展 rad（越大手离大腿越远）")
    ap.add_argument("--arm-swing-abd", type=float, default=0.0,
                    help="交会相额外肩外展 rad（峰值不受影响，只撑开中间路径）")
    ap.add_argument("--arm-swing-fold", type=float, default=0.0,
                    help="交会相额外折肘 rad（同上）")
    ap.add_argument("--arm-swing-open", type=float, default=0.50,
                    help="交会相抹掉的肩内旋比例 0~1（0.5 = 小臂改指前方，防两拳相撞）")
    ap.add_argument("--hip-smooth", type=float, default=1.8,
                    help="髋高高斯平滑 sigma（帧）")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()

    p = Params(cycles=args.cycles, cycle=args.cycle, leg_r=args.leg_r,
               step=args.step, kick_lift=args.kick,
               stance_back_deg=args.stance_back,
               arm_fwd_deg=args.arm_fwd, arm_back_deg=args.arm_back,
               arm_medial=args.arm_medial, arm_out=args.arm_out,
               arm_swing_abd=args.arm_swing_abd,
               arm_swing_fold=args.arm_swing_fold,
               arm_swing_open=args.arm_swing_open,
               hip_smooth=args.hip_smooth, lean_deg=args.lean,
               balance=not args.no_balance)

    gait, info = build_gait(p, in_place=False)
    walk = with_attention(p, gait)
    report("走正步（前进）", walk, info, p)

    # 原地版用独立的「钉脚 + 前踢收回」轨迹（不是 v=0 的滑步版）
    inplace, info_ip = build_gait(p, in_place=True)
    inplace = inplace[:-1]                 # 去掉与首帧重复的末帧 -> 无缝循环
    report("走正步（原地循环）", inplace, info_ip, p)

    if args.check_only:
        return
    out = Path(args.out_dir)
    base = {**asdict(p), **info}
    save_traj(out / "goose_step_traj.npz", walk, "goose_step",
              {**base, "label": "脚本动作 · 走正步（前进）",
               "loop": False, "in_place": False,
               "gait_start": p.n_hold + p.n_blend,
               "gait_frames": info["frames"]})
    save_traj(out / "goose_step_inplace_traj.npz", inplace, "goose_step_inplace",
              {**asdict(p), **info_ip, "label": "脚本动作 · 原地正步（循环）",
               "loop": True, "in_place": True,
               "gait_start": 0, "gait_frames": len(inplace)})
    print("\n回放：python squat_replay.py --traj goose_step_traj.npz --mode kin --loop")
    print("烘焙：python web/tools/bake_actions.py")


if __name__ == "__main__":
    main()
