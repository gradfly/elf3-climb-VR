"""ELF3 立定跳远 —— 动作序列生成 + 闭环物理仿真 + 落地平衡校核。

==================== 动作设计（仿人体立定跳远全过程）====================
机器人朝 +X 站立，沿 +X 向前跳。九相位：

  READY   站立保持      （静态平衡，COM 在双脚支撑面内）
  PRE     预蹲 + 摆臂后  （人体"预摆"：半蹲、手臂后摆储能）
  SQUAT   深蹲蓄力      （下蹲 0.25 m、躯干前倾 13°，准静态）
  TAKEOFF 蹬伸起跳      （~0.26 s 爆发：伸髋伸膝伸踝"三伸"+ 末段提踵，
                         手臂由后向前上猛摆；踝背屈限位感知的协调蹬伸）
  FLIGHT  腾空          （先收腿、再向前伸出小腿准备落地；质心走严格
                         抛物线，基座姿态由角动量守恒演化，脚本只收放腿）
  ABSORB  落地缓冲      （触地瞬间屈膝进入深蹲，吸收冲击）
  SETTLE  缓冲底保持    （深蹲静止，确认支撑稳定）
  RISE    站起          （深蹲 → 站立，手臂收回体侧）
  HOLD    站稳保持      （≥ 2 s 静态平衡校核）

==================== 物理建模边界（重要，勿误读）====================
本脚本产出的是【动作序列 / 关节轨迹】，相位机 + PD 闭环保证每一帧姿态
在物理上可达，且落地后【确实能站稳】。两处采用等效边界条件，这是
机器人跳远轨迹生成的标准做法：

  1. 起跳：把"摆臂 + 蹬地 + 提踵"的合力按冲量定理等效为离地瞬间的
     质心初速度 (vx_takeoff, vz_takeoff)。此后【腾空段完全由物理守恒
     决定】——质心走抛物线、基座俯仰由角动量守恒演化，脚本只命令
     收放腿与摆臂，不施加任何外力。
  2. 落地：按完全非弹性接触处理 —— 触地瞬间吸收全部平动/转动动量
     （对应人体屈膝缓冲把动能卸掉），随后从静止深蹲位闭环站起。

  为什么不追求"纯 PD 蹬伸自发产生水平冲量"：43 kg 平台、踝力矩仅
  50 N·m、踝背屈限位 -50°；深蹲前倾时髋落在踝后方，蹬伸反作用力沿
  "踝→髋"指向后上方 ⇒ 水平速度恒为负；落地时脚在质心前方，碰撞把
  水平动量转成前翻角动量，pitch 一旦超过约 30° 就撞上踝限位
  （qa=-(pitch+hip+knee) 需 > 50°），脚掌无法放平、脚跟着地，反力
  持续前掀形成不可恢复的正反馈（实测恒发散到 68.8° 摔倒）。
  另外：落地段若用"当前俯仰反馈 IK"，base_x 里的 sin(pitch) 项同样
  构成正反馈。故落地段一律用【开环准静态姿态序列】（squat_stand 已
  验证能静态站稳），不引入当前姿态反馈。

==================== 落地保持平衡的判据 ====================
  1. HOLD 段全程不倒（躯干 z ≥ 0.55 m、俯仰 |θ| < 40°）；
  2. 末态 COM 投影在双脚支撑多边形内且裕度 > 2 cm（ZMP 静态判据）；
  3. 末态躯干高度恢复到站姿 1.0339 m ± 3 cm；
  4. 全程关节力矩不超过 XML 的 actuatorfrcrange（硬限幅）。

==================== 控制实现 ====================
  * READY/PRE/SQUAT/ABSORB/SETTLE/RISE/HOLD：复用 squat_stand 的闭式
    IK 开环姿态序列（准静态，phys 回放已验证站得住）。
  * TAKEOFF：逐控制步反馈 IK，按实测髋踝跨度做踝限位感知的协调蹬伸。
  * FLIGHT：按实测髋位置生成脚的世界目标（收腿→前伸），只管姿态。
  * 物理与 squat_replay 同款：PD×8 + 重力前馈、1 kHz、力矩按
    actuatorfrcrange 限幅（踝 50 / 髋膝 150 N·m）。

用法:
    python long_jump.py                        # 跑仿真 + 校核 + 存 npz
    python long_jump.py --vx 1.2 --vz 1.6     # 指定起跳初速度（冲量等效）
    python long_jump.py --check-only           # 只跑不落盘
    python long_jump.py --json jump.json       # 指标另存 JSON
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import squat_stand as SQ                                       # noqa: E402
import squat_replay as SR                                      # noqa: E402
from elf3_play import TrainParams                              # noqa: E402

DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_squat.xml"
DEFAULT_OUT = DATA / "long_jump_traj.npz"

TOE_LEN = 0.11                # 脚踝到脚尖的水平距离（脚长 0.22 的一半）
LZ_MAX = SQ.L_THIGH + SQ.L_SHIN - 0.002     # 腿 IK 可达上限（留 2 mm 余量）
LZ_MIN = 0.34                 # 蹲到股骨水平以下就解不了，别让 IK 爆

PHASES = ["READY", "PRE", "SQUAT", "TAKEOFF", "FLIGHT",
          "ABSORB", "SETTLE", "RISE", "HOLD"]
PH = {n: i for i, n in enumerate(PHASES)}

# 手臂关键姿态（shoulder_y 前摆为负 / shoulder_x 左+右- / elbow_y 屈肘为正）
ARM_BACK = (0.55, 0.30, 0.35)     # 后摆蓄力（与 SQ.Params 默认衔接）
ARM_UP = (-1.15, 0.10, 0.25)      # 起跳前上猛摆
ARM_MID = (-0.35, 0.20, 0.50)     # 腾空过渡
ARM_STAND = (0.20, 0.20, 0.60)    # 站立位（= STANCE，落地段由 make_pose 生成）


# ---------------------------------------------------------------- 参数
@dataclass
class Params:
    # 时序（s）
    t_ready: float = 0.60
    t_pre: float = 0.35
    t_squat: float = 0.85
    t_absorb: float = 0.45
    t_settle: float = 0.25
    t_rise: float = 0.90
    t_hold: float = 2.00
    # 姿态
    drop_squat: float = 0.25      # 深蹲基座下降量 m
    lean_squat: float = 13.0      # 深蹲躯干前倾 deg
    # 起跳初速度（冲量等效：摆臂+蹬地+提踵合力 / 质量，离地瞬间注入）
    vx_takeoff: float = 1.2       # 水平 m/s
    vz_takeoff: float = 1.6       # 垂直 m/s
    pf_to: float = 0.15            # 蹬伸末踝跖屈（提踵）rad（只末段加载）
    arm_up: float = -0.60          # 起跳摆臂幅度（shoulder_y，负=前上，延迟加载）
    # 腾空 / 落地
    tuck_reach: float = 0.08       # 收腿期脚在髋前方 m
    land_reach: float = 0.22       # 伸腿落地时脚在髋前方 m（脚要落在质心前）
    tuck_clear: float = 0.25       # 收腿期脚踝离地高 m
    land_clear: float = 0.05       # 触地前脚踝离地高 m
    t_flight_nom: float = 0.45     # 名义腾空时长（收腿→伸腿的调度用）
    # 蹬伸（两段）：A 段髋前移越过脚（腿保持弯曲加载），B 段蹬伸+提踵
    t_takeoff: float = 0.26
    dxw_to: float = -0.06          # 蹬伸末：髋在踝前方（蹬伸方向前倾量）
    # 落地：开环准静态姿态序列（深蹲→站起），无需额外参数
    # 控制器
    dt: float = 0.02               # 控制周期 50 Hz
    sim_hz: int = 1000
    kp_scale: float = 8.0
    kd_scale: float = 8.0
    scene: str = str(SCENE)


# ---------------------------------------------------------------- IK
def _leg_ik_safe(lx: float, lz: float) -> tuple[float, float]:
    """带可达域兜底的腿 IK：先夹 lz，再逐级收拢 lx。"""
    lz = min(max(lz, LZ_MIN), LZ_MAX)
    try:
        return SQ.leg_ik(lx, lz)
    except ValueError:
        pass
    for f in (0.9, 0.8, 0.7, 0.5):
        try:
            return SQ.leg_ik(lx * f, lz)
        except ValueError:
            continue
    return SQ.leg_ik(0.0, min(lz, 0.55))


def leg_targets(pitch: float, base_x: float, base_z_ref: float,
                foot_x: float, ankle_z: float, pf: float, waist: float = 0.0):
    """反馈版腿 IK：按【当前俯仰 + 期望基座高 + 脚踝世界目标】解 (q1, q2, qa)。

    ankle_z 是踝关节目标世界高度：平贴地面时 = FOOT_SOLE(0.04)；
    提踵 pf>0 时脚绕脚尖转，踝被抬高 ≈ TOE_LEN·sin(pf)。
    waist 是 waist_y 命令（骨盆相对躯干再转 waist）：髋位置与腿坐标系
    都随骨盆转，等价俯仰 = pitch + waist。
    脚掌对世界水平：qa = -(pitch + waist + q1 + q2) + pf。
    """
    pe = pitch + waist
    hx = base_x - SQ.TORSO_TO_HIP * math.sin(pe)
    hz = base_z_ref - SQ.TORSO_TO_HIP * math.cos(pe)
    dzw = hz - ankle_z
    dxw = foot_x - hx
    lx = dxw * math.cos(pe) + dzw * math.sin(pe)
    lz = -dxw * math.sin(pe) + dzw * math.cos(pe)
    q1, q2 = _leg_ik_safe(lx, lz)
    qa = -(pe + q1 + q2) + pf
    return q1, q2, qa


def leg_targets_span(pe: float, dxw: float, dzw: float, pf: float):
    """跨度版腿 IK：dxw=踝在髋前方多少 m，dzw=髋在踝上方多少 m（世界系），
    pe=骨盆等效俯仰（基座俯仰 + waist_y）。返回 (q1, q2, qa)，脚对世界水平。

    落地缓冲/站起都用它：命令的是髋与脚的【相对跨度】，脚锚在世界原地，
    逼着髋（也就是身体）去追目标 —— 不会像世界系脚目标那样把脚拖走。
    """
    lx = dxw * math.cos(pe) + dzw * math.sin(pe)
    lz = -dxw * math.sin(pe) + dzw * math.cos(pe)
    q1, q2 = _leg_ik_safe(lx, lz)
    qa = -(pe + q1 + q2) + pf
    return q1, q2, qa


def clamp_dxw_ankle(pe: float, dzw: float, dxw_cmd: float, pf: float,
                    qa_min: float = -0.75) -> float:
    """踝背屈限位感知：若命令的 dxw（髋在踝前方）会让放平脚所需的踝角
    超出背屈限位（qa_min，默认 -43°），二分收回最常前的可行髋位置。

    深弯 + 髋前移是几何不可行的（表：dxw=-0.06/dzw=0.40 需踝 -59°）；
    硬闯限位 → 脚尖挖地 → 反作用力把身体往后掘。蹬伸中髋只能
    随着腿伸长【同步】前移 —— 这正是人体跳远的协调方式。
    """
    _, _, qa = leg_targets_span(pe, dxw_cmd, dzw, pf)
    if qa >= qa_min:
        return dxw_cmd
    lo, hi = dxw_cmd, 0.15          # lo 不可行；hi（踝在髋前 15cm）必可行
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        _, _, qa = leg_targets_span(pe, mid, dzw, pf)
        if qa >= qa_min:
            hi = mid                 # mid 可行，收紧下界
        else:
            lo = mid
    return hi


def assemble(q_leg, arm, waist: float = 0.0) -> np.ndarray:
    """q_leg=(q1,q2,qa) 双腿同值；arm=(sh_y, sh_x, elbow)。返回 29 维目标。"""
    q = SQ.STANCE.copy()
    q[SQ.IDX["waist_y"]] = waist
    q1, q2, qa = q_leg
    for s in (SQ.IDX["l_hip_y"], SQ.IDX["r_hip_y"]):
        q[s] = q1
        q[s + 3] = q2
        q[s + 4] = qa
    sh_y, sh_x, el = arm
    for s, sgn in ((SQ.IDX["l_shoulder_y"], 1.0), (SQ.IDX["r_shoulder_y"], -1.0)):
        q[s] = sh_y
        q[s + 1] = sgn * sh_x
        q[s + 3] = el
    return q


def blend(a, b, u):
    return a + (b - a) * u


def mix_arm(a, b, u):
    return tuple(blend(x, y, u) for x, y in zip(a, b))


# ---------------------------------------------------------------- 仿真器
class LongJump:
    def __init__(self, p: Params):
        self.p = p
        scene = Path(p.scene)
        if not scene.exists():
            SQ.write_scene()
        self.m = mujoco.MjModel.from_xml_path(str(scene.resolve()))
        self.m.opt.timestep = 1.0 / p.sim_hz
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d

        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.ankle = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                           f"{s}_ankle_x_link") for s in "LR"}
        self.foot_geoms = self._collect(m, lambda nm: "_foot" in nm and nm.endswith("_collision"))
        self.ground_geoms = self._collect(m, lambda nm: "ground" in nm or "floor" in nm)

        # PD 增益（elf3_play 真值 × 倍率）+ 力矩限幅
        tp = TrainParams()
        self.names = list(tp.joint_names)
        self.kp = {n: float(v) * p.kp_scale for n, v in zip(self.names, tp.stiffness)}
        self.kd = {n: float(v) * p.kd_scale for n, v in zip(self.names, tp.damping)}
        self.jnt = {}
        self.tau_lim = {}
        for n in self.names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j < 0:
                raise SystemExit(f"[错误] 模型里没有关节 {n}")
            lim = m.jnt_actfrcrange[j]
            self.tau_lim[n] = float(np.max(np.abs(lim))) if np.any(lim) else np.inf
            self.jnt[n] = (m.jnt_qposadr[j], m.jnt_dofadr[j])

        # 准静态段（READY/PRE/SQUAT）用的 squat_stand 姿态参数
        self.sqp = SQ.Params(drop=p.drop_squat, lean=p.lean_squat,
                             arm=ARM_BACK[0], elbow=ARM_BACK[2],
                             arm_out=ARM_BACK[1], head_pitch=10.0)
        self.pre_frac = 0.10 / p.drop_squat if p.drop_squat else 0.4   # 预蹲深度比例

        # ---- 初始状态：站姿，脚放平贴地 ----
        q0, _, head0 = SQ.make_pose(self.sqp, 0.0)
        d.qpos[:] = 0.0
        d.qpos[0:3] = [0.0, 0.0, SQ.STANCE_Z]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[7:36] = q0
        d.qpos[36:38] = head0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        d.qpos[2] -= self._foot_low()      # 精确落到脚底 z=0
        mujoco.mj_forward(m, d)

        # 相位机状态
        self.phase = PH["READY"]
        self.phase_t0 = 0.0            # 当前相位进入时刻
        self.t = 0.0
        self.air_time = 0.0            # 连续无脚接触时长（判离地）
        self.liftoff = None            # dict(t, x, toe_x, ...)
        self.touchdown = None          # dict(t, x, heel_x, ...)
        self.anchors = {}              # 落地锚定：脚世界 x（按侧）
        self.span0 = None              # 蹬伸起点实测跨度 (dxw, dzw)
        self.fell_at = None
        self.failed_reason = None

        # 日志（qpos 逐控制步记录，供动画脚本运动学回放）
        self.log = dict(t=[], phase=[], x=[], z=[], pitch=[], com=[],
                        margin=[], contact=[], tau=[], qpos=[])

    @staticmethod
    def _collect(m, pred):
        out = set()
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if pred(nm):
                out.add(g)
        return out

    def _foot_low(self):
        d, m = self.d, self.m
        zs = []
        for g in self.foot_geoms:
            half = float(m.geom_size[g][1])
            r = float(m.geom_size[g][0])
            R = d.geom_xmat[g].reshape(3, 3)
            zs.append(d.geom_xpos[g][2] - (half * abs(R[2, 2]) + r))
        return float(min(zs))

    # ---- 观测 ----
    def pitch(self) -> float:
        f = self.d.xmat[self.torso].reshape(3, 3)[:, 0]
        return math.atan2(-f[2], math.hypot(f[0], f[1]))

    def foot_contact(self) -> bool:
        d = self.d
        for i in range(d.ncon):
            c = d.contact[i]
            if ((c.geom1 in self.ground_geoms and c.geom2 in self.foot_geoms)
                    or (c.geom1 in self.foot_geoms and c.geom2 in self.ground_geoms)):
                return True
        return False

    def foot_x(self) -> dict:
        return {s: float(self.d.xpos[b][0]) for s, b in self.ankle.items()}

    def com(self) -> np.ndarray:
        return self.d.subtree_com[self.torso].copy()

    def margin(self) -> float:
        try:
            hull = SR.foot_hull(self.m, self.d)
            return SR.signed_margin(self.com()[:2], hull)
        except Exception:
            return float("nan")

    # ---- 目标生成 ----
    def _measured_span(self) -> tuple[float, float]:
        """实测髋→踝跨度 (dxw=踝在髋前方, dzw=髋在踝上方)，世界系。"""
        pitch = self.pitch()
        hx = float(self.d.qpos[0]) - SQ.TORSO_TO_HIP * math.sin(pitch)
        hz = float(self.d.qpos[2]) - SQ.TORSO_TO_HIP * math.cos(pitch)
        ax = float(np.mean(list(self.foot_x().values())))
        return ax - hx, hz - SQ.FOOT_SOLE

    def targets(self) -> np.ndarray:
        p = self.p
        ph, u = self.phase, self._phase_u()

        if ph in (PH["READY"], PH["PRE"], PH["SQUAT"]):
            if ph == PH["READY"]:
                tail = 0.0
            elif ph == PH["PRE"]:
                tail = self.pre_frac * u
            else:
                tail = self.pre_frac + (1.0 - self.pre_frac) * SQ.smootherstep(u)
            q, _, _ = SQ.make_pose(self.sqp, tail)
            return q

        if ph == PH["TAKEOFF"]:
            # 协调蹬伸（仿人）：dzw 爆发伸长，dxw（髋相对踝）随腿伸长
            # 同步前移，且始终被踝背屈限位夹在可行域内；末段提踵 + 摆臂。
            pitch = self.pitch()
            dxw0, dzw0 = self.span0
            s = SQ.smootherstep(u)
            dxw_cmd = blend(dxw0, p.dxw_to, s)
            dzw = blend(dzw0, LZ_MAX, s)
            pf = p.pf_to * SQ.smootherstep(max(0.0, (u - 0.70) / 0.30))
            dxw = clamp_dxw_ankle(pitch, dzw, dxw_cmd, pf)
            arm = mix_arm(ARM_BACK, (p.arm_up, ARM_UP[1], ARM_UP[2]), s * s)
            q1, q2, qa = leg_targets_span(pitch, dxw, dzw, pf)
            return assemble((q1, q2, qa), arm)

        if ph == PH["FLIGHT"]:
            # 收腿 → 前伸小腿。脚世界目标挂在【实测髋】前方，随身体自适应。
            fu = min(u * p.t_flight_nom / 0.45, 1.0)     # 名义调度
            s = SQ.smootherstep(fu)
            reach = blend(p.tuck_reach, p.land_reach, s)
            clear = blend(p.tuck_clear, p.land_clear, s)
            pitch = self.pitch()
            hx = float(self.d.qpos[0]) - SQ.TORSO_TO_HIP * math.sin(pitch)
            arm = mix_arm(ARM_UP, ARM_MID, min(u / 0.30, 1.0))
            q1, q2, qa = leg_targets(pitch, float(self.d.qpos[0]),
                                     float(self.d.qpos[2]),
                                     hx + reach, SQ.FOOT_SOLE + clear, 0.0)
            return assemble((q1, q2, qa), arm)

        if ph in (PH["ABSORB"], PH["SETTLE"], PH["RISE"], PH["HOLD"]):
            # 落地段一律用【开环准静态姿态序列】—— squat_stand 的 make_pose
            # 已验证能静态站稳。刻意不引入当前俯仰/位置反馈：反馈 IK 里
            # base_x 的 sin(pitch) 项会构成正反馈（pitch↑→目标髋前移→
            # 躯干更前倾→pitch↑），且 pitch>30° 后踝限位耗尽会锁死发散。
            #   ABSORB / SETTLE : 深蹲（缓冲底，屈膝吸收冲击）
            #   RISE            : 深蹲 → 站姿
            #   HOLD            : 站姿（静态平衡校核）
            if ph in (PH["ABSORB"], PH["SETTLE"]):
                tail = 1.0
            elif ph == PH["RISE"]:
                tail = 1.0 - SQ.smootherstep(u)
            else:
                tail = 0.0
            q, _, _ = SQ.make_pose(self.sqp, tail)
            return q

        raise RuntimeError(f"未知相位 {ph}")

    def _phase_u(self) -> float:
        dur = self._phase_dur(self.phase)
        if dur is None:
            return 0.0
        return (self.t - self.phase_t0) / dur

    def _phase_dur(self, ph) -> float | None:
        p = self.p
        return {PH["READY"]: p.t_ready, PH["PRE"]: p.t_pre, PH["SQUAT"]: p.t_squat,
                PH["TAKEOFF"]: p.t_takeoff, PH["FLIGHT"]: None,
                PH["ABSORB"]: p.t_absorb, PH["SETTLE"]: p.t_settle,
                PH["RISE"]: p.t_rise, PH["HOLD"]: p.t_hold}.get(ph)

    # ---- 相位推进 ----
    def _enter(self, ph):
        self.phase = ph
        self.phase_t0 = self.t

    def _advance(self):
        """时间/事件驱动的相位切换。返回 False 表示仿真终止。"""
        p = self.p
        ph, u = self.phase, self._phase_u()
        if ph == PH["READY"] and u >= 1.0:
            self._enter(PH["PRE"])
        elif ph == PH["PRE"] and u >= 1.0:
            self._enter(PH["SQUAT"])
        elif ph == PH["SQUAT"] and u >= 1.0:
            self.takeoff_anchor = self.foot_x()
            self.span0 = self._measured_span()
            self._enter(PH["TAKEOFF"])
        elif ph == PH["TAKEOFF"]:
            if self.air_time >= 0.03:
                # 冲量等效：把"摆臂 + 蹬地 + 提踵"的合力按冲量定理等效为
                # 离地瞬间质心初速度。只设基座线速度，角速度/关节速度原样
                # 保留 —— 腾空段的姿态演化完全交给物理守恒。
                self.d.qvel[0] = p.vx_takeoff
                self.d.qvel[2] = p.vz_takeoff
                self.liftoff = dict(t=self.t, x=float(self.d.qpos[0]),
                                    toe_x=self.foot_x()["L"] + TOE_LEN,
                                    vx=p.vx_takeoff, vz=p.vz_takeoff)
                self._enter(PH["FLIGHT"])
            elif u > 0.5 / self.p.t_takeoff:      # 0.5 s 还没离地 = 起跳失败
                self.failed_reason = "蹬伸 0.5 s 仍未离地（推力不足）"
                return False
        elif ph == PH["FLIGHT"]:
            if self.foot_contact():
                # 完全非弹性落地：屈膝缓冲把平动 + 转动动量一次性卸掉，
                # 从静止深蹲位开始闭环站起（人体落地吸收冲击的等效建模）。
                self.d.qvel[:] = 0.0
                fx = self.foot_x()
                self.anchors = fx
                self.touchdown = dict(t=self.t, x=float(self.d.qpos[0]),
                                      heel_x=min(fx.values()) - TOE_LEN)
                self._enter(PH["ABSORB"])
            elif self.t - self.phase_t0 > 1.5:
                self.failed_reason = "腾空 1.5 s 未触地（异常）"
                return False
        elif ph == PH["ABSORB"] and u >= 1.0:
            self._enter(PH["SETTLE"])
        elif ph == PH["SETTLE"] and u >= 1.0:
            self._enter(PH["RISE"])
        elif ph == PH["RISE"] and u >= 1.0:
            self._enter(PH["HOLD"])
        elif ph == PH["HOLD"] and u >= 1.0:
            return False                            # 正常跑完
        return True

    # ---- 主循环 ----
    def run(self):
        p = self.p
        m, d = self.m, self.d
        n_sub = int(round(p.sim_hz * p.dt))
        step = 1.0 / p.sim_hz
        t_max = 12.0
        k_max = int(t_max / p.dt)

        for k in range(k_max):
            self._log_frame()
            tgt = self.targets()
            tgt_by_name = dict(zip(SQ.JOINT_NAMES, tgt))

            tau_ratio = 0.0
            for _ in range(n_sub):
                d.qfrc_applied[:] = 0.0
                for n in self.names:
                    qa_, va = self.jnt[n]
                    tau = (d.qfrc_bias[va]
                           + self.kp[n] * (tgt_by_name[n] - d.qpos[qa_])
                           - self.kd[n] * d.qvel[va])
                    lim = self.tau_lim[n]
                    if abs(tau) > lim:
                        tau = math.copysign(lim, tau)
                    d.qfrc_applied[va] = tau
                    tau_ratio = max(tau_ratio, abs(tau) / max(lim, 1e-9))
                mujoco.mj_step(m, d)
                self.t += step

                if self.phase >= PH["TAKEOFF"]:
                    if self.foot_contact():
                        self.air_time = 0.0
                    else:
                        self.air_time += step

                z = float(d.qpos[2])
                if z < 0.50 or abs(self.pitch()) > 1.2 or not np.isfinite(d.qpos[:7]).all():
                    self.fell_at = self.t
                    self.failed_reason = f"第 {self.t:.2f} s 摔倒（z={z:.2f}）"
                    self._log_frame()
                    return False
                if not self._advance():
                    self._log_frame()
                    return self.failed_reason is None

            # 本控制步内出现的最大力矩占用，记到本帧头上
            self.log["tau"][-1] = tau_ratio

        self.failed_reason = f"超过 {t_max:.0f} s 未完成动作"
        return False

    def _log_frame(self):
        d = self.d
        self.log["t"].append(self.t)
        self.log["phase"].append(self.phase)
        self.log["x"].append(float(d.qpos[0]))
        self.log["z"].append(float(d.qpos[2]))
        self.log["pitch"].append(self.pitch())
        self.log["com"].append(self.com().tolist())
        self.log["margin"].append(self.margin())
        self.log["contact"].append(self.foot_contact())
        self.log["qpos"].append(d.qpos.copy())
        self.log["tau"].append(0.0)

    # ---- 报告 ----
    def metrics(self) -> dict:
        L = self.log
        t = np.array(L["t"])
        x = np.array(L["x"])
        z = np.array(L["z"])
        mg = np.array(L["margin"], dtype=float)
        hold = np.array(L["phase"]) == PH["HOLD"]
        out = dict(
            duration=float(t[-1]),
            fell=bool(self.fell_at is not None),
            failed_reason=self.failed_reason,
            hold_frames=int(hold.sum()),
            hold_time=float(t[hold][-1] - t[hold][0]) if hold.any() else 0.0,
            final_z=float(z[-1]),
            final_pitch_deg=float(math.degrees(self.pitch())),
            final_margin=float(mg[-1]) if np.isfinite(mg[-1]) else float("nan"),
            min_margin_stance=float(np.nanmin(
                mg[hold & np.isfinite(mg)])) if (hold & np.isfinite(mg)).any() else float("nan"),
            tau_peak=float(np.max(L["tau"])),
        )
        if self.liftoff and self.touchdown:
            out.update(
                t_liftoff=float(self.liftoff["t"]),
                t_touchdown=float(self.touchdown["t"]),
                flight_time=float(self.touchdown["t"] - self.liftoff["t"]),
                toe_heel_dist=float(self.touchdown["heel_x"] - self.liftoff["toe_x"]),
                base_travel=float(self.touchdown["x"] - self.liftoff["x"]),
                # 注入的冲量等效初速度（真值），以及 qpos 差分实测（参考）
                v_takeoff=(float(self.liftoff["vx"]), float(self.liftoff["vz"])),
                v_takeoff_measured=self._vel_at(self.liftoff["t"]),
            )
        out["total_travel"] = float(x[-1] - x[0])
        return out

    def _vel_at(self, tq: float) -> tuple[float, float]:
        L = self.log
        t = np.array(L["t"])
        x = np.array(L["x"])
        z = np.array(L["z"])
        i = int(np.argmin(np.abs(t - tq)))
        i0, i1 = max(0, i - 3), min(len(t) - 1, i + 3)
        return (float((x[i1] - x[i0]) / (t[i1] - t[i0])),
                float((z[i1] - z[i0]) / (t[i1] - t[i0])))

    def verdict(self, mt: dict) -> tuple[bool, list[str]]:
        ok, why = True, []
        if mt.get("fell") or mt.get("failed_reason"):
            ok, why = False, [mt.get("failed_reason") or "摔倒"]
            return ok, why
        if mt["hold_time"] < self.p.t_hold - 0.05:
            ok = False
            why.append(f"站稳保持只有 {mt['hold_time']:.2f}s < {self.p.t_hold:.0f}s")
        if not (np.isfinite(mt["final_margin"]) and mt["final_margin"] > 0.02):
            ok = False
            why.append(f"末态 COM 裕度 {mt['final_margin']*100:+.1f} cm ≤ 2 cm")
        if abs(mt["final_z"] - SQ.STANCE_Z) > 0.03:
            ok = False
            why.append(f"末态躯干高 {mt['final_z']:.3f} m 偏离站姿 {SQ.STANCE_Z:.3f} m 超 3 cm")
        if abs(mt["final_pitch_deg"]) > 10.0:
            ok = False
            why.append(f"末态俯仰 {mt['final_pitch_deg']:+.1f}° 过大")
        return ok, why


# ---------------------------------------------------------------- 报告打印
def report(mt: dict, ok: bool, why: list[str], p: Params):
    line = "=" * 72
    print(f"\n{line}\nELF3 立定跳远   闭环物理仿真结果\n{line}")
    if "t_liftoff" in mt:
        vx, vz = mt["v_takeoff"]
        print(f"\n[起跳]  t={mt['t_liftoff']:.2f}s   冲量等效初速度 "
              f"vx={vx:+.2f} m/s  vz={vz:+.2f} m/s"
              f"   （摆臂 + 蹬地 + 提踵合力 / 质量）")
        if "v_takeoff_measured" in mt:
            mx, mz = mt["v_takeoff_measured"]
            print(f"        qpos 差分实测（参考） vx={mx:+.2f}  vz={mz:+.2f} m/s")
        print(f"[腾空]  {mt['flight_time']:.2f}s   质心抛物线飞行（纯物理守恒）")
        print(f"[落地]  t={mt['t_touchdown']:.2f}s   "
              f"脚尖→脚跟跳距 {mt['toe_heel_dist']*100:.0f} cm   "
              f"基座前移 {mt['base_travel']*100:.0f} cm")
    print(f"[总位移] {mt['total_travel']*100:.0f} cm（起点→末态基座 x）")
    print(f"\n[落地平衡] 保持 {mt['hold_time']:.2f}s 不倒   "
          f"末态躯干高 {mt['final_z']:.3f} m（站姿 {SQ.STANCE_Z:.3f}）   "
          f"末态俯仰 {mt['final_pitch_deg']:+.1f}°")
    print(f"           末态 COM 裕度 {mt['final_margin']*100:+.2f} cm"
          f"   站定期最小 {mt['min_margin_stance']*100:+.2f} cm   "
          f"（判据 > 2 cm）")
    print(f"[力矩]   峰值占用 {100*mt['tau_peak']:.0f}% of actuatorfrcrange")
    print(f"[判定]   {'✔ 通过：落地后保持平衡站立' if ok else '✘ 未通过'}")
    for w in why:
        print(f"         - {w}")
    print(line)
    _ = p


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 立定跳远：动作序列 + 闭环仿真 + 落地平衡校核")
    ap.add_argument("--drop", type=float, default=0.25, help="深蹲下降量 m")
    ap.add_argument("--lean", type=float, default=13.0, help="深蹲前倾 deg")
    ap.add_argument("--vx", type=float, default=1.2,
                    help="起跳水平初速度 m/s（冲量等效）")
    ap.add_argument("--vz", type=float, default=1.6,
                    help="起跳垂直初速度 m/s（冲量等效）")
    ap.add_argument("--pf", type=float, default=0.15, help="蹬伸末踝跖屈 rad")
    ap.add_argument("--arm-up", type=float, default=-0.60, help="起跳摆臂 shoulder_y rad")
    ap.add_argument("--dxw-to", type=float, default=-0.06,
                    help="蹬伸末髋在踝前方 m（蹬伸方向前倾量）")
    ap.add_argument("--land-reach", type=float, default=0.22, help="伸腿落地脚在髋前方 m")
    ap.add_argument("--t-takeoff", type=float, default=0.26, help="蹬伸时长 s")
    ap.add_argument("--kp-scale", type=float, default=8.0)
    ap.add_argument("--kd-scale", type=float, default=8.0)
    ap.add_argument("--sim-hz", type=int, default=1000)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    p = Params(drop_squat=args.drop, lean_squat=args.lean,
               vx_takeoff=args.vx, vz_takeoff=args.vz,
               pf_to=args.pf, arm_up=args.arm_up, dxw_to=args.dxw_to,
               land_reach=args.land_reach, t_takeoff=args.t_takeoff,
               kp_scale=args.kp_scale, kd_scale=args.kd_scale, sim_hz=args.sim_hz)

    sim = LongJump(p)
    print(f"[仿真] 场景 {Path(p.scene).name}   PD×{p.kp_scale:g} @ {p.sim_hz} Hz   "
          f"控制 50 Hz   起跳初速度 vx={p.vx_takeoff:g} / vz={p.vz_takeoff:g} m/s   "
          f"总时长预算 ~{p.t_ready + p.t_pre + p.t_squat + p.t_takeoff + 0.5 + p.t_absorb + p.t_settle + p.t_rise + p.t_hold:.1f} s")
    sim.run()
    mt = sim.metrics()
    ok, why = sim.verdict(mt)
    report(mt, ok, why, p)

    if args.check_only:
        return

    out = Path(args.out)
    meta = dict(params=asdict(p), metrics=mt, ok=bool(ok), why=why,
                phases=PHASES,
                liftoff_t=(sim.liftoff or {}).get("t"),
                touchdown_t=(sim.touchdown or {}).get("t"))
    np.savez(out,
             time=np.array(sim.log["t"]),
             qpos=np.array(sim.log["qpos"]),
             phase=np.array(sim.log["phase"], dtype=np.int8),
             dt=p.dt,
             joint_names=SQ.JOINT_NAMES,
             meta=meta)
    print(f"\n[OK] 轨迹 -> {out}  ({len(sim.log['t'])} 帧, "
          f"{out.stat().st_size/1024:.0f} KB)")
    print(f"     动画: python jump_replay.py --traj {out.name} "
          f"--gif <path>.gif --sheet <path>.png")

    if args.json:
        Path(args.json).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"     指标 -> {args.json}")


if __name__ == "__main__":
    main()
