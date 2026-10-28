"""螃蟹步横移爬楼梯 —— 四相位准静态脚本控制器。

==================== 动作方案 ====================
机器人朝 +X 站立，沿 +Y 侧向横移上下楼。左腿(L) 在 +Y 侧，是【前导腿】。

落脚面序列（platforms）：
    [起步地面] up_0 ... up_N [中间平台] dn_0 ... dn_M [终点地面]
    —— 下楼段由 gen_crab_stairs.py 保证总降高 == 总升高，末端与地面齐平。

每个落脚面给 L / R 各分配【一个固定落点】（中心 ∓ off），于是每次转换
只有一只脚在动：
    GATHER(P_i)      : L = c_i + off_i , R = c_i - off_i
    SPLIT(P_i→P_i+1) : L = c_{i+1} + off_{i+1} , R = c_i - off_i
    GATHER(P_i+1)    : L = c_{i+1} + off_{i+1} , R = c_{i+1} - off_{i+1}
    => GATHER→SPLIT 只动 L，SPLIT→GATHER 只动 R。

每一"步"内部再分四个相位：
    LOAD    : 重心平移到支撑脚正上方（摆动脚不动）
    SWING   : 摆动脚抬到 max(z0,z1)+clear，横移过踏面边界
    LAND    : 摆动脚落到目标踏面
    SETTLE  : 重心回到双脚中点，骨盆侧倾回到目标值

==================== 关键结论（来自 crab_ik.py 的可行性回归）====================
  * 不做前后错位。窄踏面靠"双脚对称放在踏面中心 ±off"解决，不需要叉开。
  * 级高的瓶颈是平脚 + 踝限位，靠 waist_x 骨盆侧倾解决：
    25 cm 级高只需 10.4 deg 侧倾（限位 15 deg）。
  * 脚掌保持水平时 ankle_x = -hip_x，而 ankle_x 限位 ±20 deg
    => 单脚相对髋的侧移 |Y| <= rho * sin20deg ≈ 0.21 m。
  * 关节角不是简单相加（hip_y 会改变 hip_x 的轴方向），
    所以闭式解只作种子，每一步都跑 DLS 数值 IK 收敛到 0.00 mm。
  * 髋高必须用【世界系】插值。用"相对某只脚"会被摆腿时那只脚的
    (级高 + 抬腿余量) ≈ 0.3 m 带飞。

用法:
    python crab_climb.py                                  # 默认跑 seed 42 地形
    python crab_climb.py --out traj.npz --step-time 1.0
    python crab_climb.py --check-only                     # 只做可行性体检，不跑轨迹
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import crab_ik as K                                              # noqa: E402
from bxi_example_py_elf3.policies.joints import (                # noqa: E402
    ELF3_ISAAC_PARAMETERS as P,
)

DATA = HERE.parent / "data" / "mujoco_simulation"
DEFAULT_SCENE = DATA / "elf3_crab.xml"
DEFAULT_JSON = DATA / "terrains" / "universal_test" / "crab_stairs.json"

FOOT_HALF_Y = 0.04          # 脚掌 Y 向半宽（8 cm 宽）
EDGE_MARGIN = 0.01          # 脚掌边缘离踏面边缘的最小余量
SEP_GATHER = 0.14           # 双脚同踏面时的目标间距（再大会把 SPLIT 站姿撑爆）
SEP_GATHER_MIN = 0.10       # 双脚不重叠所需的最小间距
CLEARANCE = 0.055           # 摆动脚越过踏面边界时的离地余量
FLOOR_DEPTH = 0.30          # 两端地面段的"有效进深"

FOOT_BODY = {"L": "l_ankle_x_link", "R": "r_ankle_x_link"}
HIP_BODY = {"L": "l_hip_y_link", "R": "r_hip_y_link"}


# ---------------------------------------------------------------- 落脚面 / 站姿
@dataclass
class Platform:
    """一个可落脚的水平面。"""
    name: str
    y0: float
    y1: float
    top: float

    @property
    def center(self) -> float:
        return 0.5 * (self.y0 + self.y1)

    @property
    def depth(self) -> float:
        return self.y1 - self.y0

    @property
    def off(self) -> float:
        """双脚同踏面时，各自相对中心的 Y 偏移（半间距）。

        两脚都完全落在踏面内且不重叠：
            2*off + 2*FOOT_HALF_Y <= depth - 2*EDGE_MARGIN
            2*off >= SEP_GATHER_MIN
        """
        room = self.depth - 2.0 * EDGE_MARGIN - 2.0 * FOOT_HALF_Y
        g = min(SEP_GATHER, max(SEP_GATHER_MIN, room))
        return float(g / 2.0)

    @property
    def yL(self) -> float:
        return self.center + self.off

    @property
    def yR(self) -> float:
        return self.center - self.off


@dataclass
class Stance:
    """一个静态站姿：两脚各自踩在哪个面上（落点由 Platform 决定）。"""
    kind: str                      # "GATHER" | "SPLIT"
    plat: dict                     # {"L": Platform, "R": Platform}
    tilt: float = 0.0              # waist_x 侧倾（+ = L 髋抬高）
    # 髋高一律用【世界系】存，见文件头说明
    hip_z: float = 0.65
    hip_win: tuple = field(default=(0.0, 0.0))   # 可行髋高区间（世界系）

    @property
    def y(self) -> dict:
        return {"L": self.plat["L"].yL, "R": self.plat["R"].yR}

    @property
    def z(self) -> dict:
        return {"L": self.plat["L"].top, "R": self.plat["R"].top}

    @property
    def rise(self) -> float:
        return self.z["L"] - self.z["R"]

    @property
    def sep(self) -> float:
        return abs(self.y["L"] - self.y["R"])

    @property
    def foot_mid(self) -> float:
        return 0.5 * (self.y["L"] + self.y["R"])


def build_platforms(meta: dict, floor_depth: float = FLOOR_DEPTH) -> list[Platform]:
    """由真值表构造落脚面序列，两端各补一块地面。"""
    treads = meta["treads"]
    if not treads:
        raise SystemExit("[错误] 真值表里没有台阶")
    start_y = float(treads[0]["y0"])
    end_y = float(treads[-1]["y1"])
    pls = [Platform("floor_start", start_y - floor_depth, start_y, 0.0)]
    for t in treads:
        pls.append(Platform(t["name"], float(t["y0"]), float(t["y1"]),
                            float(t["top"])))
    pls.append(Platform("floor_end", end_y, end_y + floor_depth, 0.0))
    return pls


def build_stances(pls: list[Platform]) -> list[Stance]:
    """构造站姿序列：G0 -> S0 -> G1 -> S1 -> ... -> Gn（每步只动一只脚）。"""
    out = [Stance("GATHER", {"L": pls[0], "R": pls[0]})]
    for i in range(len(pls) - 1):
        lo, hi = pls[i], pls[i + 1]
        out.append(Stance("SPLIT", {"L": hi, "R": lo}))     # L 上到 hi
        out.append(Stance("GATHER", {"L": hi, "R": hi}))    # R 跟上到 hi
    return out


def plan_all(stances: list[Stance]) -> bool:
    """给整条站姿序列定侧倾与髋高。

    两遍：
      1) 每个站姿取自己髋高窗口的中值（保证可行）
      2) GATHER 的窗口很宽，把它的【世界】髋高拉向前后两个 SPLIT 的中值，
         再裁回窗口内 —— 这样骨盆高度平滑，不会每一步都上下弹一次。
    """
    for st in stances:
        if st.kind == "GATHER":
            win = K.split_window(0.0, st.sep, 0.0)
            st.tilt = 0.0
        else:
            t = K.tilt_for_rise(st.rise, sep=st.sep)
            if t is None:
                print(f"[失败] {st.kind} 级高 {st.rise:+.3f} m、"
                      f"脚间距 {st.sep:.3f} m 无解")
                return False
            st.tilt, win = float(t[0]), t[2]
        if win is None:
            print(f"[失败] {st.kind} 级高 {st.rise:+.3f} m 无可行髋高")
            return False
        zr = st.z["R"]
        st.hip_win = (float(zr + win[0]), float(zr + win[1]))
        st.hip_z = 0.5 * (st.hip_win[0] + st.hip_win[1])

    for i, st in enumerate(stances):
        if st.kind != "GATHER":
            continue
        nb = [stances[j].hip_z for j in (i - 1, i + 1) if 0 <= j < len(stances)]
        if not nb:
            continue
        lo, hi = st.hip_win
        st.hip_z = float(min(max(float(np.mean(nb)), lo + 1e-3), hi - 1e-3))
    return True


# ---------------------------------------------------------------- 位姿求解
class CrabClimber:
    """把站姿 / 摆腿中间态解成完整 qpos。"""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.ik = K.CrabIK(model, data)
        self.jid = {}
        for n in K.LEG_JOINTS + ("waist_x_joint", "waist_y_joint", "waist_z_joint"):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j < 0:
                raise SystemExit(f"[错误] 模型里没有关节 {n}")
            self.jid[n] = j
        self.lim = {n: (model.jnt_range[self.jid[n]][0], model.jnt_range[self.jid[n]][1])
                    for n in K.LEG_JOINTS}
        self.b_foot = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                       for s, n in FOOT_BODY.items()}
        self.b_hip = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                      for s, n in HIP_BODY.items()}
        for n, q0 in zip(P.layout.names, P.default_position):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j >= 0:
                self.d.qpos[model.jnt_qposadr[j]] = q0
        self.d.qpos[3:7] = [1, 0, 0, 0]

    def solve(self, foot_yz, tilt, hip_z, com_y, rot_w=(0.6, 0.6),
              swing: str | None = None, iters=150):
        """foot_yz: {"L": (y, z_tread), "R": (y, z_tread)}，hip_z 为世界系髋平均高。

        rot_w: (w_L, w_R)，摆动脚那一侧给 0 即"释放脚掌姿态"。
        swing: "L"/"R"/None —— 哪只脚在空中。它的姿态误差不计入合格判据。
        返回 (q_leg, 诊断dict)。root 位姿也写进了 self.d.qpos。
        """
        m, d = self.m, self.d
        d.qpos[m.jnt_qposadr[self.jid["waist_x_joint"]]] = tilt
        d.qpos[m.jnt_qposadr[self.jid["waist_y_joint"]]] = 0.0
        d.qpos[m.jnt_qposadr[self.jid["waist_z_joint"]]] = 0.0
        d.qpos[0] = 0.0
        d.qpos[1] = com_y - K.HIP_DROP * np.sin(tilt)
        d.qpos[2] = 1.0339
        for _ in range(8):                       # 迭代对齐髋的平均高度
            mujoco.mj_kinematics(m, d)
            hm = 0.5 * (d.xpos[self.b_hip["L"]][2] + d.xpos[self.b_hip["R"]][2])
            d.qpos[2] += hip_z - hm
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        hl = d.xpos[self.b_hip["L"]].copy()
        hr = d.xpos[self.b_hip["R"]].copy()
        tl = np.array([0.0, foot_yz["L"][0], foot_yz["L"][1] + K.ANKLE_SOLE_DROP])
        tr = np.array([0.0, foot_yz["R"][0], foot_yz["R"][1] + K.ANKLE_SOLE_DROP])
        q0 = self.ik.seed(hl, hr, tl, tr)
        q, el, er, rl, rr = self.ik.solve(
            q0, [(tl, K.foot_flat_quat()), (tr, K.foot_flat_quat())],
            iters=iters, rot_w=rot_w)
        bad = [n for i, n in enumerate(K.LEG_JOINTS)
               if q[i] < self.lim[n][0] - 1e-6 or q[i] > self.lim[n][1] + 1e-6]
        rot = {"L": rl, "R": rr}
        return q, {
            "err_pos": max(el, er),
            # 支撑脚必须严格平贴；摆动脚的姿态是故意放开的，单独记
            "err_rot": max(v for s, v in rot.items() if s != swing) if swing else max(rl, rr),
            "err_rot_swing": rot.get(swing, 0.0),
            "hip_l": hl, "hip_r": hr, "bad": bad,
            "com": d.subtree_com[0].copy(),
        }

    def com(self):
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.subtree_com[0].copy()


# ---------------------------------------------------------------- 时间插值
def smoothstep(u: float) -> float:
    u = float(min(max(u, 0.0), 1.0))
    return u * u * (3.0 - 2.0 * u)


def swing_z(p: float, z0: float, z1: float, clear: float,
            up: float = 0.18, down: float = 0.82) -> float:
    """摆动脚高度：快速抬到 max(z0,z1)+clear，平飞过边界，再落下。"""
    zt = max(z0, z1) + clear
    if p <= 0.0:
        return z0
    if p >= 1.0:
        return z1
    if p < up:
        return z0 + (zt - z0) * smoothstep(p / up)
    if p > down:
        return zt + (z1 - zt) * smoothstep((p - down) / (1.0 - down))
    return zt


# ---------------------------------------------------------------- 主流程
# 回退梯度：(重心转移比例 lam, 摆动脚姿态权重 rot_w)
#   宽站姿下把重心完全移到支撑脚会让摆动腿超伸（|Y| 太大 -> rho > 0.64），
#   而摆动脚离地后不必保持水平（rot_w=0 解锁 ankle_x 的 ±20deg）。
#   先试"完全转移 + 略保持水平"，不行再逐级放松。
LADDER = ((1.00, 0.30), (1.00, 0.00), (0.70, 0.00), (0.45, 0.00), (0.25, 0.00))


def solve_step(cl: CrabClimber, a: Stance, b: Stance, n: int,
               clearance: float, lam: float, rot_w_swing: float):
    """解一个"步"的全部控制帧。返回 (qpos 列表, 诊断列表, 最差误差)。"""
    mv = "L" if abs(b.y["L"] - a.y["L"]) > 1e-9 else "R"
    stf = "R" if mv == "L" else "L"
    y0, z0 = a.y[mv], a.z[mv]
    y1, z1 = b.y[mv], b.z[mv]
    mid_a, mid_b = a.foot_mid, b.foot_mid
    stance_y = a.y[stf]
    com_peak = mid_a + lam * (stance_y - mid_a)
    # 重心时序：尽早移到支撑脚正上方，在摆动脚【高位横移】期间停住不动，
    # 等脚快落地了再移向新的双脚中点。这样单脚支撑期里质心始终在支撑面内。
    p_a, p_b = 0.32, 0.78
    out_q, out_dg, worst = [], [], 0.0
    for i in range(n):
        u = i / (n - 1)
        p = smoothstep(min(u / 0.80, 1.0))
        # 起步瞬间脚还贴着踏面，必须保持水平；离地后再平滑放开姿态约束
        blend = smoothstep(min(p / 0.06, 1.0))
        rw = {"L": 0.6, "R": 0.6}
        rw[mv] = 0.6 * (1.0 - blend) + rot_w_swing * blend
        if p <= p_a:
            com_y = mid_a + (com_peak - mid_a) * (p / p_a)
        elif p <= p_b:
            com_y = com_peak
        else:
            com_y = com_peak + (mid_b - com_peak) * ((p - p_b) / (1.0 - p_b))
        fy = {"L": a.y["L"], "R": a.y["R"]}
        fz = {"L": a.z["L"], "R": a.z["R"]}
        fy[mv] = y0 + (y1 - y0) * p
        fz[mv] = swing_z(p, z0, z1, clearance)
        tilt = a.tilt + (b.tilt - a.tilt) * smoothstep(u)
        hip_z = a.hip_z + (b.hip_z - a.hip_z) * smoothstep(u)
        q, diag = cl.solve({"L": (fy["L"], fz["L"]), "R": (fy["R"], fz["R"])},
                           tilt, hip_z, com_y, rot_w=(rw["L"], rw["R"]),
                           swing=(mv if p > 1e-6 else None))
        diag["phase"] = "SWING" if p < 0.98 else "LAND"
        diag["com_y"] = com_y
        # 支撑面：p==0（还没抬）或 p==1（已落足）时摆动脚都在踏面上
        feet = [f"{stf}"] + ([f"{mv}"] if (p < 1e-6 or p > 1.0 - 1e-9) else [])
        ys = [(fy[s] - FOOT_HALF_Y, fy[s] + FOOT_HALF_Y) for s in feet]
        diag["sup"] = (min(y[0] for y in ys), max(y[1] for y in ys))
        out_q.append(cl.d.qpos.copy())
        out_dg.append(diag)
        # 支撑脚的【位置 + 姿态】都要达标；摆动脚姿态不计
        worst = max(worst, diag["err_pos"], diag["err_rot"])
        if diag["bad"]:
            worst = max(worst, 1.0)
    return out_q, out_dg, worst


def build_trajectory(cl: CrabClimber, stances: list[Stance],
                     step_time: float, dt: float, clearance: float,
                     hold: float = 0.5):
    """逐控制步解算，返回 (times, qpos 数组, 诊断列表, 每步信息)。"""
    times, qpos_out, diags, steps = [], [], [], []
    t_now = 0.0

    def emit(diag):
        times.append(t_now)
        qpos_out.append(cl.d.qpos.copy())
        diags.append(diag)

    st = stances[0]
    q, diag = cl.solve({"L": (st.y["L"], st.z["L"]), "R": (st.y["R"], st.z["R"])},
                       st.tilt, st.hip_z, st.foot_mid)
    diag["phase"] = "HOLD"
    diag["sup"] = (min(st.y["L"], st.y["R"]) - FOOT_HALF_Y,
                   max(st.y["L"], st.y["R"]) + FOOT_HALF_Y)
    n_hold = max(1, int(round(hold / dt)))
    for _ in range(n_hold):
        emit(diag)
        t_now += dt

    n = max(2, int(round(step_time / dt)))
    for k in range(len(stances) - 1):
        a, b = stances[k], stances[k + 1]
        picked = None
        for lam, rw in LADDER:
            qs, dgs, worst = solve_step(cl, a, b, n, clearance, lam, rw)
            if worst < 1e-3:
                picked = (lam, rw, qs, dgs)
                break
        if picked is None:
            lam, rw = LADDER[-1]
            qs, dgs, worst = solve_step(cl, a, b, n, clearance, lam, rw)
            print(f"  [警告] 第 {k} 步降级到 lam={lam} 仍不收敛（误差 "
                  f"{worst*1000:.0f} mm）")
        else:
            lam, rw, qs, dgs = picked
        steps.append({"k": k, "mv": "L" if abs(b.y["L"] - a.y["L"]) > 1e-9 else "R",
                      "lam": lam, "rot_w": rw})
        for dg, qq in zip(dgs, qs):
            cl.d.qpos[:] = qq
            emit(dg)
            t_now += dt

    return np.array(times), np.array(qpos_out), diags, steps


def report(stances: list[Stance], diags: list[dict], dt: float,
           steps: list[dict] | None = None) -> bool:
    print()
    print("=== 站姿规划 ===")
    print(f"{'#':>3} {'类型':<7}{'L落点':>8}{'R落点':>8}{'级高':>7}{'脚间距':>7}"
          f"{'侧倾':>8}{'髋高':>7}{'髋高窗口':>16}")
    for i, st in enumerate(stances):
        win = f"{st.hip_win[0]:.3f}~{st.hip_win[1]:.3f}"
        print(f"{i:>3} {st.kind:<7}{st.y['L']:>8.3f}{st.y['R']:>8.3f}"
              f"{st.rise:>7.3f}{st.sep:>7.3f}"
              f"{np.degrees(st.tilt):>7.1f}d{st.hip_z:>7.3f}{win:>16}")

    errs = np.array([dg["err_pos"] for dg in diags])
    rots = np.array([dg["err_rot"] for dg in diags])
    bads = [i for i, dg in enumerate(diags) if dg["bad"]]
    worse = int(np.argmax(errs))
    print()
    print("=== 轨迹体检 ===")
    print(f"  帧数 {len(diags)}   最大位置误差 {errs.max()*1000:.3f} mm"
          f"   最大姿态误差 {rots.max():.5f} rad")
    if errs.max() >= 1e-3:
        print(f"  最差帧 #{worse} t={worse*dt:.2f}s 相位={diags[worse].get('phase')} "
              f"误差 {errs[worse]*1000:.1f} mm")
    if bads:
        print(f"  !! {len(bads)} 帧超关节限位，例如 #{bads[0]}: "
              f"{', '.join(diags[bads[0]]['bad'])}")
    else:
        print("  关节限位：全程未超")
    # ---- 质心 vs 支撑面（沿 Y，横移方向是唯一要担心的轴）----
    marg = []
    for dg in diags:
        lo, hi = dg.get("sup", (np.nan, np.nan))
        cy = float(dg["com"][1])
        marg.append(min(cy - lo, hi - cy))
    marg = np.array(marg)
    neg = int((marg < 0).sum())
    print(f"  质心支撑裕度: 最小 {marg.min()*1000:+.0f} mm"
          f"（负=质心在支撑面外）  越界帧 {neg}/{len(marg)}"
          f" = {100.0*neg/len(marg):.0f}%")
    if neg:
        # 越界最长连续段
        runs, cur = [], 0
        for v in marg < 0:
            cur = cur + 1 if v else 0
            runs.append(cur)
        print(f"    最长连续越界 {max(runs)} 帧 = {max(runs)*0.02:.2f} s"
              "  —— 抬脚与重心转移同时进行，这段需要动力学支撑（RL 的用武之地）")
    if steps:
        lams = [s["lam"] for s in steps]
        full = sum(1 for x in lams if x >= 0.999)
        print(f"  重心转移: {full}/{len(lams)} 步做到完全移到支撑脚正上方；"
              f"最小 lam={min(lams):.2f}")
        if full < len(lams):
            print("    lam < 1 的步意味着单腿支撑期间重心在支撑面外"
                  " —— 纯准静态站不住，需要动力学/RL 兜底")
    return (not bads) and errs.max() < 1e-3 and rots.max() < 5e-3


def main() -> None:
    ap = argparse.ArgumentParser(description="螃蟹步横移爬楼梯（准静态脚本控制器）")
    ap.add_argument("--scene", default=str(DEFAULT_SCENE))
    ap.add_argument("--terrain-json", default=str(DEFAULT_JSON))
    ap.add_argument("--out", default=str(DATA / "crab_climb_traj.npz"))
    ap.add_argument("--step-time", type=float, default=0.9, help="每步时长 s")
    ap.add_argument("--dt", type=float, default=1.0 / 50.0, help="控制周期 s")
    ap.add_argument("--clearance", type=float, default=CLEARANCE, help="摆腿离地余量 m")
    ap.add_argument("--hold", type=float, default=0.5, help="起步保持时间 s")
    ap.add_argument("--check-only", action="store_true", help="只体检不写轨迹")
    args = ap.parse_args()

    meta = json.loads(Path(args.terrain_json).read_text(encoding="utf-8"))
    pls = build_platforms(meta)
    stances = build_stances(pls)
    if not plan_all(stances):
        return

    m = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    d = mujoco.MjData(m)
    cl = CrabClimber(m, d)

    times, qpos, diags, steps = build_trajectory(
        cl, stances, args.step_time, args.dt, args.clearance, args.hold)
    ok = report(stances, diags, args.dt, steps)
    print()
    print("结论: " + ("可行，可进入烘焙/回放" if ok else "不可行，需要调参或降难度"))

    if not args.check_only:
        np.savez(args.out, time=times, qpos=qpos, dt=args.dt)
        print(f"[OK] 轨迹 -> {args.out}  ({len(times)} 帧, {times[-1]:.1f} s)")


if __name__ == "__main__":
    main()
