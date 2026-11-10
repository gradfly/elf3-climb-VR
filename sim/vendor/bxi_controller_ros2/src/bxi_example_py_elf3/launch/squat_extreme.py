"""ELF3 极限蹲（膝 149°，忽略大腿/小腿碰撞）—— 躯干压到最低。

`squat_stand.py`（标准）和 `squat_deep.py`（膝 121.8° 贴碰撞胶囊）两个版本都不动，
这是第三个：按需求把膝角直接推到 **149°**（离机械限位 150° 只差 1°），
不让碰撞胶囊提前把腿挡住。

==================== 为此放开的两样东西 ====================
1. **大腿/小腿的碰撞**。两个胶囊半径各 5 cm，121.8° 就贴上；149° 时穿插 4 cm。
   场景里用 `<contact><exclude>` 把这对 body 的碰撞关掉（左右各一条），
   视觉网格会互相穿过 —— 这是"忽略碰撞半径"的直接后果，不是 bug。
2. **IK 的 q1 搜索范围**。原来限制在 −100°（正常蹲姿够用），但 149° 时大腿要
   折到 −150° 左右（膝盖高过髋），必须放宽到 −160° 才能解出来。

==================== 为什么 149° 反而更"低" ====================
膝角锁死后髋-踝的距离只剩 `2×0.32×cos(74.5°) = 0.171 m`。此时
`ankle = −(θ+hip+knee)`：前倾 θ 越大，hip 越负，踝需求从 −74.5°（θ=0）
一路回收到 −17.6°（θ=22°），才重新落进 −50° 的限位里。
θ>22° 时腿长不够（脚踝够不到原位），所以 22° 就是底。

    lean 0°  → 踝 −74.5°（超限 24.5°）  torso 0.594
    lean 12° → 踝 −46.8°（达标）        torso 0.566
    lean 18° → 踝 −30.8°                torso 0.527
    lean 22° → 踝 −17.6°                torso **0.488**  ← 自动搜到的底

==================== 用法 ====================
    python squat_extreme.py                 # 膝 149°，自动搜最低前倾
    python squat_extreme.py --knee 140      # 换个膝角
    python squat_extreme.py --return        # 站→蹲→站 完整循环
    python squat_replay.py --traj squat_extreme_traj.npz --mode phys
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import mujoco

import squat_stand as base
from squat_stand import (STANCE, IDX, JOINT_NAMES, Kin, convex_hull, margin_of,
                         smootherstep, limit_violations)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE_SRC = DATA / "elf3_squat.xml"
SCENE = DATA / "elf3_squat_x.xml"
DEFAULT_OUT = DATA / "squat_extreme_traj.npz"

L_THIGH = 0.32
L_SHIN = 0.32
H2 = 0.3825
SOLE = 0.04
ANKLE_LIMIT = -0.87266
EXCLUDES = [("l_hip_z_link", "l_knee_y_link"), ("r_hip_z_link", "r_knee_y_link")]


# ---------------------------------------------------------------- 宽范围 IK
def leg_ik_wide(lx: float, lz: float):
    """两连杆 IK，q1 搜索下限放到 −160°（149° 膝角时大腿要折过 −140°）。"""
    def lz_of(q1: float) -> float:
        s = -(lx + L_THIGH * math.sin(q1)) / L_SHIN
        if abs(s) > 1.0:
            return -1e9
        return L_THIGH * math.cos(q1) + L_SHIN * math.cos(math.asin(s))

    lo, hi = -2.80, 0.02
    if not (lz_of(lo) <= lz <= lz_of(hi)):
        return None
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if lz_of(mid) > lz:
            hi = mid
        else:
            lo = mid
    q1 = 0.5 * (lo + hi)
    s = -(lx + L_THIGH * math.sin(q1)) / L_SHIN
    return q1, math.asin(max(-1.0, min(1.0, s))) - q1


def joints_wide(drop: float, lean_deg: float):
    """给定下降量和前倾角，解 (hip, knee, ankle, 基座高)；不可达返回 None。"""
    th = math.radians(lean_deg)
    h = base.STANCE_Z - drop
    hip_x = -H2 * math.sin(th)
    hip_z = h - H2 * math.cos(th)
    dxw, dzw = -hip_x, hip_z - SOLE
    lx = dxw * math.cos(th) + dzw * math.sin(th)
    lz = -dxw * math.sin(th) + dzw * math.cos(th)
    r = leg_ik_wide(lx, lz)
    if r is None:
        return None
    q1, q2 = r
    return q1, q2, -(th + q1 + q2), h


def drop_for_knee(lean_deg: float, knee: float):
    """二分下降量使膝角 = knee。两端都可能不可达，先粗扫出可行区间。"""
    prev = None
    for dd in np.arange(0.02, 0.62, 0.004):
        r = joints_wide(dd, lean_deg)
        if r is None:
            prev = None
            continue
        if r[1] >= knee:
            if prev is None:
                return None
            lo, hi = prev, dd
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                rr = joints_wide(mid, lean_deg)
                if rr is None or rr[1] < knee:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        prev = dd
    return None


# ---------------------------------------------------------------- 场景
def write_scene(force: bool = False) -> None:
    """从 elf3_squat.xml 派生 elf3_squat_x.xml：加 <exclude> 关掉大腿-小腿碰撞。"""
    if not SCENE_SRC.exists():
        base.write_scene()
    src = SCENE_SRC.read_text(encoding="utf-8")
    if "squat_x" in src and "<exclude" in src and not force:
        return
    block = ["  <contact>"]
    block += [f'    <exclude body1="{a}" body2="{b}"/>' for a, b in EXCLUDES]
    block.append("    <!-- 膝 149° 时大腿/小腿穿插 4 cm，按需求忽略它们的碰撞 -->")
    block.append("  </contact>")
    block = "\n".join(block) + "\n"
    if "<contact>" in src:                       # 已有就合并
        src = src.replace("<contact>\n", "<contact>\n" + block.split("<contact>\n")[1])
    else:
        src = src.replace("</mujoco>", block + "</mujoco>")
    SCENE.write_text(src, encoding="utf-8")
    print(f"[场景] 写入 {SCENE.name}（已排除大腿-小腿碰撞）")


# ---------------------------------------------------------------- 参数
@dataclass
class Params:
    knee_deg: float = 149.0
    min_margin: float = 0.035
    ank_limit: float = ANKLE_LIMIT
    arm: float = -1.10          # 大臂前摆（负=向前）
    elbow: float = 1.65         # 前臂往下折回（限位 1.658）
    arm_out: float = 0.22       # 大臂外展（越小越贴身）；0.45 那版张得太开
    reach_max: float = 0.40     # 手腕前伸上限（m），约束自动搜臂
    min_elbow_gap: float = 0.09 # 肘-膝/肘-大腿碰撞体最小间隙（m）
    min_torso_gap: float = 0.06 # 躯干-大腿碰撞体最小间隙（m），低于它算穿插
    max_hip_deg: float = 140.0  # 髋关节屈曲上限（deg），再大就是大腿压到肚子上
    arm_lead: float = 1.0     # 手臂前摆的提前量：<1 表示比腿先到位
    head: float = 14.0
    pre_frac: float = 0.35
    t_pre: float = 0.50
    t_desc: float = 1.55
    t_hold: float = 0.70
    do_return: bool = False
    t_rise: float = 1.65
    t_settle: float = 0.35
    dt: float = 0.02
    scene: str = str(SCENE)


_GAP = {}


def torso_gap(kin: Kin) -> float:
    """躯干与大腿碰撞体的最小距离（负数=已经穿插）。

    只看 self_contacts 是不够的：它只在"已经接触"时报，蹲到底前那几厘米的
    贴脸状态抓不到。用户看到的穿插，实测碰撞体间隙只剩 1.7 cm。
    """
    m = kin.m
    if not _GAP:
        GE = mujoco.mjtObj.mjOBJ_GEOM
        _GAP["torso"] = [mujoco.mj_name2id(m, GE, n)
                         for n in ("torso_collision", "torso_2_collision")]
        _GAP["thigh"] = [mujoco.mj_name2id(m, GE, n) for side in "lr"
                         for n in (f"{side}_hip_z_1_collision",
                                   f"{side}_hip_z_2_collision")]
    return min(float(mujoco.mj_geomDistance(m, kin.d, a, b, 2.0, None))
               for a in _GAP["torso"] for b in _GAP["thigh"])


def eval_pose(kin: Kin, drop, lean, p: Params):
    r = joints_wide(drop, lean)
    if r is None:
        return None
    q1, q2, qa, h = r
    q = STANCE.copy()
    for s in (IDX["l_hip_y"], IDX["r_hip_y"]):
        q[s] = q1; q[s + 3] = q2; q[s + 4] = qa
    for s, sgn in ((IDX["l_shoulder_y"], 1.0), (IDX["r_shoulder_y"], -1.0)):
        q[s] = p.arm
        q[s + 1] = sgn * p.arm_out
        q[s + 3] = p.elbow
    th = math.radians(lean)
    z = kin.place(q, np.array([0.0, math.radians(p.head)]), th)
    com = kin.com()
    mg, _ = margin_of(com[:2], convex_hull(kin.foot_points()))
    return dict(q1=q1, q2=q2, qa=qa, h=h, z=z, q=q, com=com, margin=mg,
                lean=lean, drop=drop)


def search_arm(kin: Kin, p: Params, drop: float, lean: float):
    """扫大臂前摆角，最大化蹲姿稳定裕度；跳过自碰撞和手插地的候选。

    前摆把两侧手臂（占整机质量 22%）的质心往前送，COM 从脚掌后部往脚心挪，
    裕度几乎翻倍。−0.2~−0.6 rad 区间手臂会扫过髋部，靠 arm_out=0.35 外展绕开。
    """
    wrist = mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, "l_wrist_z_link")
    GE = mujoco.mjtObj.mjOBJ_GEOM
    g_el = mujoco.mj_name2id(kin.m, GE, "l_elbow_y_collision")
    g_kn = mujoco.mj_name2id(kin.m, GE, "l_knee_y_collision")
    g_th = mujoco.mj_name2id(kin.m, GE, "l_hip_z_1_collision")
    best, best_arm = None, None
    for arm in np.arange(-1.60, 0.25, 0.02):
        pp = replace(p, arm=float(arm))
        r = eval_pose(kin, drop, lean, pp)
        if r is None or kin.self_contacts():
            continue
        w = kin.d.xpos[wrist]
        if float(w[2]) < 0.03:                      # 手别插地
            continue
        if float(w[0]) > p.reach_max:               # 手别伸太远
            continue
        gap_kn = float(mujoco.mj_geomDistance(kin.m, kin.d, g_el, g_kn, 1.0, None))
        gap_th = float(mujoco.mj_geomDistance(kin.m, kin.d, g_el, g_th, 1.0, None))
        if min(gap_kn, gap_th) < p.min_elbow_gap:   # 肘别蹭到膝/大腿
            continue
        if best is None or r["margin"] > best["margin"]:
            best, best_arm = r, float(arm)
    return best_arm, best


def search(kin: Kin, p: Params, knee: float):
    best, tried = None, []
    for lean in np.arange(-10.0, 26.0, 0.5):
        d = drop_for_knee(lean, knee)
        if d is None:
            continue
        r = eval_pose(kin, d, lean, p)
        if r is None:
            continue
        tried.append((lean, r))
        if r["qa"] < p.ank_limit - 1e-6:
            continue
        if r["margin"] < p.min_margin:
            continue
        if -r["q1"] > math.radians(p.max_hip_deg):
            continue                      # 髋折得太狠 → 大腿压肚子
        if torso_gap(kin) < p.min_torso_gap:
            continue                      # 躯干-大腿间隙不足
        if best is None or r["z"] < best["z"]:
            best = r
    return best, tried


# ---------------------------------------------------------------- 序列
def build(p: Params, kin: Kin, target: dict) -> dict:
    keys = [(0.0, 0.0)]
    t = p.t_pre
    keys.append((t, p.pre_frac))
    t += p.t_desc
    keys.append((t, 1.0))
    t += p.t_hold
    keys.append((t, 1.0))
    if p.do_return:
        t += p.t_rise
        keys.append((t, 0.0))
        t += p.t_settle
        keys.append((t, 0.0))

    n = int(round(t / p.dt)) + 1
    times = np.arange(n) * p.dt
    qpos = np.zeros((n, 38))
    qpos[:, 3] = 1.0
    pitches = np.zeros(n)

    for i, tt in enumerate(times):
        k = 0
        while k < len(keys) - 2 and tt >= keys[k + 1][0]:
            k += 1
        t0, t1 = keys[k][0], keys[k + 1][0]
        s = smootherstep(0.0 if t1 <= t0 else (tt - t0) / (t1 - t0))
        tail = keys[k][1] + s * (keys[k + 1][1] - keys[k][1])
        # 中间帧也用 IK 重解，脚踝才不滑
        r = joints_wide(target["drop"] * tail, target["lean"] * tail)
        q1, q2, qa, _ = r
        q = STANCE.copy()
        for s_ in (IDX["l_hip_y"], IDX["r_hip_y"]):
            q[s_] = q1; q[s_ + 3] = q2; q[s_ + 4] = qa
        # 手臂前摆比腿早一点到位（arm_lead<1）：下蹲中段 COM 还偏后，
        # 手臂先摆过去能把这半秒的裕度也顶住
        at = tail ** p.arm_lead
        for s_, sgn in ((IDX["l_shoulder_y"], 1.0), (IDX["r_shoulder_y"], -1.0)):
            q[s_] = 0.2 + (p.arm - 0.2) * at
            q[s_ + 1] = sgn * (0.2 + (p.arm_out - 0.2) * at)
            q[s_ + 3] = 0.6 + (p.elbow - 0.6) * at
        qpos[i, 7:36] = q
        qpos[i, 36:38] = [0.0, math.radians(p.head) * tail]
        pitches[i] = math.radians(target["lean"] * tail)

    margins = np.zeros(n)
    torso_z = np.zeros(n)
    coms = np.zeros((n, 3))
    self_col: list[str] = []
    for i in range(n):
        torso_z[i] = kin.place(qpos[i, 7:36], qpos[i, 36:38], pitches[i])
        qpos[i, 2] = torso_z[i]
        qpos[i, 3] = math.cos(pitches[i] / 2)
        qpos[i, 5] = math.sin(pitches[i] / 2)
        coms[i] = kin.com()
        margins[i], _ = margin_of(coms[i, :2], convex_hull(kin.foot_points()))
        for c in kin.self_contacts():
            if c not in self_col:
                self_col.append(c)

    return dict(time=times, qpos=qpos, pitch=pitches, margin=margins,
                torso_z=torso_z, com=coms, n=n, self_contacts=self_col)


def report(res, p, target, kin, knee, base_meta):
    i_sq = min(int(round((p.t_pre + p.t_desc) / p.dt)), res["n"] - 1)
    tz, mg = res["torso_z"], res["margin"]
    print("\n" + "=" * 74)
    print(f"ELF3 极限蹲 — 膝 {p.knee_deg:.0f}°（忽略大腿/小腿碰撞）")
    print("=" * 74)
    print(f"\n[蹲姿] 前倾 {target['lean']:+.1f}°   基座下降 {target['drop']:.3f} m")
    print(f"       躯干 z = {tz[i_sq]:.4f} m （站姿 {tz[0]:.4f}，下降 {tz[0]-tz[i_sq]:.4f} m，"
          f"{100*(1-tz[i_sq]/tz[0]):.1f}%）")
    print(f"       髋 {math.degrees(target['q1']):+.1f}°  膝 {math.degrees(target['q2']):+.1f}°  "
          f"踝 {math.degrees(target['qa']):+.1f}°（余 {math.degrees(target['qa']-p.ank_limit):.1f}°）")
    print(f"       髋高 {target['h']-H2*math.cos(math.radians(target['lean'])):.3f} m")
    print(f"       大臂前摆 {math.degrees(p.arm):+.0f}°  外展 {math.degrees(p.arm_out):.0f}°  "
          f"屈肘 {math.degrees(p.elbow):.0f}°")
    kin.place(target["q"], np.array([0.0, math.radians(p.head)]), math.radians(target["lean"]))
    print(f"       躯干-大腿间隙 {torso_gap(kin)*100:.1f} cm   "
          f"髋屈曲 {math.degrees(-target['q1']):.1f}°（上限 {p.max_hip_deg:.0f}°）")
    if base_meta:
        print(f"\n[对比] 标准版 {base_meta.get('squat_z', 0):.4f} m → 本版 "
              f"{tz[i_sq]:.4f} m（再低 {(base_meta.get('squat_z', 0)-tz[i_sq])*100:.1f} cm）")
    print(f"\n[静态稳定] COM 投影 = ({res['com'][i_sq,0]:+.4f}, {res['com'][i_sq,1]:+.4f}) m")
    print(f"           稳定裕度 = {mg[i_sq]*100:.2f} cm（脚长 {100*mg[i_sq]/0.22:.0f}%）")
    print(f"           全程最小 {mg.min()*100:.2f} cm @ t={res['time'][int(mg.argmin())]:.2f} s")
    print(f"           ZMP 判据：{'满足' if mg.min() > 0.02 else '不满足'}")

    kin.place(res["qpos"][i_sq, 7:36], res["qpos"][i_sq, 36:38], res["pitch"][i_sq])
    tau = np.abs(kin.joint_torque())
    limits = np.array([abs(kin.m.jnt_actfrcrange[
        mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_JOINT, nm)])[1] for nm in JOINT_NAMES])
    ratio = tau / np.maximum(limits, 1e-9)
    print(f"\n[关节力矩] 最大占用 {100*ratio.max():.1f}%  ({JOINT_NAMES[int(ratio.argmax())]})")
    for i in np.argsort(-ratio)[:3]:
        print(f"           {JOINT_NAMES[i]:20s} {tau[i]:7.2f} N·m  {100*ratio[i]:5.1f}% of {limits[i]:.0f}")
    lim = limit_violations(kin, res["qpos"])
    print(f"\n[关节限位] {'无越界' if not lim else lim}")
    print(f"[自碰撞] {'无' if not res['self_contacts'] else res['self_contacts']}")
    print(f"[说明] 大腿与小腿的碰撞已在场景里 exclude，穿插约 4 cm（按需求忽略）")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 极限蹲（膝 149°，忽略大腿/小腿碰撞）")
    ap.add_argument("--knee", type=float, default=149.0, help="目标膝角 deg")
    ap.add_argument("--min-margin", type=float, default=0.035)
    ap.add_argument("--arm", type=float, default=-1.10, help="蹲姿大臂前摆 rad（负=向前）")
    ap.add_argument("--elbow", type=float, default=1.65, help="前臂折回角（限位 1.658）")
    ap.add_argument("--arm-out", type=float, default=0.22,
                    help="大臂外展，越小越贴身（0.45 是上一版，张得开）")
    ap.add_argument("--reach-max", type=float, default=0.40,
                    help="手腕前伸上限 m，越小手臂越贴身")
    ap.add_argument("--min-elbow-gap", type=float, default=0.09,
                    help="肘-膝/肘-大腿碰撞体最小间隙 m；调大→手臂更往前摆")
    ap.add_argument("--min-torso-gap", type=float, default=0.06,
                    help="躯干-大腿碰撞体最小间隙 m")
    ap.add_argument("--max-hip", type=float, default=140.0,
                    help="髋关节屈曲上限 deg")
    ap.add_argument("--no-optimize-arm", dest="opt_arm", action="store_false",
                    help="关掉大臂前摆角自动搜索")
    ap.add_argument("--head", type=float, default=14.0)
    ap.add_argument("--return", dest="do_return", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    p = Params(knee_deg=args.knee, min_margin=args.min_margin, arm=args.arm,
               elbow=args.elbow, arm_out=args.arm_out, reach_max=args.reach_max,
               min_elbow_gap=args.min_elbow_gap, min_torso_gap=args.min_torso_gap,
               max_hip_deg=args.max_hip, head=args.head,
               do_return=args.do_return)

    write_scene()
    kin = Kin(Path(p.scene))
    knee = math.radians(p.knee_deg)

    best, tried = search(kin, p, knee)
    if best is None:
        alt = [r for _, r in tried if r["qa"] >= p.ank_limit]
        mg_best = max((r["margin"] for r in alt), default=-1)
        raise SystemExit(f"[错误] 膝 {p.knee_deg:.0f}° 找不到同时满足踝限位和裕度下限的姿态"
                         f"（最好裕度 {mg_best*100:.1f} cm）。试着调小 --min-margin")
    print(f"[搜索] 最优前倾 {best['lean']:+.1f}°，下降 {best['drop']:.3f} m，"
          f"躯干 {best['z']:.4f} m")

    if args.opt_arm:
        arm_val, r_arm = search_arm(kin, p, best["drop"], best["lean"])
        r_std = eval_pose(kin, best["drop"], best["lean"],
                          replace(p, arm=0.20, elbow=0.60, arm_out=0.20))
        if r_arm is not None:
            base_mg = r_std["margin"] if r_std else best["margin"]
            print(f"[手臂] 大臂前摆 {math.degrees(arm_val):+.0f}°（{arm_val:+.2f} rad）："
                  f"蹲姿裕度 {base_mg*100:.2f} → {r_arm['margin']*100:.2f} cm"
                  f"（手臂保持站姿角度时为 {base_mg*100:.2f}）")
            p.arm = arm_val
            best = r_arm

    base_meta = {}
    bt = DATA / "squat_stand_traj.npz"
    if bt.exists():
        try:
            base_meta = np.load(bt, allow_pickle=True)["meta"].item()
        except Exception:
            pass

    res = build(p, kin, best)
    report(res, p, best, kin, knee, base_meta)

    if args.check_only:
        return

    out = Path(args.out) if args.out else DEFAULT_OUT
    i_sq = min(int(round((p.t_pre + p.t_desc) / p.dt)), res["n"] - 1)
    meta = dict(drop=best["drop"], lean_deg=best["lean"], knee_deg=p.knee_deg,
                duration=float(res["time"][-1]), stance_z=float(res["torso_z"][0]),
                squat_z=float(res["torso_z"][i_sq]),
                squat_margin=float(res["margin"][i_sq]),
                min_margin=float(res["margin"].min()),
                hip_deg=float(math.degrees(best["q1"])),
                ankle_deg=float(math.degrees(best["qa"])),
                squat_com=[float(x) for x in res["com"][i_sq]],
                t_pre=p.t_pre, t_desc=p.t_desc, t_hold=p.t_hold, t_rise=p.t_rise,
                thigh_collision_ignored=True)
    np.savez(out, time=res["time"], qpos=res["qpos"], dt=p.dt,
             joint_names=JOINT_NAMES, meta=meta)
    print(f"\n[OK] 轨迹 -> {out}  ({res['n']} 帧, {out.stat().st_size/1024:.0f} KB)")
    print(f"     回放: python squat_replay.py --traj {out.name} --mode phys")
    if args.json:
        Path(args.json).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
