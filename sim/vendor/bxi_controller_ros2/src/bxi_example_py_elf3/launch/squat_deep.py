"""ELF3 极限折叠蹲（大腿贴小腿）+ 静态稳定校核。

`squat_stand.py` 的「更深」版本：把腿折到大腿与小腿刚好接触，躯干压到最低，
然后检查这个姿态还站不站得住。原版本一行不动。

==================== 深度是怎么被卡住的 ====================
折叠深度受三条边界夹击，脚本会自动找到交集：

1. **膝**：`l_hip_z`（大腿）与 `l_knee_y`（小腿）的碰撞胶囊半径各 5 cm，
   膝角到 **≈122°** 时两者贴合（间隙 2 mm），再折就穿插。
2. **踝**：`ankle_y` 限位 −50°。脚掌要平贴，就得 `ankle = −(θ+hip+knee)`，
   膝折到 122° 时这个值很容易冲到 −60° 以下。
3. **腿长**：脚踝必须留在世界原位（脚不能滑），髋到踝的距离被膝角锁死。

关键在于**前倾角 θ 是解药不是毒药**：θ 增大时臀部后移、脚踝相对髋前移，
小腿相对竖直的夹角反而变小，`ankle` 需求从 −61°（θ=0）一路回升到 −42°（θ=15°）。
所以越往前倾，越能既折死膝盖又保住踝关节。θ>20° 后腿长不够，脚踝够不到原位。

==================== 结果 ====================
标准版（squat_stand）: 躯干 0.7839 m，膝 107°，踝 −40°，裕度 9.1 cm
本版（deep）:          躯干 **0.704 m**，膝 122°（贴合），踝 −42°，裕度 8.6 cm

躯干再低 8 cm，膝盖折到物理极限，静态裕度仍有 8.6 cm —— 不需要手臂撑地。
（手臂也够不到：肩高 0.79 m、臂展 0.51 m，手最低只能到 0.28 m，差 26 cm。
 这个结论在脚本的 `--report-arm` 里会打印出来。）

==================== 用法 ====================
    python squat_deep.py                    # 自动搜最低姿态 + 生成轨迹
    python squat_deep.py --check-only       # 只算不落盘
    python squat_deep.py --lean 12          # 手动指定前倾
    python squat_deep.py --return           # 站→蹲→站 完整循环
    python squat_replay.py --traj squat_deep_traj.npz --gif out.gif
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import mujoco

import squat_stand as base
from squat_stand import (STANCE, IDX, JOINT_NAMES, Kin, convex_hull, margin_of,
                         squat_joints, smootherstep, limit_violations)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_squat.xml"
DEFAULT_OUT = DATA / "squat_deep_traj.npz"

GE = mujoco.mjtObj.mjOBJ_GEOM


# ---------------------------------------------------------------- 几何边界
def knee_contact_angle(kin: Kin, gap_mm: float = 2.0) -> float:
    """二分膝角，使大腿-小腿碰撞胶囊的最近距离 = gap_mm。返回弧度。"""
    m, d = kin.m, kin.d
    g_thigh = mujoco.mj_name2id(m, GE, "l_hip_z_1_collision")
    g_shin = mujoco.mj_name2id(m, GE, "l_knee_y_collision")

    def gap(q2):
        q = STANCE.copy()
        q1 = -q2 / 2.0
        for s in (IDX["l_hip_y"], IDX["r_hip_y"]):
            q[s] = q1; q[s + 3] = q2; q[s + 4] = -(q1 + q2)
        kin.place(q, np.zeros(2), 0.0)
        return float(mujoco.mj_geomDistance(m, d, g_thigh, g_shin, 1.0, None))

    lo, hi = math.radians(80), math.radians(150)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if gap(mid) > gap_mm / 1000.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def drop_for_knee(lean_deg: float, knee: float) -> float | None:
    """二分下降量，使 IK 解出的膝角等于 knee。不可达返回 None。

    drop 太小时腿「够不着」脚踝目标（lean 越大越明显）、太大时腿「折不动」，
    两头都会抛腿长不可达；所以先粗扫出第一个可行点，再在最后一步区间里二分。
    """
    prev = None
    for dd in np.arange(0.02, 0.47, 0.004):
        try:
            q2 = squat_joints(dd, lean_deg, 0.0)[1]
        except ValueError:
            prev = None
            continue
        if q2 >= knee:
            if prev is None:
                return None                     # 一进入可行域就已超过
            lo, hi = prev, dd
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if squat_joints(mid, lean_deg, 0.0)[1] < knee:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        prev = dd
    return None


def eval_pose(kin: Kin, drop: float, lean: float, arm: float, elbow: float,
              arm_out: float, head: float):
    q1, q2, qa, h = squat_joints(drop, lean, 0.0)
    q = STANCE.copy()
    for s in (IDX["l_hip_y"], IDX["r_hip_y"]):
        q[s] = q1; q[s + 3] = q2; q[s + 4] = qa
    for s, sgn in ((IDX["l_shoulder_y"], 1.0), (IDX["r_shoulder_y"], -1.0)):
        q[s] = arm
        q[s + 1] = sgn * arm_out
        q[s + 3] = elbow
    th = math.radians(lean)
    z = kin.place(q, np.array([0.0, math.radians(head)]), th)
    com = kin.com()
    hull = convex_hull(kin.foot_points())
    mg, _ = margin_of(com[:2], hull)
    return dict(q1=q1, q2=q2, qa=qa, h=h, z=z, q=q, com=com, margin=mg,
                lean=lean, drop=drop)


# ---------------------------------------------------------------- 参数
@dataclass
class Params:
    knee_gap_mm: float = 2.0     # 大腿-小腿目标间隙
    min_margin: float = 0.04     # 静态裕度下限
    lean: float | None = None    # None = 自动搜
    ank_limit: float = -0.87266  # ankle_y 下限
    arm: float = -0.50
    elbow: float = 0.80
    arm_out: float = 0.26
    head_pitch: float = 14.0
    pre_frac: float = 0.35
    t_pre: float = 0.45
    t_desc: float = 1.35
    t_hold: float = 0.70
    do_return: bool = False
    t_rise: float = 1.45
    t_settle: float = 0.35
    dt: float = 0.02
    scene: str = str(SCENE)


def search(kin: Kin, p: Params, knee: float) -> dict | None:
    """扫描前倾角，找满足裕度下限、躯干最低的那个。"""
    best = None
    for lean in np.arange(0.0, 26.0, 0.5):
        drop = drop_for_knee(lean, knee)
        if drop is None:
            continue
        try:
            r = eval_pose(kin, drop, lean, p.arm, p.elbow, p.arm_out, p.head_pitch)
        except ValueError:
            continue
        if r["qa"] < p.ank_limit - 1e-6:      # 踝超限，脚掌压不平
            continue
        if r["margin"] < p.min_margin:
            continue
        if best is None or r["z"] < best["z"]:
            best = r
    return best


# ---------------------------------------------------------------- 序列
def build(p: Params, kin: Kin, target: dict) -> dict:
    key_specs = [(0.0, 0.0)]
    t = p.t_pre
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

    def pose_at(tail: float):
        drop = target["drop"] * tail
        lean = target["lean"] * tail
        q1, q2, qa, _ = squat_joints(drop, lean, 0.0)
        q = STANCE.copy()
        for s in (IDX["l_hip_y"], IDX["r_hip_y"]):
            q[s] = q1; q[s + 3] = q2; q[s + 4] = qa
        for s, sgn in ((IDX["l_shoulder_y"], 1.0), (IDX["r_shoulder_y"], -1.0)):
            q[s] = 0.2 + (p.arm - 0.2) * tail
            q[s + 1] = sgn * (0.2 + (p.arm_out - 0.2) * tail)
            q[s + 3] = 0.6 + (p.elbow - 0.6) * tail
        head = np.array([0.0, math.radians(p.head_pitch) * tail])
        return q, math.radians(lean), head

    qpos = np.zeros((n, 38))
    qpos[:, 3] = 1.0
    pitches = np.zeros(n)
    for i, tt in enumerate(times):
        k = 0
        while k < len(key_specs) - 2 and tt >= key_specs[k + 1][0]:
            k += 1
        t0 = key_specs[k][0]
        t1 = key_specs[k + 1][0]
        s = smootherstep(0.0 if t1 <= t0 else (tt - t0) / (t1 - t0))
        tail = key_specs[k][1] + s * (key_specs[k + 1][1] - key_specs[k][1])
        q, pitch, head = pose_at(tail)
        qpos[i, 7:36] = q
        qpos[i, 36:38] = head
        pitches[i] = pitch

    margins = np.zeros(n)
    torso_z = np.zeros(n)
    coms = np.zeros((n, 3))
    gaps = np.zeros(n)
    g_thigh = mujoco.mj_name2id(kin.m, GE, "l_hip_z_1_collision")
    g_shin = mujoco.mj_name2id(kin.m, GE, "l_knee_y_collision")
    self_col: list[str] = []
    for i in range(n):
        torso_z[i] = kin.place(qpos[i, 7:36], qpos[i, 36:38], pitches[i])
        qpos[i, 2] = torso_z[i]
        qpos[i, 3] = math.cos(pitches[i] / 2)
        qpos[i, 5] = math.sin(pitches[i] / 2)
        coms[i] = kin.com()
        margins[i], _ = margin_of(coms[i, :2], convex_hull(kin.foot_points()))
        gaps[i] = float(mujoco.mj_geomDistance(kin.m, kin.d, g_thigh, g_shin, 1.0, None))
        for c in kin.self_contacts():
            if c not in self_col:
                self_col.append(c)

    return dict(time=times, qpos=qpos, pitch=pitches, margin=margins,
                torso_z=torso_z, com=coms, gap=gaps, n=n,
                self_contacts=self_col)


def report(res, p, kin, target, knee_deg, base_meta) -> None:
    i_sq = min(int(round((p.t_pre + p.t_desc) / p.dt)), res["n"] - 1)
    tz = res["torso_z"]
    qpos = res["qpos"]
    print("\n" + "=" * 74)
    print("ELF3 极限折叠蹲（大腿贴小腿）")
    print("=" * 74)

    print(f"\n[边界] 膝接触极限 {math.degrees(knee_deg):.1f}°（间隙 {p.knee_gap_mm:.1f} mm）")
    print(f"       踝限位 {math.degrees(p.ank_limit):.1f}°   腿长约束：脚踝必须留在原位")
    print(f"\n[蹲姿] 前倾 {target['lean']:.1f}°   基座下降 {target['drop']:.3f} m")
    print(f"       躯干 z = {tz[i_sq]:.4f} m  （站姿 {tz[0]:.4f}，下降 {tz[0]-tz[i_sq]:.4f} m，"
          f"{100*(1-tz[i_sq]/tz[0]):.1f}%）")
    print(f"       髋 {math.degrees(target['q1']):+.1f}°  膝 {math.degrees(target['q2']):+.1f}°  "
          f"踝 {math.degrees(target['qa']):+.1f}°（余 {math.degrees(target['qa']-p.ank_limit):.1f}°）")
    print(f"       大腿-小腿间隙 {res['gap'][i_sq]*1000:+.2f} mm")
    print(f"\n[对比] 标准版 squat_stand: 躯干 {base_meta.get('squat_z', 0):.4f} m "
          f"（本版再低 {(base_meta.get('squat_z', 0)-tz[i_sq])*100:.1f} cm）")

    print(f"\n[静态稳定] COM 投影 = ({res['com'][i_sq,0]:+.4f}, {res['com'][i_sq,1]:+.4f}) m")
    print(f"           稳定裕度 = {res['margin'][i_sq]*100:.2f} cm "
          f"（脚长 {100*res['margin'][i_sq]/0.22:.0f}%）")
    print(f"           全程最小 {res['margin'].min()*100:.2f} cm @ t="
          f"{res['time'][int(res['margin'].argmin())]:.2f} s")
    print(f"           ZMP 判据：{'满足' if res['margin'].min() > 0.02 else '不满足'}")

    kin.place(qpos[i_sq, 7:36], qpos[i_sq, 36:38], res["pitch"][i_sq])
    tau = np.abs(kin.joint_torque())
    limits = np.array([abs(kin.m.jnt_actfrcrange[
        mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_JOINT, nm)])[1] for nm in JOINT_NAMES])
    ratio = tau / np.maximum(limits, 1e-9)
    print(f"\n[关节力矩] 最大占用 {100*ratio.max():.1f}%  ({JOINT_NAMES[int(ratio.argmax())]})")
    for i in np.argsort(-ratio)[:3]:
        print(f"           {JOINT_NAMES[i]:20s} {tau[i]:7.2f} N·m  {100*ratio[i]:5.1f}% of {limits[i]:.0f}")

    lim = limit_violations(kin, qpos)
    print(f"\n[关节限位] {'无越界' if not lim else lim}")
    print(f"[自碰撞] {'无' if not res['self_contacts'] else res['self_contacts']}")

    # 手臂能不能撑地
    g_sh = mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, "l_shoulder_y_link")
    g_wr = mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, "l_wrist_z_link")
    sh = kin.d.xpos[g_sh].copy()
    q0 = qpos[i_sq, 7:36].copy()
    q0[IDX["l_shoulder_y"]] = 0.0
    q0[IDX["l_shoulder_x"]] = 0.0
    q0[IDX["l_elbow_y"]] = 1.66
    kin.place(q0, qpos[i_sq, 36:38], res["pitch"][i_sq])
    wr = kin.d.xpos[g_wr].copy()
    reach = sh[2] - wr[2]
    print(f"\n[手臂撑地] 肩 z={sh[2]:.3f} m，最大下垂 {reach:.3f} m → 手最低 "
          f"{sh[2]-reach:.3f} m")
    print(f"           {'可以撑地' if sh[2]-reach < 0.05 else '够不到地面（差 %.0f cm）' % ((sh[2]-reach)*100)}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 极限折叠蹲姿态生成器")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--lean", type=float, default=None, help="手动指定前倾 deg")
    ap.add_argument("--knee-gap", type=float, default=2.0, help="大腿小腿目标间隙 mm")
    ap.add_argument("--min-margin", type=float, default=0.04, help="静态裕度下限 m")
    ap.add_argument("--arm", type=float, default=-0.50)
    ap.add_argument("--elbow", type=float, default=0.80)
    ap.add_argument("--head", type=float, default=14.0)
    ap.add_argument("--return", dest="do_return", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    p = Params(knee_gap_mm=args.knee_gap, min_margin=args.min_margin, lean=args.lean,
               arm=args.arm, elbow=args.elbow, head_pitch=args.head,
               do_return=args.do_return, scene=args.scene)

    if not Path(p.scene).exists():
        base.write_scene()
    kin = Kin(Path(p.scene))

    knee = knee_contact_angle(kin, p.knee_gap_mm)
    print(f"[边界] 大腿-小腿接触极限膝角 = {math.degrees(knee):.1f}°")

    if p.lean is None:
        target = search(kin, p, knee)
        if target is None:
            # 报错时把"实际能到的最大裕度"也测出来，免得用户瞎试
            best_mg = max((eval_pose(kin, d, l, p.arm, p.elbow, p.arm_out, p.head_pitch)["margin"]
                           for l in np.arange(0.0, 26.0, 1.0)
                           for d in [drop_for_knee(l, knee)] if d is not None), default=0.0)
            raise SystemExit(f"[错误] 找不到满足裕度下限的姿态：本次能到的最好裕度 "
                            f"≈ {best_mg*100:.1f} cm，把 --min-margin 调小到它以下")
        print(f"[搜索] 最优前倾 {target['lean']:.1f}°，下降 {target['drop']:.3f} m，"
              f"躯干 {target['z']:.4f} m")
    else:
        drop = drop_for_knee(p.lean, knee)
        if drop is None:
            raise SystemExit(f"[错误] lean={p.lean} 时膝折不到 "
                             f"{math.degrees(knee):.1f}°，把前倾调小")
        target = eval_pose(kin, drop, p.lean, p.arm, p.elbow, p.arm_out, p.head_pitch)

    base_meta = {}
    bt = DATA / "squat_stand_traj.npz"
    if bt.exists():
        try:
            base_meta = np.load(bt, allow_pickle=True)["meta"].item()
        except Exception:
            base_meta = {}

    res = build(p, kin, target)
    report(res, p, kin, target, knee, base_meta)

    if args.check_only:
        return

    out = Path(args.out) if args.out else DEFAULT_OUT
    i_sq = min(int(round((p.t_pre + p.t_desc) / p.dt)), res["n"] - 1)
    meta = dict(
        drop=target["drop"], lean_deg=target["lean"], knee_gap_mm=p.knee_gap_mm,
        min_margin=p.min_margin, duration=float(res["time"][-1]),
        stance_z=float(res["torso_z"][0]), squat_z=float(res["torso_z"][i_sq]),
        squat_margin=float(res["margin"][i_sq]), min_margin_all=float(res["margin"].min()),
        min_margin_observed=float(res["margin"].min()),
        knee_deg=float(math.degrees(target["q2"])),
        hip_deg=float(math.degrees(target["q1"])),
        ankle_deg=float(math.degrees(target["qa"])),
        thigh_shin_gap_mm=float(res["gap"][i_sq] * 1000),
        t_pre=p.t_pre, t_desc=p.t_desc, t_hold=p.t_hold, t_rise=p.t_rise,
        squat_com=[float(x) for x in res["com"][i_sq]],
    )
    # 与 base 的字段名对齐，replay 的 HUD / 判定都能直接用
    meta["min_margin"] = meta["min_margin_observed"]
    np.savez(out, time=res["time"], qpos=res["qpos"], dt=p.dt,
             joint_names=JOINT_NAMES, meta=meta)
    print(f"\n[OK] 轨迹 -> {out}  ({res['n']} 帧, {out.stat().st_size/1024:.0f} KB)")
    print(f"     回放: python squat_replay.py --traj {out.name} --gif out.gif")

    if args.json:
        Path(args.json).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
