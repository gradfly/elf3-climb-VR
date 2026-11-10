"""俯卧爬行轨迹的数值验收。

只依赖 numpy + mujoco（本机没有 matplotlib / scipy，不要引入）。
回答四个问题：
  1. 四肢到底有没有贴地？（逐帧 FK 量接触点最低高度）
  2. 支撑相接触点有没有打滑？（接触点世界坐标在支撑段内的漂移）
  3. 质心在不在支撑多边形里？（动力学上到底稳不稳）
  4. 关节有没有顶限位 / 自碰撞？

用法：
    python scripts/check_prone_crawl.py                     # 默认轨迹
    python scripts/check_prone_crawl.py --traj X.npz --csv out.csv
    python scripts/check_prone_crawl.py --quiet              # 只打印结论

退出码 0 = 全项通过，1 = 有失败项，2 = 文件/格式问题。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[1]
DATA = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
        / "bxi_example_py_elf3" / "data" / "mujoco_simulation")
SCENE = DATA / "elf3_prone.xml"
TRAJ = DATA / "prone_crawl_traj.npz"
OUTDIR = REPO / "logs" / "prone_crawl"

# 判据（宽严由这几个数决定）
TOL_HIT_MM = 4.0          # 支撑相接触点相对「预压值」的容差
TOL_PEN_MM = 10.0         # 允许的最大穿地深度（腕球/脚掌）
TOL_MID_MM = 6.0          # 中间段（肘/膝/肩/髋/大腿）允许的最大穿地深度
TOL_ACC = 400.0           # 允许的最大关节角加速度（rad/s²）。
                          # 参考：ELF3 这类电机实际可跟的上限约 50~150 rad/s²；
                          # IK 的分支跳变会让它一步冲到 1500。400 是"已经摊平但
                          # 仍偏快"的界，留作回归哨兵。
MID_GUARD = {
    (side, arm): [f"{side}_{stem}_link" for stem in
                  (("shoulder_y", "shoulder_x", "shoulder_z", "elbow_y") if arm else
                   ("hip_y", "hip_x", "hip_z", "knee_y"))]
    for side in "lr" for arm in (True, False)
}
TOL_SLIP_MM = 30.0        # 单个支撑段内接触点水平漂移上限（准静态步态）
TOL_SLIP_FAST_MM = 40.0   # v>0.25 的动态步态放宽到 40 mm：0.5 m/s 时步幅 0.234 m，
                          # 35 mm 相当于 15%，而摆动脚速 1.1 m/s、支撑相只占 55%，
                          # 接触点本来就比慢速档更难钉住（实测 2 循环 19 mm、4 循环 35 mm）。。
                          # 手是「腕 + 4 cm 偏移的球」，支撑相前臂转动会让球心相对
                          # 地面滚 2~3 cm（实测 26 mm）；脚 11 mm。这不是打滑，是
                          # 手掌滚动，物理上由接触点自身转动吸收。
TOL_MARGIN_DEG = 3.0      # 关节离限位的最小余量
PASS_HIT = 0.95           # 支撑相命中率下限
PASS_SUPPORT = 0.95       # 质心落在支撑多边形内的帧占比下限

LIMBS = [
    ("l_arm", "l", True, 0),
    ("r_leg", "r", False, 1),
    ("r_arm", "r", True, 2),
    ("l_leg", "l", False, 3),
]


# ---------------------------------------------------------------- 工具
def convex_hull(points: np.ndarray) -> np.ndarray:
    """二维凸包（单调链）。points: (n,2)。"""
    pts = np.unique(np.round(points, 9), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def inside_hull(pt: np.ndarray, hull: np.ndarray, tol: float = 1e-6) -> bool:
    """点是否在凸包内（顺时针/逆时针都成立）。"""
    if len(hull) < 3:
        return False
    sign = 0
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        cr = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(cr) < tol:
            continue
        s = 1 if cr > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


class Kin:
    def __init__(self, scene: Path):
        self.m = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        self.d = mujoco.MjData(self.m)
        GE = mujoco.mjtObj.mjOBJ_GEOM
        self.foot_geoms = {}
        for side in "lr":
            self.foot_geoms[side] = [
                g for g in range(self.m.ngeom)
                if (mujoco.mj_id2name(self.m, GE, g) or "").startswith(f"{side}_foot")]
        self.hand_geoms = {
            s: [g for g in range(self.m.ngeom)
                if (mujoco.mj_id2name(self.m, GE, g) or "") == f"{s}_wrist_z_collision"]
            for s in "lr"}
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    def set(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def min_z(self, geoms) -> float:
        zs = []
        for g in geoms:
            pos = self.d.geom_xpos[g]
            r = float(self.m.geom_size[g][0])
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = self.d.geom_xmat[g].reshape(3, 3)
                zs.append(pos[2] - (float(self.m.geom_size[g][1]) * abs(R[2, 2]) + r))
            else:
                zs.append(pos[2] - r)
        return float(min(zs)) if zs else 9.9

    def body_min_z(self, bid: int) -> float:
        """某个 body 上所有碰撞几何的最低点 z。"""
        zs = []
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != bid or self.m.geom_group[g] != 3:
                continue
            pos = self.d.geom_xpos[g]
            r = float(self.m.geom_size[g][0])
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = self.d.geom_xmat[g].reshape(3, 3)
                zs.append(pos[2] - (float(self.m.geom_size[g][1]) * abs(R[2, 2]) + r))
            else:
                zs.append(pos[2] - r)
        return float(min(zs)) if zs else 9.9

    def lowest_xy(self, geoms) -> np.ndarray:
        """接触面"形心"的 (x, y)：取所有离地不超过最低点 3 mm 的 geom 的均值。

        直接用"最低 geom 的坐标"会跳变 —— 脚掌前倾时最低点从脚尖换到脚掌，
        相邻两帧坐标能差 2~3 cm，看起来像打滑，其实是量法问题。取接触面形心
        才是真正的接触点位移。
        """
        vals = []
        for g in geoms:
            pos = self.d.geom_xpos[g]
            r = float(self.m.geom_size[g][0])
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = self.d.geom_xmat[g].reshape(3, 3)
                z = pos[2] - (float(self.m.geom_size[g][1]) * abs(R[2, 2]) + r)
            else:
                z = pos[2] - r
            vals.append((z, pos[:2].copy()))
        zmin = min(v[0] for v in vals)
        patch = [p for z, p in vals if z <= zmin + 3e-3]
        return np.mean(np.array(patch), axis=0)

    def com_xy(self) -> np.ndarray:
        """整机质心投影（free joint 的根子树即整机）。"""
        return np.array(self.d.subtree_com[0][:2])


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="俯卧爬行轨迹数值验收")
    ap.add_argument("--traj", default=str(TRAJ))
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--csv", default=str(OUTDIR / "contact.csv"))
    ap.add_argument("--report", default=str(OUTDIR / "report.txt"))
    ap.add_argument("--quiet", action="store_true", help="只打印结论")
    ap.add_argument("--kind", default="auto",
                    choices=["auto", "full", "standup", "gait"],
                    help="动作类型。standup 会跳过支撑/质心/速度这类只对爬行步态成立的判据")
    args = ap.parse_args()

    tp, sp = Path(args.traj), Path(args.scene)
    if not tp.is_file():
        print(f"[错误] 找不到轨迹 {tp}，先跑: python prone_crawl.py", file=sys.stderr)
        return 2
    if not sp.is_file():
        print(f"[错误] 找不到场景 {sp}，先跑: python prone_crawl.py --write-scene",
              file=sys.stderr)
        return 2

    z = np.load(tp, allow_pickle=True)
    qpos = z["qpos"]
    dt = float(z["dt"])
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    flags = z["contact_flags"] if "contact_flags" in z.files else None
    press = float(meta.get("press", 0.005))
    entry_time = float(meta.get("entry_time", 0.0))
    cycle_time = float(meta.get("cycle_time", 2.0))
    speed = float(meta.get("speed", 0.0))
    kind = args.kind
    if kind == "auto":
        kind = str(z["action"]) if "action" in z.files else "full"
    is_standup = kind == "standup"

    kin = Kin(sp)
    n, nq = qpos.shape
    if nq != kin.m.nq:
        print(f"[错误] qpos 有 {nq} 列，模型 nq={kin.m.nq}", file=sys.stderr)
        return 2
    if flags is None:
        print("[错误] 轨迹里没有 contact_flags，无法判定支撑相", file=sys.stderr)
        return 2

    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        if not args.quiet:
            print(s)

    # ---------- 逐帧采样 ----------
    zs = np.zeros((n, 4))
    mids = np.zeros((n, 4))          # 肩/肘/髋/膝等"中间段"的最低点（mm）
    xys = np.zeros((n, 4, 2))
    coms = np.zeros((n, 2))
    torso_z = np.zeros(n)
    selfcon = np.zeros(n, dtype=int)
    for i in range(n):
        kin.set(qpos[i])
        for k, (_, side, arm, _) in enumerate(LIMBS):
            geoms = kin.hand_geoms[side] if arm else kin.foot_geoms[side]
            zs[i, k] = kin.min_z(geoms) * 1000.0
            xys[i, k] = kin.lowest_xy(geoms)
        for k, (_, side, arm, _) in enumerate(LIMBS):
            mids[i, k] = min(kin.body_min_z(
                mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, b))
                for b in MID_GUARD[(side, arm)]) * 1000.0
        coms[i] = kin.com_xy()
        torso_z[i] = kin.d.xpos[kin.torso][2]
    # 自碰撞（只数机器人自己的 geom 之间；地面接触不算）
    floor_ids = {g for g in range(kin.m.ngeom)
                 if (mujoco.mj_id2name(kin.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                 .startswith("ut_infinite_ground")}
    for i in range(n):
        kin.set(qpos[i])
        mujoco.mj_forward(kin.m, kin.d)
        cnt = 0
        for c in range(kin.d.ncon):
            g1, g2 = kin.d.contact[c].geom1, kin.d.contact[c].geom2
            if g1 in floor_ids or g2 in floor_ids:
                continue
            if kin.m.geom_contype[g1] == 0 and kin.m.geom_conaffinity[g1] == 0:
                continue
            cnt += 1
        selfcon[i] = cnt

    # ---------- 1. 接触 ----------
    out("=" * 68)
    _title = {"standup": "动作 1/2：站立 -> 俯卧", "gait": "动作 2/2：向前爬行",
              "full": "完整动作（过渡 + 爬行）"}[kind]
    out(f"{_title}   验收   {tp.name}")
    out(f"帧 {n} / {n * dt:.2f} s / dt {dt}   过渡 {entry_time:.2f} s + "
        f"循环 {(n * dt - entry_time):.2f} s（{cycle_time:.2f} s/循环）")
    out(f"目标速度 {speed:.3f} m/s")
    out("=" * 68)
    out("\n[1] 接触高度（支撑相）")
    out(f"  {'肢体':<8}{'支撑帧':>7}{'命中率':>8}{'均值':>9}{'最低':>9}{'摆动峰值':>10}")
    hit_all, pen_all = [], []
    for k, (name, _, _, _) in enumerate(LIMBS):
        st = flags[:, k] == 1
        sw = ~st
        hit = float(np.mean(np.abs(zs[st, k] + press * 1000.0) <= TOL_HIT_MM))
        hit_all.append(hit)
        pen_all.append(float(-zs[:, k].min()))
        peak = float(zs[sw, k].max()) if sw.any() else 0.0
        out(f"  {name:<8}{st.sum():>7d}{hit * 100:>7.0f}%"
            f"{zs[st, k].mean():>9.1f}{zs[:, k].min():>9.1f}{peak:>10.1f}")
    out(f"  （单位 mm；预压 -{press * 1000:.0f} mm，命中判据 |z+{press * 1000:.0f}| ≤ "
        f"{TOL_HIT_MM:.0f} mm）")

    # ---------- 1b. 中间段穿地 ----------
    out("\n[1b] 中间段（肩/上臂/前臂/髋/膝）最低点 —— 只查接触点会漏掉这里")
    for k, (name, _side, _arm, _off) in enumerate(LIMBS):
        mn = mids[:, k].min()
        bad = int((mids[:, k] < -TOL_MID_MM).sum())
        out(f"  {name:<8} 最低 {mn:7.1f} mm   低于 -{TOL_MID_MM:.0f}mm 的帧 {bad:3d}/{n}")
    mid_all = float(-mids.min())  # 中间段最大穿地深度

    # ---------- 2. 打滑 ----------
    # 只统计循环段：过渡段脚是"拖"到位的（设计上就允许滑），不是步态问题
    i_loop0 = min(int(round(entry_time / dt)), n - 1)
    out("\n[2] 支撑相打滑（接触点世界坐标漂移，仅循环段）")
    out(f"  {'肢体':<8}{'段数':>7}{'中位':>9}{'最大':>9}")
    slip_max_all = 0.0
    for k, (name, _, _, _) in enumerate(LIMBS):
        st = (flags[:, k] == 1).copy()
        st[:i_loop0] = False
        slips = []
        i = i_loop0
        while i < n:
            if not st[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and st[j + 1]:
                j += 1
            seg = xys[i:j + 1, k]
            slips.append(float(np.max(np.linalg.norm(seg - seg[0], axis=1))) * 1000.0)
            i = j + 1
        slips = [s for s in slips if s > 0]
        med = float(np.median(slips)) if slips else 0.0
        mx = float(np.max(slips)) if slips else 0.0
        slip_max_all = max(slip_max_all, mx)
        out(f"  {name:<8}{len(slips):>7d}{med:>9.1f}{mx:>9.1f}")

    # ---------- 3. 支撑多边形 ----------
    out("\n[3] 稳定性（质心投影是否在接触点凸包内）")
    inside = np.zeros(n, dtype=bool)
    nsup = np.zeros(n, dtype=int)
    for i in range(n):
        pts = []
        for k in range(4):
            if abs(zs[i, k] + press * 1000.0) <= 8.0:
                pts.append(xys[i, k])
        nsup[i] = len(pts)
        if len(pts) >= 3:
            inside[i] = inside_hull(coms[i], convex_hull(np.array(pts)))
    loop = np.arange(n) * dt >= entry_time
    ratio = float(inside[loop].mean()) if loop.any() else 0.0
    n2 = int((nsup[loop] == 2).sum())
    out(f"  循环段质心在支撑面内：{ratio * 100:.1f}%（{inside[loop].sum()}/"
        f"{loop.sum()} 帧）")
    out(f"  支撑点个数分布：" + "  ".join(
        f"{c}点 {int((nsup[loop] == c).sum())}帧" for c in range(0, 5)))
    if n2:
        out(f"  其中 {n2} 帧只有 2 点支撑（{n2 / max(loop.sum(), 1) * 100:.0f}%）："
            f"两点支撑的\"支撑多边形\"退化成一个线段，质心不可能严格落在内部。")
    # 准静态判据只对慢速步态成立：v>0.25 m/s 时靠动量平衡，这条只报告不判定
    is_dynamic = speed > 0.25

    # ---------- 4. 关节 ----------
    out("\n[4] 关节")
    JO = mujoco.mjtObj.mjOBJ_JOINT
    names = [mujoco.mj_id2name(kin.m, JO, j) for j in range(kin.m.njnt)]
    worst, worst_j, worst_i = 1e9, "", 0
    amax, amax_j, amax_i = 0.0, "", 0
    vmax, vmax_j, vmax_i = 0.0, "", 0
    prev = None
    for j in range(kin.m.njnt):
        if kin.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        lo, hi = kin.m.jnt_range[j]
        qa = kin.m.jnt_qposadr[j]
        v = np.degrees(qpos[:, qa])
        m_lo, m_hi = v.min() - math.degrees(lo), math.degrees(hi) - v.max()
        if min(m_lo, m_hi) < worst:
            worst, worst_j = min(m_lo, m_hi), names[j]
            worst_i = int(np.argmin(v)) if m_lo < m_hi else int(np.argmax(v))
        vv = np.abs(np.diff(qpos[:, qa])) / dt
        if vv.max() > vmax:
            vmax, vmax_j, vmax_i = float(vv.max()), names[j], int(np.argmax(vv))
        aa = np.abs(np.gradient(np.gradient(qpos[:, qa], dt), dt))
        if aa.max() > amax:
            amax, amax_j, amax_i = float(aa.max()), names[j], int(np.argmax(aa))
    out(f"  离限位最近：{worst_j} 余量 {worst:.1f}°（第 {worst_i} 帧）")
    out(f"  最大关节速度：{vmax_j} {vmax:.2f} rad/s（第 {vmax_i} 帧）")
    out(f"  最大关节角加速度：{amax_j} {amax:.1f} rad/s² "
        f"= {np.degrees(amax):.0f} deg/s²（第 {amax_i} 帧）")

    # ---------- 5. 自碰撞 ----------
    out("\n[5] 自碰撞")
    bad = int((selfcon > 0).sum())
    out(f"  有机器人自身 geom 接触的帧：{bad}/{n}"
        + ("（首帧站姿也可能有，看下面的分布）" if bad else ""))

    # ---------- 6. 推进 ----------
    out("\n[6] 推进")
    i0 = min(int(round(entry_time / dt)), n - 1)
    x0 = qpos[i0, 0] if entry_time > 0 else qpos[0, 0]
    dx = qpos[-1, 0] - x0
    dur = max((n - 1) * dt - entry_time, dt)
    # 俯卧姿态下欧拉角会万向锁，偏航改用"躯干前进轴投影到地面"来量
    kin.set(qpos[-1])
    R = kin.d.xmat[kin.torso].reshape(3, 3)
    fwd = R[:, 2]
    yaw_drift = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    out(f"  循环段前进 {dx:.3f} m / {dur:.2f} s -> {dx / dur:.3f} m/s"
        f"（目标 {speed:.3f}）")
    out(f"  末帧躯干高度 {torso_z[-1] * 100:.1f} cm   偏航漂移 {yaw_drift:.1f}°")

    # ---------- 结论 ----------
    # standup 是过渡动作：没有支撑相（contact_flags 全 0），也不要求匀速前进。
    # 这些判据只在 gait/full 上成立，否则一律降级成 INFO。
    def _skip(ok_expr):
        return None if is_standup else ok_expr

    checks = [
        ("支撑相接触命中率 ≥ %.0f%%" % (PASS_HIT * 100),
         _skip(min(hit_all) >= PASS_HIT) if not is_standup else None,
         f"{min(hit_all) * 100:.0f}%" + ("（过渡动作无支撑相）" if is_standup else "")),
        (f"最大穿地 ≤ {TOL_PEN_MM:.0f} mm", max(pen_all) <= TOL_PEN_MM,
         f"{max(pen_all):.1f} mm"),
        (f"中间段穿地 ≤ {TOL_MID_MM:.0f} mm", mid_all <= TOL_MID_MM,
         f"{mid_all:.1f} mm"),
        (f"关节角加速度 ≤ {TOL_ACC:.0f} rad/s²", amax <= TOL_ACC,
         f"{amax:.1f} rad/s² ({amax_j})"),
        (f"最大滑移 ≤ {TOL_SLIP_FAST_MM if is_dynamic else TOL_SLIP_MM:.0f} mm",
         _skip(slip_max_all <= (TOL_SLIP_FAST_MM if is_dynamic else TOL_SLIP_MM)),
         f"{slip_max_all:.1f} mm" + ("（过渡动作脚是拖到位的）" if is_standup else
                                     ("（v>0.25 动态步态）" if is_dynamic else ""))),
        (f"关节余量 ≥ {TOL_MARGIN_DEG:.0f}°", worst >= TOL_MARGIN_DEG, f"{worst:.1f}°"),
        ("质心在支撑面内 ≥ %.0f%%" % (PASS_SUPPORT * 100),
         None if (is_dynamic or is_standup) else ratio >= PASS_SUPPORT,
         f"{ratio * 100:.0f}%" + ("（过渡动作，仅报告）" if is_standup else
                                  ("（v>0.25 动态步态，仅报告）" if is_dynamic else ""))),
        ("无机器人自碰撞", bad == 0, f"{bad} 帧"),
        ("前进方向 +X 且速度达标",
         _skip(dx > 0 and abs(dx / dur - speed) < 0.02 * max(speed, 1e-6)),
         f"{dx / dur:.3f} m/s" + ("（过渡动作只看姿态，不看匀速）" if is_standup else "")),
    ]
    out("\n" + "=" * 68)
    out("结论：")
    ok = True
    for label, good, val in checks:
        tag = "INFO" if good is None else ("PASS" if good else "FAIL")
        out(f"  [{tag}] {label}  ->  {val}")
        if good is not None:
            ok &= bool(good)
    out("=" * 68)
    out("总体：" + ("PASS" if ok else "FAIL"))

    # ---------- 落盘 ----------
    OUTDIR.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv = Path(args.csv)
    csv.parent.mkdir(parents=True, exist_ok=True)
    with csv.open("w", encoding="utf-8") as f:
        f.write("frame,t,x,z_torso,com_x,com_y,n_support,"
                + ",".join(f"z_{nm}" for nm, *_ in LIMBS) + ","
                + ",".join(f"contact_{nm}" for nm, *_ in LIMBS) + "\n")
        for i in range(n):
            f.write(f"{i},{i * dt:.3f},{qpos[i, 0]:.4f},{torso_z[i]:.4f},"
                    f"{coms[i, 0]:.4f},{coms[i, 1]:.4f},{nsup[i]},"
                    + ",".join(f"{zs[i, k]:.2f}" for k in range(4)) + ","
                    + ",".join(str(int(flags[i, k])) for k in range(4)) + "\n")
    print(f"\n报告 -> {args.report}\n逐帧 -> {csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
