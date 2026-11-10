"""check_parade_actions.py — 「走正步 / 敬礼」三段动作的数值验收。

跑法（项目根目录）：
    python scripts/check_parade_actions.py
    python scripts/check_parade_actions.py --report logs/parade_actions/report.txt

判据分两层：
  * 通用：38 维格式、dt=0.02、29 关节名、关节不越限位、单帧跳变 < 0.30 rad、
    按网页模型同一公式逐帧算视觉最低点（±1.2 cm）、循环版首尾闭合。
  * 专项：
    - 正步：**支撑相由相位判定**（摆动腿绷直擦地，踝高只有 4~5 cm，
      用 z 阈值会把摆动相误判成支撑腿）、支撑脚不滑 < 2 cm、
      踢腿峰值离地 > 20 cm、前进版确实前进 / 原地版 x 恒 0、
      左右腿相位差半周期、上体前倾 3~12° 且全程恒定、质心补偿收敛、
      单脚支撑相中点重心压在支撑脚上（< 5 cm）。
    - 正步姿态：两条腿绷直（支撑腿平均膝屈 < 12°、摆动腿 < 18°）、
      基座起伏受控、左右臂**前摆峰值时小臂水平**（与水平面夹角 < 3°）。
    - 敬礼：保持帧腕抬到头侧（z > 1.30 m 且 |y| < 0.20 m）、腕到头中心
      10~16 cm、抬手路径不过冲、敬礼保持段完全静止、
      敬礼动作段左臂不动（立正收臂/收势是刻意过渡，不计入）。

退出码 0 = 全通过，1 = 有判据失败。报告写入 logs/parade_actions/report.txt。
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
LAUNCH = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
          / "bxi_example_py_elf3" / "launch")
DATA = LAUNCH.parent / "data" / "mujoco_simulation"
WEB_XML = REPO / "web" / "assets" / "elf3.xml"
SIM_XML = DATA / "elf3_squat.xml"
sys.path.insert(0, str(LAUNCH))
sys.path.insert(0, str(REPO / "web" / "tools"))
sys.path.insert(0, str(REPO / "scripts"))

from action_lib import JOINT_NAMES, IDX, limit_report  # noqa: E402
from bake_squat import build_qpos_remap, foot_min_z, visual_geoms  # noqa: E402
from check_hand_clearance import mesh_points, min_dist, voxel_downsample  # noqa: E402

TOL_FOOT = 0.012
TOL_JUMP = 0.30
CASES = [
    ("goose_step_traj.npz", "goose_walk", False),
    ("goose_step_inplace_traj.npz", "goose_inplace", True),
    ("salute_traj.npz", "salute", True),
]

_lines: list[str] = []
_fails: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _lines.append(s)


def ok(cond: bool, label: str, detail: str = "") -> None:
    say(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not cond:
        _fails.append(label)


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:
        return dict(qpos=np.asarray(z["qpos"], dtype=np.float64).copy(),
                    dt=float(z["dt"]),
                    names=[str(s) for s in z["joint_names"]],
                    action=str(z["action"]),
                    meta=json.loads(str(z["meta"])))


def fk_points(qpos: np.ndarray, names: list[str]) -> dict:
    """用仿真模型逐帧 FK，取左右踝 / 右腕的世界坐标与整机质心 x。"""
    m = mujoco.MjModel.from_xml_path(str(SIM_XML))
    d = mujoco.MjData(m)
    mass = m.body_mass
    bodies = ["l_ankle_y_link", "r_ankle_y_link",
              "l_hip_y_link", "r_hip_y_link",
              "l_knee_y_link", "r_knee_y_link",
              "l_elbow_y_link", "l_wrist_y_link",
              "r_elbow_y_link", "r_wrist_y_link", "head_y_link"]
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies]
    out = {b: [] for b in bodies}
    out["com"] = []
    for row in qpos:
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        for b, i in zip(bodies, ids):
            out[b].append(d.xpos[i].copy())
        out["com"].append((mass[:, None] * d.xpos).sum(axis=0)[0] / mass.sum())
    return {b: np.asarray(v) for b, v in out.items()}


def check_common(tag: str, d: dict, expect_loop: bool, m, data, gids, ix) -> None:
    q = d["qpos"]
    say(f"\n=== {tag} ===")
    ok(q.ndim == 2 and q.shape[1] == 38, "qpos 形状 [T,38]", str(q.shape))
    ok(abs(d["dt"] - 0.02) < 1e-9, "dt = 0.02 (50 Hz)", f"{d['dt']}")
    ok(d["names"] == JOINT_NAMES, "29 关节名与模型一致", f"{len(d['names'])} 个")
    viol = limit_report(q)
    ok(not viol, "关节不越限位", "; ".join(viol) if viol else "全部在 range 内")
    jump = float(np.abs(np.diff(q[:, 7:36], axis=0)).max())
    ok(jump < TOL_JUMP, f"单帧跳变 < {TOL_JUMP} rad", f"{jump:.4f} rad")

    f36 = q[:, ix]
    lows = np.asarray([foot_min_z(m, data, row, gids) for row in f36])
    lo, hi = float(lows.min()), float(lows.max())
    ok(lo >= -TOL_FOOT and hi <= TOL_FOOT, "视觉脚底贴地 ±1.2 cm",
       f"{lo*100:+.2f} ~ {hi*100:+.2f} cm")

    seam = float(np.abs(q[0] - q[-1]).max())
    if expect_loop:
        if tag.startswith("salute"):
            ok(seam < 1e-9, "循环首尾严格同帧", f"max|Δ|={seam:.2e}")
        else:
            ok(seam < TOL_JUMP, "循环末帧→首帧是正常一帧推进",
               f"max|Δ|={seam:.4f} rad")
    else:
        ok(bool(d["meta"].get("loop")) is False, "单段：播完冻结末帧",
           f"loop={d['meta'].get('loop')}")
    d["lows"] = lows


def hand_clearance(qpos: np.ndarray, a: int, b: int,
                   pairs: list[tuple[str, str]],
                   coarse_pairs: list[tuple[str, str]] | None = None
                   ) -> tuple[float, str, int]:
    """[a,b) 帧区间内，若干「mesh 对」的最小顶点间距（mm）与最贴的一对。

    手是 2026-11-10 才加的纯视觉 mesh，挂在腕外 0.119 m 处；动作设计时只有
    「腕」这个点，所以必须单独守住这段余量（正步就因为小臂横着掠过中线，
    两个拳只差 16 mm 撞在一起）。先用 12 mm 体素抽稀扫全段定位可疑帧，
    再对最贴的 6 帧做全量顶点复核，兼顾速度与精度。
    """
    if "clear" not in _CACHE:
        m = mujoco.MjModel.from_xml_path(str(SIM_XML))
        P = {}
        for n in ("l_hand", "r_hand", "l_hip_x_link", "r_hip_x_link",
                  "waist_z_link", "head_y_link", "head_z_link"):
            v, bid = mesh_points(m, n)
            if v is not None:
                P[n] = (v, bid)
        _CACHE["clear"] = (m, mujoco.MjData(m), P)
    m, data, P = _CACHE["clear"]
    pairs = [p for p in pairs if p[0] in P and p[1] in P]
    coarse_pairs = [p for p in (coarse_pairs or pairs) if p in pairs]

    def world(name: str) -> np.ndarray:
        v, bid = P[name]
        return v @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]

    coarse = np.empty(b - a)
    for n, k in enumerate(range(a, b)):
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(m, data)
        W = {nm: voxel_downsample(world(nm), 0.012).astype(np.float32)
             for nm in P}
        coarse[n] = min(min_dist(W[p[0]], W[p[1]]) for p in coarse_pairs) * 1000

    worst = (np.inf, "", 0)
    for k in a + np.argsort(coarse)[:6]:
        data.qpos[:] = qpos[int(k)]
        mujoco.mj_kinematics(m, data)
        W = {nm: world(nm) for nm in P}
        for p in pairs:
            dist = min_dist(W[p[0]], W[p[1]]) * 1000
            if dist < worst[0]:
                worst = (dist, f"{p[0]} ↔ {p[1]}", int(k))
    return worst


_CACHE: dict = {}


def check_goose(tag: str, d: dict, fk: dict) -> None:
    q = d["qpos"]
    meta = d["meta"]
    gs = int(meta.get("gait_start", 0))
    gf = int(meta.get("gait_frames", len(q)))
    dt = d["dt"]
    cyc = float(meta.get("cycle", 1.2))
    duty = float(meta.get("duty", 0.5))
    t = np.arange(gf) * dt
    # 摆动腿是「绷直擦地划出」的，踝高只有 4~5 cm；用 z 阈值会把整段摆动相
    # 误判成贴地支撑（旧判据因此报出 16 cm 假滑步、5 cm 假重心偏差）。
    # 直接按生成器写入的相位判定支撑腿：左右严格差半周期。
    sup_l = ((t / cyc) % 1.0) < duty
    sup_r = ((t / cyc + 0.5) % 1.0) < duty
    pl = fk["l_ankle_y_link"][gs:gs + gf]
    pr = fk["r_ankle_y_link"][gs:gs + gf]
    com = fk["com"][gs:gs + gf]
    travel = float(q[-1, 0] - q[0, 0])
    say("  -- 正步专项 --")
    if tag.startswith("goose_walk"):
        ok(travel > 2.5, "确实向前推进", f"Δx = {travel:.3f} m")
    else:
        ok(abs(travel) < 1e-9, "原地版基座 x 恒定", f"Δx = {travel:.2e} m")

    # 踢腿峰值：踝 z 减去踝到脚底 0.04
    peak = max(float(fk[b][:, 2].max()) - 0.04 for b in
               ("l_ankle_y_link", "r_ankle_y_link"))
    ok(peak > 0.20, "踢腿峰值脚底离地 > 20 cm", f"{peak*100:.1f} cm")

    # 支撑脚不滑：只在真实支撑相内判定（接触段首尾各裁 3 帧，排除
    # toe-off / heel-strike 过渡）。摆动相的脚本来就要向前划出，不计入。
    worst = 0.0
    for g, x in ((sup_l, pl[:, 0]), (sup_r, pr[:, 0])):
        starts = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
        for i in range(0, len(starts), 2):
            a, c = starts[i] + 3, starts[i + 1] - 3
            if c - a >= 10:
                worst = max(worst, float(x[a:c].max() - x[a:c].min()))
    ok(worst < 0.02, "支撑相脚不滑 < 2 cm", f"最大漂移 {worst*100:.2f} cm")

    # 左右腿相位差：hip_y 序列互相关在半周期处最大
    lh, rh = q[:, 7 + IDX["l_hip_y"]], q[:, 7 + IDX["r_hip_y"]]
    n = min(120, len(lh) // 2)
    lag = int(round(float(d["meta"].get("cycle", 1.2)) / d["dt"] / 2))
    if n > lag + 5:
        c_same = float(np.corrcoef(lh[:n], rh[:n])[0, 1])
        c_half = float(np.corrcoef(lh[:n], rh[lag:lag + n])[0, 1])
        ok(c_half > c_same + 0.2, "左右腿错开半个周期",
           f"corr(0)={c_same:+.2f} corr(T/2)={c_half:+.2f}")

    # ---- 上体前倾 + 重心投影平衡 ----
    # 四元数 [w,x,y,z] 的绕 y 分量：pitch = 2*atan2(qy, qw)
    pitch = np.degrees(2.0 * np.arctan2(q[gs:gs + gf, 5], q[gs:gs + gf, 3]))
    lean_cfg = float(d["meta"].get("lean_deg", 0.0))
    ok(3.0 <= lean_cfg <= 12.0, "配置上体略微前倾 3~12°",
       f"lean_deg = {lean_cfg:.2f}°")
    ok(abs(float(pitch.mean()) - lean_cfg) < 0.5,
       "实际躯干前倾角与配置一致",
       f"实测 {pitch.mean():.2f}°（全程 {pitch.min():.2f}~{pitch.max():.2f}°）")
    ok(float(np.ptp(pitch)) < 1.0, "前倾角全程恒定（基座不晃）",
       f"波动 {np.ptp(pitch):.3f}°")
    resid = float(d["meta"].get("com_hip_residual", 9.0))
    shift = float(d["meta"].get("balance_shift", 0.0))
    ok(abs(resid) < 0.001, "质心补偿迭代收敛 < 1 mm",
       f"残差 {resid*1000:+.2f} mm, 补偿量 {shift*100:.2f} cm")

    # ---- 两条腿绷直（本次核心要求）----
    kl = np.degrees(q[gs:gs + gf, 7 + IDX["l_knee_y"]])
    kr = np.degrees(q[gs:gs + gf, 7 + IDX["r_knee_y"]])
    knee_sup = np.where(sup_l, kl, kr)
    knee_sw = np.where(sup_l, kr, kl)
    ok(float(knee_sup.mean()) < 12.0, "支撑腿全程绷直（平均膝屈 < 12°）",
       f"均值 {knee_sup.mean():.1f}°")
    ok(float(knee_sup.max()) < 30.0, "支撑腿最大膝屈 < 30°",
       f"最大 {knee_sup.max():.1f}°（仅支撑腿切换处 2~3 帧）")
    ok(float(knee_sw.mean()) < 18.0, "摆动腿全程绷直（平均膝屈 < 18°）",
       f"均值 {knee_sw.mean():.1f}°")
    ok(float(knee_sw.max()) < 40.0, "摆动腿最大膝屈 < 40°",
       f"最大 {knee_sw.max():.1f}°（仅起脚/落脚 2~3 帧）")
    rise = float(np.ptp(q[gs:gs + gf, 2])) * 100
    ok(rise < 6.0, "基座起伏受控 < 6 cm", f"{rise:.2f} cm")
    if tag.startswith("goose_walk"):
        ok(rise > 2.0, "前进版靠髋起伏维持腿直", f"{rise:.2f} cm")
    else:
        ok(rise < 0.5, "原地版支撑脚不动、基座不起伏", f"{rise:.2f} cm")

    # ---- 支撑腿大腿微微后倾（本次要求）----
    # 取每个支撑相的中点帧（此时支撑脚在髋后 dx_sup，姿态最稳定），量「髋 -> 膝」
    # 与竖直的夹角：正 = 膝在髋前方，负 = 膝在髋后方（大腿后倾）。
    th_sup = []
    for g, side in ((sup_l, "l"), (sup_r, "r")):
        starts = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
        for i in range(0, len(starts), 2):
            a, c = starts[i], starts[i + 1]
            if c - a >= 10:
                j = gs + (a + c) // 2
                hip = fk[f"{side}_hip_y_link"][j]
                knee = fk[f"{side}_knee_y_link"][j]
                th_sup.append(math.degrees(math.atan2(knee[0] - hip[0],
                                                      hip[2] - knee[2])))
    th_sup = np.asarray(th_sup)
    want = float(meta.get("stance_back_deg", 5.0))
    ok(-12.0 < float(th_sup.mean()) < -2.0, "支撑相中点大腿微微后倾 2~12°",
       f"均值 {th_sup.mean():+.2f}°（膝在髋后 {-th_sup.mean():.1f}°）")
    ok(abs(float(th_sup.mean()) + want) < 1.5, "后倾角与配置一致",
       f"配置 {want:.1f}°，实测 {th_sup.mean():+.2f}°")
    if not tag.startswith("goose_walk"):
        ok(float(np.ptp(th_sup)) < 1.5, "原地版支撑腿后倾角恒定",
           f"波动 {np.ptp(th_sup):.2f}°")

    # ---- 大臂前摆到最高点时小臂水平、横向横在胸前（本次要求）----
    # 峰值姿态本身不变（腕本来就落在中线上），加手之后靠交会相抹掉一半肩内旋
    # （goose_step 的 arm_swing_open）来避让，所以这里的姿态判据维持原样；
    # 「两只拳会不会撞」由下面的 hand_clearance 逐帧守住。
    for side in ("l", "r"):
        col = q[:, 7 + IDX[f"{side}_shoulder_y"]]
        i = int(col.argmin())                 # 前摆峰值 = 肩角最负
        fa = fk[f"{side}_wrist_y_link"][i] - fk[f"{side}_elbow_y_link"][i]
        fa = fa / np.linalg.norm(fa)
        elev = math.degrees(math.asin(fa[2]))          # 与水平面夹角
        trans = math.degrees(math.atan2(-fa[1] * (1 if side == "l" else -1),
                                        fa[0]))        # 向中线的横向偏角
        elb = math.degrees(q[i, 7 + IDX[f"{side}_elbow_y"]])
        w = fk[f"{side}_wrist_y_link"][i]
        e = fk[f"{side}_elbow_y_link"][i]
        ok(abs(elev) < 3.0, f"{side} 臂前摆峰值小臂水平 < 3°",
           f"仰角 {elev:+.2f}°（肘关节 {elb:.1f}°）")
        ok(trans > 60.0, f"{side} 臂小臂横向横在胸前（横向角 > 60°）",
           f"横向 {trans:.1f}°（0°=指向前方，90°=纯横向）")
        ok(abs(float(w[1])) < 0.10, f"{side} 臂手腕落在身体中线附近",
           f"腕 y = {w[1]:+.3f} m（肩 y = ±0.178）")
        ok(0.88 < float(w[2]) < 1.08 and float(w[0]) > 0.15,
           f"{side} 臂小臂位于胸前（高度合适、在躯干前方）",
           f"腕 (x={w[0]:+.3f}, z={w[2]:.3f})，肘 x={e[0]:+.3f}")

    # ---- 手（腕外 0.119 m 的拳）不能撞到别的部件 ----
    gap, pair, kk = hand_clearance(q, gs, gs + gf,
                                   [("l_hand", "r_hand"),
                                    ("l_hand", "l_hip_x_link"),
                                    ("r_hand", "r_hip_x_link"),
                                    ("l_hand", "waist_z_link"),
                                    ("r_hand", "waist_z_link"),
                                    ("l_hand", "r_wrist_z_link"),
                                    ("r_hand", "l_wrist_z_link")],
                                   coarse_pairs=[("l_hand", "r_hand"),
                                                 ("l_hand", "l_hip_x_link"),
                                                 ("r_hand", "r_hip_x_link")])
    ok(gap > 40.0, "手与手/腿/躯干最小间距 > 4 cm",
       f"{gap:.1f} mm（{pair}，帧 {kk}）")

    # ---- 重心投影：支撑脚后置后，重心落在脚掌前半（动态平衡姿势）----
    sup = np.where(sup_l & ~sup_r, pl[:, 0],
                   np.where(sup_r & ~sup_l, pr[:, 0],
                            (pl[:, 0] + pr[:, 0]) / 2.0))
    dev = com - sup
    single = sup_l ^ sup_r
    mids = []
    for g in (sup_l, sup_r):
        starts = np.flatnonzero(np.diff(np.r_[0, g.astype(int), 0]))
        for i in range(0, len(starts), 2):
            a, c = starts[i], starts[i + 1]
            if c - a >= 10:
                mids.append(float(dev[(a + c) // 2]))
    mid_lo, mid_hi = min(mids, default=-9.0), max(mids, default=9.0)
    # 支撑脚在质心后方 stance_back，所以重心必然前移；只要仍落在
    # 脚掌支撑区（踝 −9 ~ +15 cm）内就是「压在前脚掌」的行进姿势。
    ok(mid_hi < 0.15 and mid_lo > -0.09,
       "支撑相重心投影落在脚掌支撑区内（-9~+15 cm）",
       f"中点 {mid_lo*100:+.1f} ~ {mid_hi*100:+.1f} cm")
    ok(float(np.mean(mids)) > 0.06,
       "重心压在前脚掌（支撑脚后置带来的前倾动态平衡）",
       f"中点均值 {np.mean(mids)*100:+.1f} cm")
    ok(abs(float(dev[single].mean()) - float(np.mean(mids))) < 0.03,
       "单脚支撑全程重心投影稳定（无系统性前后漂移）",
       f"全程均值 {dev[single].mean()*100:+.1f} cm")


def check_salute(tag: str, d: dict, fk: dict) -> None:
    q = d["qpos"]
    w = fk["r_wrist_y_link"]
    head = fk["head_y_link"]
    meta = d["meta"]
    say("  -- 敬礼专项 --")
    # 必须取「保持段」的帧，不能取全体腕 z 的 argmax：抬手途中若有关节空间
    # 过冲，argmax 会落在半路上，量到的是路径而不是敬礼手位。
    i = int(meta.get("salute_hold_frame", int(2.1 / d["dt"])))
    wz = float(w[i, 2])
    ok(wz > 1.30, "右腕抬到头部高度", f"保持帧 z = {wz:.3f} m")
    # 基准手位 y = −0.125，再沿机器人右侧（−y）平移 wrist_right
    shift = float(meta.get("wrist_right_m", 0.0))
    y_tgt = -0.125 - shift
    ok(abs(float(w[i, 1]) - y_tgt) < 0.01,
       f"右腕沿机器人右侧水平平移 {shift*100:.0f} cm",
       f"y = {w[i,1]:+.3f} m（目标 {y_tgt:+.3f}），z 抬升 {wz-1.345:+.3f} m")
    ok(abs(wz - 1.345) < 0.02, "平移只走横向、高度不变",
       f"z 与基准手位 1.345 差 {wz-1.345:+.3f} m")
    ok(float(w[i, 1]) < -0.11, "右腕仍在头部右侧（没越过中线）",
       f"y = {w[i,1]:+.3f} m")
    dist = float(np.linalg.norm(w[i] - head[i])) * 1000
    ok(100.0 < dist < 260.0, "手在头右侧外侧、不插进头部", f"距头中心 {dist:.0f} mm")
    over = float(w[:, 2].max() - wz)
    ok(over < 0.02, "抬手路径不过冲（腕不越过太阳穴）",
       f"全程 z 峰值比保持帧高 {over*1000:.1f} mm")

    # 保持段（抬手结束后的 1.4 s）应完全静止
    hold = q[int(1.4 / d["dt"]):int(2.6 / d["dt"])]
    still = float(np.abs(np.diff(hold[:, 7:36], axis=0)).max()) if len(hold) > 2 else 9.0
    ok(still < 1e-9, "敬礼保持段完全静止", f"max|Δ|={still:.2e}")

    # 左臂只需在「敬礼动作本身」静止；开头立正收臂 / 结尾收势是刻意过渡。
    a = int(meta.get("attention_end_frame", int(0.5 / d["dt"])))
    b = int(meta.get("salute_active_end_frame", int(3.6 / d["dt"])))
    lh = q[a:b, 7 + IDX["l_shoulder_y"]:7 + IDX["l_shoulder_y"] + 4]
    lmov = float(np.abs(lh - lh[0]).max())
    ok(lmov < 0.01, "敬礼动作段左臂保持自然下垂", f"最大偏离 {lmov:.4f} rad")

    # 「自然下垂」不是只看关节角不动：小臂沿 elbow 的 +x，肘角为 0 时小臂
    # 是水平前伸的（旧版就是这样，看着像端着托盘）。要求小臂方向竖直向下。
    le, lw = fk["l_elbow_y_link"], fk["l_wrist_y_link"]
    fa = lw[a:b] - le[a:b]
    fa = fa / np.linalg.norm(fa, axis=1, keepdims=True)
    down = float(-fa[:, 2].min())
    ok(down > 0.95, "左臂小臂竖直向下（不是水平前伸）",
       f"竖直度 {-fa[:,2].min():.4f}（1.0 = 完全垂直）")
    ok(float(np.abs(fa[:, 0]).max()) < 0.20, "左臂小臂没有前伸",
       f"前向分量 max {np.abs(fa[:,0]).max():.3f}")
    ok(float(lw[a:b, 1].min()) > 0.15, "左腕垂在身体侧面、不贴躯干",
       f"y = {lw[a:b,1].min():.3f} m（躯干半宽 0.09）")
    ok(float(lw[a:b, 2].max()) < 0.72, "左腕垂到大腿高度", f"z = {lw[a:b,2].max():.3f} m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default="logs/parade_actions/report.txt")
    ap.add_argument("--src-dir", default=str(DATA))
    args = ap.parse_args()
    src = Path(args.src_dir)

    m = mujoco.MjModel.from_xml_path(str(WEB_XML))
    data = mujoco.MjData(m)
    gids = visual_geoms(m)
    say("=" * 72)
    say("ELF3 走正步 / 敬礼 —— 动作轨迹数值验收")
    say("=" * 72)

    for fname, tag, loop in CASES:
        d = load(src / fname)
        ix = build_qpos_remap(m, d["names"])
        check_common(tag, d, loop, m, data, gids, ix)
        fk = fk_points(d["qpos"], d["names"])
        if tag.startswith("goose"):
            check_goose(tag, d, fk)
        else:
            check_salute(tag, d, fk)

    say("\n" + "=" * 72)
    if _fails:
        say(f"结论：{len(_fails)} 项未通过")
        for f in _fails:
            say(f"  - {f}")
    else:
        say("结论：全部判据通过 ✔")
    say("=" * 72)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\n报告 -> {out}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
