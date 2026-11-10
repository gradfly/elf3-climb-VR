"""ELF3 敬礼（右手军礼）开环动作序列生成器。

姿态怎么定
----------
站姿下右腕在 ``(0.129, -0.258, 0.694)``，头中心 ``(0, 0, 1.333)``、右肩
``(0, -0.178, 1.121)``。军礼要求：

* 手抬到**右太阳穴旁**：腕目标 ``(0.035, -0.125, 1.345)``；
* 上臂抬平略高、肘向外侧撑开：肘目标 ``(0.06, -0.36, 1.27)``；
* 手掌向下、指梢朝眉梢：腕 3 个关节给一个固定朝向。

肩到腕只有 0.245 m，而大臂/小臂各约 0.255 m，肘必须折回来。只用「腕位置」
一个约束时 IK 会收敛到任意肘分支（常见的是肘插进躯干或反关节），所以这里
用 **腕 + 肘双约束的加权 DLS**：腕权重 1.0、肘权重 0.35，一次解出目标姿态。

抬手/礼毕两段用**统一 smootherstep 的关节空间插值**（整条右臂同一时序，
端点零速度）。早先给肩/肘/腕错峰时，手沿关节弧线先甩到 z=1.394 m —— 比
太阳穴目标高 5 cm、外偏 9 cm，像「越过头顶再落回来」；改成同一时序后过冲
只剩 1.7 cm。也试过笛卡尔目标 + 逐帧 IK：4 自由度腕肘双约束存在等价分支，
路径中途会翻到另一组肩角（实测单帧 180° 跳变），反而更差。
只在端点解一次 IK、中间纯插值，既无分支漂移也不跳帧。

时间序列（首尾都是立正前的 STAND_Q，可无缝循环）
------------------------------------------------
    立正 0.5 s -> 抬手 0.9 s -> 敬礼保持 1.4 s
    -> 礼毕放下 0.8 s -> 收势 0.5 s -> 静止 0.4 s

立正/收势两段是自然站姿 <-> 贴裤缝的过渡；敬礼动作本身（抬手、保持、礼毕）
左臂完全静止，腰/腿/头全程不动，基座 x/z 不变。

    python salute.py
    python salute.py --check-only
    python squat_replay.py --traj salute_traj.npz --mode kin --loop
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import mujoco

from action_lib import (DATA, DT, IDX, IK, STAND_Z, limit_report,
                        make_qpos_row, save_traj, smootherstep, stand_q)

# 世界坐标（基座在 x=0, z=STAND_Z，pitch=0）
# 肘目标不再写死，而是由腕目标按「两段等长」的几何自洽式反解（见
# elbow_target_for）。早先手写 (0.06,-0.36,1.27) 时肩->肘只有 0.243 m
# （比大臂短 1.2 cm），约束互相打架，解直接跑到反关节分支，腕误差 20 cm。
WRIST_BASE = np.array([0.035, -0.125, 1.345])   # 旧手位：腕贴右太阳穴
HEAD_CENTER = np.array([0.000, 0.000, 1.333])
SHOULDER_R = np.array([0.000, -0.178, 1.121])   # 站姿右肩轴（实测 FK）
ARM_LEN = 0.256                                  # 大臂/小臂同长

R_ARM = ["r_shoulder_y_joint", "r_shoulder_x_joint",
         "r_shoulder_z_joint", "r_elbow_y_joint"]

# 多起点重启：4 自由度腕+肘双约束有局部极小，实测从「立正姿态」出发
# 收敛到腕误差 0.2 mm 的自然军礼解；其余起点是兜底。
IK_INITS = [None, [-1.0, -1.2, 0.0, 1.6], [0.8, -1.0, -1.5, 1.0]]


def wrist_target(p: "Params") -> np.ndarray:
    """敬礼手位：基准手位再沿机器人**右侧**（−y）水平平移 ``wrist_right``。"""
    t = WRIST_BASE.copy()
    t[1] -= p.wrist_right
    return t


def elbow_target_for(wrist: np.ndarray, shoulder: np.ndarray = SHOULDER_R,
                     length: float = ARM_LEN) -> np.ndarray:
    """给定腕目标，反解几何自洽的肘目标（|肩−肘| = |肘−腕| = length）。

    肘落在「以肩、腕为球心、半径 length 的两球交圆」上，圆半径
    ``r = sqrt(L² − (|SW|/2)²)``。取圆上**最靠机器人右侧**（−y）的点：
    军礼要求肘向外撑开，而不是插进躯干或反关节。手写肘目标很容易让
    两段长度对不上，IK 就会在两球之间反复拉扯。
    """
    d = np.asarray(wrist, float) - np.asarray(shoulder, float)
    n = float(np.linalg.norm(d))
    if n >= 2.0 * length:
        raise ValueError(f"腕目标超出臂展：|SW|={n:.3f} > {2*length:.3f}")
    d = d / n
    a = n / 2.0
    r = math.sqrt(max(0.0, length * length - a * a))
    out = np.array([0.0, -1.0, 0.0])          # 机器人右侧
    out -= d * float(out @ d)                 # 投影到圆平面
    out /= np.linalg.norm(out)
    return np.asarray(shoulder, float) + d * a + out * r


@dataclass
class Params:
    elbow_weight: float = 0.35       # 肘位置约束权重（相对腕=1.0）
    prior_w: float = 3e-5            # 轻微零空间正则，抑制反关节
    wrist_x: float = 0.00            # 敬礼手掌朝向
    wrist_y: float = -0.30
    wrist_z: float = 0.25
    wrist_right: float = 0.10        # 腕相对基准手位向右（−y）水平移动量
    # 左臂 / 立正姿态的自然下垂：大臂沿 −z 垂下，小臂必须靠 elbow_y≈90°
    # 才跟着垂下 —— 手臂链里小臂是沿 elbow 的 **+x**，elbow_y=0 时小臂
    # 水平前伸（旧版就是这样，看着像端着托盘，不是自然下垂）。
    l_sh_y: float = 0.04             # 肩前抬 2.3°
    l_sh_x: float = 0.08             # 肩外展 4.6°
    l_elbow: float = 1.52            # 肘 87°，小臂竖直略前
    t_attention: float = 0.50
    t_raise: float = 0.90
    t_hold: float = 1.40
    t_lower: float = 0.80
    t_settle: float = 0.50
    t_final: float = 0.40
    dt: float = DT


def attention_q(p: Params) -> np.ndarray:
    """立正/自然站姿：双臂自然下垂（大臂垂直、小臂也跟着垂直），腿脚不动。"""
    q = stand_q()
    for side, sgn in (("l", 1.0), ("r", -1.0)):
        i = IDX[f"{side}_shoulder_y"]
        q[i] = p.l_sh_y
        q[i + 1] = sgn * p.l_sh_x
        q[i + 2] = 0.0
        q[i + 3] = p.l_elbow
        q[i + 4:i + 7] = 0.0
    return q


def solve_salute(ik: IK, p: Params) -> tuple[np.ndarray, dict]:
    wrist_t = wrist_target(p)
    elbow_t = elbow_target_for(wrist_t)
    targets = [("r_wrist_y_link", wrist_t, 1.0),
               ("r_elbow_y_link", elbow_t, p.elbow_weight)]
    best: tuple[float, np.ndarray, float] | None = None
    for ini in IK_INITS:
        q0 = attention_q(p)
        if ini is not None:
            for n, v in zip(R_ARM, ini):
                q0[IDX[n.replace("_joint", "")]] = v
        ik.set_base(0.0, STAND_Z, 0.0)
        q, resid = ik.solve(q0, targets, R_ARM,
                            iters=400, lam=1e-5, step=0.12)
        ik.d.qpos[7:36] = q
        mujoco.mj_forward(ik.m, ik.d)
        w = ik.body_pos("r_wrist_y_link")
        e = ik.body_pos("r_elbow_y_link")
        score = (float(np.linalg.norm(w - wrist_t))
                 + 0.3 * float(np.linalg.norm(e - elbow_t)))
        # 肩大幅后翻的解也数学可行，但看着不像敬礼，加一点惩罚
        if abs(q[IDX["r_shoulder_y"]]) > 2.0:
            score += 0.2
        if best is None or score < best[0]:
            best = (score, q.copy(), resid)

    assert best is not None
    q = best[1]
    resid = best[2]
    # 手掌朝向（腕 3 个关节不参与腕的位置 IK，直接给）
    i = IDX["r_shoulder_y"]
    q[i + 4], q[i + 5], q[i + 6] = p.wrist_x, p.wrist_y, p.wrist_z
    ik.d.qpos[7:36] = q
    mujoco.mj_forward(ik.m, ik.d)
    wrist = ik.body_pos("r_wrist_y_link").copy()
    elbow = ik.body_pos("r_elbow_y_link").copy()
    # 左臂（不参与 IK）顺手量一下：自然下垂要求小臂方向几乎竖直向下
    lw = ik.body_pos("l_wrist_y_link").copy()
    le = ik.body_pos("l_elbow_y_link").copy()
    lfa = lw - le
    info = dict(wrist_err_mm=round(float(np.linalg.norm(wrist - wrist_t)) * 1000, 2),
                elbow_err_mm=round(float(np.linalg.norm(elbow - elbow_t)) * 1000, 2),
                wrist_to_head_mm=round(float(np.linalg.norm(wrist - HEAD_CENTER)) * 1000, 1),
                wrist_y_m=round(float(wrist[1]), 4),
                wrist_right_m=p.wrist_right,
                left_forearm_down=round(float(-lfa[2] / np.linalg.norm(lfa)), 4),
                left_wrist=[round(float(v), 3) for v in lw],
                ik_residual_mm=round(resid * 1000, 2),
                salute_joints_deg={n.split("_joint")[0]: round(math.degrees(q[IDX[n.replace("_joint", "")]]), 1)
                                   for n in R_ARM})
    return q, info


def joint_path(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """整条 29 维姿态统一 smootherstep 插值，末帧严格等于 b。

    所有关节同一时序（不再给肩/肘/腕错峰），端点一阶导数为 0，
    既避免腕部关节弧线过冲，也不会出现逐帧 IK 的分支翻转。
    """
    return [a + (b - a) * smootherstep(k / max(1, n - 1)) for k in range(n)]


def build(p: Params, ik: IK) -> tuple[np.ndarray, dict]:
    # 「自然站姿」与「立正」在这里是同一个姿态：双臂自然下垂（大臂垂直、
    # 小臂也垂直贴身）。旧的 stand_q 给肘 0.6 rad，小臂前伸 34°，
    # 循环首尾看着像端着东西，不是自然下垂。
    q_att = attention_q(p)
    q_stand = q_att.copy()
    q_sal, info = solve_salute(ik, p)          # 只在这里解一次 IK

    def frames(dur: float) -> int:
        return int(round(dur / p.dt))

    qs: list[np.ndarray] = []
    # 立正收臂 -> 抬手 -> 敬礼保持 -> 礼毕放下 -> 收势 -> 静止
    qs.extend(joint_path(q_stand, q_att, frames(p.t_attention)))
    qs.extend(joint_path(q_att, q_sal, frames(p.t_raise)))
    qs.extend([q_sal.copy() for _ in range(frames(p.t_hold))])
    qs.extend(joint_path(q_sal, q_att, frames(p.t_lower)))
    qs.extend(joint_path(q_att, q_stand, frames(p.t_settle)))
    qs.extend([q_stand.copy() for _ in range(frames(p.t_final))])
    qs.append(q_stand.copy())                     # 末帧 == 首帧，循环闭合
    qpos = np.stack([make_qpos_row(q, 0.0, STAND_Z) for q in qs])

    seam = float(np.abs(qpos[0] - qpos[-1]).max())
    info["loop_seam_max_abs"] = seam
    info["frames"] = len(qpos)
    info["attention_end_frame"] = frames(p.t_attention)
    info["salute_hold_frame"] = (frames(p.t_attention) + frames(p.t_raise)
                                 + frames(p.t_hold) // 2)
    info["salute_active_end_frame"] = (frames(p.t_attention) + frames(p.t_raise)
                                       + frames(p.t_hold) + frames(p.t_lower))
    return qpos, info


def report(qpos: np.ndarray, info: dict, p: Params) -> None:
    print("\n=== 敬礼（右手军礼） ===")
    print(f"  帧数 {len(qpos)}  时长 {len(qpos)*p.dt:.2f} s  基座不动")
    print(f"  腕目标误差 {info['wrist_err_mm']:.1f} mm  "
          f"肘目标误差 {info['elbow_err_mm']:.1f} mm  "
          f"IK 残差 {info['ik_residual_mm']:.2f} mm")
    print(f"  腕 -> 头中心 {info['wrist_to_head_mm']:.0f} mm（手位在头右侧外）")
    print(f"  右腕向右平移 {info['wrist_right_m']*100:.0f} cm  ->  y = {info['wrist_y_m']:+.3f} m")
    print(f"  左臂小臂竖直度 {info['left_forearm_down']:.4f}"
          f"（1.0 = 完全垂直向下），左腕 {info['left_wrist']}")
    print(f"  右臂解算角：{info['salute_joints_deg']}")
    jump = float(np.abs(np.diff(qpos[:, 7:36], axis=0)).max())
    print(f"  单帧最大关节跳变 {jump:.4f} rad ({math.degrees(jump):.2f}°)")
    print(f"  循环接缝 max|Δ| = {info['loop_seam_max_abs']:.2e}")
    viol = limit_report(qpos)
    print(f"  关节限位：{'全部通过' if not viol else '越界 -> ' + '; '.join(viol)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 敬礼动作生成器")
    ap.add_argument("--elbow-weight", type=float, default=0.35)
    ap.add_argument("--hold", type=float, default=1.40, help="敬礼保持秒数")
    ap.add_argument("--wrist-right", type=float, default=0.10,
                    help="右手腕沿机器人右侧（−y）水平移动的米数")
    ap.add_argument("--l-elbow", type=float, default=1.52,
                    help="左臂肘关节角（rad），1.57≈90° 才让小臂垂下")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()

    p = Params(elbow_weight=args.elbow_weight, t_hold=args.hold,
               wrist_right=args.wrist_right, l_elbow=args.l_elbow)
    ik = IK()
    qpos, info = build(p, ik)
    report(qpos, info, p)

    if args.check_only:
        return
    out = Path(args.out_dir) / "salute_traj.npz"
    save_traj(out, qpos, "salute",
              {**asdict(p), **info, "label": "脚本动作 · 敬礼（右手军礼）",
               "loop": True,
               "wrist_target": wrist_target(p).tolist(),
               "elbow_target": elbow_target_for(wrist_target(p)).tolist()})
    print("\n回放：python squat_replay.py --traj salute_traj.npz --mode kin --loop")
    print("烘焙：python web/tools/bake_actions.py")


if __name__ == "__main__":
    main()
