"""ELF3 立定跳远 · 第二版（双臂后伸蓄力 + 前倾蹬伸 + 双臂前摆 + 跳距 > 1.2 m）

==================== 相对 v1（long_jump.py）的动作编排差异 ====================
v1 保持原样、本文件不改它。本版只改【动作编排】与【起跳初速度】，
落地平衡判据、控制器、物理建模（冲量等效起跳 + 完全非弹性落地 +
开环准静态站起）与 v1 完全一致，详见 long_jump.py 头部说明。

  * 预蹲 / 深蹲阶段：双臂向后伸直后摆（arm_back，比 v1 更靠后、更接近
    伸直），即人类跳远的"预摆蓄能"姿态 —— 双臂向后伸。
  * 蹬伸起跳阶段：躯干在深蹲前倾基础上再叠加 waist_y 前倾（takeoff_lean，
    身体明显前压，"身体前倾"）；同时双臂由后快速向前摆出（arm_up 前伸、
    屈肘，像短跑摆臂），把质心往前送。
  * 起跳初速度提高（vx / vz 更大），使跳距 > 1.2 m。

用法:
    python long_jump_v2.py                       # 跑仿真 + 校核 + 存 v2 npz
    python long_jump_v2.py --vx 2.4 --vz 1.9     # 指定起跳初速度（冲量等效）
    python long_jump_v2.py --check-only          # 只跑不落盘
    python long_jump_v2.py --json jump_v2.json   # 指标另存 JSON

    # 动画（复用 jump_replay.py，指定 v2 轨迹）
    python jump_replay.py --traj long_jump_v2_traj.npz \
        --gif ../long_jump_v2.gif --sheet ../long_jump_v2_sheet.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import long_jump as LJ                                          # noqa: E402
import squat_stand as SQ                                        # noqa: E402

DATA = HERE.parent / "data" / "mujoco_simulation"
DEFAULT_OUT = DATA / "long_jump_v2_traj.npz"

TOE_LEN = LJ.TOE_LEN
SQ_STANCE_Z = SQ.STANCE_Z


# ---------------------------------------------------------------- v2 编排
class LongJumpV2(LJ.LongJump):
    """复用 v1 的闭环仿真 / 物理 / 落地校核，只重写动作编排。"""

    # v2 手臂编排（shoulder_y 前摆为负 / shoulder_x 外张 / elbow_y 屈肘为正）
    arm_back = (0.85, 0.18, 0.12)   # 双臂向后伸（预蹲/深蹲 后摆蓄力，近伸直）
    arm_up   = (-1.05, 0.15, 0.55)  # 双臂快速向前摆（蹬伸 前伸 + 屈肘）
    arm_mid  = (-0.30, 0.20, 0.50)  # 腾空过渡
    takeoff_lean = 0.30              # 蹬伸期额外前倾（waist_y）rad（身体前倾，比 v2 初版更大）

    def __init__(self, p: LJ.Params):
        super().__init__(p)
        # 让预蹲 / 深蹲的躯干手臂姿态用 v2 的"后伸"姿态（tail=1 时即 arm_back）
        self.sqp = SQ.Params(drop=p.drop_squat, lean=p.lean_squat,
                             arm=self.arm_back[0], elbow=self.arm_back[2],
                             arm_out=self.arm_back[1], head_pitch=10.0)

    def targets(self) -> np.ndarray:
        p = self.p
        ph, u = self.phase, self._phase_u()

        if ph in (LJ.PH["READY"], LJ.PH["PRE"], LJ.PH["SQUAT"]):
            if ph == LJ.PH["READY"]:
                tail = 0.0
            elif ph == LJ.PH["PRE"]:
                tail = self.pre_frac * u
            else:
                tail = self.pre_frac + (1.0 - self.pre_frac) * SQ.smootherstep(u)
            q, _, _ = SQ.make_pose(self.sqp, tail)
            return q

        if ph == LJ.PH["TAKEOFF"]:
            # 蹬伸起跳：躯干在前倾基础上再叠加 waist_y 前倾（身体前倾），
            # 双臂由后（arm_back）快速向前摆（arm_up），腿做踝限位感知的协调蹬伸。
            pitch = self.pitch()
            lean_t = self.takeoff_lean * u          # 蹬伸中逐渐前倾
            pe = pitch + lean_t
            dxw0, dzw0 = self.span0
            s = SQ.smootherstep(u)
            dxw_cmd = LJ.blend(dxw0, p.dxw_to, s)
            dzw = LJ.blend(dzw0, LJ.LZ_MAX, s)
            pf = p.pf_to * SQ.smootherstep(max(0.0, (u - 0.70) / 0.30))
            dxw = LJ.clamp_dxw_ankle(pe, dzw, dxw_cmd, pf)
            arm = LJ.mix_arm(self.arm_back, self.arm_up, s * s)
            q1, q2, qa = LJ.leg_targets_span(pe, dxw, dzw, pf)
            return LJ.assemble((q1, q2, qa), arm, waist=lean_t)

        if ph == LJ.PH["FLIGHT"]:
            # 收腿 → 前伸小腿（与 v1 一致：脚目标挂在实测髋前方，随身体自适应）
            fu = min(u * p.t_flight_nom / 0.45, 1.0)
            s = SQ.smootherstep(fu)
            reach = LJ.blend(p.tuck_reach, p.land_reach, s)
            clear = LJ.blend(p.tuck_clear, p.land_clear, s)
            pitch = self.pitch()
            hx = float(self.d.qpos[0]) - SQ.TORSO_TO_HIP * math.sin(pitch)
            arm = LJ.mix_arm(self.arm_up, self.arm_mid, min(u / 0.30, 1.0))
            q1, q2, qa = LJ.leg_targets(pitch, float(self.d.qpos[0]),
                                        float(self.d.qpos[2]),
                                        hx + reach, SQ.FOOT_SOLE + clear, 0.0)
            return LJ.assemble((q1, q2, qa), arm)

        if ph in (LJ.PH["ABSORB"], LJ.PH["SETTLE"], LJ.PH["RISE"], LJ.PH["HOLD"]):
            # 落地段：开环准静态姿态序列（squat_stand 已验证能静态站稳）
            if ph in (LJ.PH["ABSORB"], LJ.PH["SETTLE"]):
                tail = 1.0
            elif ph == LJ.PH["RISE"]:
                tail = 1.0 - SQ.smootherstep(u)
            else:
                tail = 0.0
            q, _, _ = SQ.make_pose(self.sqp, tail)
            return q

        raise RuntimeError(f"未知相位 {ph}")


# ---------------------------------------------------------------- 报告打印
def report(mt: dict, ok: bool, why: list[str], p: LJ.Params):
    line = "=" * 72
    print(f"\n{line}\nELF3 立定跳远 · v2（双臂后伸蓄力 + 前倾前摆 + >1.2m）   仿真结果\n{line}")
    if "t_liftoff" in mt:
        vx, vz = mt["v_takeoff"]
        print(f"\n[起跳]  t={mt['t_liftoff']:.2f}s   冲量等效初速度 "
              f"vx={vx:+.2f} m/s  vz={vz:+.2f} m/s")
        print(f"[腾空]  {mt['flight_time']:.2f}s   质心抛物线飞行（纯物理守恒）")
        print(f"[落地]  t={mt['t_touchdown']:.2f}s   "
              f"脚尖→脚跟跳距 {mt['toe_heel_dist']*100:.0f} cm   "
              f"基座前移 {mt['base_travel']*100:.0f} cm")
    print(f"[总位移] {mt['total_travel']*100:.0f} cm（起点→末态基座 x）")
    print(f"\n[落地平衡] 保持 {mt['hold_time']:.2f}s 不倒   "
          f"末态躯干高 {mt['final_z']:.3f} m（站姿 {SQ_STANCE_Z:.3f}）   "
          f"末态俯仰 {mt['final_pitch_deg']:+.1f}°")
    print(f"           末态 COM 裕度 {mt['final_margin']*100:+.2f} cm"
          f"   站定期最小 {mt['min_margin_stance']*100:+.2f} cm   "
          f"（判据 > 2 cm）")
    dist_ok = mt.get("toe_heel_dist", 0.0) >= 1.2
    print(f"[跳距]   {mt.get('toe_heel_dist', 0.0)*100:+.0f} cm   "
          f"{'✔ ≥ 1.2 m' if dist_ok else '✘ < 1.2 m'}")
    print(f"[力矩]   峰值占用 {100*mt['tau_peak']:.0f}% of actuatorfrcrange")
    print(f"[判定]   {'✔ 通过：落地后保持平衡站立且跳距达标' if ok and dist_ok else '✘ 未通过'}")
    for w in why:
        print(f"         - {w}")
    if not dist_ok:
        print(f"         - 跳距 {mt.get('toe_heel_dist',0.0)*100:.0f} cm < 120 cm")
    print(line)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="ELF3 立定跳远 v2：双臂后伸蓄力 + 前倾前摆 + 跳距>1m")
    ap.add_argument("--vx", type=float, default=3.0,
                    help="起跳水平初速度 m/s（冲量等效，默认 3.0 以达 >1.2m）")
    ap.add_argument("--vz", type=float, default=1.7,
                    help="起跳垂直初速度 m/s（冲量等效；≥1.9 会使落地腿伸太前而前翻）")
    ap.add_argument("--takeoff-lean", type=float, default=0.30,
                    help="蹬伸期额外前倾（waist_y）rad")
    ap.add_argument("--drop", type=float, default=0.25, help="深蹲下降量 m")
    ap.add_argument("--lean", type=float, default=13.0, help="深蹲前倾 deg")
    ap.add_argument("--pf", type=float, default=0.15, help="蹬伸末踝跖屈 rad")
    ap.add_argument("--dxw-to", type=float, default=-0.06, help="蹬伸末髋在踝前方 m")
    ap.add_argument("--land-reach", type=float, default=0.22, help="伸腿落地脚在髋前方 m")
    ap.add_argument("--kp-scale", type=float, default=8.0)
    ap.add_argument("--kd-scale", type=float, default=8.0)
    ap.add_argument("--sim-hz", type=int, default=1000)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    p = LJ.Params(drop_squat=args.drop, lean_squat=args.lean,
                  vx_takeoff=args.vx, vz_takeoff=args.vz,
                  pf_to=args.pf, dxw_to=args.dxw_to,
                  land_reach=args.land_reach,
                  kp_scale=args.kp_scale, kd_scale=args.kd_scale, sim_hz=args.sim_hz)

    sim = LongJumpV2(p)
    sim.takeoff_lean = args.takeoff_lean
    print(f"[仿真 v2] 场景 {Path(p.scene).name}   PD×{p.kp_scale:g} @ {p.sim_hz} Hz   "
          f"控制 50 Hz   起跳初速度 vx={p.vx_takeoff:g} / vz={p.vz_takeoff:g} m/s   "
          f"蹬伸前倾 {math.degrees(args.takeoff_lean):.0f}°")
    sim.run()
    mt = sim.metrics()
    ok, why = sim.verdict(mt)
    if mt.get("toe_heel_dist", 0.0) < 1.2:
        ok = False
    report(mt, ok, why, p)

    if args.check_only:
        return

    out = Path(args.out)
    meta = dict(params=asdict(p), metrics=mt, ok=bool(ok), why=why,
                phases=LJ.PHASES, version="v2",
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
