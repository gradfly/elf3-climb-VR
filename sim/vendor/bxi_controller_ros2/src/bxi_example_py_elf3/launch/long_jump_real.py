"""ELF3 立定跳远 · 物理诚实版（纯自驱，无冲量注入）

================ 重要结论（实测 50+ 组参数扫描）================
本版去掉了 v1/v2 在离地瞬间【强行注入基座初速度】的做法，完全靠 43kg /
踝 50 / 髋膝 150 N·m 的执行力真实蹬地。结果：垂直自驱很强（vz 峰值
+1.0~+1.4 m/s，能蹬离地、落地能站稳），但**水平前向自驱速度上限仅
≈ +0.05~+0.15 m/s**，净跳距 ≈ 0 甚至略后退（−4~−15 cm）。

根因：要向前推，质心须在支撑脚前方、靠"脚向后蹬地"的反作用；而本机深蹲
+ 低踝力矩下髋/质心恒在脚后方，腿一伸反作用力就是"上+后"，水平分量恒为负。
⇒ 这台 ELF3 在站立起跳下**无法自身完成有意义的前向跳远**；v1/v2 的 1.2m
只能靠注入冲量达成（那是"动作编排参考"，非机器人真实能力）。

要真向前跳，需要：助跑 / 更大力矩的执行器 / 明确标注的外援冲量。

================ 与 v1/v2 的核心改动（只在 TAKEOFF 相位）================
  * 旧 v2：leg_targets_span(pe, dxw, dzw, pf) —— 脚相对髋的"跨度"，
    髋往前脚就跟着往前，脚对地没有向后的推力 ⇒ 永远只能往后溜。
  * 本版：leg_targets(pitch, base_x, base_z↑, foot_x固定, ankle_z, pf)
    —— 脚世界 x 钉在起跳瞬间位置（planted），base_z 由蹲位爆炸式抬到站姿
    以上，并加 anchor_shift<0 让脚目标落在髋后方（命令"向后蹬地"），
    躯干前倾 ⇒ 腿伸直时把脚往后蹬、身体往前上蹿。已尽力逼近前向，
    但受平台力矩所限仍只能做到近似原地/微退的小跳。

用法：
    python long_jump_real.py                  # sweep 多组参数，打印真实自驱指标
    python long_jump_real.py --cfg rise=1.04 pf=0.25 anchor_shift=-0.10 --lean 8
    python long_jump_real.py --inject 2.0 1.7 # 叠加外援冲量（明确标注，非自驱）
    # 纯自驱“更高跳”：加大摆臂幅度 + 提高蹬腿刚度（仍零注入，落站稳）
    python long_jump_real.py --arm-scale 1.8 --kp 16 --lean 8 --anchor -0.10 --out ../data/.../long_jump_real_hi.npz

================ 跳高可调性结论（实测）================
在【零冲量注入、落地站稳】前提下，跳跃高度可用两枚旋钮显著抬高：
  * 摆臂幅度 arm_scale（1.0→1.8）：离地 vz +1.39→+1.74，相对站立跳高 17.6→40.7 cm
  * 蹬腿刚度 kp（8→16）：同幅度下再 +约 10 cm
  机制：摆臂上挥通过肩部反作用把躯干往上顶（不占腿力矩预算）；更高 kp 让腿在
  力矩限幅内更猛地伸。两者叠加可使 COM 峰值抬到 ~1.19 m（站姿 0.76 → +43 cm），
  且仍落地开环站起不倒。注意：纯竖直(anchor=0)摆臂过猛会给躯干带角动量→空中翻滚
  落不稳；保持微前倾+微后蹬的着陆友好基姿(lean≈8, anchor≈-0.10)即可稳落。
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
DEFAULT_OUT = DATA / "long_jump_real_traj.npz"


# ---------------------------------------------------------------- 真实自驱版
class LongJumpReal(LJ.LongJump):
    """TAKEOFF 改为世界系脚掌定点蹬伸；其余相位复用基类（飞行=真实抛物线，
    落地=完全非弹性 + 开环准静态站起）。默认【不注入】冲量。"""

    def __init__(self, p: LJ.Params, cfg: dict | None = None):
        super().__init__(p)
        self.cfg = dict(rise=1.04, pf_end=0.40, tt=0.30,
                        arm_back=LJ.ARM_BACK, arm_up=LJ.ARM_UP,
                        arm_scale=1.0, arm_rate=2.0)
        if cfg:
            self.cfg.update(cfg)
        self._squat_z = None
        self.inject = tuple(self.cfg.get("inject", (0.0, 0.0)))  # (vx, vz) 可叠加

    # 起跳前记录蹲位躯干高，供抬升曲线
    def _enter_takeoff(self):
        self._squat_z = float(self.d.qpos[2])
        self.takeoff_anchor = self.foot_x()
        self.span0 = self._measured_span()

    def targets(self) -> np.ndarray:
        p = self.p
        ph, u = self.phase, self._phase_u()

        if ph in (LJ.PH["READY"], LJ.PH["PRE"], LJ.PH["SQUAT"]):
            return super().targets()

        if ph == LJ.PH["TAKEOFF"]:
            cfg = self.cfg
            pitch = self.pitch()
            s = SQ.smootherstep(u)
            # 脚世界 x：默认钉在起跳瞬间位置；anchor_shift<0 表示命令脚"往后蹬"
            # （脚目标在髋后方），这样腿伸直时把地往后推、身体往前上蹿。
            ash = cfg.get("anchor_shift", 0.0)
            fx = {k: v + ash for k, v in self.takeoff_anchor.items()}
            # 躯干高度：从蹲位爆炸式抬到站姿以上（rise>1 → 略高于站姿）
            z0 = self._squat_z if self._squat_z else (SQ.STANCE_Z - p.drop_squat)
            base_z = LJ.blend(z0, SQ.STANCE_Z * cfg["rise"], s)
            # 末段提踵（站到脚尖），脚掌绕脚尖转、踝被抬高
            pf = cfg["pf_end"] * SQ.smootherstep(max(0.0, (u - 0.55) / 0.45))
            ankle_z = SQ.FOOT_SOLE + LJ.TOE_LEN * math.sin(pf)
            # 双臂：后摆蓄力 → 前摆（仿人摆臂）。
            #   arm_rate: 摆臂进度曲线指数，越大越集中在蹬伸末段(越猛)
            #   arm_scale: 在 sh_y 上放大/缩小摆幅(绕 arm_back 基准)，>1 摆得更开
            ab = cfg["arm_back"]; au = cfg["arm_up"]
            u_arm = s ** cfg.get("arm_rate", 2.0)
            arm = LJ.mix_arm(ab, au, u_arm)
            sc = cfg.get("arm_scale", 1.0)
            if sc != 1.0:
                sh_y = ab[0] + (arm[0] - ab[0]) * sc
                arm = (sh_y, arm[1], arm[2])
            # 双腿各自解 IK（脚定点、躯干抬升），结果取平均
            q_leg = [None, None, None]
            for i, side in enumerate("LR"):
                foot_x = fx[side]
                q1, q2, qa = LJ.leg_targets(pitch, float(self.d.qpos[0]),
                                            base_z, foot_x, ankle_z, pf)
                if i == 0:
                    q_leg = [q1, q2, qa]
                else:
                    q_leg = [(a + b) / 2 for a, b in zip(q_leg, (q1, q2, qa))]
            return LJ.assemble(tuple(q_leg), arm)

        # FLIGHT / ABSORB / SETTLE / RISE / HOLD 复用基类
        return super().targets()

    def _advance(self):
        """默认不注入冲量；若 self.inject 非零，则在离地瞬间叠加（明确标注）。"""
        p = self.p
        ph, u = self.phase, self._phase_u()
        if ph == LJ.PH["READY"] and u >= 1.0:
            self._enter(LJ.PH["PRE"])
        elif ph == LJ.PH["PRE"] and u >= 1.0:
            self._enter(LJ.PH["SQUAT"])
        elif ph == LJ.PH["SQUAT"] and u >= 1.0:
            self._enter_takeoff()
            self._enter(LJ.PH["TAKEOFF"])
        elif ph == LJ.PH["TAKEOFF"]:
            if self.air_time >= 0.03 and self.liftoff is None:
                self.liftoff = dict(t=self.t, x=float(self.d.qpos[0]),
                                    toe_x=self.foot_x()["L"] + LJ.TOE_LEN,
                                    vx=float(self.d.qvel[0]), vz=float(self.d.qvel[2]))
                # 可选叠加冲量（默认 0）
                self.d.qvel[0] += self.inject[0]
                self.d.qvel[2] += self.inject[1]
                self._enter(LJ.PH["FLIGHT"])
            elif u > 0.6 / self.p.t_takeoff and self.liftoff is None:
                self.failed_reason = "蹬伸 0.6 s 仍未离地（推力不足）"
                return False
        elif ph == LJ.PH["FLIGHT"]:
            if self.foot_contact() and self.liftoff is not None:
                self.d.qvel[:] = 0.0
                fx = self.foot_x()
                self.anchors = fx
                self.touchdown = dict(t=self.t, x=float(self.d.qpos[0]),
                                      heel_x=min(fx.values()) - LJ.TOE_LEN)
                self._enter(LJ.PH["ABSORB"])
            elif self.t - self.phase_t0 > 1.5:
                self.failed_reason = "腾空 1.5 s 未触地（异常）"
                return False
        elif ph == LJ.PH["ABSORB"] and u >= 1.0:
            self._enter(LJ.PH["SETTLE"])
        elif ph == LJ.PH["SETTLE"] and u >= 1.0:
            self._enter(LJ.PH["RISE"])
        elif ph == LJ.PH["RISE"] and u >= 1.0:
            self._enter(LJ.PH["HOLD"])
        elif ph == LJ.PH["HOLD"] and u >= 1.0:
            return False
        return True


# ---------------------------------------------------------------- 指标
def evaluate(p: LJ.Params, cfg: dict) -> dict:
    sim = LongJumpReal(p, cfg)
    sim.run()
    mt = sim.metrics()
    # 真实自驱起跳速度（inject=0 时即纯 PD 产出）
    if sim.liftoff:
        vx, vz = sim.liftoff["vx"], sim.liftoff["vz"]
    else:
        vx = vz = float("nan")
    ok, why = sim.verdict(mt)
    if sim.liftoff and sim.touchdown:
        real_dist = mt.get("toe_heel_dist", 0.0)
    else:
        real_dist = 0.0
    # 质心峰值高度（跳高核心指标）：从日志里取 subtree_com 的 z 极大值
    com = np.array(sim.log["com"], dtype=float)
    start_z = float(com[0, 2]) if com.size else float("nan")
    peak_z = float(np.max(com[:, 2])) if com.size else float("nan")
    liftoff_z = (float(np.interp(sim.liftoff["t"], sim.log["t"], com[:, 2]))
                 if sim.liftoff else float("nan"))
    return dict(cfg=cfg, vx=vx, vz=vz, dist=real_dist,
                peak_com_z=peak_z, start_com_z=start_z,
                jump_height=peak_z - start_z,           # 相对站立起点的跳高
                aerial_rise=vz * vz / (2 * 9.81) if sim.liftoff else float("nan"),
                liftoff_com_z=liftoff_z,
                flight=mt.get("flight_time", 0.0),
                hold=mt.get("hold_time", 0.0),
                final_z=mt.get("final_z", float("nan")),
                final_pitch=mt.get("final_pitch_deg", float("nan")),
                margin=mt.get("final_margin", float("nan")),
                tau=mt.get("tau_peak", float("nan")),
                ok=bool(ok and sim.liftoff is not None),
                fail=sim.failed_reason)


def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 真实自驱立定跳远（无冲量注入）")
    ap.add_argument("--cfg", default=None,
                    help="覆盖参数 k=v,k=v（rise/pf/tt/arm_up/...）")
    ap.add_argument("--inject", nargs=2, type=float, default=None,
                    metavar=("VX", "VZ"), help="叠加冲量（明确标注，默认 0）")
    ap.add_argument("--rise", type=float, default=None)
    ap.add_argument("--lean", type=float, default=13.0,
                    help="深蹲前倾 deg（向后蹬版用 8 左右效果最好）")
    ap.add_argument("--pf", type=float, default=None)
    ap.add_argument("--tt", type=float, default=None)
    ap.add_argument("--arm-scale", type=float, default=None,
                    help="摆臂幅度放大系数（绕后摆基准；>1 摆得更开、上挥更猛）")
    ap.add_argument("--arm-rate", type=float, default=None,
                    help="摆臂集中度指数；越大越集中在蹬伸末段")
    ap.add_argument("--kp", type=float, default=None,
                    help="蹬腿 PD 刚度倍率（力矩限幅内决定蹬伸力度）")
    ap.add_argument("--drop", type=float, default=None,
                    help="深蹲下降量 m（更深=腿伸得更长=冲量更大）")
    ap.add_argument("--anchor", type=float, default=None,
                    help="脚目标相对髋的世界 x 偏移 m（<0=命令向后蹬地）")
    ap.add_argument("--vx", type=float, default=0.0, help="注入 vx（默认 0=纯自驱）")
    ap.add_argument("--vz", type=float, default=0.0, help="注入 vz（默认 0=纯自驱）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--json", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="扫多组 rise/pf/tt，打印真实自驱指标表")
    args = ap.parse_args()

    base = LJ.Params(drop_squat=0.25, lean_squat=13.0,
                     vx_takeoff=0.0, vz_takeoff=0.0,
                     pf_to=0.15, arm_up=-0.60, dxw_to=-0.06,
                     land_reach=0.22, kp_scale=8.0, kd_scale=8.0, sim_hz=1000)

    def parse_cfg(s):
        cfg = {}
        if s:
            for kv in s.split(","):
                k, v = kv.split("=")
                cfg[k.strip()] = float(v)
        return cfg

    if args.sweep:
        rises = [1.04, 1.08]
        pfs = [0.25, 0.45]
        tts = [0.30]
        shifts = [-0.08, -0.12, -0.16, -0.20, -0.24]
        base.lean_squat = 8.0
        print(f"{'shift':>6} {'rise':>5} {'pf':>5} | {'vx':>7} {'vz':>7} "
              f"{'dist':>7} {'flight':>7} {'hold':>5} {'pitch':>6} {'margin':>7} {'tau':>5}  ok")
        print("-" * 78)
        best = None
        for sh in shifts:
            for r in rises:
                for pf in pfs:
                    for tt in tts:
                        base.t_takeoff = tt
                        cfg = dict(rise=r, pf_end=pf, anchor_shift=sh)
                        if args.arm_scale is not None:
                            cfg["arm_scale"] = args.arm_scale
                        if args.arm_rate is not None:
                            cfg["arm_rate"] = args.arm_rate
                        if args.inject:
                            cfg["inject"] = tuple(args.inject)
                        res = evaluate(base, cfg)
                        tag = "Y" if res["ok"] else ("-" if res["fail"] else "n")
                        print(f"{sh:6.2f} {r:5.2f} {pf:5.2f} | "
                              f"{res['vx']:+7.2f} {res['vz']:+7.2f} "
                              f"{res['dist']*100:6.0f}cm {res['flight']:7.2f} "
                              f"{res['hold']:5.2f} {res['final_pitch']:+6.1f} "
                              f"{res['margin']*100:6.1f}cm {res['tau']*100:4.0f}%  {tag}")
                        if res["ok"] and (best is None or res["dist"] > best["dist"]):
                            best = res
        print("-" * 78)
        if best:
            print(f"[最优纯自驱] shift={best['cfg'].get('anchor_shift')} "
                  f"rise={best['cfg']['rise']} pf={best['cfg']['pf_end']} → "
                  f"真实跳距 {best['dist']*100:.0f} cm   "
                  f"(vx={best['vx']:+.2f}, vz={best['vz']:+.2f} m/s)")
        return

    # 单组运行
    cfg = parse_cfg(args.cfg or "")
    if args.rise is not None:
        cfg["rise"] = args.rise
    if args.lean is not None:
        base.lean_squat = args.lean
    if args.pf is not None:
        cfg["pf_end"] = args.pf
    if args.tt is not None:
        base.t_takeoff = args.tt
    if args.arm_scale is not None:
        cfg["arm_scale"] = args.arm_scale
    if args.arm_rate is not None:
        cfg["arm_rate"] = args.arm_rate
    if args.kp is not None:
        base.kp_scale = args.kp
    if args.drop is not None:
        base.drop_squat = args.drop
    if args.anchor is not None:
        cfg["anchor_shift"] = args.anchor
    if args.inject:
        cfg["inject"] = tuple(args.inject)
    elif args.vx or args.vz:
        cfg["inject"] = (args.vx, args.vz)

    sim = LongJumpReal(base, cfg)
    sim.run()
    mt = sim.metrics()
    # 单跑路径补充“跳高”指标（ experiment 的 evaluate() 已含，这里补齐）
    _com = np.array(sim.log["com"], dtype=float)
    if _com.size:
        mt["start_com_z"] = float(_com[0, 2])
        mt["peak_com_z"] = float(np.max(_com[:, 2]))
        mt["jump_height"] = mt["peak_com_z"] - mt["start_com_z"]
    ok, why = sim.verdict(mt)
    if sim.liftoff and sim.touchdown:
        mt["toe_heel_dist"] = mt.get("toe_heel_dist", 0.0)
    vx = sim.liftoff["vx"] if sim.liftoff else float("nan")
    vz = sim.liftoff["vz"] if sim.liftoff else float("nan")
    print(f"\n[真实自驱跳远] rise={cfg.get('rise')} pf_end={cfg.get('pf_end')} "
          f"t_takeoff={base.t_takeoff}  注入={cfg.get('inject',(0,0))}")
    if sim.liftoff:
        print(f"[离地] vx={vx:+.2f} m/s  vz={vz:+.2f} m/s  (纯 PD 产出，无注入时为真实值)")
        print(f"[跳高] 离地后自由上升 {vz*vz/(2*9.81)*100:5.1f} cm   "
              f"质心峰值 {mt.get('peak_com_z',float('nan')):.3f} m   "
              f"相对站立起点跳高 {mt.get('jump_height',float('nan'))*100:5.1f} cm")
    else:
        print(f"[失败] {sim.failed_reason}")
    print(f"[跳距] {mt.get('toe_heel_dist',0)*100:.0f} cm   保持 {mt.get('hold_time',0):.2f}s")
    print(f"[判定] {'✔ 通过' if ok else '✘ ' + str(why)}")

    out = Path(args.out)
    np.savez(out,
             time=np.array(sim.log["t"]),
             qpos=np.array(sim.log["qpos"]),
             phase=np.array(sim.log["phase"], dtype=np.int8),
             dt=base.dt,
             joint_names=SQ.JOINT_NAMES,
             meta=dict(params=asdict(base), cfg=cfg, metrics=mt, ok=bool(ok),
                       vx_liftoff=float(vx) if sim.liftoff else None,
                       vz_liftoff=float(vz) if sim.liftoff else None,
                       version="real-selfpowered"))
    print(f"[OK] 轨迹 -> {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(cfg=cfg, metrics=mt),
                                             indent=2, ensure_ascii=False),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
