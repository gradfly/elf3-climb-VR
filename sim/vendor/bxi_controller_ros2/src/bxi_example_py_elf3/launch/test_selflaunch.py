"""实测：关掉冲量注入，纯靠 PD + 这套动作，ELF3 自己能蹿多快。

用于回答"137cm 是不是真实物理能跳出来的"——直接测机器人自身
执行力能给质心多大的起跳速度，再反推真实可达跳距。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import mujoco
import long_jump as LJ
import long_jump_v2 as LJ2

# ---- 物理常量（从模型里读真值，不靠记忆）----
import squat_stand as SQ

# 1) 临时把 base._advance 包一层：起跳段每控制步记录真实基座线速度
_orig_advance = LJ.LongJump._advance


def patched(self):
    if self.phase == LJ.PH["TAKEOFF"]:
        self._dbg.append((self.t,
                          float(self.d.qpos[2]),
                          float(self.d.qvel[0]),   # 真实水平速度
                          float(self.d.qvel[2]),   # 真实垂直速度
                          float(self.air_time)))
    return _orig_advance(self)


LJ.LongJump._advance = patched

# 2) 关键：让"注入的冲量" = 0，看 PD 自己能产出多少
p = LJ.Params(vx_takeoff=0.0, vz_takeoff=0.0,
              drop_squat=0.25, lean_squat=13.0,
              pf_to=0.15, dxw_to=-0.06, land_reach=0.22,
              kp_scale=8.0, kd_scale=8.0, sim_hz=1000)

sim = LJ2.LongJumpV2(p)
sim._dbg = []

# 读真实质量 / 力矩限幅
m = sim.m
total_mass = float(m.body_mass.sum())
total_mass = float(np.sum([m.body_mass[i] for i in range(m.nbody)]))
# actuatorfrcrange
names = sim.names
tau = {n: sim.tau_lim[n] for n in names}

sim.run()
dbg = np.array(sim._dbg) if sim._dbg else np.zeros((0, 5))

print("=" * 64)
print("ELF3 自身执行力（无冲量注入）实测")
print("=" * 64)
print(f"[模型] 总质量     = {total_mass:.1f} kg")
print(f"[力矩限幅] 踝={tau.get('l_ankle_y_joint',0):.0f}  髋={tau.get('l_hip_y_joint',0):.0f}  "
      f"膝={tau.get('l_knee_y_joint',0):.0f}  N·m（左右同值）")

if dbg.size:
    t, z, vx, vz, air = dbg[:, 0], dbg[:, 1], dbg[:, 2], dbg[:, 3], dbg[:, 4]
    peak_vz = float(np.max(vz))
    peak_vx = float(np.max(vx))
    min_vx = float(np.min(vx))
    max_air = float(np.max(air))
    lift_off = bool(np.any(air >= 0.03))
    print(f"\n[起跳段 真实速度]")
    print(f"  垂直 vz 峰值   = {peak_vz:+.3f} m/s   （>0 才离得开地）")
    print(f"  水平 vx 峰值   = {peak_vx:+.3f} m/s   （正值=向前）")
    print(f"  水平 vx 谷值   = {min_vx:+.3f} m/s   （负值=向后，说明腿在往后推）")
    print(f"  连续离地时长   = {max_air:.3f} s   （≥0.03 才算真离地）")
    print(f"  → 真实是否离地 = {'是' if lift_off else '否（始终贴地/最多微颤）'}")
    if lift_off:
        tflight = 2 * peak_vz / 9.81
        d_real = peak_vx * tflight
        print(f"  → 若靠自身力：跳距≈ {d_real*100:.0f} cm（vx*t_flight，t={tflight:.2f}s）")
    else:
        print("  → 机器人自己根本起不来，137cm 完全靠脚本注入的 3.0/1.7 m/s 冲量")
else:
    print("[无起跳数据]")

print("=" * 64)
