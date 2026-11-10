"""诊断动作轨迹的「柔顺性」：角加速度 / jerk（加加速度）分布与切换点跳变。

用法：
  python scripts/check_prone_smooth.py [traj.npz]
默认 fast 档动作 1。
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
TR = sys.argv[1] if len(sys.argv) > 1 else "prone_standup_fast_traj.npz"

# 阶段切换归一化时刻（动作 1，v9 四阶段；与 prone_crawl.ENTRY_STAGES 对齐）
U_CROUCH = 0.24         # 蹲底（脚仍在站姿触地点）
U_PLANT = 0.44          # 双手撑地（此后手是固定支点）
U_KICK = 0.74           # 双脚后伸到位
U_EXIT = U_KICK         # 收腿窗口开始（关节空间收敛到 gait 首帧）
JN = None


def grad(a, dt):
    return np.gradient(a, dt, axis=0)


def main():
    global JN
    z = np.load(DATA / TR, allow_pickle=True)
    q = z["qpos"].astype(float)
    dt = float(z["dt"])
    names = [str(x) for x in z["joint_names"]] if "joint_names" in z else None
    if names is None:
        nq = q.shape[1]
        names = [f"j{i}" for i in range(nq)]
    JN = names
    nctrl = len(names)
    J = q[:, 7:7 + nctrl]             # 受控关节（去掉 7 维 free joint）
    vel = grad(J, dt)
    acc = grad(vel, dt)
    jerk = grad(acc, dt)
    T = dt * max(1, len(q) - 1)       # 不写死 5 s，normal/fast 都能用

    # ---- 角加速度峰值 ----
    amag = np.abs(acc)
    print(f"轨迹 {TR}  n={len(q)} dt={dt}  关节数={J.shape[1]}")
    print("\n== 角加速度峰值 top 18 (帧, 关节, rad/s^2, u) ==")
    idx = np.dstack(np.unravel_index(np.argsort(amag.ravel())[::-1][:18], amag.shape))[0]
    for i, j in idx:
        print(f"  f{i:3d} u={i*dt/T:.2f}  {names[j]:22s} {amag[i,j]:9.1f}")
    print(f"  -> 角加速度全局峰值 {amag.max():.1f} rad/s^2")

    # ---- jerk 峰值 ----
    jmag = np.abs(jerk)
    print("\n== jerk（角加速度的变化率）峰值 top 12 (帧, 关节, rad/s^3) ==")
    idxj = np.dstack(np.unravel_index(np.argsort(jmag.ravel())[::-1][:12], jmag.shape))[0]
    for i, j in idxj:
        print(f"  f{i:3d} u={i*dt/T:.2f}  {names[j]:22s} {jmag[i,j]:11.0f}")
    print(f"  -> jerk 全局峰值 {jmag.max():.0f} rad/s^3")

    # ---- 切换点附近的加速度/jerk 剖面 ----
    print("\n== 切换点附近的角加速度（每 5 帧打印峰值，单位 rad/s^2）==")
    for label, u0, u1 in [("蹲下", 0.00, U_CROUCH),
                          ("双手撑地", U_CROUCH, U_PLANT),
                          ("双脚后伸", U_PLANT, U_KICK),
                          ("趴姿/对齐", U_EXIT, 1.00)]:
        i0, i1 = int(u0 * len(q)), int(u1 * len(q))
        win = amag[i0:i1]
        print(f"  {label:8s} u={u0:.2f}~{u1:.2f}: 峰值 {win.max():7.1f}  "
              f"中位 {np.median(win):6.1f}  jerk峰值 {jmag[i0:i1].max():9.0f}")

    # ---- 速度剖面是否出现「平台」（连续多帧速度≈恒定 = 被限速削顶）----
    spd = np.linalg.norm(vel, axis=1)
    vmax_j = np.abs(vel).max(axis=0)
    print("\n== 单关节最大速度 top 8 (关节, rad/s, 帧) ==")
    for j in np.argsort(vmax_j.ravel())[::-1][:8]:
        print(f"  {names[j]:22s} {vmax_j[j]:7.2f}  (f{np.argmax(np.abs(vel[:,j]))})")
    # 检测限速平台：某帧整段速度接近某上限且前后恒定
    thr = 9.0  # rad/s 约 515 deg/s
    flat = 0
    for i in range(2, len(q) - 2):
        if spd[i] > thr and abs(spd[i] - spd[i - 1]) < 0.05 * spd[i] and abs(spd[i] - spd[i + 1]) < 0.05 * spd[i]:
            flat += 1
    print(f"\n  速度 > {thr} rad/s 且前后基本恒定的「平台」帧数：{flat}（多=限速削顶造成卡顿观感）")
    print(f"  整段速度峰值 {spd.max():.2f} rad/s")


if __name__ == "__main__":
    main()
