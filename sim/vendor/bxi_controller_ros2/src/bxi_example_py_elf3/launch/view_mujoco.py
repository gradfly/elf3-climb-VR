"""MuJoCo 原生查看器 —— 用于核对机器人各部件的**原始装配状态**。

用途
----
把 elf3.xml（已 <include> scene_bx.xml，含地面与楼梯）直接用 MuJoCo 自己的
OpenGL 查看器打开，**完全不经过 vuer / three.js 的任何坐标变换**。
这样就能一眼判断：

  * 如果这里部件角度正常  -> 模型本身没问题，问题在渲染端（vuer/three.js）的换算；
  * 如果这里部件角度也歪  -> 问题在 XML / STL 网格本身。

用法
----
  python view_mujoco.py              # 默认 hold：静态保持初始装配姿态（推荐先用这个核对角度）
  python view_mujoco.py --sim        # 真实物理：跑 mj_step（无控制器，自由基座会倒地）
  python view_mujoco.py --pin        # 钉住躯干位姿，只让关节在重力下自由摆动
  python view_mujoco.py --drop       # 先把基座下移，消除脚底悬空（落地对齐）
  python view_mujoco.py --zero       # 关节全部归零（MuJoCo 默认位形）
  python view_mujoco.py --dump       # 仅在终端打印各部件位姿表，不开窗口

窗口内快捷键（鼠标左键旋转 / 滚轮缩放 / 右键平移）
--------------------------------------------------
  G  切换 物理运行 / 静止
  B  切换 基座 钉住 / 自由
  R  复位到初始位形
  Z  关节全部归零
  D  落地对齐（消除脚底悬空）
  M  在终端打印当前各部件位姿表
  （左上角 "Pause" 按钮也可暂停）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

CWD = Path(__file__).resolve().parent
# elf3.xml（连同 meshes/、sensors/、terrains/）位于同级 data/mujoco_simulation/ 下
MODEL_DIR = CWD.parent / "data" / "mujoco_simulation"
DEFAULT_XML = MODEL_DIR / "elf3.xml"

# MuJoCo 查看器把 key_callback 收到的键码交给我们，这里只用字母键。
GLFW_G = 71
GLFW_B = 66
GLFW_R = 82
GLFW_Z = 90
GLFW_D = 68
GLFW_M = 77

VISUAL_GROUP = 2  # elf3.xml 里 class="visual" 的 mesh geom 都在 group 2


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def visual_geoms(model):
    """返回所有视觉 mesh geom 的 (gid, 所属 body 名, 名称) 列表。"""
    out = []
    for g in range(model.ngeom):
        if int(model.geom_group[g]) != VISUAL_GROUP:
            continue
        bid = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        out.append((g, bname, gname))
    return out


def foot_clearance(model, data):
    """脚底碰撞 geom 的世界最低点相对 z=0 的高度（>0 表示悬空）。"""
    lowest = float("inf")
    for g in range(model.ngeom):
        if int(model.geom_group[g]) != 3:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        gtype = int(model.geom_type[g])
        if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            # capsule 在 geom_xmat 的局部 z 轴上：fromto 已烘焙进 xpos/xmat
            half = float(model.geom_size[g][1])
            ax = data.geom_xmat[g].reshape(3, 3)[:, 2]
            z = data.geom_xpos[g][2] - half * abs(float(ax[2]))
            r = float(model.geom_size[g][0])
            lowest = min(lowest, z - r)
        elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            r = float(model.geom_size[g][0])
            lowest = min(lowest, float(data.geom_xpos[g][2]) - r)
    return lowest


def qpos_base_index(model):
    """返回自由关节在 qpos 中的起始下标（取第一个 freejoint）。"""
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(model.jnt_qposadr[j]), j
    return None, None


def dump_pose_table(model, data, title="部件原始位姿"):
    """打印所有视觉 mesh geom 的世界位姿（MuJoCo Z-up 世界系）。"""
    geoms = visual_geoms(model)
    print(f"\n=== {title}（MuJoCo 世界系 Z-up，共 {len(geoms)} 个部件）===")
    print(f"{'gid':>4} {'body':<22} {'geom':<24} "
          f"{'X':>9} {'Y':>9} {'Z':>9}   {'R 对角(接近1=无扭转)':>22}")
    for g, bname, gname in geoms:
        p = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        diag = f"{R[0,0]:+.3f} {R[1,1]:+.3f} {R[2,2]:+.3f}"
        print(f"{g:>4} {bname:<22} {gname:<24} "
              f"{p[0]:+9.4f} {p[1]:+9.4f} {p[2]:+9.4f}   {diag:>22}")
    print()


def body_pose_summary(model, data):
    """按 body 汇总：位置 + 姿态（四元数），便于核对层级装配。"""
    print(f"\n=== 各 body 世界位姿（共 {model.nbody} 个）===")
    print(f"{'bid':>4} {'body':<24} {'X':>9} {'Y':>9} {'Z':>9}   {'quat(w,x,y,z)':>28}")
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        p = data.xpos[b]
        q = data.xquat[b]
        print(f"{b:>4} {name:<24} {p[0]:+9.4f} {p[1]:+9.4f} {p[2]:+9.4f}   "
              f"{q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}")
    print()


# --------------------------------------------------------------------------
# 主程序
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="MuJoCo 原生查看器：核对机器人原始装配状态",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--xml", default=str(DEFAULT_XML), help="模型文件（默认 elf3.xml）")
    ap.add_argument("--sim", action="store_true", help="真实物理：持续 mj_step，基座自由")
    ap.add_argument("--pin", action="store_true", help="钉住躯干位姿，关节自由摆动")
    ap.add_argument("--drop", action="store_true", help="落地对齐：基座下移消除脚底悬空")
    ap.add_argument("--zero", action="store_true", help="关节全部归零")
    ap.add_argument("--dump", action="store_true", help="只打印位姿表，不开窗口")
    args = ap.parse_args()

    xml = Path(args.xml)
    if not xml.is_absolute():
        xml = (MODEL_DIR / xml).resolve()
    if not xml.exists():
        print(f"[错误] 找不到模型文件: {xml}")
        return 1

    print(f"加载模型: {xml}")
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    print(f"  nbody={model.nbody}  ngeom={model.ngeom}  "
          f"njnt={model.njnt}  nu={model.nu}  nq={model.nq}")

    # ---- 初始位形 ----
    if args.zero:
        data.qpos[:] = 0.0
        adr, _ = qpos_base_index(model)
        if adr is not None:
            data.qpos[adr + 2] = 1.1  # 自由关节的 z 给个合理值，否则会从 z=0 起
        print("  [位形] 关节全部归零")
    mujoco.mj_forward(model, data)

    # ---- 落地对齐 ----
    if args.drop:
        gap = foot_clearance(model, data)
        adr, _ = qpos_base_index(model)
        if adr is not None and np.isfinite(gap):
            data.qpos[adr + 2] -= gap
            mujoco.mj_forward(model, data)
            print(f"  [落地] 脚底悬空 {gap:+.5f} m -> 基座下移，z={data.qpos[adr+2]:.5f}")

    gap = foot_clearance(model, data)
    print(f"  脚底相对 z=0 的高度: {gap:+.5f} m "
          f"({'悬空' if gap > 1e-4 else '贴地' if abs(gap) <= 1e-4 else '插入地面'})")

    dump_pose_table(model, data, "开机初始位姿")

    if args.dump:
        body_pose_summary(model, data)
        return 0

    # ---- 运行模式 ----
    if args.sim or args.pin:
        mode = "pin" if args.pin else "sim"
    else:
        mode = "hold"
    print(f"[模式] {mode}  "
          f"({'静止保持装配姿态' if mode == 'hold' else '钉住躯干+关节自由' if mode == 'pin' else '完整物理仿真（会倒）'})")

    qpos0 = data.qpos.copy()
    base_adr, _ = qpos_base_index(model)
    state = {"run": mode != "hold", "pin": mode == "pin"}

    def do_reset(d, keep_flags=True):
        d.qpos[:] = qpos0
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        state["run"] = False

    def do_zero(d):
        d.qpos[:] = 0.0
        if base_adr is not None:
            d.qpos[base_adr + 2] = 1.1
            d.qpos[base_adr + 3] = 1.0  # quat w
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        state["run"] = False

    def do_drop(d):
        g = foot_clearance(model, d)
        if base_adr is not None and np.isfinite(g):
            d.qpos[base_adr + 2] -= g
            mujoco.mj_forward(model, d)
        state["run"] = False

    def on_key(keycode):
        if keycode == GLFW_G:
            state["run"] = not state["run"]
            print(f"[切换] 物理运行 = {state['run']}")
        elif keycode == GLFW_B:
            state["pin"] = not state["pin"]
            print(f"[切换] 钉住基座 = {state['pin']}")
        elif keycode == GLFW_R:
            do_reset(data)
            print("[复位] 回到初始位形")
        elif keycode == GLFW_Z:
            do_zero(data)
            print("[归零] 所有关节 = 0")
        elif keycode == GLFW_D:
            do_drop(data)
            print("[落地] 已消除脚底悬空")
        elif keycode == GLFW_M:
            dump_pose_table(model, data, "当前位姿")

    print("\n窗口已打开。快捷键: G=运行/静止  B=钉住基座  R=复位  Z=归零  D=落地  M=打印位姿")
    print("鼠标: 左键旋转 / 滚轮缩放 / 右键平移。关闭窗口即退出。\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        # 相机：稍微拉远一点看全身
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.85]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0

        last = time.perf_counter()
        while viewer.is_running():
            t0 = time.perf_counter()
            if state["run"]:
                if state["pin"] and base_adr is not None:
                    data.qpos[base_adr:base_adr + 7] = qpos0[base_adr:base_adr + 7]
                    data.qvel[base_adr:base_adr + 6] = 0.0
                mujoco.mj_step(model, data)
                if state["pin"] and base_adr is not None:
                    data.qpos[base_adr:base_adr + 7] = qpos0[base_adr:base_adr + 7]
                    data.qvel[base_adr:base_adr + 6] = 0.0
                    mujoco.mj_forward(model, data)
            viewer.sync()

            # 限制到 ~60 fps，免得空转吃满 CPU
            dt = time.perf_counter() - t0
            sleep = (1.0 / 60.0) - dt
            if sleep > 0:
                time.sleep(sleep)
            last = t0

    print("查看器已关闭。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
