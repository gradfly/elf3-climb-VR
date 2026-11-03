#!/usr/bin/env python3
"""播训练用的参考步态表 `crab_ref_v1.npz`（不是 crab_climb_traj.npz）。

训练时 ref_table.py 按 (rise, tread, phase) 查这张表插值出 q_ref；本脚本把同一张表
的一个档位按整周期循环播出来，用来肉眼检查"参考步态到底长什么样"。

表里除了 29 个关节角 q，还存了规划器解算出的躯干位姿 root_y / root_z 和每周期
横移 dy_cycle，所以能还原真实上楼运动：一个 90 帧周期（1.8 s / 两步）躯干沿 +Y
前进一个 tread、升高一个 rise，脚踩在对应踏面上。

和 crab_replay.py 的区别：crab_replay 播的是 crab_climb.py 在某条具体楼梯上的
全程轨迹；本脚本播的是训练真正查表用的周期步态模板。

用法
----
  python scripts/show_ref_gait.py                   # 默认 rise=0.18 tread=0.30
  python scripts/show_ref_gait.py --rise 0.10       # 换台阶高（就近选档）
  python scripts/show_ref_gait.py --tread 0.36 --cycles 4
  python scripts/show_ref_gait.py --speed 0.3 --loop
  python scripts/show_ref_gait.py --no-gui          # 无窗口自检（服务器/无显示环境）

窗口快捷键：空格 暂停/继续、R 重播、[ ] 减速/加速、T 相机跟随开关。
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
ROBOT_XML = DATA / "elf3_robot_only.xml"
DEFAULT_TABLE = REPO / "train/assets/ref_tables/crab_ref_v1.npz"

SCENE_WIDTH = 1.2        # 楼梯净宽（沿 X），与 gen_crab_stairs 默认一致


def _load_table(path: Path) -> dict:
  z = np.load(path, allow_pickle=True)
  return {k: z[k] for k in z.files}


def _nearest(arr: np.ndarray, val: float) -> int:
  return int(np.argmin(np.abs(np.asarray(arr, dtype=np.float64) - val)))


def _scene_xml(rise: float, tread: float, n_steps: int) -> str:
  """机器人 + 平面地 + 与 (rise, tread) 匹配的均匀楼梯。

  表里的平台坐标约定：第 i 级占 y ∈ [i*tread, (i+1)*tread]、顶面 z = (i+1)*rise。
  rise=0 时直接不放台阶（平地横移）。
  """
  boxes = []
  if rise > 1e-6:
    for i in range(n_steps):
      top = (i + 1) * rise
      boxes.append(
          f'<geom name="step_{i}" type="box" pos="0 {(i + 0.5) * tread:.4f} {top / 2:.4f}" '
          f'size="{SCENE_WIDTH / 2:.3f} {tread / 2:.4f} {top / 2:.4f}" '
          f'rgba="0.62 0.58 0.52 1" contype="1" conaffinity="1"/>')
  return (
      '<mujoco model="crab_ref_gait">\n'
      f'  <include file="{ROBOT_XML.as_posix()}"/>\n'
      '  <worldbody>\n'
      '    <geom name="floor" type="plane" size="3 9 0.05" pos="0 3 0" rgba="0.30 0.30 0.32 1"/>\n'
      f'    {chr(10).join(boxes)}\n'
      '  </worldbody>\n'
      '</mujoco>\n')


def _load_model(xml: str) -> mujoco.MjModel:
  """from_xml_string 的 include 里 meshdir="./meshes/" 按当前目录解析，
  所以临时切到 DATA 目录加载，再切回来。"""
  cwd = os.getcwd()
  os.chdir(DATA)
  try:
    return mujoco.MjModel.from_xml_string(xml)
  finally:
    os.chdir(cwd)


def _build_frames(model: mujoco.MjModel, tb: dict, ir: int, it: int, cycles: int) -> np.ndarray:
  """把表里的一个档位拼成连续 qpos 序列（每周期 +tread 横移、+rise 升高）。"""
  q = tb["q"][ir, it]                       # (P, 29)
  ry = tb["root_y"][ir, it]                 # (P,)
  rz = tb["root_z"][ir, it]
  dy = float(tb["dy_cycle"][ir, it])
  rise = float(tb["rise"][ir])
  period = q.shape[0]

  adr = [int(model.jnt_qposadr[
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(n))])
      for n in tb["joint_names"]]
  frames = np.zeros((cycles * period, model.nq))
  for c in range(cycles):
    for k in range(period):
      f = frames[c * period + k]
      f[0:7] = (0.0, ry[k] + c * dy, rz[k] + c * rise, 1.0, 0.0, 0.0, 0.0)
      f[adr] = q[k]
  return frames


def _run_gui(model, data, frames, torso_id, args):
  import mujoco.viewer as mjv

  state = {"paused": False, "speed": float(args.speed), "restart": False, "track": True}

  def on_key(keycode: int):
    if keycode == 32:                                   # 空格
      state["paused"] = not state["paused"]
    elif keycode in (ord("R"), ord("r")):
      state["restart"] = True
    elif keycode == ord("["):
      state["speed"] = max(0.1, state["speed"] * 0.7)
      print(f"[info] speed = {state['speed']:.2f}x")
    elif keycode == ord("]"):
      state["speed"] = min(4.0, state["speed"] / 0.7)
      print(f"[info] speed = {state['speed']:.2f}x")
    elif keycode in (ord("T"), ord("t")):
      state["track"] = not state["track"]

  viewer = mjv.launch_passive(model, data, key_callback=on_key)
  with viewer:
    viewer.cam.distance = args.dist
    viewer.cam.azimuth = args.az
    viewer.cam.elevation = args.el
    n = len(frames)
    dt = args.dt
    t0 = time.perf_counter()
    i = 0
    while viewer.is_running():
      if state["restart"]:
        t0 = time.perf_counter()
        state["restart"] = False
        state["paused"] = False
      if not state["paused"]:
        i = int((time.perf_counter() - t0) / (dt / state["speed"]))
        if i >= n:
          if args.loop:
            i %= n
          else:
            i = n - 1
            state["paused"] = True
      data.qpos[:] = frames[i]
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      if state["track"]:
        viewer.cam.lookat[:] = data.xpos[torso_id]
      viewer.sync()
      time.sleep(0.004)


def main() -> int:
  ap = argparse.ArgumentParser(description="播放训练参考步态表 crab_ref_v1.npz")
  ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
  ap.add_argument("--rise", type=float, default=0.18, help="目标台阶高，m（就近选档）")
  ap.add_argument("--tread", type=float, default=0.30, help="目标踏面进深，m（就近选档）")
  ap.add_argument("--cycles", type=int, default=3, help="播几个步态周期（1 周期=2 步=1.8 s）")
  ap.add_argument("--speed", type=float, default=1.0, help="播放倍速")
  ap.add_argument("--loop", action="store_true", help="循环播放")
  ap.add_argument("--dist", type=float, default=3.0, help="相机距离")
  ap.add_argument("--az", type=float, default=90.0, help="相机方位角")
  ap.add_argument("--el", type=float, default=-45.0, help="相机俯仰角（负值=从上往下看）")
  ap.add_argument("--no-gui", action="store_true", help="不开窗口，只做数据/模型自检")
  args = ap.parse_args()

  tb = _load_table(args.table)
  ir = _nearest(tb["rise"], args.rise)
  it = _nearest(tb["tread"], args.tread)
  rise = float(tb["rise"][ir])
  tread = float(tb["tread"][it])
  ok = bool(tb["ok"][ir, it])
  period = int(tb["period_frames"])
  dt = float(tb["dt"])
  args.dt = dt
  print(f"[info] 档位 rise={rise:.3f} m  tread={tread:.3f} m  "
        f"周期={period} 帧 / {period * dt:.2f} s  v_ref={float(tb['v_ref'][ir, it]):.3f} m/s")
  print(f"[info] 每周期横移 dy_cycle={float(tb['dy_cycle'][ir, it]):.3f} m，"
        f"IK 最大误差 err_pos_max={float(tb['err_pos_max'][ir, it]) * 1000:.2f} mm  ok={ok}")
  if not ok or float(tb["err_pos_max"][ir, it]) > 0.01:
    print("[warn] 这档 IK 解算质量差（ok=False 或误差 >10 mm），动画里的脚可能会离踏面")

  model = _load_model(_scene_xml(rise, tread, args.cycles + 2))
  data = mujoco.MjData(model)
  frames = _build_frames(model, tb, ir, it, args.cycles)
  torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
  if torso_id < 0:
    torso_id = 1
  print(f"[info] qpos 帧 {len(frames)} 个，总时长 {len(frames) * dt:.2f} s")

  if args.no_gui:
    for f in frames:
      data.qpos[:] = f
      mujoco.mj_kinematics(model, data)
    data.qpos[:] = frames[-1]
    mujoco.mj_kinematics(model, data)
    print(f"[info] 自检OK：末帧躯干 y={data.xpos[torso_id][1]:.3f} "
          f"z={data.xpos[torso_id][2]:.3f}（起点 y={frames[0][1]:.3f} z={frames[0][2]:.3f}）")
    return 0

  try:
    _run_gui(model, data, frames, torso_id, args)
  except Exception as e:
    print(f"[跳过] 打不开 MuJoCo 窗口（{e}）；可用 --no-gui 自检")
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
