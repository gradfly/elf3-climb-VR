#!/usr/bin/env python3
"""生成 mjlab 可用的纯机器人资产：elf3_robot_only.xml

从 data/mujoco_simulation/elf3.xml 派生，做两件事：
  1) 删掉地形 include（mjlab 的 SceneCfg 自己管地形）
  2) 删掉 <actuator> 段（mjlab 用 BuiltinPositionActuatorCfg 注入位置执行器）

输出到同目录（保证 meshdir="./meshes/" 仍可用），并做一次 MuJoCo 编译自检。

用法：
    python train/assets/make_robot_xml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # 仓库根
DATA = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
SRC = DATA / "elf3.xml"
DST = DATA / "elf3_robot_only.xml"

TERRAIN_INCLUDE = '<include file="terrains/universal_test/universal_test_terrain_compact.xml"/>'


def main() -> int:
    if not SRC.exists():
        print(f"[ERR] 找不到 {SRC}")
        return 1

    txt = SRC.read_text(encoding="utf-8")

    # 1) 去地形 include
    before = txt.count(TERRAIN_INCLUDE)
    txt = txt.replace(TERRAIN_INCLUDE, "<!-- terrain removed for mjlab -->")

    # 2) 去真实 <actuator> 段（注意：文件里另有一段被 <!-- --> 注释掉的 actuator，不能误伤）
    marker = "\n  <actuator>"
    end = "\n  </actuator>"
    if marker in txt:
        head, rest = txt.split(marker, 1)
        tail = rest.split(end, 1)[1] if end in rest else ""
        txt = head + "\n<!-- actuators injected by mjlab -->" + tail
        removed = True
    else:
        removed = False

    DST.write_text(txt, encoding="utf-8")
    print(f"[OK] 写出 {DST}")
    print(f"     地形 include 替换 {before} 处；actuator 段删除={removed}")

    # 3) 编译自检
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(DST))
    print(f"[OK] 编译通过: nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody}")
    if m.nu != 0:
        print("[WARN] nu != 0，说明 actuator 没删干净")
    # 机器人应无地形 geom：统计 plane 数量
    planes = sum(1 for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    print(f"     plane geom 数量 = {planes}（纯机器人应为 0）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
