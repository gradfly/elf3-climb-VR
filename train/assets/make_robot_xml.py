#!/usr/bin/env python3
"""生成 mjlab 可用的纯机器人资产：elf3_robot_only.xml

从 data/mujoco_simulation/elf3.xml 派生，做三件事：
  1) 删掉地形 include（mjlab 的 SceneCfg 自己管地形）
  2) 删掉 <actuator> 段（mjlab 用 BuiltinPositionActuatorCfg 注入位置执行器）
  3) **把视觉 mesh 从 geom group 0 挪到 group 2**（见下）

为什么第 3 步必须做
------------------
`terrain_scan` 用 `include_geom_groups=(0,)` 只对 group 0 打射线，地形在 group 0。
但机器人 XML 的 `<default class="elf3">` 里视觉 mesh 也是 `group="0"`，
而**碰撞胶囊是 group="3"**。于是从躯干往下打的 height_scan 会打到自己的
腰/腿网格：实测 y=±0.1 处高度从 104.9 cm 掉到 62.8 cm，差 42 cm，
是真实台阶（15 cm）的近 3 倍。`exclude_parent_body=True` 只能排掉
`torso_link` 自己的网格，排不掉子 body（腿）的。

挪到 group 2 后：group 0 = 只剩地形，group 2 = 机器人视觉（viewer 默认仍然
显示 group 0/1/2，所以外观不变），group 3 = 碰撞体。

⚠️ 改了 group 后 height_scan 的数值会变（不再含自身遮挡），
   观测分布随之改变 -> 需要重新训练，不要指望旧 checkpoint 还能用。

输出到同目录（保证 meshdir="./meshes/" 仍可用），并做一次 MuJoCo 编译自检。

用法：
    python train/assets/make_robot_xml.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # 仓库根
DATA = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
SRC = DATA / "elf3.xml"
DST = DATA / "elf3_robot_only.xml"

TERRAIN_INCLUDE = '<include file="terrains/universal_test/universal_test_terrain_compact.xml"/>'

# default class="elf3" 里的视觉 geom：所有 body 的 <geom rgba=.. mesh=../> 都继承它
VISUAL_GEOM_OLD = '<geom type="mesh" contype="0" conaffinity="0" group="0" density="0" material="silver"/>'
VISUAL_GEOM_NEW = '<geom type="mesh" contype="0" conaffinity="0" group="2" density="0" material="silver"/>'


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

    # 3) 视觉 mesh 挪出 group 0：height_scan 只打 group 0，不能打到机器人自己
    n_visual = txt.count(VISUAL_GEOM_OLD)
    if n_visual == 0:
        print("[ERR] 没找到视觉 geom 的 group=\"0\" 定义，XML 结构可能变了。")
        print("      期望这一行：", VISUAL_GEOM_OLD)
        print("      不改会让 height_scan 打到机器人自己的腿（rise 估计被顶到上限）。")
        return 1
    if n_visual > 1:
        print(f"[WARN] 找到 {n_visual} 处视觉 geom group 定义，全部替换为 group 2")
    txt = txt.replace(VISUAL_GEOM_OLD, VISUAL_GEOM_NEW)

    DST.write_text(txt, encoding="utf-8")
    print(f"[OK] 写出 {DST}")
    print(f"     地形 include 替换 {before} 处；actuator 段删除={removed}；"
          f"视觉 mesh group 0 -> 2（{n_visual} 处）")

    # 4) 编译自检
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(DST))
    print(f"[OK] 编译通过: nq={m.nq} nv={m.nv} nu={m.nu} njnt={m.njnt} nbody={m.nbody}")
    if m.nu != 0:
        print("[WARN] nu != 0，说明 actuator 没删干净")
    # 机器人应无地形 geom：统计 plane 数量
    planes = sum(1 for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
    print(f"     plane geom 数量 = {planes}（纯机器人应为 0）")

    groups = Counter(int(m.geom_group[g]) for g in range(m.ngeom))
    print(f"     geom group 分布 = {dict(sorted(groups.items()))}"
          f"  （视觉 mesh 应在 2，碰撞在 3；group 0 必须为空，否则 height_scan 会打到机器人）")
    if groups.get(0, 0) > 0:
        print("[ERR] group 0 里还有机器人 geom —— height_scan 会打到它，参考步态估计会被污染")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
