"""侧向（螃蟹步）楼梯地形。

用 mjlab 的**倒置**金字塔台阶（`pyramid_stairs_inv`）：中心是**最低的方形平地**，
四面逐级向外抬升，像一只碗。机器人出生在碗底那块平地上、面朝 +X，收到 +Y 方向
的速度指令 —— 往 +Y 走就是「侧向上台阶」，正好对应螃蟹步，无需自定义地形生成器。

难度分级用 step_height_range（级高）与 step_width（踏面进深）控制，
与仓库 crab_config.json 的范围一致（rise 0.10–0.22、tread 0.24–0.40）。

⚠️ 2026-10-29 修正（这是之前训练一直学不会的直接原因）
------------------------------------------------------
原来用的是正金字塔 `pyramid_stairs` —— mjlab 源码里它的 origin 是
`+(num_steps+1) * step_height`，也就是**出生在最高处的平台上**，往任何方向走都是
**下楼梯**。后果一整串：

* `ref_table.estimate` 找的是 `dh = h[iy] - h[iy+1] > 0`（+Y 方向地形抬升）。
  从顶上往外走地形只会降低 -> 永远 `dh < 0` -> `rise_est` 实测全程 0.000；
* 参考步态永远停在「平地那一档」，爬楼动作全靠残差自己憋；
* `|action|` 均值 3.33（限幅约 1.5）-> 76% 的输出顶死在限幅上，退化成 bang-bang；
* 机器人实际是在往下坡挪，于是站一会就滑/摔。

换 `pyramid_stairs_inv` 后出生点在碗底，往外走全程上台阶，上面每一条都反过来。

平台宽度（`platform_width`）的选择
---------------------------------
倒置碗底那块平地的实际尺寸 = `terrainsize - 2*num_steps*step_width`，由平台宽度
间接决定（num_steps 取整）。要让它落在 height_scan（±1.0 m）与参考表检测带
（|y| ∈ 0.15~1.00 m）里，否则又会出现「测不到台阶」。下面四档实测半宽：

    flat 0.60 m / easy 0.80 m / mid 0.52 m（两级可见）/ hard 0.62 m（两级可见）
"""
from __future__ import annotations

from mjlab.terrains.config import pyramid_stairs_inv
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

CRAB_STAIRS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    curriculum=True,
    sub_terrains={
        # 近平地：先学会不摔、学会横着迈步
        "flat": pyramid_stairs_inv(
            proportion=0.20,
            step_height_range=(0.0, 0.02),
            step_width=0.40,
            platform_width=1.2,     # -> 碗底平台半宽 0.60 m
            border_width=1.0,
        ),
        # 缓台阶
        "easy_stairs": pyramid_stairs_inv(
            proportion=0.35,
            step_height_range=(0.02, 0.10),
            step_width=0.40,
            platform_width=1.2,     # -> 半宽 0.80 m
            border_width=0.8,
        ),
        # 标准（对应 rise 0.10–0.16、tread 0.32）
        "mid_stairs": pyramid_stairs_inv(
            proportion=0.30,
            step_height_range=(0.10, 0.16),
            step_width=0.32,
            platform_width=1.0,     # -> 半宽 0.52 m；两级边缘 0.52 / 0.84 都在 1 m 内
            border_width=0.6,
        ),
        # 难（对应 rise 0.16–0.22、tread 0.24）
        "hard_stairs": pyramid_stairs_inv(
            proportion=0.15,
            step_height_range=(0.16, 0.22),
            step_width=0.24,
            platform_width=1.0,     # -> 半宽 0.62 m；两级边缘 0.62 / 0.86
            border_width=0.5,
        ),
    },
    add_lights=True,
)
