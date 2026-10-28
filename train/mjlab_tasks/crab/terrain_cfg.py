"""侧向（螃蟹步）楼梯地形。

用 mjlab 自带的方形金字塔台阶：从中心平台向四周逐级抬升。
机器人站在中心、面朝 +X，收到 +Y 方向的速度指令 —— 向 +Y 走就是「侧向上台阶」，
正好对应螃蟹步，无需自定义地形生成器。

难度分级用 step_height_range（级高）与 step_width（踏面进深）控制，
与仓库 crab_config.json 的范围一致（rise 0.10–0.22、tread 0.24–0.40）。
"""
from __future__ import annotations

from mjlab.terrains.config import pyramid_stairs
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

CRAB_STAIRS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    curriculum=True,
    sub_terrains={
        # 近平地：先学会不摔
        "flat": pyramid_stairs(
            proportion=0.20,
            step_height_range=(0.0, 0.02),
            step_width=0.40,
            platform_width=2.5,
            border_width=1.0,
        ),
        # 缓台阶
        "easy_stairs": pyramid_stairs(
            proportion=0.35,
            step_height_range=(0.02, 0.10),
            step_width=0.40,
            platform_width=2.2,
            border_width=0.8,
        ),
        # 标准（对应 rise 0.10–0.16）
        "mid_stairs": pyramid_stairs(
            proportion=0.30,
            step_height_range=(0.10, 0.16),
            step_width=0.32,
            platform_width=2.0,
            border_width=0.6,
        ),
        # 难（对应 rise 0.16–0.22、tread ~0.24）
        "hard_stairs": pyramid_stairs(
            proportion=0.15,
            step_height_range=(0.16, 0.22),
            step_width=0.24,
            platform_width=1.8,
            border_width=0.5,
        ),
    },
    add_lights=True,
)
