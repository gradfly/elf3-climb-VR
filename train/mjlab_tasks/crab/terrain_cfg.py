"""侧向（螃蟹步）楼梯地形。

用 mjlab 的**倒置**金字塔台阶（`pyramid_stairs_inv`）：中心是**最低的方形平地**，
四面逐级向外抬升，像一只碗。机器人出生在碗底那块平地上、面朝 +X，收到 +Y 方向
的速度指令 —— 往 +Y 走就是「侧向上台阶」，正好对应螃蟹步，无需自定义地形生成器。

难度分级用 step_height_range（级高）与 step_width（踏面进深）控制，
与仓库 crab_config.json 的范围对齐（目标 rise 7–25 cm、tread 0.20–0.40；
训练地形实际覆盖 0–25 cm、tread 0.24–0.40，见文件末尾 hard_stairs 的注释）。

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

行列语义（2026-10-29 更正，之前这里写的是错的）
----------------------------------------------
`curriculum=True` 时 mjlab 把 `num_cols` 覆盖成 `len(sub_terrains)`：

    **列 = 地形类型，行 = 难度**

`terrain.terrain_types`（列）在环境创建时随机分配，**一个 env 终身待在同一列**；
课程 `update_env_origins(env_ids, move_up, move_down)` 只改 `terrain_levels`（行）。
`terrain_levels_vel` 里那段 `mask = types == i` 的分组统计就是证据。

所以「level 0」**不等于「平地」**。行 0 在四列上的实际台阶高度是：

    level 0 行：flat 0.00 / easy 0.00 / mid 0.01 / hard 0.02   （2026-10-31 调整后）

⚠️ 但这条是按 `step_height_range` 与 difficulty 推算的，**实测对不上**：
`--terrain-level 0` 下八个 env 的真实台阶是 0 / 6.1 / 7.8 / 7.8 / 8.0 / 12.3 / 12.5 / 14.2 cm
（`terrain_type` 分别为 0/0/1/1/2/2/3/3，但与高度**不对应** —— type0 同时出现 12.5 和 8.0）。
说明「列 = 类型、行 = 难度」这个映射在本版 mjlab 上并不成立，`terrain_types` 也不是列号。

既然映射不可靠，就不去依赖它：**把四档的低端全部对齐到 0**。这样无论 env 落在哪一列、
哪一行，只要 difficulty ≈ 0 就一定是平地；而高端（0.10 / 0.16 / 0.25）一概不动，
最终难度不损失。这比继续猜映射更省事，也更稳。

也就是说，即使课程把所有人钉在 level 0，仍有 **25% 的 env 开局就站在
8 cm 台阶上、25% 在 5 cm 上**。这一条直接解释了为什么
`Curriculum/terrain_levels/mean = 0` 能和 `Metrics/rise_est = 0.14` 同时成立 ——
两者都是真的：所有 env 都在行 0，但类型分散在四列，`rise_est` 是四列混在一起
的加权平均。

（调整前 mid 起点 0.10 / hard 起点 0.16，即 25% 的 env 一出生就面对 16 cm 台阶，
另一半是 10 cm —— 那才是「5 秒必摔、走不动」的直接原因。）

行内插值：`step_height = h_min + (h_max-h_min) * difficulty`，difficulty 是该行
在本 sub_terrain 内的归一化位置（第一行 -> 0，最后一行 -> 1）。所以下限降低
**不会**削弱最终难度，hard 的顶格仍是 0.25 m。

`proportion` 在 curriculum 模式下不参与分列（每型固定占一列），留着他处兼容。
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
        # ⚠️ 2026-10-31：低端 0.02 -> 0.00。实测 `--terrain-level 0`（课程最底层）
        # 八个 env 的实际台阶是 0 / 6.1 / 7.8 / 7.8 / 8.0 / 12.3 / 12.5 / 14.2 cm，
        # **只有 1 个是平地** —— 策略从来没有"在平地上站稳"的阶段，直接就被丢上台阶，
        # 所以 ep_len 连 10 s 都到不了，怎么调奖励都没用。
        # 把四档的**低端全部对齐到 0**，行 0 就真的是平地；高端一概不动，
        # 最终难度不损失。这样也不依赖"行/列到底怎么映射"这个还没搞清的问题：
        # 低端为 0 时，任何一列的最难行之外都不会出现台阶。
        "easy_stairs": pyramid_stairs_inv(
            proportion=0.35,
            step_height_range=(0.00, 0.10),
            step_width=0.40,
            platform_width=1.2,     # -> 半宽 0.80 m
            border_width=0.8,
        ),
        # 标准（行 0 是 1 cm，最后一行到 16 cm）
        # ⚠️ 起点必须低：curriculum 只调行，环境一出生就在本列的第 0 行，
        # 起点 0.05 意味着 1/4 的 env 从没在低于 5 cm 的台阶上练过。
        "mid_stairs": pyramid_stairs_inv(
            proportion=0.30,
            step_height_range=(0.01, 0.16),
            step_width=0.32,
            platform_width=1.0,     # -> 半宽 0.52 m；两级边缘 0.52 / 0.84 都在 1 m 内
            border_width=0.6,
        ),
        # 难（行 0 是 2 cm，顶格 25 cm）
        # ⚠️ 2026-10-31：顶格 0.22 -> 0.25。目标台阶范围是 7~25 cm（crab_config.json 的
        # rise_min/max = 0.12/0.25），而原来 22 cm 封顶 —— 25 cm 那 3 cm 完全没练过，
        # 参考表也 clamp 在 0.22（表已同步重建到 0.26，见 train/assets/ref_tables/）。
        # 低端 0.08 -> 0.02 同上：行 0 八分之七的 env 面对 8~14 cm 就是这么来的。
        "hard_stairs": pyramid_stairs_inv(
            proportion=0.15,
            step_height_range=(0.02, 0.25),
            step_width=0.24,
            platform_width=1.0,     # -> 半宽 0.62 m；两级边缘 0.62 / 0.86
            border_width=0.5,
        ),
    },
    add_lights=True,
)
