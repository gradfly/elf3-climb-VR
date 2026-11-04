"""注册 elf3 螃蟹步爬楼梯任务。

放到 mjlab/tasks/crab/ 后，`mjlab.tasks.import_packages` 会自动导入本文件，
任务即出现在 `mjl list_envs` / `train` 的可选列表里。

任务 ID：
  - Elf3-Crab-Stairs        训练
  - Elf3-Crab-Stairs-Play   回放/评测
"""
from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import crab_env_cfg
from .env_cfg_residual import crab_residual_env_cfg
from .rl_cfg import elf3_crab_ppo_runner_cfg, elf3_squat_ppo_runner_cfg
from .squat_env_cfg import squat_env_cfg

register_mjlab_task(
    task_id="Elf3-Crab-Stairs",
    env_cfg=crab_env_cfg(play=False),
    play_env_cfg=crab_env_cfg(play=True),
    rl_cfg=elf3_crab_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Elf3-Crab-Stairs-Play",
    env_cfg=crab_env_cfg(play=False),
    play_env_cfg=crab_env_cfg(play=True),
    rl_cfg=elf3_crab_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ---- 残差任务（按 docs/crab_rl_training.md 的思路）----
register_mjlab_task(
    task_id="Elf3-Crab-Residual",
    env_cfg=crab_residual_env_cfg(play=False),
    play_env_cfg=crab_residual_env_cfg(play=True),
    rl_cfg=elf3_crab_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Elf3-Crab-Residual-Play",
    env_cfg=crab_residual_env_cfg(play=False),
    play_env_cfg=crab_residual_env_cfg(play=True),
    rl_cfg=elf3_crab_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

# ---- 蹲下 / 起立跟踪任务（参考轨迹 + 残差，见 squat_env_cfg.py）----
for _which in ("down", "up"):
    _tag = _which.capitalize()
    register_mjlab_task(
        task_id=f"Elf3-Squat-{_tag}",
        env_cfg=squat_env_cfg(which=_which, play=False),
        play_env_cfg=squat_env_cfg(which=_which, play=True),
        rl_cfg=elf3_squat_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=f"Elf3-Squat-{_tag}-Play",
        env_cfg=squat_env_cfg(which=_which, play=False),
        play_env_cfg=squat_env_cfg(which=_which, play=True),
        rl_cfg=elf3_squat_ppo_runner_cfg(),
        runner_cls=VelocityOnPolicyRunner,
    )
