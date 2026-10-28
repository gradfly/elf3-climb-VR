"""残差 RL 环境配置：在基线 crab_env_cfg 上做增量修改。

为什么不复制一份
----------------
基线有 400 行、且还在演进；复制会导致两份配置漂移。
这里只在返回的 cfg 上替换动作/观测/奖励/指令/课程，其他（地形、传感器、终止、
事件、仿真参数）完全沿用基线。

任务 ID
-------
    Elf3-Crab-Residual        训练
    Elf3-Crab-Residual-Play   回放/评测

⚠️ 观测维度 290 -> 393，动作语义也变了，**不能加载旧 checkpoint**。
   旧任务 Elf3-Crab-Stairs 保持原样，可继续用 model_9297.pt 对比。
"""
from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from . import mdp_extra as X
from .env_cfg import crab_env_cfg
from .residual_action import ResidualGaitActionCfg
from .robot_cfg import ACTION_SCALE_MAP, DEFAULT_JOINT_POS, REPO

SCAN_SIZE = (1.2, 2.0)          # 前后 ±0.6 / 左右 ±1.0（螃蟹步加宽侧向）
TABLE_PATH = str(REPO / "train/assets/ref_tables/crab_ref_v1.npz")

ACTOR_ORDER = [
    "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos",
    "joint_pos_ref_err", "joint_vel", "actions", "command",
    "phase", "ref_meta", "height_scan",
]


def _rebuild_observations(cfg) -> None:
    a = cfg.observations["actor"].terms
    new_actor = {
        "base_lin_vel": a["base_lin_vel"],
        "base_ang_vel": a["base_ang_vel"],
        "projected_gravity": a["projected_gravity"],
        "joint_pos": a["joint_pos"],
        "joint_pos_ref_err": ObservationTermCfg(func=X.joint_pos_ref_err),
        "joint_vel": a["joint_vel"],
        "actions": a["actions"],
        "command": a["command"],
        "phase": ObservationTermCfg(func=X.phase_sincos),
        "ref_meta": ObservationTermCfg(func=X.ref_meta),
        "height_scan": a["height_scan"],
    }
    cfg.observations["actor"].terms = new_actor

    c = cfg.observations["critic"].terms
    new_critic = dict(new_actor)
    new_critic["joint_pos"] = c["joint_pos"]          # critic 用无偏版本
    for k in ("foot_height", "foot_air_time", "foot_contact", "foot_contact_forces"):
        if k in c:
            new_critic[k] = c[k]
    new_critic["ref_joint_pos"] = ObservationTermCfg(func=X.ref_joint_pos)
    cfg.observations["critic"].terms = new_critic


def _rebuild_rewards(cfg) -> None:
    r = cfg.rewards

    # 去掉与参考步态冲突/无意义的项
    r.pop("feet_swing_height", None)
    r.pop("track_angular_velocity", None)

    # 残差动的幅度本来就小，平滑惩罚减半
    if "action_rate_l2" in r:
        r["action_rate_l2"].weight = -0.05

    r["forward_progress_y"] = RewardTermCfg(
        func=X.forward_progress_y, weight=1.0,
        params={"command_name": "twist"},
    )
    r["ref_tracking"] = RewardTermCfg(
        func=X.ref_tracking, weight=0.6,
        params={"action_name": "joint_pos", "sigma": 0.35},
    )
    r["yaw_drift"] = RewardTermCfg(func=X.yaw_drift, weight=-0.5, params={})
    r["lateral_drift"] = RewardTermCfg(func=X.lateral_drift, weight=-0.2, params={})
    r["is_terminated"] = RewardTermCfg(func=X.is_terminated, weight=-5.0, params={})


def _retune_commands(cfg, play: bool) -> None:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_y = (0.15, 0.24)     # 与 v_ref≈0.18 同量级
    twist.ranges.ang_vel_z = (0.0, 0.1 if play else 0.0)
    # play 模式下 viser 滑块要求每个轴 max>=0.1；训练模式保持 0（不指令转向）


def _retune_curriculum(cfg) -> None:
    if "command_vel" not in cfg.curriculum:
        return
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_y": (0.10, 0.16)},
                {"step": 400 * 24, "lin_vel_y": (0.14, 0.20)},
                {"step": 1200 * 24, "lin_vel_y": (0.15, 0.24)},
            ],
        },
    )


def crab_residual_env_cfg(play: bool = False):
    cfg = crab_env_cfg(play=play, scan_size=SCAN_SIZE)

    # ---- 动作：时变 offset 的残差关节位置 ----
    cfg.actions["joint_pos"] = ResidualGaitActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=ACTION_SCALE_MAP,
        offset=0.0,
        use_default_offset=False,     # offset 由参考表每步给出
        clip=None,                    # 限幅按弧度自己做
        table_path=TABLE_PATH,
        period_frames=90,
        hold_steps=25,
        leg_clip_rad=0.30,
        waist_clip_rad=0.15,
        arm_clip_rad=0.15,
        scan_size=SCAN_SIZE,
        scan_resolution=0.1,
        terrain_refresh_steps=5,
        v_rate_range=(0.6, 1.6),
        phase_offset_choices=(0.0, 0.5),
        # 相位平移：phase=0 -> 表帧 33（双支撑起点），phase=0.5 -> 帧 78，
        # 避免 episode 一开头就要求单腿支撑。
        phase_shift=33.0 / 90.0,
        # α=0 时参考退化为默认站姿 —— 与基线任务「动作全零」等价，
        # 早期训练难度和已验证的基线一致，再随课程渐变为脚本步态。
        ref_stand=tuple(DEFAULT_JOINT_POS),
        ramp_steps=1500 * 24,           # 1500 iter 线性升到满幅
        alpha_fixed=1.0 if play else -1.0,   # play 直接用满幅步态
    )

    _rebuild_observations(cfg)
    _rebuild_rewards(cfg)
    _retune_commands(cfg, play)
    _retune_curriculum(cfg)
    return cfg
