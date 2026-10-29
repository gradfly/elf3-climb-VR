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

import inspect

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

    # 2000 轮实测（每步加权值）：upright 0.92 + ref_tracking 0.49 + track_lin 0.86
    # 合计 2.3 分几乎不用动就能拿到，而"走快"只多拿 forward_progress 0.107。
    # 结果就是策略优先生存而不是前进 —— 这里把"站着也能拿的分"压下去。
    if "upright" in r:
        r["upright"].weight = 0.5          # 1.0 -> 0.5（原值 0.92/步，白送分）
    if "action_rate_l2" in r:
        r["action_rate_l2"].weight = -0.03  # -0.05 -> -0.03（原 -0.80/步，压步幅）

    r["forward_progress_y"] = RewardTermCfg(
        func=X.forward_progress_y, weight=3.0,     # 1.0 -> 3.0，位移成为主奖励
        params={"command_name": "twist"},
    )
    r["ref_tracking"] = RewardTermCfg(
        func=X.ref_tracking, weight=0.3,           # 0.6 -> 0.3（原值已 0.81 饱和）
        params={"action_name": "joint_pos", "sigma": 0.35},
    )
    r["yaw_drift"] = RewardTermCfg(func=X.yaw_drift, weight=-0.5, params={})
    r["lateral_drift"] = RewardTermCfg(func=X.lateral_drift, weight=-0.2, params={})
    # ⚠️ 奖励按 dt 缩放：一次性惩罚实际值 = weight × 0.02。
    # 原来 −5 只等于每次摔倒扣 0.1 分（日志 is_terminated −0.005 = −0.1/20s），
    # 而一个 episode 回报 ~10 分 —— 摔倒几乎免费，策略自然选择"用速度换摔"。
    # −150 × 0.02 = −3.0/次，约占单回合回报 30%，够痛但不主导。
    r["is_terminated"] = RewardTermCfg(func=X.is_terminated, weight=-150.0, params={})

    # 探针：只把 rise_est / tread_est / scan_ok 写进 Metrics/*，不影响学习。
    # 权重必须非零（0 会让 mjlab 整项跳过、连 func 都不调用）。
    r["ref_probe"] = RewardTermCfg(
        func=X.ref_probe, weight=1e-9, params={"action_name": "joint_pos"},
    )


def _retune_commands(cfg, play: bool) -> None:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    twist.ranges.lin_vel_y = (0.15, 0.24)     # 与 v_ref≈0.18 同量级
    twist.ranges.ang_vel_z = (0.0, 0.1 if play else 0.0)
    # play 模式下 viser 滑块要求每个轴 max>=0.1；训练模式保持 0（不指令转向）


def _params_for(func, params: dict) -> dict:
    """按函数的真实签名裁剪 params，丢掉它不认的参数。

    ⚠️ 2026-10-29：直接照 GitHub main 的签名硬塞 `promotion_fraction=0.6` 会在
    服务器上抛 `TypeError: terrain_levels_vel() got an unexpected keyword argument`。
    原因是**服务器装的 mjlab 比 main 旧**，那个参数还没有。以后凡是给 mjlab 内置
    函数传 params，一律先过这里，别再信 main 的签名。
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(params)
    ps = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps):
        return dict(params)
    allowed = {p.name for p in ps
               if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY)}
    kept = {k: v for k, v in params.items() if k in allowed}
    dropped = sorted(set(params) - set(kept))
    if dropped:
        print(f"[curriculum][warn] mjlab 版本的 {getattr(func, '__name__', func)} "
              f"不接受 {dropped}，已忽略（不影响训练，只是这档调节不生效）")
    return kept


def _retune_curriculum(cfg) -> None:
    # 地形升级默认要求 distance > 0.8 × commanded_distance，而 commanded_distance
    # 含"早摔剩余时间应走的距离"。2000 轮时 81% 的回合在 9.3 s 摔掉，几乎必然降级，
    # terrain 卡在 0.08、看不到台阶。放宽到 0.6，让"能走一段"的 env 先升上去。
    tl = cfg.curriculum.get("terrain_levels") if cfg.curriculum else None
    if tl is not None:
        want = {**(tl.params or {}), "promotion_fraction": 0.6}
        tl.params = _params_for(tl.func, want)

    if "command_vel" not in cfg.curriculum:
        return
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=mdp.commands_vel,
        params=_params_for(mdp.commands_vel, {
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_y": (0.10, 0.16)},
                {"step": 400 * 24, "lin_vel_y": (0.14, 0.20)},
                {"step": 1200 * 24, "lin_vel_y": (0.15, 0.24)},
            ],
        }),
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
        leg_clip_rad=0.40,
        waist_clip_rad=0.20,
        arm_clip_rad=0.20,
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
