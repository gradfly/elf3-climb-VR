"""蹲下 / 起立「参考轨迹跟踪」环境配置。

两个任务共用一份代码，只差 `which` 与参考轨迹：

    Elf3-Squat-Down   站姿 -> 极限蹲（squat_extreme_down_traj，2.46 s）
    Elf3-Squat-Up     极限蹲 -> 站姿（squat_extreme_up_traj，2.02 s）

为什么不复制一份 crab_env_cfg 改改
--------------------------------
基线有 400 多行、且是在服务器上反复调试过的（求解器参数、摩擦/质心随机化、
接触传感器定义都在里面）。这里**沿用**它，只覆盖与本任务相关的部分：

    地形   台阶 -> 平地（本任务不爬楼）
    指令   无速度指令（原地蹲起）
    观测   地形 height_scan -> 参考轨迹误差/进度/高度误差
    动作   JointPositionAction -> TrajTrackAction（残差叠在参考轨迹上）
    奖励   速度/步态类 -> 关节跟踪 + 高度跟踪 + 躯干朝向跟踪
    终止   加「参考播完」与「躯干塌到 0.28 m」

⚠️ 观测维度、动作语义都变了，**不能加载 Elf3-Crab-* 的 checkpoint**。
"""
from __future__ import annotations

import importlib
import math

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp_extra as X
from . import squat_mdp as S
from .env_cfg import crab_env_cfg
from .robot_cfg import ACTION_SCALE_MAP, JOINT_NAMES, REPO
from .traj_action import TrajTrackActionCfg
from .traj_ref import load_traj

REF_TABLE = str(REPO / "train/assets/ref_tables/squat_ref_v1.npz")

# 任务名 -> (参考表里的 key, 中文说明)
TASKS = {
    "down": ("down", "站姿 -> 极限蹲"),
    "up": ("up", "极限蹲 -> 站姿"),
}

# 极限蹲的躯干高度是 0.523 m；塌到这个高度以下判定为摔倒
MIN_Z = {"down": 0.28, "up": 0.30}


def _flat_terrain() -> TerrainGeneratorCfg:
    """一块平地（训练/回放都不需要台阶）。

    优先用 mjlab 自带的 flat 预设；取不到就退回「0 高度倒置金字塔」——
    后者用的是本项目已经实测过的 `pyramid_stairs_inv`，几何上等价于平地。
    两条路都失败时明确报错，不做静默降级（没有地面 = 永远在自由落体）。
    """
    sub = None
    for mod_name in ("mjlab.terrains.config", "mjlab.terrains"):
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "flat", None)
            if callable(fn):
                sub = fn(proportion=1.0)
                print(f"[squat] 平地地形：{mod_name}.flat()")
                break
        except Exception as e:                      # noqa: BLE001
            print(f"[squat][warn] {mod_name}.flat 不可用（{e!r}）")
    if sub is None:
        try:
            from mjlab.terrains.config import pyramid_stairs_inv
            sub = pyramid_stairs_inv(
                proportion=1.0, step_height_range=(0.0, 0.0),
                step_width=0.40, platform_width=1.2, border_width=1.0,
            )
            print("[squat] 平地地形：退化为 0 高度倒置金字塔（等价平地）")
        except Exception as e:                      # noqa: BLE001
            raise RuntimeError(f"拿不到任何平地地形实现：{e!r}") from e

    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        curriculum=False,
        sub_terrains={"flat": sub},
        add_lights=True,
    )


def _build_observations(cfg, which: str) -> None:
    a = cfg.observations["actor"].terms
    actor = {
        "base_lin_vel": ObservationTermCfg(
            func=a["base_lin_vel"].func, params=a["base_lin_vel"].params,
            noise=Unoise(n_min=-0.3, n_max=0.3),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=a["base_ang_vel"].func, params=a["base_ang_vel"].params,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel, params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        # 参考轨迹的三路信息：29 关节误差 / 高度误差 / 进度
        "traj_joint_err": ObservationTermCfg(
            func=S.traj_ref_err, noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "traj_height_err": ObservationTermCfg(
            func=S.traj_height_err, noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        "traj_progress": ObservationTermCfg(func=S.traj_progress_sincos),
    }
    cfg.observations["actor"].terms = actor

    critic = dict(actor)
    critic["joint_pos"] = ObservationTermCfg(func=mdp.joint_pos_rel)   # 无偏版本
    critic["traj_joint_err"] = ObservationTermCfg(func=S.traj_ref_err)
    critic["traj_height_err"] = ObservationTermCfg(func=S.traj_height_err)
    critic["ref_joint_pos"] = ObservationTermCfg(func=S.traj_ref_joint_pos)
    critic["traj_pitch_err"] = ObservationTermCfg(func=S.traj_pitch_err)
    cfg.observations["critic"].terms = critic


def _build_rewards(cfg) -> None:
    r = cfg.rewards
    r.clear()

    # 主干：29 关节跟踪（σ 0.35 rad ≈ 20°，早期也有梯度）
    r["traj_track"] = RewardTermCfg(
        func=S.traj_track, weight=1.5, params={"sigma": 0.35},
    )
    # 腿单独再给一份、更紧（σ 0.30）：站不站得住由腿决定
    r["leg_track"] = RewardTermCfg(
        func=S.traj_track, weight=2.0, params={"sigma": 0.30, "legs_only": True},
    )
    # 蹲多深 = 任务目标本身
    r["height_track"] = RewardTermCfg(
        func=S.height_track, weight=2.0, params={"sigma": 0.05},
    )
    # 躯干前倾 18.5° 是参考的一部分，不能拿 upright 罚
    r["up_track"] = RewardTermCfg(
        func=S.up_track, weight=1.0, params={"sigma": 0.25},
    )

    r["joint_pos_limits"] = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0)
    # -0.01：与爬楼任务同量级（那版实测 -0.006 足够抑制抖动，又不至于奖励「别动」）
    r["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    # 活着就给分，抵消「早点摔掉少挨罚」的退化解
    r["alive"] = RewardTermCfg(func=S.alive, weight=0.5)
    # 摔倒一次性重罚（按 dt 缩放后 ≈ -2/次；单回合满分约 60）
    r["is_terminated"] = RewardTermCfg(func=X.is_terminated, weight=-100.0, params={})
    # 探针：Metrics/traj_*
    r["squat_probe"] = RewardTermCfg(func=S.squat_probe, weight=1e-9, params={})


def _build_terminations(cfg, which: str) -> None:
    cfg.terminations.clear()
    cfg.terminations["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(55.0)},
    )
    cfg.terminations["torso_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "torso_ground_contact", "force_threshold": 20.0},
    )
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=S.base_too_low, params={"min_z": MIN_Z[which]},
    )
    cfg.terminations["drifted"] = TerminationTermCfg(
        func=S.base_drifted, params={"max_xy": 1.2},
    )
    # 参考播完 = 成功结束，不算摔倒（time_out=True）
    cfg.terminations["traj_finished"] = TerminationTermCfg(
        func=S.traj_finished, time_out=True,
    )


def _retune_events(cfg, which: str, play: bool) -> None:
    if "reset_base" in cfg.events:
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (0.0, 0.002),
            "yaw": (-0.05, 0.05),
        }
    push = cfg.events.get("push_robot")
    if push is not None:
        # 回合才 2.5 s，推力不能太密：3~6 s 一次等于整场都推不到，1.0~2.5 s 一次
        # 每个回合刚好挨一次，量级 ±0.15 m/s（相对蹲起速度 0.2 m/s 是小扰动）。
        push.interval_range_s = (1.0, 2.5)
        if isinstance(getattr(push, "params", None), dict):
            push.params["velocity_range"] = {
                "x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (-0.08, 0.08),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.08, 0.08),
            }
    if play:
        cfg.events.pop("push_robot", None)


def _set_init_pose(cfg, which: str) -> None:
    """出生姿态取参考第 0 帧（down = 站姿，up = 极限蹲）。"""
    traj = load_traj(REF_TABLE, TASKS[which][0])
    q0 = traj.q[0].cpu().numpy()
    z0 = float(traj.z[0].cpu())
    ent = cfg.scene.entities["robot"]
    ent.init_state.pos = (0.0, 0.0, z0)
    ent.init_state.joint_pos = {n: float(v) for n, v in zip(JOINT_NAMES, q0)}
    print(f"[squat] {which}: 出生 z={z0:.4f} m，"
          f"膝={math.degrees(float(q0[JOINT_NAMES.index('l_knee_y_joint')])):.1f}°")


def squat_env_cfg(which: str = "down", play: bool = False):
    if which not in TASKS:
        raise ValueError(f"which 只能是 {sorted(TASKS)}，收到 {which!r}")

    cfg = crab_env_cfg(play=play)

    # ---- 地形：平地 ----
    flat = _flat_terrain()
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=flat,
        max_init_terrain_level=0,
    )
    # 只留躯干接触传感器（终止用）。地形射线 / 足底接触在本任务里没用，
    # 去掉能省掉 273 条射线 × 每个 env 的每帧开销。
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors if getattr(s, "name", "") == "torso_ground_contact"
    )
    if not cfg.scene.sensors:
        raise RuntimeError("没找到 torso_ground_contact 传感器，终止项会失效")

    # ---- 无速度指令 ----
    cfg.commands = {}

    # ---- 动作：参考轨迹 + 残差 ----
    cfg.actions["joint_pos"] = TrajTrackActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=ACTION_SCALE_MAP,
        offset=0.0,
        use_default_offset=False,
        clip=None,
        traj_path=REF_TABLE,
        traj_key=TASKS[which][0],
        leg_clip_rad=0.35,
        waist_clip_rad=0.15,
        arm_clip_rad=0.25,
    )

    _build_observations(cfg, which)
    _build_rewards(cfg)
    _build_terminations(cfg, which)
    _retune_events(cfg, which, play)
    _set_init_pose(cfg, which)

    # ---- 课程：无 ----
    cfg.curriculum = {}

    # ---- 回合长度：参考时长 + 0.5 s 余量（正常会在 traj_finished 提前结束）----
    traj = load_traj(REF_TABLE, TASKS[which][0])
    cfg.episode_length_s = float(traj.duration + 0.5)

    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
        distance=2.2,
        elevation=-6.0,
        azimuth=200.0,
    )

    if play:
        cfg.episode_length_s = int(1e6)     # 回放：播完保持蹲姿/站姿，不自动重置
                                            # （1e6 s ≈ 5e7 步，仍在 int32 内）
        # 「参考播完」这条终止在训练里是对的（回合该结束就结束），但回放里必须去掉：
        # 否则 2.46 s 一到就 reset 回初始站姿，看起来就是「蹲下去又站起来」的无限循环
        # —— 其实是重置，不是策略在起身。去掉后参考停在末帧（极限蹲），一直保持。
        cfg.terminations.pop("traj_finished", None)
        print("[squat][play] 已移除 traj_finished 终止：播完保持末帧，不重置")
        cfg.observations["actor"].enable_corruption = False
    return cfg


def squat_task_meta(which: str) -> dict:
    """训练脚本/文档用：返回该任务的参考轨迹信息。"""
    traj = load_traj(REF_TABLE, TASKS[which][0])
    return dict(
        which=which,
        desc=TASKS[which][1],
        frames=traj.T,
        duration=traj.duration,
        z0=float(traj.z[0].cpu()),
        z_end=float(traj.z[-1].cpu()),
        meta=dict(traj.meta),
    )
