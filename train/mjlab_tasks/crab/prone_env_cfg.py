"""趴姿三个动作的「参考轨迹 + 残差」跟踪环境。

    Elf3-Prone-Stand2Crawl   站立 -> 趴姿（含 1 s 冻结尾）
    Elf3-Prone-Crawl         趴姿向前爬行（含 1 s 冻结尾）
    Elf3-Prone-Crawl2Stand   趴姿 -> 站立（含 1 s 冻结尾）

沿用 squat 那套：基线用 crab_env_cfg，覆盖地形/观测/动作/奖励/终止。
与 squat 的关键差异见 prone_mdp 的模块注释——趴姿不能沿用
「base_too_low / bad_orientation / base_anchor」这组原地站立的判据。

⚠️ 观测与动作语义和 Elf3-Squat-*、Elf3-Crab-* 都不同，checkpoint 不能互相加载。
"""
from __future__ import annotations

import os

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp_extra as X
from . import prone_mdp as P
from . import squat_mdp as S
from .env_cfg import crab_env_cfg
from .robot_cfg import ACTION_SCALE_MAP, JOINT_NAMES, REPO
from .squat_env_cfg import _flat_terrain
from .traj_action import TrajTrackActionCfg
from .traj_ref import load_traj

# normal 档用 prone_ref_v1.npz；ELF3_PRONE_PRESET=fast 用 prone_ref_v1_fast.npz
PRESET = os.environ.get("ELF3_PRONE_PRESET", "normal").strip().lower()
_SUF = "" if PRESET != "fast" else "_fast"
REF_TABLE = str(REPO / f"train/assets/ref_tables/prone_ref_v1{_SUF}.npz")

# 训练时是否在回合中推机器人（模拟踢/滑）
PUSH = os.environ.get("ELF3_PRONE_PUSH", "1") not in ("0", "false", "False")

# 任务名 -> (参考表 key, 中文说明)
TASKS = {
    "stand2crawl": ("stand2crawl", "站立 -> 蹲下 -> 撑地 -> 趴姿"),
    "crawl": ("crawl", "趴姿向前爬行"),
    "crawl2stand": ("crawl2stand", "趴姿 -> 撑地 -> 蹲 -> 站立"),
}


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
        # 参考轨迹四路信息：29 关节误差 / 高度误差 / 水平误差 / 进度
        "traj_joint_err": ObservationTermCfg(
            func=S.traj_ref_err, noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "traj_height_err": ObservationTermCfg(
            func=S.traj_height_err, noise=Unoise(n_min=-0.005, n_max=0.005),
        ),
        # 爬行时「我相对该在的位置偏了多少」——没有这一路，策略只能靠积分
        # 速度盲推，长回合必然累积漂移。
        "traj_xy_err": ObservationTermCfg(
            func=P.base_xy_err, noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "traj_progress": ObservationTermCfg(func=S.traj_progress_sincos),
    }
    cfg.observations["actor"].terms = actor

    critic = dict(actor)
    critic["joint_pos"] = ObservationTermCfg(func=mdp.joint_pos_rel)   # 无偏版本
    critic["traj_joint_err"] = ObservationTermCfg(func=S.traj_ref_err)
    critic["traj_height_err"] = ObservationTermCfg(func=S.traj_height_err)
    critic["traj_xy_err"] = ObservationTermCfg(func=P.base_xy_err)
    critic["ref_joint_pos"] = ObservationTermCfg(func=S.traj_ref_joint_pos)
    critic["traj_pitch_err"] = ObservationTermCfg(func=S.traj_pitch_err)
    cfg.observations["critic"].terms = critic


def _build_rewards(cfg) -> None:
    r = cfg.rewards
    r.clear()

    # 主干：29 关节跟踪（σ 0.35 rad ≈ 20°，早期也有梯度）
    r["traj_track"] = RewardTermCfg(func=S.traj_track, weight=1.5,
                                    params={"sigma": 0.35})
    # 腿再给一份更紧的（σ 0.30）：撑地和爬行都由腿决定
    r["leg_track"] = RewardTermCfg(func=S.traj_track, weight=2.0,
                                   params={"sigma": 0.30, "legs_only": True})
    # 高度：趴下要真的趴下去，站起来要真的站直（σ 0.06 m）
    r["height_track"] = RewardTermCfg(func=S.height_track, weight=1.5,
                                      params={"sigma": 0.06})
    # 躯干朝向：趴/站切换的核心信号，σ 0.25 对应中等角度容差
    r["up_track"] = RewardTermCfg(func=S.up_track, weight=1.5,
                                  params={"sigma": 0.25})
    # 水平位置：爬行段参考前进 0.48 m，不罚「原地比划」
    r["xy_track"] = RewardTermCfg(func=P.xy_track, weight=1.0,
                                  params={"sigma": 0.10})

    r["joint_pos_limits"] = RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0)
    r["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02)
    # 活着给分，抵消「早点结束少挨罚」的退化解
    r["alive"] = RewardTermCfg(func=S.alive, weight=0.5)
    # 终止一次性重罚
    r["is_terminated"] = RewardTermCfg(func=X.is_terminated, weight=-100.0, params={})
    # 探针：把跟踪质量写进 Metrics/*，权重接近 0
    r["prone_probe"] = RewardTermCfg(func=P.prone_probe, weight=1e-9, params={})


def _build_terminations(cfg) -> None:
    t = cfg.terminations
    t.clear()
    t["time_out"] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # 参考播完（末尾 1 s 冻结帧也在参考里，所以等于「动作做完再站住 1 s」）
    t["traj_finished"] = TerminationTermCfg(func=S.traj_finished, time_out=True)
    # 趴姿不能沿用 base_too_low(0.28 m)：正常趴姿只有 0.10 m，会每帧误杀。
    # 改成「高度偏离参考」和「水平偏离参考」。
    t["traj_z_far"] = TerminationTermCfg(
        func=P.traj_z_far, params={"max_err": 0.35},
    )
    t["traj_xy_far"] = TerminationTermCfg(
        func=P.traj_xy_far, params={"max_dist": 1.0},
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
        # 回合 2.5~5.5 s，2~4 s 一次保证每回合至少挨一次
        push.interval_range_s = (2.0, 4.0)
        if isinstance(getattr(push, "params", None), dict):
            push.params["velocity_range"] = {
                "x": (-0.20, 0.20), "y": (-0.20, 0.20), "z": (-0.10, 0.10),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            }
    if play or not PUSH:
        cfg.events.pop("push_robot", None)


def _set_init_pose(cfg, which: str) -> None:
    """出生姿态 = 参考第 0 帧（站立段从站姿起，爬行段从趴姿起）。"""
    traj = load_traj(REF_TABLE, TASKS[which][0])
    q0 = traj.q[0].cpu().numpy()
    z0 = float(traj.z[0].cpu())
    xy0 = traj.xy[0].cpu().numpy() if traj.xy is not None else (0.0, 0.0)
    ent = cfg.scene.entities["robot"]
    ent.init_state.pos = (float(xy0[0]), float(xy0[1]), z0)
    ent.init_state.joint_pos = {n: float(v) for n, v in zip(JOINT_NAMES, q0)}
    print(f"[prone] {which}: 出生 (x={xy0[0]:+.3f}, y={xy0[1]:+.3f}, z={z0:.3f}) m")


def prone_env_cfg(which: str = "crawl", play: bool = False):
    if which not in TASKS:
        raise ValueError(f"which 只能是 {sorted(TASKS)}，收到 {which!r}")

    cfg = crab_env_cfg(play=play)

    # ---- 地形：平地，初始等级 0 ----
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=_flat_terrain(),
        max_init_terrain_level=0,
    )
    # 本任务不用地形射线/足底/躯干接触，全部去掉（省每帧射线开销）
    cfg.scene.sensors = ()

    # ---- 无速度指令 ----
    cfg.commands = {}

    # ---- 动作：参考轨迹 + 残差 ----
    # 手臂要负责把身体撑起来、撑住（趴<->站的核心），所以明显放宽；
    # 爬行时腿的修正量也大（膝盖贴地拖），腿给 0.40。
    cfg.actions["joint_pos"] = TrajTrackActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=ACTION_SCALE_MAP,
        offset=0.0,
        use_default_offset=False,
        clip=None,
        traj_path=REF_TABLE,
        traj_key=TASKS[which][0],
        leg_clip_rad=0.40,
        waist_clip_rad=0.20,
        arm_clip_rad=0.45,
    )

    _build_observations(cfg, which)
    _build_rewards(cfg)
    _build_terminations(cfg)
    _retune_events(cfg, which, play)
    _set_init_pose(cfg, which)

    # ---- 课程：无 ----
    cfg.curriculum = {}

    # ---- 回合长度：参考时长（已含 1 s 冻结尾）+ 0.5 s 余量 ----
    traj = load_traj(REF_TABLE, TASKS[which][0])
    cfg.episode_length_s = float(traj.duration + 0.5)
    print(f"[prone] {which}: 参考 {traj.duration:.2f} s（含冻结尾）"
          f" -> 回合 {cfg.episode_length_s:.2f} s")

    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
        distance=2.4,
        elevation=-8.0,
        azimuth=190.0,
    )

    if play:
        cfg.episode_length_s = int(1e6)
        cfg.terminations.pop("traj_finished", None)
        cfg.terminations.pop("traj_z_far", None)
        cfg.terminations.pop("traj_xy_far", None)
        cfg.observations["actor"].enable_corruption = False
        print("[prone][play] 已移除 traj_finished / traj_z_far / traj_xy_far 终止")
    return cfg


def prone_task_meta(which: str) -> dict:
    traj = load_traj(REF_TABLE, TASKS[which][0])
    return dict(
        which=which,
        desc=TASKS[which][1],
        frames=traj.T,
        duration=traj.duration,
        z0=float(traj.z[0].cpu()),
        z_end=float(traj.z[-1].cpu()),
        preset=PRESET,
        meta=dict(traj.meta),
    )
