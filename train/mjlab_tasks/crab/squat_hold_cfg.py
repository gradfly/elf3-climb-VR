"""极限蹲 / 站姿「静态保持」独立任务（全过程保持一个位形，不播参考）。

为什么要单独拆出来
------------------
`Elf3-Squat-Down/Up` 的回合是「跟踪 2.46 s + 保持 5 s」：跟踪段占 62% 的步数，
策略的梯度主要花在「跟着时间轴动」上，「站住」只在最后 5 s 被要求，而且是
**同一个网络兼顾两件事**的折中结果。想让「蹲住不动」成为第一目标，就得让
整个回合都是保持段 —— 这个文件把它拆成独立任务，单独训一个网络。

两个任务（分别训练，各自一份实验目录）：

    Elf3-Squat-Hold   极限蹲保持：出生 = down 末帧，z = 0.523 m，膝 149°
    Elf3-Stand-Hold   站姿保持  ：出生 = down 首帧，z = 1.034 m

实现方式：**不新增参考表**。直接复用 `squat_ref_v1.npz` 的 down 表，把帧号
冻结在两端（`TrajTrackActionCfg.freeze_frame`）。冻结后 `traj_track /
height_track / up_track / symmetry` 全部自动退化成「保持这个位形」，
奖励代码一行没改 —— 这也是不复刻一份奖励函数的原因：两份实现早晚会漂移。

与跟踪任务的四处差异
--------------------
1. 回合长度 = 保持段本身（默认 20 s，`ELF3_HOLD_EPISODE`），没有跟踪段；
2. 扰动更强更密（每 0.8~2.0 s 推一次 ±0.25 m/s）：保持任务如果不推，
   参考位形本身准静态稳定，策略什么都不做也能拿满分，学不到抗扰；
3. 初始状态带扰动（关节 ±0.05 rad、基座 ±2 cm / ±4.6°）：否则策略只会
   记住「初始状态就是这个」，一被推就崩；
4. 加了 `quiet`（关节速度 RMS 越小越高分）与更重的 `action_rate_l2`：
   「保持」的敌人不是姿势错，而是抖 —— 抖动从脚底传回机身，几秒就积累成漂移。
"""
from __future__ import annotations

import math
import os

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from . import squat_mdp as S
from .env_cfg import crab_env_cfg
from .robot_cfg import ACTION_SCALE_MAP
from .squat_env_cfg import (
    ARM_GUARD,
    REF_TABLE,
    _arm_leg_sensor,
    _build_observations,
    _build_rewards,
    _flat_terrain,
    _set_init_pose,
)
from .traj_action import TrajTrackActionCfg

# kind -> 参考帧（down 表：0 = 站姿、-1 = 极限蹲）、塌陷阈值、中文名
KINDS = {
    "squat": dict(frame=-1, min_z=0.35, desc="极限蹲保持"),
    "stand": dict(frame=0, min_z=0.90, desc="站姿保持"),
}

# 回合长度（秒）。20 s 里大约挨 14 次推，足够学到抗扰，又不至于让每次 episode
# 长到采样效率崩掉。想更狠：ELF3_HOLD_EPISODE=40 bash scripts/run_squat_train.sh hold
EPISODE_S = float(os.environ.get("ELF3_HOLD_EPISODE", "20.0") or 20.0)

# 「别走」阈值：跟踪任务是 1.2 m（允许一点位移），保持任务只给 0.5 m
MAX_DRIFT_M = 0.5


def _retune_events(cfg, play: bool) -> None:
    """把基线的事件改成「保持任务该有的扰动强度」。"""
    rb = cfg.events.get("reset_base")
    if rb is not None and isinstance(getattr(rb, "params", None), dict):
        # 这些是**相对出生点**的偏移（基线 z=(0,0.03) 就是抬 0~3 cm，不是绝对高度）
        rb.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.005),
            "yaw": (-0.08, 0.08),          # ±4.6°
        }

    rj = cfg.events.get("reset_robot_joints")
    if rj is not None and isinstance(getattr(rj, "params", None), dict):
        # ±0.05 rad ≈ 2.9°，正好是「PD 能拉回来但需要主动发力」的量级。
        # 太大（>0.15）会让一半 env 出生就撞腿/翻倒，采样全是失败回合。
        rj.params["position_range"] = (-0.05, 0.05)
        rj.params["velocity_range"] = (0.0, 0.0)

    push = cfg.events.get("push_robot")
    if push is not None:
        # 基线是 (1,3) s / ±0.3；这里更密但稍小：密度决定「恢复」的样本量，
        # 幅度决定难度。保持任务没有跟踪段分摊步数，可以吃得更密。
        push.interval_range_s = (0.8, 2.0)
        if isinstance(getattr(push, "params", None), dict):
            push.params["velocity_range"] = {
                "x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (-0.12, 0.12),
                "roll": (-0.15, 0.15), "pitch": (-0.15, 0.15), "yaw": (-0.15, 0.15),
            }
    if play:
        cfg.events.pop("push_robot", None)


def _build_terminations(cfg, kind: str) -> None:
    """保持任务的终止：比跟踪任务严（没有人推着走，漂 0.5 m 就是错的）。"""
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
        func=S.base_too_low, params={"min_z": KINDS[kind]["min_z"]},
    )
    cfg.terminations["drifted"] = TerminationTermCfg(
        func=S.base_drifted, params={"max_xy": MAX_DRIFT_M},
    )
    # 注意：**不注册 traj_finished**。参考帧是冻结的，`finished` 恒 False，
    # 注册了也只是每步算一个常 False 的张量；回合该由时长/摔倒决定。


def squat_hold_env_cfg(kind: str = "squat", play: bool = False):
    if kind not in KINDS:
        raise ValueError(f"kind 只能是 {sorted(KINDS)}，收到 {kind!r}")
    K = KINDS[kind]

    cfg = crab_env_cfg(play=play)

    # ---- 地形：平地 ----
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=_flat_terrain(),
        max_init_terrain_level=0,
    )
    # 只留躯干接触传感器。足底接触/地形射线在本任务里没用（保持不爬台阶），
    # 去掉能省掉几百条射线 × 每个 env 的每帧开销。
    cfg.scene.sensors = tuple(
        s for s in cfg.scene.sensors if getattr(s, "name", "") == "torso_ground_contact"
    )
    if not cfg.scene.sensors:
        raise RuntimeError("没找到 torso_ground_contact 传感器，终止项会失效")

    arm_guard = bool(ARM_GUARD)
    if arm_guard:
        try:
            cfg.scene.sensors = tuple(cfg.scene.sensors) + (_arm_leg_sensor(),)
            print("[hold] 已加手臂-腿接触传感器 arm_leg_contact（碰到罚 -0.6/步）")
        except Exception as e:                    # noqa: BLE001
            arm_guard = False
            print(f"[hold][warn] 手臂-腿接触传感器建不起来（{e!r}），跳过接触惩罚")

    # ---- 无速度指令 ----
    cfg.commands = {}

    # ---- 动作：冻结帧 + 残差 ----
    cfg.actions["joint_pos"] = TrajTrackActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=ACTION_SCALE_MAP,
        offset=0.0,
        use_default_offset=False,
        clip=None,
        traj_path=REF_TABLE,
        traj_key="down",
        freeze_frame=int(K["frame"]),
        # 保持任务里残差只用来救平衡，不需要「照顾时间轴」的余量：
        # 腿 0.35 保留（推一下得能把重心拉回来），腰 0.15、臂 0.12 收紧，
        # 免得策略又把手臂压到大腿上。
        leg_clip_rad=0.35,
        waist_clip_rad=0.15,
        arm_clip_rad=0.12,
    )

    _build_observations(cfg, "down", with_progress=False)
    # 复用跟踪任务的整套奖励：参考冻结后它们就是「保持位形」的奖励
    _build_rewards(cfg, arm_guard=arm_guard)
    # 再叠两条只有「保持」才需要的：别抖、别乱动
    cfg.rewards["quiet"] = RewardTermCfg(
        func=S.quiet, weight=0.7, params={"sigma": 0.8},
    )
    cfg.rewards["action_rate_l2"].weight = -0.05

    _build_terminations(cfg, kind)
    _retune_events(cfg, play)
    _set_init_pose(cfg, "down", frame=int(K["frame"]))

    cfg.curriculum = {}

    # ---- 回合长度 = 保持段本身 ----
    cfg.episode_length_s = max(2.0, float(EPISODE_S))
    from .traj_ref import load_traj
    _T = load_traj(REF_TABLE, "down").T
    print(f"[hold] {kind}（{K['desc']}）：{cfg.episode_length_s:.1f} s/回合，"
          f"参考冻结在 down 第 {int(K['frame']) % _T}/{_T - 1} 帧，"
          f"塌陷阈值 z<{K['min_z']:.2f} m，漂移阈值 {MAX_DRIFT_M:.1f} m")

    cfg.viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name="torso_link",
        distance=2.2,
        elevation=-6.0,
        azimuth=200.0,
    )

    if play:
        cfg.episode_length_s = int(1e6)     # 回放：一直保持，不自动重置
        cfg.observations["actor"].enable_corruption = False
        print("[hold][play] 回合长度拉到 1e6 s、去掉扰动：可以一直看着它站/蹲")
    return cfg


def squat_hold_meta(kind: str) -> dict:
    """训练脚本/文档用。"""
    from .squat_env_cfg import squat_task_meta

    m = squat_task_meta("down")
    f = int(KINDS[kind]["frame"]) % int(m["frames"])
    return dict(
        kind=kind,
        desc=KINDS[kind]["desc"],
        frame=f,
        z=float(m["z0"] if f == 0 else m["z_end"]),
        episode_s=EPISODE_S,
        min_z=KINDS[kind]["min_z"],
    )
