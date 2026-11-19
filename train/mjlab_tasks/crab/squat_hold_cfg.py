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

与跟踪任务的六处差异
--------------------
1. 回合长度 = 保持段本身（默认 20 s，`ELF3_HOLD_EPISODE`），没有跟踪段；
2. 扰动更强更密（每 0.8~2.0 s 推一次 ±0.25 m/s）：保持任务如果不推，
   参考位形本身准静态稳定，策略什么都不做也能拿满分，学不到抗扰；
3. 初始状态带扰动（关节 ±0.05 rad、基座 ±2 cm / ±4.6°）：否则策略只会
   记住「初始状态就是这个」，一被推就崩；
4. 加了 `quiet`（关节速度 RMS 越小越高分）与更重的 `action_rate_l2`：
   「保持」的敌人不是姿势错，而是抖 —— 抖动从脚底传回机身，几秒就积累成漂移。
5. 观测与真机对齐（2026-11-19 重训需求）：去掉 `base_lin_vel`（真机无 odom）
   与 `traj_height_err`（真机无基座高度测量），actor 观测 130 -> 126 维。
   训练输入分布 = 真机能提供的全部信息，不再依赖部署侧「填 0」的近似
   （本地闭环实测：deploy 填 0 明显比 train 真值差，见
   docs/squat_onnx_sim_verify.md）。critic 是特权观测不上真机，
   保留 `traj_height_err` 帮价值函数判断「蹲没蹲到位」。
6. 外力触碰扰动 `touch_robot`（2026-11-19 重训需求）：随机方向 ±35 N 的
   真实力持续 0.2~0.5 s 推躯干（`apply_body_impulse`，自动到期清零），
   模拟真人触碰/碰撞 —— 与 `push_robot` 的瞬时速度冲击互补：
   速度冲击是「被撞一下」，持续力是「被人推一把」，带倾覆力矩（力臂感）。

另有一组「收紧训练压力」的改动（终止 55°->30°、塌陷 0.35->0.48、姿态奖励
加大带宽收紧、quiet σ 0.8->0.4），动机与标定见 docs/squat_hold_retrain.md：
旧版终止松到允许「漂着活完回合」，hold 学成了每秒 +2.2° 的慢发散。
"""
from __future__ import annotations

import math
import os

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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
# 2026-11-19 重训：squat 塌陷阈值 0.35 -> 0.48（蹲姿 z=0.523，掉 4 cm 即终止）。
# 旧阈值 0.35 允许策略「漂着活完回合」，是 hold 学成慢发散的根因之一。
KINDS = {
    "squat": dict(frame=-1, min_z=0.48, desc="极限蹲保持"),
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
    # 55 -> 30：蹲姿固有前倾 18.5° + 12° 余量。允许姿态波动，不允许持续前倾漂移
    # （旧 55° 下「每秒 +2.2° 漂 20 s」能活完回合，docs/squat_hold_retrain.md §一.1）
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(30.0)},
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


def _add_touch_event(cfg, play: bool) -> None:
    """外力触碰扰动：随机方向的持续力推躯干（模拟真人触碰/碰撞）。

    与 push_robot（瞬时改速度）互补，两类扰动都保留：
      - push_by_setting_velocity = 速度阶跃，「被撞了一下」，无质量/力臂感；
      - apply_body_impulse       = 真实力持续 N 秒，「被人推一把」，
        offset×force 产生倾覆力矩（推在胸口会让人往后仰），到期自动清零。

    量级标定（整机 43.2 kg、足底 μ≈1.0、双足打滑极限 ~420 N）：
      每分量 ±35 N（世界系均匀采样）→ 典型水平合幅 ~26 N，持续 0.2~0.5 s
      → 冲量 5~13 N·s，与 push_robot 的 ±0.25 m/s（≈10.8 N·s）同量级。
      力含竖直分量，顺带覆盖「按压肩膀 / 腋下托举」类扰动。

    ⚠️ apply_body_impulse 是 mjlab main 分支才有的类事件（事件管理器检测到
    class 自动实例化并自管生命周期，mode 必须是 "step"）。旧版训练机没有：
    明确报错让人升级 mjlab，不做静默降级。
    """
    if play:
        return  # 回放不做扰动（与 push_robot 的 play 行为一致）
    impulse_cls = getattr(mdp, "apply_body_impulse", None)
    if impulse_cls is None:
        raise RuntimeError(
            "当前 mjlab 没有 mdp.apply_body_impulse（外力触碰事件，main 分支才有）。\n"
            "训练机升级：bash scripts/bootstrap_gpu.sh 重装 mjlab（GitHub main），\n"
            "或：uv pip install -U 'mjlab @ git+https://github.com/mujocolab/mjlab.git'"
        )
    cfg.events["touch_robot"] = EventTermCfg(
        func=impulse_cls,
        mode="step",                     # 类事件自管生命周期（冷却->触发->持续->清零）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "force_range": (-35.0, 35.0),        # 每分量 ±N（世界系）
            "torque_range": (-5.0, 5.0),         # 附加小扭矩（Nm）
            "duration_s": (0.2, 0.5),            # 一次触碰持续 0.2~0.5 s
            "cooldown_s": (1.5, 3.0),            # 两次触碰间隔 1.5~3.0 s
            # 力作用点沿躯干 z 上移 12 cm（胸口/后背高度）：offset×force 产生
            # 倾覆力矩，模拟「推在身上」而不是「推在质心上的无磨损滑块」
            "body_point_offset": (0.0, 0.0, 0.12),
        },
    )


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
    # ---- 真机观测对齐（2026-11-19 重训需求）----
    # 去掉真机没有的观测块，训练输入分布 = 真机可提供的全部信息，
    # 不再依赖部署侧「填 0」的近似：
    #   base_lin_vel    真机无 odom（编码器/IMU 给不出世界系线速度）；
    #   traj_height_err 真机无基座高度测量（部署只能填 0，而训练噪声仅 ±5 mm，
    #                   deploy 分布外 —— 本地实测 deploy 明显比 train 差）。
    # actor 观测 130 -> 126 维：ang_vel(3)+gravity(3)+joint_pos(31)+joint_vel(31)
    #   +actions(29)+traj_joint_err(29)。导出 ONNX 的 obs_dim 自动跟随。
    for _name in ("base_lin_vel", "traj_height_err"):
        cfg.observations["actor"].terms.pop(_name, None)
    # critic 是特权观测（不上真机）：保留 traj_height_err —— 高度误差真值
    # 帮价值函数判断「蹲没蹲到位」；base_lin_vel 在保持任务里真值 ≈0、
    # 训练噪声 ±0.3 纯属垃圾信息，一并去掉。
    cfg.observations["critic"].terms.pop("base_lin_vel", None)
    # 复用跟踪任务的整套奖励：参考冻结后它们就是「保持位形」的奖励
    _build_rewards(cfg, arm_guard=arm_guard)
    # ---- 保持任务专属奖励收紧（docs/squat_hold_retrain.md §一.2）----
    # 姿态权重加大、带宽收紧 —— 逼策略把稳态误差主动补掉，而不是顺应前倾
    # （原 up_track 1.0/σ0.25、height_track 2.0/σ0.05）。
    cfg.rewards["up_track"].weight = 3.0
    cfg.rewards["up_track"].params["sigma"] = 0.15
    cfg.rewards["height_track"].weight = 4.0
    # 再叠两条只有「保持」才需要的：别抖、别乱动
    cfg.rewards["quiet"] = RewardTermCfg(
        func=S.quiet, weight=0.7, params={"sigma": 0.4},   # 0.8 -> 0.4：更严格的「别抖」
    )
    cfg.rewards["action_rate_l2"].weight = -0.05
    # ---- 外力触碰扰动（真机对齐 + 抗扰需求，2026-11-19）----
    _add_touch_event(cfg, play)

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
    _obs_terms = tuple(cfg.observations["actor"].terms)
    _obs_dim = sum(
        {"base_ang_vel": 3, "projected_gravity": 3, "joint_pos": 31,
         "joint_vel": 31, "actions": 29, "traj_joint_err": 29,
         "traj_height_err": 1, "base_lin_vel": 3, "traj_progress": 2}.get(n, 0)
        for n in _obs_terms
    )
    _ev = "push_robot" + ("+touch_robot" if "touch_robot" in cfg.events else "")
    print(f"[hold] actor 观测 {_obs_dim} 维（{'+'.join(_obs_terms)}）—— "
          f"{'真机对齐版：无 base_lin_vel / traj_height_err' if 'base_lin_vel' not in _obs_terms else '含真机缺失项！'}")
    print(f"[hold] 扰动事件：{_ev}" + ("（play 模式无扰动）" if play else ""))

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
