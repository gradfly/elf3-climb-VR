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
import os

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
    #
    # ⚠️ 2026-10-29 修一个静默 bug：这里原来写的是 "feet_swing_height"（多一个 s），
    # 而 `crab_env_cfg` 的奖励字典里键名是 **"foot_swing_height"** —— pop 从来没
    # 匹配上，这条惩罚一直在跑（日志里就有 Episode_Reward/foot_swing_height）。
    #
    # 它和 α 课程是直接打架的：α<1 时参考步态只抬 5~12 cm，而这条的
    # target_height 是 0.18，等于一手要求"跟随参考"、一手罚"没抬到 18 cm"。
    #
    # 处理方式不是简单删掉 —— 删了抬脚就完全没人管，会退化成贴地蹭
    # （实测 Metrics/peak_height_mean 只有 1.6 cm，正是这个退化的指纹）。
    # 而是把两个高度项的目标统一压到 0.10 m：α=1 时参考抬约 18 cm，
    # 0.10 只作为"必须真的离地"的下界，不再和参考抢梯度。
    r.pop("feet_swing_height", None)          # 旧拼写，留着无害、也防止将来加回来
    r.pop("track_angular_velocity", None)

    for _name in ("foot_swing_height", "foot_clearance"):
        _term = r.get(_name)
        if _term is not None and isinstance(getattr(_term, "params", None), dict):
            _term.params["target_height"] = 0.10   # 0.18 -> 0.10（见上面的说明）

    # 2000 轮实测（每步加权值）：upright 0.92 + ref_tracking 0.49 + track_lin 0.86
    # 合计 2.3 分几乎不用动就能拿到，而"走快"只多拿 forward_progress 0.107。
    # 结果就是策略优先生存而不是前进 —— 这里把"站着也能拿的分"压下去。
    if "upright" in r:
        r["upright"].weight = 0.5          # 1.0 -> 0.5（原值 0.92/步，白送分）
    if "action_rate_l2" in r:
        # -0.05 -> -0.03 -> **-0.01**（2026-10-29 第二次实测）
        # 471 轮日志：action_rate_l2 = -0.1444 × 20 = **-2.89/回合**，
        # 而 forward_progress_y 只有 0.0657 × 20 = **+1.31/回合** ——
        # 平滑惩罚是前进奖励的 2.2 倍，等于在奖励它"别动"。
        # α 到 1 之后参考步态满幅，残差需要更大幅度的调整，这个惩罚会更伤人。
        # 再往下降没必要：0.01 时约 -0.96/回合，仍是前进奖励的 73%，够抑制抖动。
        r["action_rate_l2"].weight = -0.01

    r["forward_progress_y"] = RewardTermCfg(
        func=X.forward_progress_y, weight=3.0,     # 1.0 -> 3.0，位移成为主奖励
        params={"command_name": "twist"},
    )
    r["ref_tracking"] = RewardTermCfg(
        func=X.ref_tracking, weight=0.3,           # 0.6 -> 0.3（原值已 0.81 饱和）
        params={"action_name": "joint_pos", "sigma": 0.35},
    )
    # sin(yaw)²：转身就罚。螃蟹步的定义是「朝向不变 + 侧向平移」，
    # 所以这条不是微调，是任务约束本身。
    # -0.5 太轻：偏 30° 时每步 -0.5×0.25×0.02 = -0.0025，500 步才 -1.2，
    # 而走完全程 +1.3 —— 转个身和走完全程代价相当，等于鼓励它转身走。
    # -1.5 让 30° 偏转 ≈ -3.7/回合，明显痛于前进收益。注意 sin² 在 0 附近是
    # 平的（二次），小角度几乎无罚，不会妨碍平衡调整。
    r["yaw_drift"] = RewardTermCfg(func=X.yaw_drift, weight=-1.5, params={})
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


def _retune_viewer(cfg) -> None:
    """回放/录像的相机设置：默认**钉在世界系**，不要用跟拍。

    ⚠️ 2026-10-29 教训：ViewerConfig.origin_type 默认 AUTO（跟随机器人）。
    跟拍相机下机器人永远在画面中央，台阶跟着画面一起退，
    **肉眼完全看不出它到底走了没有** —— 一段 16 s 的视频就这么被误读成
    "一次侧步都没有"，而同一份 checkpoint 无界面实测 v_y=0.126 m/s。

    ⚠️ 2026-10-29 晚：按使用习惯把默认机位改回 **AUTO（跟拍）**，方便看脚下细节。
    代价就是上面那个坑回来了 —— 判断「到底有没有侧移」时必须显式切固定机位：

        ELF3_VIEWER_ORIGIN=world   固定机位：台阶在画面里不动，机器人相对它平移
        ELF3_VIEWER_ORIGIN=asset   跟资产根，比 auto 更近

    或者干脆别靠肉眼，看诊断脚本的 v_y / yaw 速率（README 10.5）。

    可用环境变量（不用改代码）：
        ELF3_VIEWER_ORIGIN=auto|world|asset   （默认 auto = 跟拍）
        ELF3_VIEWER_LOOKAT="0,0.8,0.3"        （世界系注视点，仅 world 模式自动设）
        ELF3_VIEWER_DIST=6 / ELEV=-18 / AZIM=90
        ELF3_VIEWER_W=1280 / H=720
    """
    v = getattr(cfg, "viewer", None)
    if v is None:
        return

    def _set(name, value):
        if not hasattr(v, name):
            print(f"[viewer][warn] 这个 mjlab 的 ViewerConfig 没有 {name}，已跳过")
            return
        try:
            setattr(v, name, value)
        except Exception as e:
            print(f"[viewer][warn] 设置 {name} 失败：{e}")

    _set("width", int(os.environ.get("ELF3_VIEWER_W", 1280)))
    _set("height", int(os.environ.get("ELF3_VIEWER_H", 720)))

    origin = os.environ.get("ELF3_VIEWER_ORIGIN", "auto").strip().lower()
    Ot = getattr(type(v), "OriginType", None)
    if Ot is not None:
        # world -> 固定机位（能看出位移）；asset -> 跟拍（能看清脚下细节）
        want = ("WORLD" if origin in ("world", "w")
                else "AUTO" if origin == "auto" else "ASSET_ROOT")
        target = None
        for n in dir(Ot):
            if n.upper() == want:
                target = getattr(Ot, n)
                break
        if target is None:        # 找不到就退回 AUTO
            for n in dir(Ot):
                if n.upper() == "AUTO":
                    target = getattr(Ot, n)
                    break
        if origin in ("world", "w") and target is not getattr(Ot, "WORLD", None):
            print("[viewer][warn] 这个 mjlab 的 OriginType 没有 WORLD，相机仍是跟拍；"
                  "看位移请改用诊断脚本（scripts/diagnose_policy.py）")
        _set("origin_type", target)
    elif origin in ("world", "w"):
        print("[viewer][warn] 这个 mjlab 的 ViewerConfig 没有 OriginType，相机无法固定，"
              "录出来的视频是跟拍的（看不出位移）。用诊断脚本的 v_y / yaw 速率判断。")

    lookat = os.environ.get("ELF3_VIEWER_LOOKAT")
    if lookat:
        try:
            _set("lookat", tuple(float(t) for t in lookat.split(",")))
        except Exception:
            print(f"[viewer][warn] ELF3_VIEWER_LOOKAT={lookat!r} 解析失败，用默认值")
    elif origin in ("world", "w"):
        # 只有世界系固定机位才需要手给注视点：机器人从碗底平地出发沿 +Y 走，
        # 放在前方一点，能容纳 ~5 m 位移。
        # 跟拍（auto/asset）模式下 lookat 是相对资产的偏移，硬塞世界坐标会把
        # 镜头甩到空地上（画面里看不到机器人） —— 所以这里不再无条件设。
        _set("lookat", (0.0, 0.8, 0.3))
    _set("distance", float(os.environ.get("ELF3_VIEWER_DIST", 6.0)))
    _set("elevation", float(os.environ.get("ELF3_VIEWER_ELEV", -18.0)))
    _set("azimuth", float(os.environ.get("ELF3_VIEWER_AZIM", 90.0)))


def _retune_curriculum(cfg) -> None:
    # ⚠️ 2026-10-29：不再靠 promotion_fraction 调门槛，**直接换掉整个课程项**。
    # 内置 terrain_levels_vel 的 commanded_distance 会把「摔倒后剩余时间应走的
    # 距离」补进去，等于摔得越早要求越高：实测 200 轮 ep_len=4.2 s 时要求走
    # 3.04 m，实际只走 0.42 m -> 永远升不上去且必然降级 -> 永远卡在 flat，
    # 一辈子见不到台阶。详见 mdp_extra.terrain_levels_time 的注释。
    if cfg.curriculum is not None and "terrain_levels" in cfg.curriculum:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=X.terrain_levels_time,
            params={"promote_frac": 0.25, "demote_frac": 0.10, "min_dist": 0.30},
        )

    # 初始等级只给 0~1。
    # ⚠️ 2026-10-29 更正：之前设成 4，理由是「让一部分 env 开局就落在 easy 档」，
    # 但那是基于「level 0 = flat」这个错误前提（见 terrain_cfg 顶部的说明）。
    # 实际上 curriculum 只调行、列（类型）是随机且终身的：level 0 就已经覆盖
    # flat/easy/mid/hard 四种类型，设成 4 等于把 hard 列的 env 直接推到
    # 8 cm + 4/9×14 cm ≈ 14 cm，而它们连平地都还站不稳。
    terr = getattr(getattr(cfg, "scene", None), "terrain", None)
    if terr is not None and hasattr(terr, "max_init_terrain_level"):
        try:
            terr.max_init_terrain_level = 1
        except Exception as e:
            print(f"[curriculum][warn] 设置 max_init_terrain_level 失败：{e}")

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


def _alpha_fixed(play: bool) -> float:
    """参考幅度 α 的取值：<0 表示走课程，>=0 表示固定。

    优先级：环境变量 ELF3_ALPHA > play 默认 1.0 > 训练走课程(-1)。

    ⚠️ 为什么训练也得能固定：续训时动作项是**新建**的，内部 `_step_i` 从 0
    开始，`_alpha_now()` 于是把 α 退回 0（参考 = 纯站姿）。可策略是在 α=1 下
    训出来的，残差量级、相位节奏全按满幅步态长出来 —— 参考突然变回站姿，
    两者错位，行为当场崩，等于把学会的东西又打回去。

    所以续训一个 α 已经满幅的 checkpoint，必须 `ELF3_ALPHA=1.0` 把它钉住：

        ELF3_ALPHA=1.0 .venv/bin/train Elf3-Crab-Residual --agent.resume True ...

    另一个方向也成立：想复现「第 N 轮时的真实水平」做录像/对拍，
    就用当时的 `Metrics/ref_alpha`（见 README 10.2.0）。
    """
    raw = os.environ.get("ELF3_ALPHA")
    if raw is not None and raw.strip() != "":
        try:
            return max(0.0, float(raw))
        except ValueError:
            print(f"[cfg][warn] ELF3_ALPHA={raw!r} 不是数字，忽略")
    return 1.0 if play else -1.0


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
        # play 默认满幅（α=1），但训练是走课程的：790 轮时 ref_alpha 才 0.527。
        # 直接回放等于「满幅参考 + 为半幅参考训出来的残差」，残差补不上差额，
        # 画面会明显比训练时差 —— 别据此判断策略退步。
        # 训练和 play 都可被 ELF3_ALPHA 覆盖（-1 以外的值 = 固定 α），
        # 续训 α 已满幅的 checkpoint 时必须 `ELF3_ALPHA=1.0`，理由见 _alpha_fixed。
        alpha_fixed=_alpha_fixed(play),
    )

    _rebuild_observations(cfg)
    _rebuild_rewards(cfg)
    _retune_commands(cfg, play)
    _retune_curriculum(cfg)
    if play:
        _retune_viewer(cfg)
    return cfg
