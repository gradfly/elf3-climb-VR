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
from mjlab.managers.termination_manager import TerminationTermCfg
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

    # ⚠️ 2026-10-30 晚：target 0.10 -> 0.15，swing_height 权重 -0.25 -> -1.0。
    # 依据是 FK 实测：参考步态（rise=0.12 / tread=0.30）的脚抬升幅度是 0.174 m，
    # 而策略的 Metrics/peak_height_mean 只有 0.029 —— 只跟了 17%。视频里能直接
    # 看到它一直在碗底平台蹭、从不抬腿上台阶。
    # 原因是抬脚「收益≈0、风险很大」：air_time 够不到门槛（见下），
    # swing_height 每次落地只罚 0.125，而抬脚失衡一次就是 -5 分。
    # 加权后：峰值停在 2.9 cm 时每回合约 -13 分（相对回报 ~32 是 41%），
    # 真抬到 15 cm 则 cost=0、不罚 —— 这才是引导而不是单纯加痛。
    for _name in ("foot_swing_height", "foot_clearance"):
        _term = r.get(_name)
        if _term is not None and isinstance(getattr(_term, "params", None), dict):
            _term.params["target_height"] = 0.15
    if "foot_swing_height" in r:
        r["foot_swing_height"].weight = -1.0

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
        # 2026-10-30 晚：leg_clip_rad 0.40 -> 0.55 之后，同样的策略输出会变成更大的
        # 实际动作，这条惩罚跟着水涨船高（等于变相加强了「别动」）。压到 -0.006
        # 抵消量纲变化。
        r["action_rate_l2"].weight = -0.006

    # ① 速度奖励：丢掉内置 track_linear_velocity，换成 EMA 平均推进 + 按指令归一化。
    #    内置那条有两个结构性问题：
    #    (a) 它是 xy_error + z_error，指令 z 恒为 0 等于要求 v_z = 0，而爬台阶
    #        必须 v_z ≈ rise/1.8 ≈ 0.08~0.10 m/s —— 正确爬楼反被判"竖直超速"。
    #    (b) 它比的是**瞬时**速度，而参考步态的 COM 是脉冲式的（v_y 均值 0.167、
    #        峰值 1.618、26% 帧在倒退、std 0.519）。任何瞬时 exp 核（std 0.055~0.15）
    #        在这种剖面下都只奖励"匀速蹭走"、重罚"真的迈步"：实测跟参考每步
    #        0.02~0.45，而 0.126 m/s 的蹭走 0.88，方向完全反了。
    #    换成 X.forward_progress_y（EMA tau=1.8 s = 一个步态周期、cap 1.0）后标定：
    #        跟参考 0.85 / 蹭走(0.126) 0.70 / 站住 0
    #    tau 不能取小（0.4 s 时跟参考只有 0.59 < 蹭走 0.70，这一项自己在奖励蹭：
    #    负速度帧被 floor 掉，只给 74% 的周期发钱 —— 详见 mdp_extra 的说明）。
    #    它区分不开"踩步"和"蹭"（平均速度只差 25%），那是 ref_tracking + air_time
    #    + 躯干接触终止的活；这一项只负责"别停、别超速"（旧实现允许 1.5 倍指令，
    #    正好让"匀速快蹭"凭超速压过真实步态）。
    r.pop("track_linear_velocity", None)
    r["forward_progress_y"] = RewardTermCfg(
        func=X.forward_progress_y, weight=5.0,     # 1.0 -> 3.0 -> 5.0，主推进项
        params={"command_name": "twist", "tau": 1.8},
    )

    # ② 抬脚从来没被奖励过：air_time 的 weight=0（mjlab 对 0 权重连 func 都不调用），
    #    而且 command_threshold=0.5 对 0.14~0.20 的指令恒为 0 —— 就算给了权重也会被
    #    整体清零。实测 peak_height_mean 只有 1.6 cm，贴地蹭走之所以是稳态解，
    #    就是因为"抬脚"在奖励函数里没有任何正收益。
    #    threshold_max=1.2：参考步态摆动期 33/90 帧，相位速率 0.6~1.8 时墙钟
    #    摆动时长 0.37~1.1 s，上限必须覆盖慢节奏那一端。
    # threshold_min 0.12 -> 0.05：实测 air_time_mean 只有 0.0704，原门槛让这条
    # 奖励恒为 0 —— 连「抬起了一点」都拿不到钱。降到 0.05 后 7 cm 的摆动就有
    # 正收益；weight 0.6 -> 1.0，让「抬脚」成为明确赚钱的项。
    _air = r.get("air_time")
    if _air is not None and isinstance(getattr(_air, "params", None), dict):
        _air.weight = 1.0
        _air.params.update(
            threshold_min=0.05, threshold_max=1.2, command_threshold=0.05,
        )

    # ③ 参考跟踪是唯一逼出「真正侧步」的项，但 0.3 的权重给的分太少：
    #    实测 ref_tracking 报 0.49（等价 12 腿 RMS 偏差 0.42 rad，已经"没在跟"）。
    #    而速度项已经退化成"平均推进"（跟参考 0.83 / 蹭走 0.70，只差 0.13），
    #    区分步态的责任全落在这条上，权重必须够：0.3 -> 1.5 -> 2.0。
    #    sigma 0.30：0.42 rad RMS 时 exp(-0.175/0.18)=0.38（仍有梯度），
    #    0.15 rad RMS 时 0.88。太紧（<=0.2）会把"参考本身动力学不可行、残差必须
    #    偏离"的那部分偏差也一起罚掉，反而逼它去跟飞。
    r["ref_tracking"] = RewardTermCfg(
        func=X.ref_tracking, weight=2.0,
        params={"action_name": "joint_pos", "sigma": 0.30},
    )
    # 帧间偏航角速度平方 (rad/s)²：转身/绕圈就罚。螃蟹步的定义是
    # 「朝向不变 + 侧向平移」，所以这条不是微调，是任务约束本身。
    # ⚠️ 2026-10-30 换了实现。原用 sin(yaw)²，它对 yaw 和 yaw+2π 取值相同 ——
    # 原地绕一整圈回到原点就「无罪」：实测 6000 轮 checkpoint 在 play 里 20 s
    # 累计转了 367°，这条只报 sin²=0.018（等价净偏航 7.5°），等于完全没罚。
    # 现在看每步的 (Δyaw/dt)²，绕圈会持续累计。
    # 标定：0.7 rad/s -> 0.49 -> 每步 -0.735（×dt×600 ≈ -8.8/回合），
    # 相对单回合回报 ~75 分够痛但不主导；0.1 rad/s 只 -0.015/步，不干扰正常步态。
    r["yaw_drift"] = RewardTermCfg(func=X.yaw_drift, weight=-1.5, params={})
    r["lateral_drift"] = RewardTermCfg(func=X.lateral_drift, weight=-0.2, params={})
    # 2026-10-31：X 向漂移实测判定为 **B（步态不对称）**。证据：三档（L3/L5/L7）机身系
    # 纵向速度的整周期均值都是 +0.10~0.13 m/s，是侧向指令(0.17)的 60%，且与难度无关；
    # 而 corr(vx, yaw) 只有 -0.20~-0.40（A 的证据很弱）。
    # 上面那条 lateral_drift 是 (x-x0)²，0.2 m 处每步只 -0.00016 分（回合 -0.003），
    # 等于没罚，所以补这一条：EMA(一个周期) 后的纵向偏置，死区 0.03 放过正常摆动。
    # 权重标定见 mdp_extra.lateral_bias 的 docstring：0.10 m/s -> 约 -4.7/回合。
    r["lateral_bias"] = RewardTermCfg(
        func=X.lateral_bias, weight=-80.0, params={"tau": 1.8, "deadzone": 0.03},
    )
    # ⚠️ 奖励按 dt 缩放：一次性惩罚实际值 = weight × 0.02。
    # 原来 −5 只等于每次摔倒扣 0.1 分，而当时一个 episode 回报 ~10 分 ——
    # 摔倒几乎免费，策略自然选择"用速度换摔"。−150 时小回合 ≈ 30%。
    # ⚠️ 2026-10-30 晚：−400 实测把策略逼成了「站着不动」。
    # 1065 轮实测：摔率 78%、单回合正奖励只有 ~0.5 分，而摔一次 −8 分 ——
    # 是整回合收益的 16 倍。只要「走」让摔倒概率上升 5%，期望收益就转负。
    # 数据上是教科书式的「用停滞换生存」：ep_len 从 5.7 s 涨到 9.4 s 的同时，
    # 给分段的推进强度从 0.39 一路掉到 0.30（320/750/800/1065 四个点单调下降）。
    # −250（−5/次）仍然疼（≈6 个回合的正奖励），但不再让「走」变成必输的赌局。
    r["is_terminated"] = RewardTermCfg(func=X.is_terminated, weight=-250.0, params={})

    # 探针：只把 rise_est / tread_est / scan_ok 写进 Metrics/*，不影响学习。
    # 权重必须非零（0 会让 mjlab 整项跳过、连 func 都不调用）。
    r["ref_probe"] = RewardTermCfg(
        func=X.ref_probe, weight=1e-9, params={"action_name": "joint_pos"},
    )


def _retune_commands(cfg, play: bool) -> None:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    # ⚠️ 指令上界必须和「步态几何能给出的速度」对齐，否则永远追不上：
    #   参考步态的位移速率 = tread / 1.8 s × 相位速率，速率又被
    #   v_rate_range 夹在 [0.6, 1.8]，于是
    #       可达最大速度 = tread/1.8 × 1.8 = tread
    #   tread=0.24 的窄踏面只有 0.24 m/s，原来 (0.15, 0.24) 的上界刚好贴着它，
    #   EMA 抖动一下就会出现「物理上不可能追上」的指令 -> 这一项永远只有
    #   部分分、梯度被削平，等于给所有行为加了一个常数偏置。
    #   收窄到 (0.14, 0.20)：窄踏面(0.24) 也能覆盖，宽踏面(0.40) 只是节奏更慢。
    twist.ranges.lin_vel_y = (0.14, 0.20)
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
                  "看位移请改用诊断脚本打印的 v_y / Δy（它的录像也是跟拍）")
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
                {"step": 0, "lin_vel_y": (0.10, 0.14)},
                {"step": 400 * 24, "lin_vel_y": (0.13, 0.17)},
                {"step": 1200 * 24, "lin_vel_y": (0.14, 0.20)},
            ],
        }),
    )


def _retune_safety(cfg) -> None:
    """堵两个「活着就能拿分」的漏洞，并把早期扰动降到可控。

    ① 躯干/腰/头 碰地即终止。
       fell_over(70°) 只挡得住完全翻倒；跪在台阶上、侧身靠着台阶蹭上去
       这类姿态倾角远不到 70°，却照样能吃到 upright 0.5 + forward_progress，
       而且残差不用再负责平衡 —— 这是残差策略最省力的局部最优，
       实测 2000 轮的"贴地蹭"稳态就是它的产物。
       阈值 20 N：躯干自重约 200 N，20 N 只相当于轻擦，不会误杀正常步态
       （正常爬楼躯干离踏面 0.5 m 以上，根本不接触）。

    ② 扰动从 (1,3) s / ±0.3 降到 (3,6) s / ±0.2。
       爬台阶的容错本来就比平地小一个量级：每 1~3 s 来一次 0.3 m/s +
       0.3 rad/s 的推力，等于整回合都在做"被撞后的恢复"，几乎采不到
       一个完整的台阶步 —— 早期只会学到"叉开腿硬扛"。
       先让它学会走，扰动后续再按回合长度加回来。
    """
    cfg.terminations["torso_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params=_params_for(mdp.illegal_contact, {
            "sensor_name": "torso_ground_contact", "force_threshold": 20.0,
        }),
    )

    push = cfg.events.get("push_robot")
    if push is not None:
        push.interval_range_s = (3.0, 6.0)
        if isinstance(getattr(push, "params", None), dict):
            push.params["velocity_range"] = {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.1, 0.1),
            }


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


def _alpha_start() -> float:
    """α 课程的起点（只在 alpha_fixed<0 的课程模式下生效）。

    为什么需要它：续训一个 α 已经涨到 0.70 的 checkpoint 时，
    · 不设任何环境变量 -> ramp 从 0 重来，参考打回站姿，策略当场崩；
    · 设 `ELF3_ALPHA=0.70` -> 那是「固定」语义，永远到不了 1.0（实测 4049 轮
      全部锁在 0.70，步幅只有脚本步态的七成，爬 22 cm 台阶不够用）。
    正确做法是 `ELF3_ALPHA_START=0.70`：从 0.70 继续线性升到 1.0。
    """
    raw = os.environ.get("ELF3_ALPHA_START")
    if raw is None or raw.strip() == "":
        return 0.0
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        print(f"[cfg][warn] ELF3_ALPHA_START={raw!r} 不是数字，按 0 处理")
        return 0.0


def _ramp_steps(play: bool) -> int:
    """α 课程总步数。可用 `ELF3_RAMP_ITERS`（单位 iteration，内部 ×24）覆盖。"""
    default = 1500 * 24
    if play:
        return default
    raw = os.environ.get("ELF3_RAMP_ITERS")
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(1, int(float(raw))) * 24
    except ValueError:
        print(f"[cfg][warn] ELF3_RAMP_ITERS={raw!r} 不是数字，用默认 1500")
        return default


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
        # 0.40 -> 0.55（2026-10-30 晚）：实测残差饱和比例 62%、mean|action| 1.953，
        # 而腿残差限幅换算下来约 2.0 action 单位 —— 策略想动但被 clamp 卡死，
        # 限幅外梯度恒为 0，再想加力也没用。放宽到 0.55（≈2.75 单位）解除天花板。
        # 代价：实际动作幅度上限 +37%，单腿支撑时更容易失衡，ep_len 短期会掉。
        leg_clip_rad=0.55,
        waist_clip_rad=0.20,
        arm_clip_rad=0.20,
        scan_size=SCAN_SIZE,
        scan_resolution=0.1,
        terrain_refresh_steps=5,
        # 相位速率夹 [0.6, 1.8]：上界 1.8 = "指令 / 参考步速" 的极限。
        # 与 _retune_commands 的指令上界配套（可达最大速度 = tread × 1.8/1.8）。
        v_rate_range=(0.6, 1.8),
        phase_offset_choices=(0.0, 0.5),
        # 相位平移：phase=0 -> 表帧 33（双支撑起点），phase=0.5 -> 帧 78，
        # 避免 episode 一开头就要求单腿支撑。
        phase_shift=33.0 / 90.0,
        # α=0 时参考退化为默认站姿 —— 与基线任务「动作全零」等价，
        # 早期训练难度和已验证的基线一致，再随课程渐变为脚本步态。
        ref_stand=tuple(DEFAULT_JOINT_POS),
        ramp_steps=_ramp_steps(play),   # 默认 1500 iter，ELF3_RAMP_ITERS 可覆盖
        alpha_start=_alpha_start(),      # 课程起点，ELF3_ALPHA_START 可覆盖
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
    _retune_safety(cfg)
    if play:
        _retune_viewer(cfg)
    return cfg
