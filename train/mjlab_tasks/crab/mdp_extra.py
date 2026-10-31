"""残差任务专用的观测项与奖励项。

所有函数签名都是 mjlab 的 `func(env, **params)`。
读取 entity data 的字段名做了兼容（mjlab 版本间 root_link_* 与 root_* 有差异），
取不到时退化为零，避免整条训练因为一个字段名挂掉。
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------- 工具
def _term(env, action_name: str = "joint_pos"):
    return env.action_manager.get_term(action_name)


def _data(env):
    return env.scene["robot"].data


def _root_pos(env) -> torch.Tensor:
    d = _data(env)
    for name in ("root_link_pos_w", "root_pos_w"):
        v = getattr(d, name, None)
        if v is not None:
            return v
    raise AttributeError("entity.data 里找不到 root pos（root_link_pos_w / root_pos_w）")


def _root_quat(env) -> torch.Tensor:
    d = _data(env)
    for name in ("root_link_quat_w", "root_quat_w"):
        v = getattr(d, name, None)
        if v is not None:
            return v
    raise AttributeError("entity.data 里找不到 root quat")


def _zeros(env, n: int) -> torch.Tensor:
    return torch.zeros(env.num_envs, n, device=env.device)


# ---------------------------------------------------------------- 观测
def joint_pos_ref_err(env, action_name: str = "joint_pos") -> torch.Tensor:
    """12 条腿相对参考轨迹的偏差 [B,12]（按 target 顺序对齐，避开实体排序差异）。"""
    t = _term(env, action_name)
    ids = t.leg_pos
    q = _data(env).joint_pos[:, t.target_joint_ids]
    return q[:, ids] - t.ref_joint_pos[:, ids]


def phase_sincos(env, action_name: str = "joint_pos") -> torch.Tensor:
    return _term(env, action_name).phase_sincos


def ref_meta(env, action_name: str = "joint_pos") -> torch.Tensor:
    """当前估计的 (rise, tread, v_ref)，做量纲归一化后给策略（都在 O(1) 量级）。"""
    t = _term(env, action_name)
    return torch.stack([t.rise_est * 5.0, t.tread_est * 2.5, t.ref_v * 5.0], dim=-1)


def ref_joint_pos(env, action_name: str = "joint_pos") -> torch.Tensor:
    """完整 29 维参考（只给 critic）。"""
    return _term(env, action_name).ref_joint_pos


# ---------------------------------------------------------------- 奖励
def forward_progress_y(env, command_name: str = "twist",
                       tau: float = 1.8, v_floor: float = 0.05) -> torch.Tensor:
    """世界 +Y 的**一个步态周期平均**推进速度，按指令归一化后的线性项，夹到 [0, 1]。

    为什么是 EMA + 线性，而不是瞬时速度 + exp 核
    -------------------------------------------
    `crab_ref_v1.npz`（rise=0.18, tread=0.30）的 root_y 是**脉冲式** COM 轨迹：
    停下 -> 1.6 m/s 猛冲 -> 倒退制动。FK 实测 v_y 均值 0.167、峰值 1.618、
    26% 的帧 v<-0.05、std 0.519。瞬时误差的 exp 核（std 0.055~0.15）在这种
    剖面下只会奖励"匀速蹭走"、重罚"真的迈步"，方向与任务目标完全相反：
    实测跟参考每步只值 0.02~0.45，而 0.126 m/s 的稳态蹭走值 0.88。

    tau 必须取一个步态周期（1.8 s），不能取 0.4 s
    -----------------------------------------
    0.4 s 只覆盖 1/4 个周期，EMA 仍随"冲-停-退"起伏，负速度帧被 floor 成 0，
    等于只给 74% 的周期发钱：跟参考 0.59 < 蹭走 0.70 —— 这一项本身就在奖励蹭。
    tau=1.8 s 把整周期平均掉（标定，cmd=0.18）：
        跟参考 0.85 / 稳态蹭走 0.126 -> 0.70 / 站住 -> 0
    tau 再大（2.5~4 s）收益很小，但指令重采样（3~8 s）时滞后更明显。
    它区分不开"踩步"和"蹭"的程度有限 —— 真正的步态考核是 ref_tracking +
    air_time + 躯干接触终止；这一项只负责两件事：
        别停（0 分）、别超速（cap 1.0。参考平均速度就等于 v_ref，允许 1.5 倍
        就会让"匀速快蹭"凭超速压过真实步态）。

    为什么用位置差分而不是 body 系线速度
    ---------------------------------
    位置差分天然在世界系、不受偏航影响，且与「净位移」同义；body 系速度在
    偏航漂移后会和世界 +Y 错位。
    """
    y = _root_pos(env)[:, 1]
    prev_y = getattr(env, "_crab_prev_y", None)
    if prev_y is None or prev_y.shape != y.shape:
        env._crab_prev_y = y.clone()
        return torch.zeros_like(y)

    dt = float(getattr(env, "step_dt", 0.02))
    dy = y - prev_y
    env._crab_prev_y = y.clone()
    v = dy / max(dt, 1e-6)

    alpha = 1.0 - math.exp(-dt / max(float(tau), 1e-6))
    ema = getattr(env, "_crab_vy_ema", None)
    if ema is None or ema.shape != v.shape:
        ema = v.clone()
    ema = ema + alpha * (v - ema)

    try:
        v_cmd = env.command_manager.get_command(command_name)[:, 1].abs()
    except Exception:
        v_cmd = torch.full_like(v, 0.16)

    # cmd≈0 的「原地踏步」env（twist.rel_standing_envs=0.1）：参考步态仍在播
    # （相位速率下限 0.6），净位移应≈0。按 v_cmd 归一化会把 5 cm/s 的任何漂移
    # 都当成满分，所以这类 env 改成"越不动越高分"，与内置 exp 核在 0 指令下的
    # 行为一致。
    # ⚠️ 2026-10-30：still 必须打折。drive 的上限是 1.0，still 若也是 1.0，
    # 这批 standing env 站着不动就是 1.0×5.0 = 5.0/步，反而高于跟参考步态的
    # 0.85×5.0 = 4.25 —— 站着比走还赚。它们由 rel_standing_envs 随机指定、
    # 策略自己选不了，但满分会把 value baseline 整体抬高、稀释"走"的梯度。
    # 打五折后站着 0.5 = 2.5/步，明显低于认真走。
    active = v_cmd > float(v_floor)
    drive = torch.clamp(ema / torch.clamp(v_cmd, min=float(v_floor)), 0.0, 1.0)
    still = 0.5 * torch.clamp(1.0 - ema.abs() / float(v_floor), 0.0, 1.0)
    r = torch.where(active, drive, still)

    # episode 刚 reset 的那一步 dy 跨了重置（含随机落点），丢弃并重置 EMA，
    # 否则一个几百 m/s 的假速度会在 EMA 里残留好几步。
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        if bool(reset.any()):
            ema = torch.where(reset, torch.zeros_like(ema), ema)
            r = torch.where(reset, torch.zeros_like(r), r)

    # EMA 冷启动屏蔽段。
    # ⚠️ 第一版按「一个完整 tau」（1.8 s = 90 步）屏蔽，实测在 ep_len 只有
    # 2.23 s 的早期，80% 的回合都被屏蔽掉：给分段几乎满分（0.95），整回合
    # 均值却只有 0.185。更糟的是它和 ep_len 形成负反馈 —— 回合越短 -> 前进
    # 奖励越少 -> 越学不会走 -> 回合继续短。
    # 改成 tau/3（0.6 s = 30 步）：短回合占 27%、长回合占 5%。冷启动残差只到
    # 1-exp(-1/3)≈28%，这部分靠「这段时间既不罚也不给分」兜住；EMA 本身照常
    # 累积，出屏蔽段时已有 30 个样本。
    if hasattr(env, "episode_length_buf"):
        warm_steps = max(1, int(round(float(tau) / 3.0 / max(dt, 1e-6))))
        warm = env.episode_length_buf < warm_steps
        r = torch.where(warm, torch.zeros_like(r), r)

    env._crab_vy_ema = ema
    return r


def ref_tracking(env, action_name: str = "joint_pos",
                 sigma: float = 0.35) -> torch.Tensor:
    """12 腿关节跟踪参考的 exp 核（用 exp 而非 L2，避免退化成行为克隆）。"""
    t = _term(env, action_name)
    ids = t.leg_pos
    q = _data(env).joint_pos[:, t.target_joint_ids]
    err = ((q[:, ids] - t.ref_joint_pos[:, ids]) ** 2).mean(dim=-1)
    return torch.exp(-err / (2.0 * sigma * sigma))


def yaw_drift(env) -> torch.Tensor:
    """帧间偏航角速度的平方 (rad/s)²：转圈要走不动。

    ⚠️ 2026-10-30 换实现。原来是 `sin(yaw)**2`，它对 yaw 和 yaw+2π 取值相同 ——
    **原地绕一整圈回到原点就「无罪」**。实测（6000 轮 checkpoint，play 20 s）
    累计偏航 367°，而这条只报 sin²=0.018（等价于净偏航 7.5°），等于完全没罚。

    现在看「每步的 yaw 变化」：先用 atan2(sinΔ, cosΔ) 把帧间差 wrap 到 [-π, π]
    （否则 ±π 处会跳变），再除 dt 得 rad/s，最后取平方 —— 平方让小抖动几乎不罚、
    大转动重罚，比绝对值更不容易把正常步态的轻微摆动也一起压死。
    """
    try:
        q = _root_quat(env)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    except Exception:
        return torch.zeros(env.num_envs, device=env.device)

    prev = getattr(env, "_crab_prev_yaw", None)
    if prev is None or prev.shape != yaw.shape:
        env._crab_prev_yaw = yaw.clone()
        return torch.zeros_like(yaw)
    d = torch.atan2(torch.sin(yaw - prev), torch.cos(yaw - prev))
    env._crab_prev_yaw = yaw.clone()
    # reset 那一步跨了 episode，丢弃
    if hasattr(env, "episode_length_buf"):
        d = torch.where(env.episode_length_buf <= 1, torch.zeros_like(d), d)
    dt = float(getattr(env, "step_dt", 0.02))
    return (d / max(dt, 1e-6)) ** 2


def lateral_drift(env) -> torch.Tensor:
    """(x - x0)^2：世界 +X 方向不该有位移。"""
    x = _root_pos(env)[:, 0]
    x0 = getattr(env, "_crab_x0", None)
    if x0 is None or x0.shape != x.shape:
        env._crab_x0 = x.clone()
        return torch.zeros_like(x)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        if bool(reset.any()):
            env._crab_x0 = torch.where(reset, x, env._crab_x0)
    return (x - env._crab_x0) ** 2


def lateral_bias(env, tau: float = 1.8, deadzone: float = 0.03) -> torch.Tensor:
    """机身系纵向速度的**一个步态周期平均**，超过门限后平方 —— 只罚偏置，不罚迈步。

    为什么需要它（2026-10-31 实测）
    ------------------------------
    三个难度档位跑 16 env × 20 s，机身系纵向速度的整周期均值都是 **+0.10 m/s**
    （L3 +0.125 / L5 +0.098 / L7 +0.109），是侧向指令 0.17 的 60%，且**与地形无关**
    —— 这是步态自带的偏置，不是台阶几何造成的。逐帧 |v_lon| 有 0.18，但那是正常
    摆动（参考步态 COM 是脉冲式，v_y 峰值 1.618、26% 帧倒退），不能拿来罚。

    为什么不能用现有的 `lateral_drift`
    --------------------------------
    它是 `(x-x0)²`、权重 -0.2：0.2 m 偏移时每步只 -0.00016 分，整个回合 -0.003 分，
    量级太小等于没罚。而且它罚的是"已经漂出去的位移"，梯度在起点最弱；
    纵向偏置是**每步都在产生**的，直接在速度层罚更早、更线性。

    为什么用 EMA + 死区
    -------------------
    逐帧罚会重罚正常迈步，把策略逼回匀速蹭走（`track_linear_velocity` 的老毛病，
    见 forward_progress_y 的注释）。tau=1.8 s 正好一个步态周期，把"冲-停-退"平均掉，
    剩下的才是净偏置。deadzone=0.03 m/s 放过正常残余摆动。

    标定（以 weight=-20 计，实测取这个值）
    ------------------------------------
        |偏置| = 0.03 -> 0                        （死区内，不罚）
        |偏置| = 0.11 -> (0.08)²=0.0064 -> -0.13/步（约为 forward_progress_y 的 15%）

    ⚠️ 2026-10-31 教训：**别用重权重**。原来 weight=-80 时，535 轮下来偏置
    从 0.086 涨到 0.114（罚了没降），而这一项占了 forward_progress_y 的 62%：
    ref_tracking 0.976 -> 0.295、upright 0.32 -> 0.08、ep_len 7.4 -> 4.4 s。
    策略的选择是"少动来避罚"而不是"走直"。步态没稳之前，这一项只当提示。
    """
    pos = _root_pos(env)[:, :2]
    prev = getattr(env, "_crab_prev_xy", None)
    if prev is None or prev.shape != pos.shape:
        env._crab_prev_xy = pos.clone()
        return torch.zeros(env.num_envs, device=pos.device)

    dt = float(getattr(env, "step_dt", 0.02))
    v_w = (pos - prev) / max(dt, 1e-6)
    env._crab_prev_xy = pos.clone()

    try:
        q = _root_quat(env)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # 机身 z 轴的世界系分量：直立 1 / 侧倒 0 / 翻转 -1
        upright = (1.0 - 2.0 * (x ** 2 + y ** 2)) > 0.6
    except Exception:
        yaw = torch.zeros(env.num_envs, device=pos.device)
        upright = torch.ones(env.num_envs, dtype=torch.bool, device=pos.device)

    # 世界系 -> 机身系：机身 +X 方向 = (cos yaw, sin yaw)
    v_lon = v_w[:, 0] * torch.cos(yaw) + v_w[:, 1] * torch.sin(yaw)

    # 摔倒时机身系分解没有意义（yaw/pitch 都是任意的），而且"向前扑倒"会被算成巨大偏置。
    # 摔倒本身已经由 is_terminated 罚过，这里不要重复计。倾倒时输入置 0，让 EMA 自然衰减。
    v_lon = torch.where(upright, v_lon, torch.zeros_like(v_lon))

    a = 1.0 - math.exp(-dt / max(float(tau), 1e-6))
    ema = getattr(env, "_crab_vlon_ema", None)
    if ema is None or ema.shape != v_lon.shape:
        ema = v_lon.clone()
    ema = ema + a * (v_lon - ema)

    # reset 那一步位置会跳好几个米（换地形块），EMA 必须丢掉这个样本
    if hasattr(env, "episode_length_buf"):
        rst = env.episode_length_buf <= 1
        if bool(rst.any()):
            ema = torch.where(rst, torch.zeros_like(ema), ema)
    env._crab_vlon_ema = ema.clone()

    return torch.clamp(ema.abs() - float(deadzone), min=0.0) ** 2 * upright.to(ema.dtype)


def swing_clearance(env, sensor_name: str = "foot_height_scan",
                    contact_sensor: str = "feet_ground_contact",
                    target: float = 0.15, max_height: float = 0.5,
                    command_name: str = "twist",
                    command_threshold: float = 0.05) -> torch.Tensor:
    """摆动腿离地高度 / target，**平方**后按脚平均 —— 正奖励。

    为什么必须给正的
    --------------
    现有配置里抬脚只有惩罚没有收益：
        air_time        weight 1.0，但门槛 0.05 s —— 贴地蹭时压根不触发，等于 0；
        foot_swing_height weight -1.0，同样只在"已经在摆"的时候才罚；
        foot_clearance  weight -2.0，target 0.15，3 cm 时每步只 -0.03 分。
    也就是说"从不抬脚"这个状态在奖励函数里是**完全中性**的（惩罚项全都不触发），
    而抬脚一旦失衡就是 is_terminated 的 -5 分。实测于是收敛到"贴地蹭/顶在台阶边上"。
    这个不对称只能用**正的、从 0 开始就有梯度**的项来破。

    为什么是平方不是线性（2026-10-31 实测教训）
    ----------------------------------------
    第一版是线性的，1400 轮后 `peak_height_mean` 从 0.029 掉到 **0.0096**，而
    `air_time_mean` 从 0.070 涨到 0.140 —— 脚离地时间翻倍、高度反而更低，正是
    "贴着地面快速蹭"在刷这条奖励（接触传感器只要没受力就判腾空，1 cm 也算）。
    平方后：1 cm -> 0.004、5 cm -> 0.11、10 cm -> 0.44、15 cm -> 1.0，
    低空悬停几乎不给钱，梯度主要落在最后几厘米。

    为什么用脚离地高度，而不是关节角
    ------------------------------
    ref_tracking 罚的是关节角误差，而残差限幅 ±0.55 rad 足以把参考的抬腿整个抵消掉，
    关节角层面的"跟得上"不等于"脚真的抬起来了"。这里直接量脚底到地形的高度。

    为什么要 max_height 截断
    ----------------------
    射线没打到地形时传感器会返回 max_distance(1.0) 或 0。不截断的话"没打到"会被当成
    "抬到 1 m 高"，直接把这条奖励刷满 —— 又是一个静默的 reward hacking。

    为什么要竖直姿态门控（2026-10-31 实测）
    ------------------------------------
    只门控"不接触 + 有指令"还不够：**摔倒过程中四肢甩在空中也满足这两条**，而且高度能
    顶到 max_height，lift=1.0 每步白拿全额。实测那次 600 轮续训后 ep_len 掉到 1.54 s、
    跟踪比 5.63、mean|action| 3.14 且 69% 顶死限幅 —— 策略学会了"扑腾"。
    现在要求机身 z 轴在世界系的分量 > 0.6（倾斜不超过 53°）才付钱。
    """
    from mjlab.tasks.velocity import mdp as _m

    try:
        h = _m.foot_height(env, sensor_name=sensor_name).to(dtype=torch.float32)
        c = _m.foot_contact(env, sensor_name=contact_sensor)
    except Exception:
        return torch.zeros(env.num_envs, device=_root_pos(env).device)

    if c.dtype == torch.bool:
        swing = (~c).to(h.dtype)
    else:
        swing = (1.0 - c.to(h.dtype)).clamp(0.0, 1.0)

    valid = ((h > 0.0) & (h < float(max_height))).to(h.dtype)
    # 平方而不是线性：线性时"脚只离地 1 cm"也能拿 0.067，而接触传感器只要没受力就判
    # 为腾空 —— 于是策略学会了"贴着地面快速蹭"（实测 air_time 从 0.070 涨到 0.140，
    # 而 peak_height 从 0.029 掉到 0.0096，就是刷这条奖励的指纹）。
    # 平方后：1 cm -> 0.004、5 cm -> 0.11、10 cm -> 0.44、15 cm -> 1.0。
    lift = torch.clamp(h / max(float(target), 1e-6), 0.0, 1.0) ** 2

    try:
        v_cmd = env.command_manager.get_command(command_name)[:, 1].abs()
        gate = (v_cmd > float(command_threshold)).to(h.dtype).unsqueeze(-1)
    except Exception:
        gate = torch.ones_like(h)

    # ⚠️ 竖直姿态门控（2026-10-31，实测踩到）：不加这道，**摔倒比走路更赚钱**。
    # 机器人翻倒时四肢甩在空中、脚离地高度能顶到 max_height，lift=1.0，每步白拿
    # weight 全额；而走路时抬 3 cm 只有 0.04。实测后果：ep_len 1.54 s、跟踪比 5.63、
    # mean|action| 3.14 且 69% 顶死限幅 —— 策略学会了"扑腾"。
    # 用机身 z 轴在世界系的分量判直立：q=(w,x,y,z) 时 up_z = 1 - 2(x²+y²)，
    # 直立为 1、侧倒为 0、翻转 -1。阈值 0.6 约等于"倾斜不超过 53°"。
    try:
        q = _root_quat(env)
        up_z = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        gate = gate * (up_z > 0.6).to(h.dtype).unsqueeze(-1)
    except Exception:
        pass

    return (lift * swing * valid * gate).mean(dim=-1)


def is_terminated(env) -> torch.Tensor:
    """终止标志（一次性惩罚用）。字段名随版本不同，取不到就返回 0。"""
    for name in ("reset_terminated", "reset_buf", "terminated"):
        v = getattr(env, name, None)
        if v is not None:
            return v.to(torch.float32)
    return torch.zeros(env.num_envs, device=env.device)


def ref_probe(env, action_name: str = "joint_pos") -> torch.Tensor:
    """把参考步态/地形估计的内部量写进 `Metrics/*`，本身不参与学习。

    为什么需要它
    ------------
    `rise_est` 是「height_scan 到底有没有读到台阶」的唯一证据 —— 它恒为 0 时
    参考步态锁死在平地档，训练再久也是平地策略（2026-10-29 就这么白训了 12000 轮）。
    没有这一路指标就只能靠猜。

    为什么权重是 1e-9 而不是 0
    -------------------------
    mjlab 的 RewardManager 对 `weight == 0.0` 的项是 `continue` —— **连 func 都不调用**，
    所以写不进任何日志。用 1e-9：函数照跑、对 reward 的污染量级 1e-9 × 0.02，可忽略。
    """
    dev = _root_pos(env).device
    log = getattr(env, "extras", None)
    log = log.get("log") if isinstance(log, dict) else None
    try:
        term = _term(env, action_name)
    except Exception:
        return torch.zeros(env.num_envs, device=dev)

    def _put(key, t):
        if log is not None:
            log[key] = torch.as_tensor(t, device=dev).float().mean()

    try:
        _put("Metrics/scan_ok", torch.tensor(1.0 if term.scan_ok else 0.0, device=dev))
        _put("Metrics/rise_est", term.rise_est)
        _put("Metrics/tread_est", term.tread_est)
        if term.rise_valid is not None:
            _put("Metrics/rise_valid_frac", term.rise_valid.to(torch.float32))
        if term.raw_rise is not None:
            _put("Metrics/rise_raw", term.raw_rise)
        _put("Metrics/ref_alpha", torch.tensor(float(term.ref_alpha), device=dev))
    except Exception:
        pass
    return torch.zeros(env.num_envs, device=dev)


def terrain_levels_time(env, env_ids, promote_frac: float = 0.25,
                        demote_frac: float = 0.40, min_dist: float = 0.90,
                        **kwargs) -> dict:
    """按「活够久 + 真的挪了」升降地形，取代内置的「走够远」。

    为什么必须换掉内置 terrain_levels_vel
    ------------------------------------
    内置的判据是：

        time_left = max_episode_length_s - ep_len
        commanded_distance = |cmd| * time_left + 已走距离   # 约等于 |cmd| * 20 s
        move_up   = distance > 0.8 * commanded_distance
        move_down = distance < 0.5 * commanded_distance

    那段 `time_left` 的设计意图是「早摔不该减轻要求」，但后果是
    **摔得越早、要求反而越高**。实测 200 轮 ep_len = 4.2 s：

        要求走 0.8 × 3.8 m = 3.04 m，实际只走 0.42 m（|v_y| ≈ 0.1 m/s）

    于是 `move_up` 永远 False、`move_down` 永远 True -> 全部 env 钉死在
    level 0（flat，台阶 0~2 cm），策略一辈子见不到真台阶，只能学会在平地
    上踉跄。2026-10-29 之前那 12000 轮无效训练是同一个机理。

    现在的判据
    ---------
        升级：活到 max_episode_length 的 25%（5 s）**且**离开出生点 >= 0.90 m
        降级：活不到 max_episode_length 的 40%（8 s）
        其余：保持不变

    ⚠️ demote_frac 为什么从 0.10 提到 0.40（2026-10-31）
    ------------------------------------------------
    0.10 对应 2.0 s，而略差的策略很自然会稳定在 2~4 s —— **正好落在
    [2 s, 5 s] 这条死带里**：活过了降级线所以不降级，又够不到升级线所以不升级。
    课程于是被冻在"进得去、撑不住"的地形上，没有任何机制把它拉回来。
    实测 1400 轮后 `Mean episode length = 125.88 步 = 2.52 s`、`rise_est = 0.15`
    （地形已经在 15 cm 级）—— 就是这个死锁。
    0.40 对应 8 s：撑不到 8 s 就降一级，让策略自己退回到能站住的地形上重新学。

    ⚠️ min_dist 为什么从 0.30 提到 0.90（2026-10-31）
    ------------------------------------------------
    倒置金字塔的碗底平台半宽是 0.52~0.80 m（flat 0.60 / easy 0.80 / mid 0.52 /
    hard 0.62）。原来的 0.30 意味着**在平台上原地蹭 0.3 m 就能升级，一步都不用爬**。
    实测后果：`Curriculum/terrain_levels/mean` 一路涨到 5.8，但 eval 时仍有一批 env
    卡在第一级台阶边缘（Δy 全部 ≈0.83 m）动不了 —— 它从来没被要求"爬上去"，
    只被要求"挪一挪"。
    0.90 在四列上都落在平台边缘之外（flat 0.60 / easy 0.80 / mid 0.52 / hard 0.62），
    即"至少要踏上第一级台阶"才够格升级。用水平距离而不是竖直位移，是因为
    `env_origins` 的 z 含义随版本不同（可能是表面高度也可能恒 0），判不准。
    若升级停住不再上涨，说明策略连第一级都上不去 —— 那是抬脚的问题，去调
    `swing_clearance`，不要把这个门槛再调回去。

    升降不对称（25% vs 10%），避免偶尔摔一次就被打回 level 0。
    `min_dist` 那一条专门防「原地罚站也能升级」——站着不动拿不到难地形。

    ⚠️ 副作用：这条判据只保证「站得住」，不保证「走得远」。推进靠
    `forward_progress_y` 奖励项拉动；如果训练后 v_y 一直上不去，说明
    该加重它，而不是把这条课程判据改回距离。
    """
    terrain = getattr(getattr(env, "scene", None), "terrain", None)
    if terrain is None or not hasattr(terrain, "update_env_origins"):
        return {}

    step_dt = float(getattr(env, "step_dt", 0.02))
    t_max = float(getattr(env, "max_episode_length_s", 20.0))

    # 注意：本函数在 `_reset_idx` 里被调用，此时 episode_length_buf 还是
    # 「刚结束那一回合的长度」，尚未清零 —— 内置实现也是这么用的。
    t_alive = env.episode_length_buf[env_ids].to(torch.float32) * step_dt

    # 离开出生点的水平距离（env_origins 此刻还是本回合的出生点）
    try:
        pos = _root_pos(env)[env_ids, :2]
        origins = getattr(env.scene, "env_origins", None)
        if origins is not None:
            dist = torch.norm(pos - origins[env_ids, :2].to(pos.dtype), dim=1)
        else:
            dist = torch.zeros_like(t_alive)
    except Exception:
        dist = torch.zeros_like(t_alive)

    move_up = (t_alive >= t_max * float(promote_frac)) & (dist >= float(min_dist))
    move_down = (~move_up) & (t_alive < t_max * float(demote_frac))

    # 首次 reset（还没走过一步）时 distance 无意义，冻结，否则会把所有 env 平白升级
    if int(getattr(env, "common_step_counter", 1) or 0) == 0:
        move_up = torch.zeros_like(move_up)
        move_down = torch.zeros_like(move_down)

    try:
        terrain.update_env_origins(env_ids, move_up, move_down)
    except TypeError as e:
        print(f"[curriculum][ERR] update_env_origins 调用失败（版本差异？）：{e}")
        return {}

    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}
