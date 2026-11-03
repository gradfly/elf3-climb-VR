#!/usr/bin/env python
"""标准化评测：给定一个或多个 checkpoint，在固定条件下跑 episode 并给出可比较的分数卡。

为什么需要它
------------
`diagnose_policy.py` 是"现场排障"（打印单次运行的原始量和分块诊断），
本脚本是"验收/回归"：每次用同一套协议（无观测噪声、固定指令、固定地形等级、
固定时长、固定 episode 数）评估，输出 JSON，便于回答两个问题：

  1) 这一版策略到底比上一版好还是坏？
  2) 它失败在哪一档地形、哪一种能力（生存 / 推进 / 朝向 / 打滑 / 步态跟踪）？

协议（默认值都可通过参数改）
----------------------------
  * play 配置（无观测噪声、无 push），可显式 --push 打开扰动版；
  * episode 固定 20 s（覆盖 play 自带的 1e9 秒"永不超时"）；
  * 指令冻结为 (0, cmd_y, 0)，不重采样、不抽 standing env；
  * 地形等级逐档跑（默认 0/2/4/6/8），每档所有 env 同级；
  * 每个 env 至少跑 --episodes 个完整回合。

主要指标
--------
  fall_rate       摔倒率（非超时终止）
  reach_rate      未摔且净 Δy >= 0.7 × cmd × T（真的在推进）
  climb_rate      未摔且净 Δy >= --success-dy 且 Δz >= --success-dz（真的上了台阶）
  track_ratio     净 Δy / (cmd × 回合时长)
  yaw_rate        存活期 |ω_z| 均值（螃蟹步应 < 0.15 rad/s）
  vlon_bias       机身系纵向速度偏置（螃蟹步应≈0）
  sat_frac        动作维度顶在残差限幅上的比例
  support_proxy   根节点投影落在支撑中心附近的帧占比（近似值，见下）
  clearance       摆动脚平均离地高度 / 峰值（看是不是贴地蹭）
  slip            [1m着地脚[0m的水平滑动速度（看是不是原地踩）

⚠️ support_proxy 是近似量
-------------------------
mjlab 的接触传感器只给 found/force，不给我们接触点；这里用两条脚 site 的
xy 位置做代理：双脚着地时以两脚中点为圆心、半径 0.18 m；单脚着地时以该脚为
圆心、半径 0.13 m。它衡量的是"重心是否大致落在支撑基底内"，不是几何精确的
支撑多边形，因此只作为诊断项，不参与 PASS/FAIL。

用法（服务器上，仓库根目录）
---------------------------
    # 0) 只做 checkpoint 静态体检（不需要 mjlab / GPU，可在本地跑）
    python scripts/eval_policy.py --checkpoint model_7047.pt --audit-only

    # 1) 单模型验收（5 档地形 × 固定 0.18 m/s × 20 s × 2 回合 × 32 env）
    MUJOCO_GL=egl .venv/bin/python scripts/eval_policy.py \
        --checkpoint logs/rsl_rl/elf3_crab/<run>/model_7047.pt \
        --num-envs 32 --levels 0,2,4,6,8 --cmds 0.18

    # 2) 新旧对比：同一套初始条件，先跑旧模型再跑新模型
    MUJOCO_GL=egl .venv/bin/python scripts/eval_policy.py \
        --checkpoint old.pt new.pt --num-envs 32 --levels 0,2,4,6,8

    # 3) 抗扰版（play 默认没有 push，这里补回来）
    MUJOCO_GL=egl .venv/bin/python scripts/eval_policy.py \
        --checkpoint ckpt.pt --push --push-scale 0.5

输出
----
  * 终端：逐条件表 + 每个 checkpoint 的总体等级与失败项；
  * JSON：--out 指定（默认 logs/eval/<第一个 checkpoint 名>.json），含
    静态体检、每个 condition 的汇总、以及每条 episode 的原始记录。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from dataclasses import asdict

import torch

EXPECTED_OBS = {"residual": 393, "base": 290}
EXPECTED_ACT = 29

# support_proxy 的半径（见文件头说明）。改动会改变口径，请不要在不同 run 之间混用。
SUPPORT_R_DOUBLE = 0.18
SUPPORT_R_SINGLE = 0.13


# ================================================================ 静态体检
def audit_checkpoint(path: str) -> dict:
    """不依赖 mjlab，读出 checkpoint 的关键元信息，并做可加载性/截断检查。

    训练被 kill / Ctrl-C 时，正在写盘的 checkpoint 会是空文件或截断文件，
    torch.load 报的是 `EOFError: Ran out of input`，看不出是文件坏了。
    这里先按大小筛一遍，并给出同目录最近的完好文件。
    """
    info: dict = {"path": path, "ok": False, "error": None}
    if not os.path.exists(path):
        info["error"] = "文件不存在"
        return info
    size = os.path.getsize(path)
    info["size_bytes"] = size
    if size < 4096:
        info["error"] = f"只有 {size} 字节，疑似训练中断时写坏的截断文件"
        d = os.path.dirname(os.path.abspath(path))
        good = [c for c in glob.glob(os.path.join(d, "model_*.pt"))
                if os.path.getsize(c) > 4096]
        info["siblings"] = sorted(good, key=os.path.getmtime, reverse=True)[:3]
        return info

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        info["error"] = f"torch.load 失败：{e!r}"
        return info

    actor = ck.get("actor_state_dict") if isinstance(ck, dict) else None
    if not isinstance(actor, dict):
        actor = ck if isinstance(ck, dict) else {}
    info["iter"] = ck.get("iter") if isinstance(ck, dict) else None
    info["top_keys"] = sorted(ck.keys()) if isinstance(ck, dict) else []

    obs_dim = action_dim = None
    hidden = []
    norm = {}
    for k, v in actor.items():
        if not hasattr(v, "shape"):
            continue
        key = k.lower()
        if "obs_normalizer" in key and any(s in key for s in ("_mean", "mean")):
            try:
                obs_dim = int(v.reshape(-1).numel())
                norm["mean_abs"] = float(v.float().abs().mean())
                norm["mean_min"] = float(v.float().min())
                norm["mean_max"] = float(v.float().max())
            except Exception:
                pass
        if "obs_normalizer" in key and any(s in key for s in ("_var", "var", "std")):
            try:
                norm["var_min"] = float(v.float().min())
                norm["var_max"] = float(v.float().max())
            except Exception:
                pass
        if "obs_normalizer" in key and "count" in key:
            try:
                norm["count"] = float(v.float().reshape(-1)[0])
            except Exception:
                pass
        if ".weight" in key and len(v.shape) == 2:
            hidden.append(int(v.shape[0]))
            if obs_dim is None:
                obs_dim = int(v.shape[1])
    if hidden:
        action_dim = hidden[-1]
        hidden = hidden[:-1]

    info.update(obs_dim=obs_dim, action_dim=action_dim, hidden=hidden, norm=norm,
                ok=True)
    return info


def print_audit(a: dict) -> None:
    print(f"[audit] {a['path']}")
    if not a["ok"]:
        print(f"[audit]   FAIL  {a['error']}")
        for c in a.get("siblings", []):
            print(f"[audit]   同目录可选：{c} ({os.path.getsize(c) // 1024} KB)")
        return
    print(f"[audit]   iter={a['iter']}  obs={a['obs_dim']}  act={a['action_dim']}  "
          f"hidden={a['hidden']}  size={a['size_bytes'] // 1024} KB")
    n = a.get("norm", {})
    print(f"[audit]   obs_normalizer: count={n.get('count', 'n/a')}  "
          f"|mean|={n.get('mean_abs', float('nan')):.3f}  "
          f"var=[{n.get('var_min', float('nan')):.3f}, {n.get('var_max', float('nan')):.3f}]")
    if a["obs_dim"] != EXPECTED_OBS["residual"]:
        print(f"[audit]   WARN obs 维度不是残差任务的 {EXPECTED_OBS['residual']}"
              f"（{a['obs_dim']}），确认 --task 是否匹配")
    if a["action_dim"] != EXPECTED_ACT:
        print(f"[audit]   WARN 动作维度不是 {EXPECTED_ACT}")
    if n.get("count", 0) and n.get("count", 0) < 1e5:
        print("[audit]   WARN 归一化样本数偏少，mean/var 可能还没收敛")


# ================================================================ mjlab 取数
def _unwrap(env):
    seen = 0
    while hasattr(env, "unwrapped") and seen < 5:
        inner = env.unwrapped
        if inner is env:
            break
        env = inner
        seen += 1
    return env


def _root_pos(env):
    d = env.scene["robot"].data
    for name in ("root_link_pos_w", "root_pos_w"):
        v = getattr(d, name, None)
        if v is not None:
            return v
    raise AttributeError("找不到 root pos")


def _root_quat(env):
    d = env.scene["robot"].data
    for name in ("root_link_quat_w", "root_quat_w", "root_link_quat"):
        v = getattr(d, name, None)
        if v is not None:
            return v
    return None


def _root_yaw(env):
    q = _root_quat(env)
    if q is None:
        return None
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _root_yaw_rate(env):
    d = env.scene["robot"].data
    for name in ("root_link_ang_vel_w", "root_ang_vel_w"):
        v = getattr(d, name, None)
        if v is not None:
            return v[:, 2]
    return None


def _pin_command(base, vec, name: str = "twist") -> bool:
    """把指令张量就地写死；返回是否成功。"""
    try:
        term = base.command_manager.get_term(name)
    except Exception:
        return False
    for attr in ("command", "_command", "vel_command", "cmd"):
        t = getattr(term, attr, None)
        if torch.is_tensor(t) and t.dim() == 2 and t.shape[-1] >= vec.numel():
            try:
                t[:, :vec.numel()] = vec
                return True
            except Exception:
                continue
    return False


def _disable_curriculum(base) -> None:
    cm = getattr(base, "curriculum_manager", None)
    if cm is None:
        return
    if getattr(cm, "_diag_disabled", False):
        return
    try:
        cm.compute = lambda *a, **k: {}
        cm._diag_disabled = True
    except Exception:
        pass


def _force_terrain_level(base, level: int) -> int | None:
    """把所有 env 钉到指定行，并同步 env_origins（否则 reset 还在旧出生点）。

    play 配置每次 reset 会调用 randomize_terrain，把 level 重新随机化；评测前
    已在 main 里移除了这个 event，所以这里设置的值会一直保持。
    等级超过地形行数时自动夹到最高行，并返回实际生效的等级。
    """
    terr = getattr(getattr(base, "scene", None), "terrain", None)
    if terr is None:
        return None
    origins = getattr(terr, "terrain_origins", None)
    if origins is not None and origins.dim() >= 2:
        level = min(int(level), int(origins.shape[0]) - 1)
    v = getattr(terr, "terrain_levels", None)
    if v is None or not torch.is_tensor(v):
        return None
    v[:] = int(level)
    if origins is not None:
        types = getattr(terr, "terrain_types", None)
        if types is not None and torch.is_tensor(types):
            terr.env_origins[:] = origins[int(level), types]
    return int(level)


def _foot_site_ids(base):
    """返回 l_foot / r_foot 在 site 数组里的下标；拿不到就 None。"""
    ent = base.scene["robot"]
    names = None
    holders = (ent, getattr(ent, "indexing", None), getattr(ent, "data", None))
    for holder in holders:
        if holder is None:
            continue
        for attr in ("site_names", "_site_names"):
            v = getattr(holder, attr, None)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                names = list(v)
                break
        if names:
            break
    if not names:
        return None
    try:
        return names.index("l_foot"), names.index("r_foot")
    except ValueError:
        return None


def _site_tensor(base, attrs):
    d = base.scene["robot"].data
    for attr in attrs:
        v = getattr(d, attr, None)
        if torch.is_tensor(v) and v.dim() == 3:
            return v
    return None


def _foot_sites(base, ids):
    if ids is None:
        return None
    v = _site_tensor(base, ("site_pos_w", "site_link_pos_w", "site_xpos", "site_pos"))
    if v is None:
        return None
    l, r = ids
    return torch.stack([v[:, l, :3], v[:, r, :3]], dim=1)


def _foot_vels(base, ids):
    if ids is None:
        return None
    v = _site_tensor(base, ("site_vel_w", "site_link_vel_w", "site_lin_vel_w", "site_vel"))
    if v is None:
        return None
    l, r = ids
    return torch.stack([v[:, l, :3], v[:, r, :3]], dim=1)


def _term_metrics(term, data):
    """参考跟踪误差与腿关节 RMS（与 ref_tracking 奖励同口径）。"""
    try:
        ids = term.leg_pos
        q = data.joint_pos[:, term.target_joint_ids]
        err = q[:, ids] - term.ref_joint_pos[:, ids]
        mse = (err ** 2).mean(dim=-1)
        track = torch.exp(-mse / (2.0 * 0.35 * 0.35))
        return track, torch.sqrt(mse)
    except Exception:
        return None, None


# ================================================================ 单条件评测
def _episode_summary(acc: dict, *, dt: float, cmd: float, level: int,
                     fall: bool, timeout: bool, seconds: float,
                     success_dy: float, success_dz: float) -> dict:
    steps = max(float(acc["steps"]), 1.0)
    ep_sec = steps * dt
    dy = float(acc["dy"])
    dx = float(acc["dx"])
    dz = float(acc["dz"])

    def _avg(key):
        return float(acc[key]) / steps

    def _ratio(num_key, cnt_key):
        den = float(acc[cnt_key])
        return float(acc[num_key]) / den if den > 0 else float("nan")

    return dict(
        level=int(level), cmd=float(cmd), ep_steps=int(steps), ep_sec=ep_sec,
        fall=bool(fall), timeout=bool(timeout),
        dy=dy, dx=dx, dz=dz,
        vy_mean=_avg("vy"), vx_mean=_avg("vx"), vlon_bias=_avg("vlon"),
        yaw_rate=_avg("wz_abs"),
        yaw_travel_deg=float(acc["yaw_travel"]) * 180.0 / math.pi,
        track_ratio=dy / max(cmd * ep_sec, 1e-6),
        reach=bool((not fall) and dy >= 0.7 * cmd * seconds),
        climb=bool((not fall) and dy >= success_dy and dz >= success_dz),
        support_proxy=_avg("support"),
        up_z=_avg("upz"),
        clearance_mean=_ratio("clearance", "clearance_cnt"),
        clearance_peak=float(acc["peak"]),
        swing_frac=_ratio("swing", "foot_cnt"),
        slip=_ratio("slip", "contact_cnt"),
        ref_track=_avg("ref_track"),
        ref_rms=_avg("ref_rms"),
        act_abs=_avg("act_abs"),
        sat_frac=_avg("sat"),
        sat_leg_frac=_avg("sat_leg"),
        sat_oth_frac=_avg("sat_oth"),
        rise_est=_avg("rise"),
    )


def evaluate_conditions(env, base, policy, *, levels, cmds, episodes, seconds,
                        seed, success_dy, success_dz) -> list[dict]:
    """每个 (level, cmd) 跑固定回合数，返回 episode 记录列表。"""
    from mjlab.tasks.velocity import mdp as vm

    B = int(base.num_envs)
    dt = float(getattr(base, "step_dt", 0.02))
    max_ep = int(getattr(base, "max_episode_length", 0) or 0)
    if max_ep <= 0:
        max_ep = int(round(float(seconds) / dt))

    try:
        term = base.action_manager.get_term("joint_pos")
    except Exception:
        term = None
    site_ids = _foot_site_ids(base)

    # 可选指标的可用性只探测一次，避免每一步都走 try/except 的慢路径。
    try:
        vm.foot_height(base, sensor_name="foot_height_scan")
        vm.foot_contact(base, sensor_name="feet_ground_contact")
        foot_ok = True
    except Exception:
        foot_ok = False
        print("[warn] 足底传感器不可用，clearance / slip / support 会缺失")

    clip_action = getattr(term, "_clip_action", None) if term is not None else None

    # 腿 12 维的下标：饱和率要按「腿 / 非腿」拆开看。
    # 为什么：29 维里 14 维是手臂，clip 只有 0.20 rad，而动作 scale ≈0.21 -> action 空间
    # 上限约 0.65~0.97，策略 std 常年 0.87，于是手臂维度天然大面积饱和。
    # 合成一个 sat_frac 会让「腿顶死」（严重）和「手臂顶死」（基本无害）分不开。
    leg_ids = None
    if term is not None:
        # ⚠️ 必须用 leg_pos（12 条腿在 **target/动作** 顺序里的下标），
        # 不能用 leg_joint_ids（那是关节名顺序的下标）。两者不一定一致，
        # 用错了会把「手臂的饱和」算到腿头上。
        for _attr in ("leg_pos", "leg_joint_ids"):
            try:
                _ids = getattr(term, _attr, None)
                _ids = list(_ids() if callable(_ids) else (_ids or []))
                if _ids:
                    leg_ids = torch.tensor([int(i) for i in _ids],
                                           dtype=torch.long, device=base.device)
                    break
            except Exception:
                continue
    ref_ok = False
    records: list[dict] = []
    cond_idx = 0

    for level in levels:
        _disable_curriculum(base)
        actual_level = _force_terrain_level(base, level)
        if actual_level is None:
            actual_level = level
        if actual_level != level:
            print(f"[warn] 地形等级 {level} 超出范围，已夹到 {actual_level}")

        for cmd in cmds:
            frozen = torch.tensor([0.0, float(cmd), 0.0], device=base.device)
            torch.manual_seed(int(seed) * 1000 + cond_idx)
            cond_idx += 1

            out = env.reset()
            obs = out[0] if isinstance(out, tuple) else out
            if not _pin_command(base, frozen):
                print("[warn] 无法写死指令（command term 结构变了）；cmd 档位将不可信")
            obs = env.get_observations()   # reset 后的 obs 还是旧指令，重算一次

            acc = {k: torch.zeros(B, device=base.device, dtype=torch.float32)
                   for k in ("steps", "dy", "dx", "dz", "yaw_travel", "wz_abs",
                             "vy", "vx", "vlon", "support", "upz",
                             "clearance", "clearance_cnt", "peak", "swing", "foot_cnt",
                             "slip", "contact_cnt", "ref_track", "ref_rms",
                             "act_abs", "sat", "sat_leg", "sat_oth", "rise")}
            ep_steps = torch.zeros(B, dtype=torch.long, device=base.device)

            p = _root_pos(base)
            prev_xy = p[:, :2].clone()
            prev_z = p[:, 2].clone()
            yw = _root_yaw(base)
            prev_yaw = yw.clone() if yw is not None else None

            for _ in range(int(episodes) * max_ep):
                _pin_command(base, frozen)
                with torch.no_grad():
                    actions = policy(obs)

                out = env.step(actions)
                extras = {}
                if len(out) == 4:
                    obs, _rew, dones, extras = out
                else:
                    obs, _rew, dones = out
                dones_b = dones.bool()
                alive = ~dones_b
                # 回合结束的 env 会在 step 内重采样指令；立刻写回，保证下一步的 obs 一致。
                _pin_command(base, frozen)

                p = _root_pos(base)
                dxy = (p[:, :2] - prev_xy) / dt
                dz_step = p[:, 2] - prev_z
                prev_xy = p[:, :2].clone()
                prev_z = p[:, 2].clone()

                try:
                    wz = _root_yaw_rate(base)
                except Exception:
                    wz = None
                if wz is None and prev_yaw is not None:
                    yw_now = _root_yaw(base)
                    dyaw = torch.atan2(torch.sin(yw_now - prev_yaw),
                                       torch.cos(yw_now - prev_yaw))
                    wz = dyaw / dt
                    prev_yaw = yw_now
                elif prev_yaw is not None:
                    prev_yaw = _root_yaw(base)

                q = _root_quat(base)
                if q is not None:
                    upz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                else:
                    upz = torch.ones(B, device=base.device)

                yw_now = prev_yaw if prev_yaw is not None else _root_yaw(base)
                if yw_now is None:
                    vlon = torch.zeros(B, device=base.device)
                else:
                    vlon = dxy[:, 0] * torch.cos(yw_now) + dxy[:, 1] * torch.sin(yw_now)

                # ---- 足底 / 支撑代理 ----
                clr = torch.zeros(B, device=base.device)
                clr_cnt = torch.zeros(B, device=base.device)
                peak = torch.zeros(B, device=base.device)
                swing = torch.zeros(B, device=base.device)
                foot_cnt = torch.full((B,), 2.0, device=base.device)
                slip = torch.zeros(B, device=base.device)
                contact_cnt = torch.zeros(B, device=base.device)
                support = torch.zeros(B, device=base.device)
                if foot_ok:
                    try:
                        h = vm.foot_height(base, sensor_name="foot_height_scan").float()
                        c = vm.foot_contact(base, sensor_name="feet_ground_contact")
                        c = (c.bool() if c.dtype == torch.bool
                             else c.to(torch.float32) > 0.5)
                        swing = (~c).float().sum(dim=-1)
                        valid = (h > 0.0) & (h < 0.5)
                        sw = (~c) & valid
                        clr = (h * sw.float()).sum(dim=-1)
                        clr_cnt = sw.float().sum(dim=-1)
                        peak = torch.where(valid, h, torch.zeros_like(h)).max(dim=-1).values

                        sites = _foot_sites(base, site_ids)
                        if sites is not None:
                            n_contact = c.float().sum(dim=-1)
                            s_xy = sites[:, :, :2]
                            # 双脚中点 / 单脚自身；无接触判为不在支撑内。
                            both = n_contact >= 2
                            one = n_contact == 1
                            center = (s_xy.sum(dim=1) / n_contact.clamp(min=1).unsqueeze(-1))
                            # 单脚时 center 已经就是那只脚；双脚时是两脚中点。
                            dist = torch.norm(p[:, :2] - center, dim=-1)
                            radius = torch.where(both,
                                                 torch.full_like(dist, SUPPORT_R_DOUBLE),
                                                 torch.full_like(dist, SUPPORT_R_SINGLE))
                            support = ((dist <= radius) & (n_contact > 0)).float()

                            vels = _foot_vels(base, site_ids)
                            if vels is not None:
                                sp = torch.norm(vels[:, :, :2], dim=-1)
                                slip = (sp * c.float()).sum(dim=-1)
                                contact_cnt = c.float().sum(dim=-1)
                    except Exception:
                        pass

                # ---- 参考跟踪 / 动作 ----
                ref_track = torch.zeros(B, device=base.device)
                ref_rms = torch.zeros(B, device=base.device)
                if term is not None:
                    try:
                        t_, r_ = _term_metrics(term, base.scene["robot"].data)
                        if t_ is not None:
                            ref_track, ref_rms = t_, r_
                            ref_ok = True
                    except Exception:
                        pass
                act_abs = actions.float().abs().mean(dim=-1)
                sat = torch.zeros(B, device=base.device)
                sat_leg = torch.zeros(B, device=base.device)
                sat_oth = torch.zeros(B, device=base.device)
                if clip_action is not None:
                    try:
                        over = (actions.float().abs() > clip_action.float()).float()
                        sat = over.mean(dim=-1)
                        if leg_ids is not None:
                            sat_leg = over[:, leg_ids].mean(dim=-1)
                            mask = torch.ones(over.shape[1], dtype=torch.bool, device=base.device)
                            mask[leg_ids] = False
                            if bool(mask.any()):
                                sat_oth = over[:, mask].mean(dim=-1)
                    except Exception:
                        pass
                rise = torch.zeros(B, device=base.device)
                if term is not None:
                    try:
                        rise = term.rise_est.float()
                    except Exception:
                        pass

                # ---- 累加（只累加存活步；dones 那一步的状态已被 reset 覆盖）----
                w = alive.float()
                acc["steps"] += w
                acc["dy"] += dxy[:, 1] * dt * w
                acc["dx"] += dxy[:, 0] * dt * w
                acc["dz"] += dz_step * w
                acc["vy"] += dxy[:, 1] * w
                acc["vx"] += dxy[:, 0] * w
                acc["vlon"] += vlon * w
                acc["wz_abs"] += (wz.abs() if wz is not None else torch.zeros(B, device=base.device)) * w
                acc["yaw_travel"] += (wz.abs() * dt if wz is not None
                                      else torch.zeros(B, device=base.device)) * w
                acc["support"] += support * w
                acc["upz"] += upz * w
                acc["clearance"] += clr * w
                acc["clearance_cnt"] += clr_cnt * w
                acc["peak"] = torch.maximum(acc["peak"], peak * w)
                acc["swing"] += swing * w
                acc["foot_cnt"] += foot_cnt * w
                acc["slip"] += slip * w
                acc["contact_cnt"] += contact_cnt * w
                acc["ref_track"] += ref_track * w
                acc["ref_rms"] += ref_rms * w
                acc["act_abs"] += act_abs * w
                acc["sat"] += sat * w
                acc["sat_leg"] += sat_leg * w
                acc["sat_oth"] += sat_oth * w
                acc["rise"] += rise * w

                # 优先用环境自己给的 time_outs；没有就按"跑满本回合步数"推断。
                t_outs = extras.get("time_outs") if isinstance(extras, dict) else None
                if t_outs is not None:
                    timeout_b = dones_b & t_outs.bool()
                else:
                    timeout_b = dones_b & (ep_steps >= max_ep - 1)
                if bool(dones_b.any()):
                    for k in torch.nonzero(dones_b, as_tuple=False).flatten().tolist():
                        snap = {key: float(acc[key][k].item()) for key in acc}
                        rec = _episode_summary(
                            snap, dt=dt, cmd=cmd, level=actual_level,
                            fall=not bool(timeout_b[k]),
                            timeout=bool(timeout_b[k]),
                            seconds=float(seconds),
                            success_dy=success_dy, success_dz=success_dz)
                        if not foot_ok:
                            for key in ("support_proxy", "clearance_mean",
                                        "clearance_peak", "swing_frac", "slip"):
                                rec[key] = float("nan")
                        if not ref_ok:
                            rec["ref_track"] = float("nan")
                            rec["ref_rms"] = float("nan")
                        records.append(rec)
                        for key in acc:
                            acc[key][k] = 0.0
                        ep_steps[k] = 0

                ep_steps += 1
                ep_steps[dones_b] = 0

            print(f"[eval] level={actual_level} cmd={cmd:.2f} 完成 "
                  f"{episodes}×{max_ep} 步 × {B} env")

    return records


# ================================================================ 汇总
def _nan(vals):
    vals = [v for v in vals if v is not None and math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


SUMMARY_KEYS = (
    "fall_rate", "timeout_rate", "reach_rate", "climb_rate", "ep_sec_median",
    "dy_mean", "dz_mean", "speed_mean", "track_ratio", "yaw_rate",
    "yaw_travel_deg", "vlon_bias", "vx_mean", "support_proxy", "up_z",
    "clearance_mean", "clearance_peak", "swing_frac", "slip", "ref_track",
    "ref_rms", "act_abs", "sat_frac", "rise_est",
)


def summarize(records: list[dict], *, seconds: float) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, **{k: float("nan") for k in SUMMARY_KEYS}}
    ep_sec = [r["ep_sec"] for r in records]
    return dict(
        n=n,
        fall_rate=sum(r["fall"] for r in records) / n,
        timeout_rate=sum(r["timeout"] for r in records) / n,
        reach_rate=sum(r["reach"] for r in records) / n,
        climb_rate=sum(r["climb"] for r in records) / n,
        ep_sec_median=statistics.median(ep_sec),
        dy_mean=_nan([r["dy"] for r in records]),
        dz_mean=_nan([r["dz"] for r in records]),
        speed_mean=_nan([r["dy"] / max(r["ep_sec"], 1e-6) for r in records]),
        track_ratio=_nan([r["track_ratio"] for r in records]),
        yaw_rate=_nan([r["yaw_rate"] for r in records]),
        yaw_travel_deg=_nan([r["yaw_travel_deg"] for r in records]),
        vlon_bias=_nan([r["vlon_bias"] for r in records]),
        vx_mean=_nan([r["vx_mean"] for r in records]),
        support_proxy=_nan([r["support_proxy"] for r in records]),
        up_z=_nan([r["up_z"] for r in records]),
        clearance_mean=_nan([r["clearance_mean"] for r in records]),
        clearance_peak=_nan([r["clearance_peak"] for r in records]),
        swing_frac=_nan([r["swing_frac"] for r in records]),
        slip=_nan([r["slip"] for r in records]),
        ref_track=_nan([r["ref_track"] for r in records]),
        ref_rms=_nan([r["ref_rms"] for r in records]),
        act_abs=_nan([r["act_abs"] for r in records]),
        sat_frac=_nan([r["sat_frac"] for r in records]),
        sat_leg_frac=_nan([r["sat_leg_frac"] for r in records]),
        sat_oth_frac=_nan([r["sat_oth_frac"] for r in records]),
        rise_est=_nan([r["rise_est"] for r in records]),
    )


def grade(s: dict, level: int, *, success_level: int) -> tuple[str, list[str]]:
    """硬指标评分；support_proxy / clearance 仅诊断，不参与。"""
    fails = []
    if s.get("n", 0) == 0:
        return "NO-DATA", ["没有完整回合"]
    if s["fall_rate"] > 0.15:
        fails.append(f"fall {s['fall_rate']:.0%}>15%")
    if s["track_ratio"] < 0.75:
        fails.append(f"track {s['track_ratio']:.2f}<0.75")
    if s["yaw_rate"] > 0.15:
        fails.append(f"yaw {s['yaw_rate']:.2f}>0.15")
    if abs(s["vlon_bias"]) > 0.05:
        fails.append(f"v_lon {s['vlon_bias']:+.2f}>0.05")
    if s["sat_frac"] > 0.60:
        # 括号里给拆分：腿顶死是真问题（步态调不动），手臂顶死基本无害
        # （14 个手臂维度 clip 只有 0.20 rad，action 空间上限 ~0.65~0.97，天然容易饱和）。
        leg = s.get("sat_leg_frac", float("nan"))
        oth = s.get("sat_oth_frac", float("nan"))
        extra = ""
        if leg == leg and oth == oth:
            extra = f" (腿 {leg:.0%} / 非腿 {oth:.0%})"
        fails.append(f"sat {s['sat_frac']:.0%}>60%{extra}")
    if level >= success_level and s["climb_rate"] < 0.80:
        fails.append(f"climb {s['climb_rate']:.0%}<80%")
    return ("PASS" if not fails else "FAIL"), fails


def print_table(cond: list[dict]) -> None:
    print("-" * 132)
    print(f"{'ckpt':<14} {'L':>2} {'cmd':>5} {'fall':>6} {'tmout':>6} {'reach':>6} "
          f"{'climb':>6} {'ep_s':>5} {'dy':>6} {'track':>6} {'yaw':>5} {'vlon':>6} "
          f"{'clr':>5} {'slip':>5} {'sat':>5} {'sL':>4} {'sO':>4} {'ref':>5} {'sup':>5}  grade")
    for c in cond:
        s = c["summary"]
        g, fails = c["grade"], c["fails"]
        print(f"{c['checkpoint']:<14} {c['level']:>2} {c['cmd']:>5.2f} "
              f"{s['fall_rate']:>6.0%} {s['timeout_rate']:>6.0%} {s['reach_rate']:>6.0%} "
              f"{s['climb_rate']:>6.0%} {s['ep_sec_median']:>5.1f} {s['dy_mean']:>6.2f} "
              f"{s['track_ratio']:>6.2f} {s['yaw_rate']:>5.2f} {s['vlon_bias']:>+6.3f} "
              f"{s['clearance_mean']:>5.3f} {s['slip']:>5.3f} {s['sat_frac']:>5.0%} "
              f"{s.get('sat_leg_frac', float('nan')):>4.0%} "
              f"{s.get('sat_oth_frac', float('nan')):>4.0%} "
              f"{s['ref_track']:>5.2f} {s['support_proxy']:>5.0%}  {g}"
              + (f"  ({'; '.join(fails)})" if fails else ""))
    print("-" * 132)


def print_compare(by_ckpt: dict, conditions: list[tuple[int, float]]) -> None:
    """第一个 checkpoint 作为基线，打印其余 checkpoint 的指标差值。"""
    names = list(by_ckpt)
    if len(names) < 2:
        return
    base_name = names[0]
    print(f"[compare] 基线 = {base_name}（负值的 fall/yaw/vlon 更好，正值的 dy/track 更好）")
    print("-" * 100)
    print(f"{'ckpt':<14} {'L':>2} {'cmd':>5} {'d_fall':>8} {'d_dy':>8} {'d_track':>8} "
          f"{'d_yaw':>8} {'d_vlon':>8} {'d_climb':>8}")
    for name in names[1:]:
        for (level, cmd) in conditions:
            a = by_ckpt[base_name].get((level, cmd))
            b = by_ckpt[name].get((level, cmd))
            if not a or not b:
                continue
            print(f"{name:<14} {level:>2} {cmd:>5.2f} "
                  f"{b['fall_rate'] - a['fall_rate']:>+8.0%} "
                  f"{b['dy_mean'] - a['dy_mean']:>+8.2f} "
                  f"{b['track_ratio'] - a['track_ratio']:>+8.2f} "
                  f"{b['yaw_rate'] - a['yaw_rate']:>+8.2f} "
                  f"{b['vlon_bias'] - a['vlon_bias']:>+8.3f} "
                  f"{b['climb_rate'] - a['climb_rate']:>+8.0%}")
    print("-" * 100)


# ================================================================ 主流程
def _setup_push(env_cfg, scale: float) -> None:
    from mjlab.envs.mdp.events import push_by_setting_velocity
    from mjlab.managers.event_manager import EventTermCfg

    r = 0.2 * float(scale)
    env_cfg.events["push_robot"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {
            "x": (-r, r), "y": (-r, r), "z": (-0.1 * scale, 0.1 * scale),
            "roll": (-0.15 * scale, 0.15 * scale),
            "pitch": (-0.15 * scale, 0.15 * scale),
            "yaw": (-0.1 * scale, 0.1 * scale),
        }},
    )
    print(f"[info] 已打开 push（3~6 s 一次，力度 ×{scale}，y 向 ±{r:.2f} m/s）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True,
                    help="一个或多个 model_XXXX.pt；第一个作为对比基线")
    ap.add_argument("--task", default="Elf3-Crab-Residual-Play")
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=2, help="每个 env 至少跑几个完整回合")
    ap.add_argument("--seconds", type=float, default=20.0, help="单个回合时长")
    ap.add_argument("--levels", default="0,2,4,6,8", help="地形等级列表，如 0,2,4,6,8")
    ap.add_argument("--cmds", default="0.18", help="固定指令 v_y 列表，如 0.14,0.18,0.20")
    ap.add_argument("--success-dy", type=float, default=2.0, help="climb 的净 Δy 门槛 (m)")
    ap.add_argument("--success-dz", type=float, default=0.10, help="climb 的 Δz 门槛 (m)")
    ap.add_argument("--success-level", type=int, default=4,
                    help=">= 该等级才把 climb_rate 纳入 PASS/FAIL")
    ap.add_argument("--push", action="store_true", help="打开扰动评测（默认无扰动）")
    ap.add_argument("--push-scale", type=float, default=0.5, help="扰动幅度倍率")
    ap.add_argument("--alpha", type=float, default=None,
                    help="覆盖 ELF3_ALPHA；应为 checkpoint 训练当时的 Metrics/ref_alpha")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-only", action="store_true",
                    help="只做 checkpoint 静态体检，不建环境（不需要 mjlab / GPU）")
    ap.add_argument("--out", default=None, help="JSON 输出路径")
    args = ap.parse_args(argv)

    audits = [audit_checkpoint(p) for p in args.checkpoint]
    for a in audits:
        print_audit(a)
    bad = [a for a in audits if not a["ok"]]
    if bad:
        print(f"[ERR] {len(bad)} 个 checkpoint 无法加载，停止。")
        return 2

    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint[0]))[0]
        prefix = "audit_" if args.audit_only else ""
        out_path = os.path.join("logs", "eval", f"{prefix}{stem}.json")

    if args.audit_only:
        result = {"audits": audits, "conditions": [], "summary": {}}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[audit] JSON -> {out_path}")
        return 0

    # 环境变量必须在注册任务之前设置：env_cfg_residual 在构造时读取它们。
    if args.alpha is not None:
        os.environ["ELF3_ALPHA"] = repr(float(args.alpha))

    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    cmds = [float(x) for x in args.cmds.split(",") if x.strip()]

    try:
        env_cfg = load_env_cfg(args.task, play=True)
        agent_cfg = load_rl_cfg(args.task)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    except KeyError:
        print(f"[ERR] 任务 {args.task!r} 未注册；先在服务器执行 bash train/install_task.sh")
        return 2

    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.episode_length_s = float(args.seconds)      # play 默认 1e9，必须覆盖
    try:
        env_cfg.observations["actor"].enable_corruption = False
    except Exception:
        pass
    if not args.push:
        try:
            env_cfg.events.pop("push_robot", None)
        except Exception:
            pass
    else:
        _setup_push(env_cfg, args.push_scale)

    # play 配置每次 reset 会随机重排地形等级；评测要"固定档位"，所以移除该 event，
    # 由 _force_terrain_level 同时改写 terrain_levels 和 env_origins。
    try:
        env_cfg.events.pop("randomize_terrain", None)
    except Exception:
        pass

    # 指令范围钉在第一个 cmd；每个条件运行时会逐帧覆写为准确值。
    tw = env_cfg.commands["twist"]
    v0 = cmds[0]
    for k, val in (("lin_vel_x", (0.0, 0.0)), ("lin_vel_y", (v0, v0)),
                   ("ang_vel_z", (0.0, 0.0))):
        try:
            setattr(tw.ranges, k, val)
        except Exception:
            pass
    for k, val in (("resampling_time_range", (1e9, 1e9)), ("rel_standing_envs", 0.0)):
        try:
            setattr(tw, k, val)
        except Exception:
            pass

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = _unwrap(env)
    B = int(base.num_envs)
    dt = float(getattr(base, "step_dt", 0.02))

    # 地形行数决定最高档；先夹一次，保证 summary 的 key 与 episode 记录一致。
    try:
        rows = int(base.scene.terrain.terrain_origins.shape[0])
        clamped = [min(lv, rows - 1) for lv in levels]
        if clamped != levels:
            print(f"[warn] 地形只有 {rows} 行，等级 {levels} 已夹到 {clamped}")
        levels = list(dict.fromkeys(clamped))
    except Exception:
        pass

    if args.alpha is not None:
        print(f"[info] ELF3_ALPHA={args.alpha}（回放口径必须对齐训练时的 ref_alpha）")
    try:
        term0 = base.action_manager.get_term("joint_pos")
        a_now = float(term0.ref_alpha)
        if a_now < 0.95:
            print("!" * 70)
            print(f"[!!] ref_alpha={a_now:.3f} < 0.95：参考步态不是满幅，评测口径与成品不符。")
            print("     用 --alpha <训练当时的 Metrics/ref_alpha> 重新跑。")
            print("!" * 70)
        else:
            print(f"[info] ref_alpha={a_now:.3f}")
    except Exception:
        pass

    all_conds: list[dict] = []
    all_eps: list[dict] = []
    by_ckpt: dict[str, dict] = {}

    for ckpt in args.checkpoint:
        name = os.path.splitext(os.path.basename(ckpt))[0]
        print(f"[info] 加载 {ckpt}")
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        try:
            runner.load(ckpt, load_cfg={"actor": True}, strict=True,
                        map_location=device)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] 加载失败（可能动作/观测语义与当前任务不匹配）：{e!r}")
            return 3
        policy = runner.get_inference_policy(device=device)

        recs = evaluate_conditions(
            env, base, policy, levels=levels, cmds=cmds,
            episodes=args.episodes, seconds=args.seconds, seed=args.seed,
            success_dy=args.success_dy, success_dz=args.success_dz)

        all_eps.extend([{**r, "checkpoint": name} for r in recs])
        by_ckpt[name] = {}
        for level in levels:
            for cmd in cmds:
                sel = [r for r in recs
                       if r["level"] == level and abs(r["cmd"] - cmd) < 1e-9]
                s = summarize(sel, seconds=args.seconds)
                g, fails = grade(s, level, success_level=args.success_level)
                entry = dict(checkpoint=name, level=level, cmd=cmd,
                             summary=s, grade=g, fails=fails)
                all_conds.append(entry)
                by_ckpt[name][(level, cmd)] = s

        del runner, policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print_table(all_conds)
    print_compare(by_ckpt, [(lv, c) for lv in levels for c in cmds])

    print("=" * 80)
    for name in by_ckpt:
        conds = [c for c in all_conds if c["checkpoint"] == name]
        passed = [c for c in conds if c["grade"] == "PASS"]
        failed = [c for c in conds if c["grade"] == "FAIL"]
        print(f"{name}: PASS {len(passed)}/{len(conds)}")
        for c in failed:
            print(f"  FAIL  L{c['level']} cmd={c['cmd']:.2f}: {'; '.join(c['fails'])}")
    print("=" * 80)

    result = {
        "meta": {
            "task": args.task, "num_envs": B, "episodes": args.episodes,
            "seconds": args.seconds, "levels": levels, "cmds": cmds,
            "push": bool(args.push), "push_scale": args.push_scale,
            "alpha": args.alpha, "seed": args.seed,
            "success_dy": args.success_dy, "success_dz": args.success_dz,
            "device": device, "dt": dt,
        },
        "audits": audits,
        "conditions": all_conds,
        "episodes": all_eps,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[info] JSON -> {out_path}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
