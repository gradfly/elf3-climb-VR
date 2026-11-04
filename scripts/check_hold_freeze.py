"""离线自检：`TrajTrackAction.freeze_frame`（不需要 mjlab / GPU）。

「保持」任务（Elf3-Squat-Hold / Elf3-Stand-Hold）完全靠 `freeze_frame` 把参考
钉死在某一帧上。这一步出错不会有任何报错，只会让策略学错东西：

    freeze 没生效（仍从第 0 帧起）  -> 蹲姿任务实际要求「保持站姿」
    冻结后还在推进帧号            -> 参考一路播到末帧，等于偷偷变成了跟踪任务
    finished 没屏蔽               -> 回合在 2.46 s 就结束，20 s 的保持段白设

这些都只有跑几小时训练、看回放才看得出来。这里用替身 mjlab 直接读参考表比对，
2 秒出结果。

用法：
    python scripts/check_hold_freeze.py
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(REPO, "train/assets/ref_tables/squat_ref_v1.npz")
CRAB = os.path.join(REPO, "train/mjlab_tasks/crab")


# ---------------------------------------------------------------- 替身 mjlab
@dataclass(kw_only=True)
class JointPositionActionCfg:
    """`traj_action.py` 只用到这几个字段。"""

    entity_name: str = "robot"
    actuator_names: tuple = (".*",)
    scale: object = 0.25
    offset: float = 0.0
    use_default_offset: bool = False
    clip: object = None

    def build(self, env):
        return JointPositionAction(self, env)


class JointPositionAction:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self._B = int(env.num_envs)
        self._D = len(env.joint_names)
        self._raw_actions = torch.zeros(self._B, self._D)
        self._scale = float(cfg.scale)
        self._target_names = list(env.joint_names)
        self._target_ids = list(range(self._D))
        self.last_raw = None

    def process_actions(self, actions):
        self.last_raw = actions.clone()

    def reset(self, env_ids=None):
        pass


def _install_stub() -> None:
    mk = types.ModuleType
    mods = {
        "mjlab": mk("mjlab"),
        "mjlab.envs": mk("mjlab.envs"),
        "mjlab.envs.mdp": mk("mjlab.envs.mdp"),
        "mjlab.envs.mdp.actions": mk("mjlab.envs.mdp.actions"),
    }
    act = mods["mjlab.envs.mdp.actions"]
    act.JointPositionAction = JointPositionAction
    act.JointPositionActionCfg = JointPositionActionCfg
    mods["mjlab"].envs = mods["mjlab.envs"]
    mods["mjlab.envs"].mdp = mods["mjlab.envs.mdp"]
    mods["mjlab.envs.mdp"].actions = act
    sys.modules.update(mods)

    # traj_action 用 `from .traj_ref import ...`，得有个包来承载相对导入
    pkg = mk("crabpkg")
    pkg.__path__ = [CRAB]
    sys.modules["crabpkg"] = pkg


class _Env:
    num_envs = 3
    step_dt = 0.02

    def __init__(self, joint_names):
        self.joint_names = joint_names
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)


# ---------------------------------------------------------------- 断言
_FAILED = []


def check(label: str, cond, info: str = "") -> None:
    cond = bool(cond)
    print(("  PASS  " if cond else "  FAIL  ") + label + (f"   {info}" if info else ""))
    if not cond:
        _FAILED.append(label)


def close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> bool:
    return torch.allclose(a.float(), b.float(), atol=tol, rtol=0)


def main() -> int:
    _install_stub()
    traj_action = importlib.import_module("crabpkg.traj_action")
    traj_ref = importlib.import_module("crabpkg.traj_ref")

    ref = traj_ref.load_traj(REF, "down")
    T = ref.T
    print(f"参考表：down，{T} 帧 / {ref.duration:.2f} s，"
          f"z {float(ref.z[0]):.4f} -> {float(ref.z[-1]):.4f} m\n")

    def build(**kw):
        env = _Env(ref.names)
        cfg = traj_action.TrajTrackActionCfg(
            traj_path=REF, traj_key="down", scale=0.25, **kw,
        )
        return env, cfg.build(env)

    # ---- 1. 冻结在末帧（= 极限蹲，Elf3-Squat-Hold）----
    print("① freeze_frame=-1（极限蹲保持）")
    env, a = build(freeze_frame=-1)
    check("参考 = 末帧关节角", close(a.ref_joint_pos, ref.q[-1].unsqueeze(0).expand(3, -1)))
    check("参考高度 = 0.523 m", abs(float(a.ref_base_z[0]) - float(ref.z[-1])) < 1e-5,
          f"z={float(a.ref_base_z[0]):.4f}")
    check("帧号 = T-1", bool((a.frame_index == T - 1).all()), f"idx={int(a.frame_index[0])}")
    for _ in range(5):
        a.process_actions(torch.zeros(3, len(ref.names)))
    check("走 5 步后参考不变", close(a.ref_joint_pos, ref.q[-1].unsqueeze(0).expand(3, -1)))
    check("走 5 步后帧号不变", bool((a.frame_index == T - 1).all()))
    check("finished 恒 False（否则回合 2.46 s 就结束）",
          not bool(a.finished.any()))
    a.reset(torch.arange(3))
    check("reset 后仍是末帧", close(a.ref_joint_pos, ref.q[-1].unsqueeze(0).expand(3, -1)))

    # ---- 2. 冻结在首帧（= 站姿，Elf3-Stand-Hold）----
    print("\n② freeze_frame=0（站姿保持）")
    env, a = build(freeze_frame=0)
    check("参考 = 首帧关节角", close(a.ref_joint_pos, ref.q[0].unsqueeze(0).expand(3, -1)))
    check("参考高度 = 1.034 m", abs(float(a.ref_base_z[0]) - float(ref.z[0])) < 1e-5,
          f"z={float(a.ref_base_z[0]):.4f}")
    a.process_actions(torch.zeros(3, len(ref.names)))
    check("走 1 步后帧号仍是 0", bool((a.frame_index == 0).all()))

    # ---- 3. 不冻结时行为不变（回归检查，别把跟踪任务改坏）----
    print("\n③ freeze_frame=None（跟踪任务，应与改动前完全一致）")
    env, a = build()
    check("起始 = 第 0 帧", close(a.ref_joint_pos, ref.q[0].unsqueeze(0).expand(3, -1)))
    # 注意语义：process_actions 先把**本步**的参考写进 _ref_q，再推进帧号。
    # 所以走 1 步后 idx=1 而 ref 仍是 q[0]（这一步执行的），下一步才是 q[1]。
    a.process_actions(torch.zeros(3, len(ref.names)))
    check("走 1 步：idx=1、参考仍是本步执行的 q[0]",
          bool((a.frame_index == 1).all())
          and close(a.ref_joint_pos, ref.q[0].unsqueeze(0).expand(3, -1)),
          f"idx={int(a.frame_index[0])}")
    a.process_actions(torch.zeros(3, len(ref.names)))
    check("走 2 步后参考到第 1 帧", close(a.ref_joint_pos, ref.q[1].unsqueeze(0).expand(3, -1)),
          f"idx={int(a.frame_index[0])}")
    check("起始未 finished", not bool(a.finished.any()))
    for _ in range(T):
        a.process_actions(torch.zeros(3, len(ref.names)))
    check("播完 finished=True", bool(a.finished.all()), f"idx={int(a.frame_index[0])}")
    check("帧号 clamp 在 T-1 不越界", bool((a.frame_index <= T - 1).all()))

    # ---- 4. 非法帧号要明确报错，不能静默取错帧 ----
    print("\n④ 越界保护")
    try:
        build(freeze_frame=999)
        check("freeze_frame=999 抛 ValueError", False)
    except ValueError:
        check("freeze_frame=999 抛 ValueError", True)

    # ---- 5. 残差限幅仍然按腿/腰/臂分开 ----
    print("\n⑤ 残差限幅（冻结模式不该把它弄丢）")
    # 按 squat_hold_cfg 的真实取值传：腿 0.35 / 腰 0.15 / 臂 0.12
    env, a = build(freeze_frame=-1, leg_clip_rad=0.35, waist_clip_rad=0.15, arm_clip_rad=0.12)
    a.process_actions(torch.full((3, len(ref.names)), 100.0))
    raw = a.last_raw
    scale = 0.25
    leg_col = ref.names.index("l_knee_y_joint")
    arm_col = ref.names.index("l_shoulder_y_joint")
    check("腿关节夹到 0.35 rad", abs(float(raw[0, leg_col]) * scale - 0.35) < 1e-4,
          f"{float(raw[0, leg_col]) * scale:.4f} rad")
    check("手臂夹到 0.12 rad", abs(float(raw[0, arm_col]) * scale - 0.12) < 1e-4,
          f"{float(raw[0, arm_col]) * scale:.4f} rad")

    print()
    if _FAILED:
        print(f"[FAIL] {len(_FAILED)} 项未通过：{_FAILED}")
        return 1
    print("[OK] 冻结帧行为全部符合预期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
