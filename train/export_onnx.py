#!/usr/bin/env python3
"""把训练好的 checkpoint 导出为 ONNX 策略，供 crab_replay / 网页回放 / 真机部署。

用法（在服务器上、已装 mjlab 的环境里）：
    python train/export_onnx.py \
        --task Elf3-Crab-Stairs \
        --checkpoint logs/rsl_rl/elf3_crab/<run>/model_3000.pt \
        --out logs/export/elf3_crab_policy.onnx

说明：
  - 依赖 rsl_rl 的 `export_policy_as_onnx`；不同版本签名略有差异，脚本已做兼容尝试。
  - 导出的 ONNX 输入为观测向量（与训练时 actor 观测同维），输出为 29 维动作。
  - 若你的 mjlab 版本 runner 自带 `export_policy_as_onnx`，会优先用它。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Elf3-Crab-Stairs")
    ap.add_argument("--checkpoint", required=True, help="model_XXXX.pt")
    ap.add_argument("--out", default="logs/export/elf3_crab_policy.onnx")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from dataclasses import asdict

    import mjlab.tasks.crab  # noqa: F401  触发任务注册
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env = ManagerBasedRlEnv(load_env_cfg(args.task, play=True), device=args.device)
    rl_cfg = load_rl_cfg(args.task)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner

    # 构造 runner 必须照抄 mjlab/scripts/train.py 的两步，少一步都会炸：
    #
    #   1) cfg 要经 dataclasses.asdict 转成 **dict**
    #      —— runner.__init__ 里有 `if key in train_cfg`，dataclass 不支持 `in`，
    #         会报 `TypeError: argument of type 'RslRlOnPolicyRunnerCfg' is not iterable`。
    #         注意 `.to_dict()` 这个方法是**不存在**的（旧版代码假设有，实际走的是 else 分支）。
    #   2) env 要包一层 RslRlVecEnvWrapper
    #      —— runner 期望的是 vecenv 接口（num_envs / step / reset 的返回约定）。
    agent_cfg = asdict(rl_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(rl_cfg, "clip_actions", None))
    runner = runner_cls(env, agent_cfg, str(out.parent), args.device)
    runner.load(args.checkpoint)
    print(f"[OK] 载入 checkpoint: {args.checkpoint}")

    # ⚠️ 方法名是 export_policy_to_onnx，不是 export_policy_as_onnx（差两个词）。
    # 而 rsl_rl 这个版本里**没有**任何 ONNX 导出函数 —— utils 下只有 LogWriter /
    # check_nan / resolve_* 这些，`from rsl_rl.utils import export_policy_as_onnx`
    # 必然 ImportError。导出能力只在 mjlab 的 runner 上。
    fn = None
    for name in ("export_policy_to_onnx", "export_policy_as_onnx"):
        cand = getattr(runner, name, None)
        if callable(cand):
            fn = cand
            print(f"[info] 使用 runner.{name}()")
            break
    if fn is None:
        have = [n for n in dir(runner) if "export" in n.lower() or "onnx" in n.lower()]
        print(f"[ERR] runner 上没有导出方法；它有的是 {have}", file=sys.stderr)
        return 3

    # 签名在不同版本间变过：有的 (path, filename=...)，有的只要 (dir,)。
    # 逐个试，第一个不抛 TypeError 的算成功。
    last_err = None
    for call in (
        lambda: fn(str(out.parent), filename=out.name),
        lambda: fn(str(out.parent), out.name),
        lambda: fn(str(out.parent)),
        lambda: fn(),
    ):
        try:
            call()
            last_err = None
            break
        except TypeError as e:
            last_err = e
    if last_err is not None:
        print(f"[ERR] 调用导出方法失败：{last_err}", file=sys.stderr)
        return 4

    if not out.exists():
        # 某些版本把文件写到 out.parent/policy.onnx
        cand = out.parent / "policy.onnx"
        if cand.exists():
            cand.rename(out)
    if out.exists():
        print(f"[OK] 导出 {out} ({out.stat().st_size/1024:.1f} KB)")
        return 0
    print("[ERR] 导出后没找到 ONNX 文件，请检查 rsl_rl 版本", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
