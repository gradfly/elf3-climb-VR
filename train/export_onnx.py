#!/usr/bin/env python3
"""把训练好的 checkpoint 导出为 ONNX 策略（爬楼 / 蹲起 / 保持任务通用）。

最省事的用法（在**仓库根目录**跑，需已装 mjlab 的 .venv）：

    # 蹲起四任务：自动找该实验目录下最新的完好 checkpoint
    .venv/bin/python train/export_onnx.py --squat down
    .venv/bin/python train/export_onnx.py --squat up
    .venv/bin/python train/export_onnx.py --squat hold-squat
    .venv/bin/python train/export_onnx.py --squat hold-stand

    # 爬楼（原来的用法不变）
    .venv/bin/python train/export_onnx.py \
        --task Elf3-Crab-Residual \
        --checkpoint logs/rsl_rl/elf3_crab/<run>/model_3000.pt \
        --out train/checkpoints/elf3_crab_residual.onnx

每次导出会同时写一份同名 `.json` 契约文件：观测项拼接顺序与总维度、
动作语义、参考表路径/帧数/控制周期、数值对拍结果。真机侧按它拼观测。

⚠️ 导出的 ONNX **输出是 29 维残差动作，不是关节目标**。
   蹲起/保持任务必须自己复现 `q_target = q_ref + scale * clip(a)`，
   并维护 20 ms 参考帧计数器 —— 详见 docs/squat_rl_training.md §6 / §7。

为什么有 torch 直出兜底
------------------------
rsl_rl 版本的 `runner.export_policy_to_onnx` 时有时无（方法名还从
`export_policy_as_onnx` 改过）。所以默认 `--engine auto` 先试 runner 自带导出，
失败就用 `runner.alg.actor` 走 `torch.onnx.export`。服务器上
`runner.alg.actor` 已经把 obs 归一化模块包在图里（见 diagnose_policy.py 的
「归一化统计量体检」），导它不会丢 running mean/var。
导出后用 onnxruntime 与 `get_inference_policy()` 做一次数值对拍，
不一致直接报错 —— 一个能加载但输出错误的 ONNX 比没有 ONNX 更危险。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REF_TABLE = REPO / "train/assets/ref_tables/squat_ref_v1.npz"

# mode -> (任务 ID, 实验目录, 参考表 key, 冻结帧)
# 与 scripts/export_squat_video.py 的 MODES 保持一致，别改出两份不同的映射。
SQUAT_MODES = {
    "down":       ("Elf3-Squat-Down-Play",   "elf3_squat_down",        "down", None),
    "up":         ("Elf3-Squat-Up-Play",     "elf3_squat_up",          "up",   None),
    "hold-squat": ("Elf3-Squat-Hold-Play",   "elf3_squat_hold_squat",  "down", -1),
    "hold-stand": ("Elf3-Stand-Hold-Play",   "elf3_squat_hold_stand",  "down", 0),
}

ACTOR_OBS_TERMS_NOTE = (
    "观测按 cfg.observations['actor'].terms 的字典顺序拼接；"
    "实际维度以本文件 obs_dim 为准，不确定时用 train/export_onnx.py 导出的契约核对"
)


# ---------------------------------------------------------------- 参数解析
def _latest_checkpoint(exp: str) -> str:
    """找该实验目录下最新 run 里迭代数最大的完好 checkpoint。"""
    root = REPO / "logs" / "rsl_rl" / exp
    if not root.is_dir():
        raise FileNotFoundError(
            f"没有实验目录 {root}。先在仓库根目录跑训练，或用 --checkpoint 指定路径。")

    def _best_in(d: Path):
        pts = []
        for p in d.glob("model_*.pt"):
            m = re.search(r"model_(\d+)\.pt$", p.name)
            # 训练被 kill 时正在写盘的 checkpoint 是空/截断文件，必须跳过
            if m and p.stat().st_size > 4096:
                pts.append((int(m.group(1)), p))
        return max(pts, key=lambda t: t[0])[1] if pts else None

    runs = sorted((p for p in root.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        hit = _best_in(r)
        if hit is not None:
            return str(hit)
    hits = [p for p in root.rglob("model_*.pt") if p.stat().st_size > 4096]
    if hits:
        return str(max(hits, key=lambda p: p.stat().st_mtime))
    raise FileNotFoundError(f"{root} 下没有完好的 model_*.pt（>4096 B）")


def _ref_info(key: str, freeze):
    """从烘焙表读出时长/帧数，避免契约里写死数字。"""
    import numpy as np

    with np.load(REF_TABLE, allow_pickle=True) as d:
        z = d[f"{key}/z"]
        T = int(d[f"{key}/q"].shape[0])
        dt = float(d[f"{key}/dt"])
        z0, z1 = float(z[0]), float(z[-1])
        f = None if freeze is None else int(freeze) % T
        z_hold = round(float(z[f]), 4) if f is not None else None
    return {
        "table": str(REF_TABLE.relative_to(REPO)).replace("\\", "/"),
        "traj_key": key,
        "frames": T,
        "dt": dt,
        "duration_s": round((T - 1) * dt, 4),
        "freeze_frame": f,
        "z_start": round(z0, 4),
        "z_end": round(z1, 4),
        # 冻结任务真正要保持的高度是这一帧，不是 z_end（末帧是极限蹲 0.523）
        "z_frozen": z_hold,
    }


def _resolve(args):
    if args.squat:
        task, exp, key, freeze = SQUAT_MODES[args.squat]
        tag = args.squat.replace("-", "_")
        out = Path(args.out) if args.out else REPO / "logs/export" / f"squat_{tag}.onnx"
        ckpt = args.checkpoint or _latest_checkpoint(exp)
        return task, out, ckpt, {"mode": args.squat, "exp": exp,
                                 "ref": _ref_info(key, freeze)}
    out = Path(args.out) if args.out else REPO / "logs/export/elf3_crab_policy.onnx"
    if not args.checkpoint:
        raise ValueError("非 --squat 模式必须给 --checkpoint（或用 --squat <mode> 自动找）")
    return args.task, out, args.checkpoint, None


# ---------------------------------------------------------------- runner 兼容层
def _build_runner(runner_cls, wrapped, rl_cfg, out: Path, device: str):
    """服务器版 mjlab 用 3 参构造；旧 rsl_rl 用 (env, cfg, log_dir, device)。"""
    agent_cfg = asdict(rl_cfg)
    try:
        return runner_cls(wrapped, agent_cfg, device=device)
    except TypeError as e:
        print(f"[warn] runner 3 参构造失败（{e!r}），退回 rsl_rl 4 参签名")
        return runner_cls(wrapped, agent_cfg, str(out.parent), device)


def _load_ckpt(runner, ckpt: str, device: str) -> None:
    """三种 load 签名都见过：mjlab 自定义 / play.py 标准 / 最简。"""
    attempts = (
        ("load_cfg/strict", lambda: runner.load(
            str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)),
        ("map_location", lambda: runner.load(str(ckpt), map_location=device)),
        ("minimal", lambda: runner.load(str(ckpt))),
    )
    for label, call in attempts:
        try:
            call()
            print(f"[OK] 载入 checkpoint（{label}）：{ckpt}")
            return
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn] load({label}) 失败：{e!r}")
    raise RuntimeError(f"三种 runner.load 签名都失败：{ckpt}")


def _find_actor(runner):
    """返回推理用的 nn.Module。服务器版是 runner.alg.actor（含 obs 归一化）。"""
    import torch

    cands = (
        getattr(getattr(runner, "alg", None), "actor", None),
        getattr(getattr(getattr(runner, "alg", None), "actor_critic", None), "actor", None),
        getattr(getattr(getattr(runner, "alg", None), "policy", None), "actor", None),
        getattr(getattr(runner, "alg", None), "actor_critic", None),
    )
    for m in cands:
        if isinstance(m, torch.nn.Module):
            return m
    return None


# ---------------------------------------------------------------- 导出
def _export_runner(runner, out: Path) -> bool:
    """优先用 runner 自带导出；方法名/签名跨版本变过，逐个试。"""
    fn = None
    for name in ("export_policy_to_onnx", "export_policy_as_onnx"):
        cand = getattr(runner, name, None)
        if callable(cand):
            fn = cand
            print(f"[info] 使用 runner.{name}()")
            break
    if fn is None:
        have = [n for n in dir(runner) if "export" in n.lower() or "onnx" in n.lower()]
        print(f"[warn] runner 上没有导出方法；它有的是 {have} → 改走 torch 直出")
        return False

    for call in (
        lambda: fn(str(out.parent), filename=out.name),
        lambda: fn(str(out.parent), out.name),
        lambda: fn(str(out.parent)),
    ):
        try:
            call()
            if out.exists():
                return True
            alt = out.parent / "policy.onnx"
            if alt.exists():
                alt.rename(out)
                return True
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn] runner 导出调用失败（{e!r}），试下一个签名")
    print("[warn] runner 导出没产出目标文件 → 改走 torch 直出")
    return False


def _export_torch(runner, obs_dim: int, out: Path, device: str) -> None:
    import torch

    actor = _find_actor(runner)
    if actor is None:
        raise RuntimeError("找不到 runner 里的 actor 模块，无法 torch.onnx.export；"
                           "可试 --engine runner 或检查 mjlab 版本")
    actor.eval()
    dummy = torch.zeros(1, obs_dim, device=device)
    with torch.no_grad():
        probe = actor(dummy)
    if isinstance(probe, (tuple, list)):
        class _First(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x)[0]

        actor = _First(actor).eval()
        print("[info] actor 返回 tuple，导出时只取第 0 项（确定性动作均值）")

    torch.onnx.export(
        actor, dummy, str(out),
        input_names=["obs"], output_names=["actions"],
        opset=17, do_constant_folding=True,
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    print(f"[OK] torch.onnx.export 完成：{out}")


def _verify(out: Path, obs_dim, runner, device: str) -> dict:
    """用 onnxruntime 与 get_inference_policy 对拍，防止导出到错的模块/丢归一化。"""
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
    except ImportError as e:
        print(f"[warn] 缺 {e.name}，跳过数值对拍；"
              f"建议 .venv/bin/pip install onnx onnxruntime")
        return {"verified": False, "reason": f"缺依赖：{e.name}"}

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = list(inp.shape)
    declared = shape[-1] if shape else None
    if obs_dim and isinstance(declared, int) and declared != obs_dim:
        return {"verified": False,
                "reason": f"ONNX 输入维度 {declared} != 训练观测 {obs_dim}"}
    print(f"[OK] ONNX 输入 {inp.name} {shape}，输出 "
          f"{[(o.name, list(o.shape)) for o in sess.get_outputs()]}")
    if not isinstance(obs_dim, int) and not isinstance(declared, int):
        return {"verified": False, "reason": "对拍异常：拿不到确定的观测维度"}

    try:
        policy = runner.get_inference_policy(device=device)
        rng = np.random.default_rng(0)
        x_np = rng.normal(0, 1, (1, obs_dim or declared)).astype("float32")
        x = torch.as_tensor(x_np, device=device)
        with torch.no_grad():
            ref = policy(x)
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
        ref_np = ref.detach().float().cpu().numpy()
        got_np = np.asarray(sess.run(None, {inp.name: x_np})[0], dtype="float32")
        err = float(np.max(np.abs(ref_np - got_np)))
        print(f"[info] 数值对拍 max|Δ| = {err:.3e}")
        return {"verified": err <= 1e-4, "max_abs_err": err,
                "reason": None if err <= 1e-4 else
                "输出对不上：多半是导到了不含 obs 归一化的模块，或 actor 选错"}
    except Exception as e:                                      # noqa: BLE001
        print(f"[warn] 数值对拍没跑成（{e!r}），只做了维度检查")
        return {"verified": False, "reason": f"对拍异常：{e!r}"}


def _write_contract(out: Path, task: str, ckpt, obs_dim, obs_terms,
                    engine: str, squat, verify: dict) -> Path:
    d = {
        "task": task,
        "checkpoint": str(ckpt),
        "onnx": str(out),
        "engine": engine,
        "obs_dim": obs_dim,
        "obs_terms_order": obs_terms,
        "obs_note": ACTOR_OBS_TERMS_NOTE,
        "action_dim": 29,
        "verify": verify,
    }
    if squat:
        d["action_semantics"] = (
            "ONNX 输出是 29 维**残差**，不是关节角。"
            "部署侧：raw = clip(a, -clip_rad/scale, +clip_rad/scale) * residual_scale; "
            "q_target = q_ref(frame) + ACTION_SCALE_MAP * raw。"
            "clip_rad：腿 0.35 / 腰 0.15 / 臂 0.12 rad，顺序与 ACTION_SCALE_MAP 一致")
        d["reference"] = squat["ref"]
        d["control_dt"] = squat["ref"]["dt"]
        d["deploy_notes"] = [
            "切模式时把参考帧计数器清零，每 20 ms 推进 1 帧，到 frames-1 停止推进",
            "拼接 actor 观测时 traj_joint_err / traj_height_err / traj_progress "
            "都从参考表当前帧算",
            "freeze_frame 非 null 时全程用该帧，不推进，也不给 traj_progress",
            "触发摔倒终止不要 reset：冻结当前关节目标 + 阻尼下电",
        ]
    else:
        d["action_semantics"] = (
            "ONNX 输出是 29 维动作；按该任务的 action 项做 scale + default_offset "
            "后才得到关节目标（见 crab_replay.py 的残差模式）")
    p = out.with_suffix(".json")
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="导出 mjlab 策略为 ONNX + 观测契约")
    ap.add_argument("--task", default="Elf3-Crab-Stairs", help="任务 ID")
    ap.add_argument("--squat", choices=sorted(SQUAT_MODES), default=None,
                    help="蹲起/保持任务快捷方式（自动找最新 checkpoint）")
    ap.add_argument("--checkpoint", default=None, help="model_XXXX.pt")
    ap.add_argument("--out", default=None, help="输出 .onnx 路径")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", choices=("auto", "runner", "torch"), default="auto")
    ap.add_argument("--obs-dim", type=int, default=None,
                    help="拿不到 env.reset() 时手工给观测维度")
    ap.add_argument("--skip-verify", action="store_true", help="跳过数值对拍")
    ap.add_argument("--no-contract", action="store_true", help="不写 .json 契约")
    args = ap.parse_args()

    try:
        task, out, ckpt, squat = _resolve(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 1
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = out.with_name(out.stem + ".prev.onnx")
        out.replace(prev)                      # 防止导出失败却把旧文件当成新产物
        print(f"[info] 旧产物已挪到 {prev.name}")

    import mjlab.tasks.crab  # noqa: F401  触发任务注册
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    env_cfg = load_env_cfg(task, play=True)
    obs_terms = []
    try:
        obs_terms = list(env_cfg.observations["actor"].terms.keys())
    except Exception:                                          # noqa: BLE001
        pass
    base = ManagerBasedRlEnv(env_cfg, device=args.device)
    obs_dim = args.obs_dim
    if obs_dim is None:
        try:
            r = base.reset()
            o = r[0] if isinstance(r, tuple) else r
            obs_dim = int(o.shape[-1])
        except Exception as e:                                  # noqa: BLE001
            print(f"[warn] env.reset() 拿不到观测维度（{e!r}）；"
                  f"runner 导出不受影响，torch 导出需要 --obs-dim")
    print(f"[info] task={task}  obs_dim={obs_dim}  观测顺序={obs_terms}")

    rl_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(
        base, clip_actions=getattr(rl_cfg, "clip_actions", None))
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = _build_runner(runner_cls, wrapped, rl_cfg, out, args.device)
    _load_ckpt(runner, ckpt, args.device)

    engine = None
    if args.engine in ("auto", "runner"):
        if _export_runner(runner, out):
            engine = "runner"
        elif args.engine == "runner":
            print("[ERR] runner 导出失败（--engine runner 不允许回退）", file=sys.stderr)
            return 3
    if engine is None:
        if obs_dim is None:
            print("[ERR] torch 直出需要观测维度：加 --obs-dim <N>", file=sys.stderr)
            return 4
        try:
            _export_torch(runner, obs_dim, out, args.device)
        except Exception as e:                                  # noqa: BLE001
            print(f"[ERR] torch.onnx.export 失败：{e!r}", file=sys.stderr)
            return 4
        engine = "torch"

    if not out.exists():
        print(f"[ERR] 导出后没找到 {out}", file=sys.stderr)
        return 2
    print(f"[OK] 导出 {out} ({out.stat().st_size / 1024:.1f} KB)，engine={engine}")

    verify = {"verified": None, "reason": "--skip-verify"}
    if not args.skip_verify:
        verify = _verify(out, obs_dim, runner, args.device)
        if verify.get("verified") is False and "缺依赖" not in str(verify.get("reason")) \
                and "对拍异常" not in str(verify.get("reason")):
            print(f"[ERR] 数值对拍没通过：{verify.get('reason')}；"
                  f"确认是误报可加 --skip-verify", file=sys.stderr)
            return 5

    if not args.no_contract:
        cj = _write_contract(out, task, ckpt, obs_dim, obs_terms, engine, squat, verify)
        print(f"[OK] 观测契约：{cj}")
    print("[next] 真机侧按契约拼观测；蹲起任务还要复现参考帧计数器与"
          " q_target = q_ref + scale * clip(a)，见 docs/squat_rl_training.md §7")
    return 0


if __name__ == "__main__":
    sys.exit(main())
