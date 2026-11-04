#!/usr/bin/env python3
"""把蹲起 / 保持任务的策略回放导出成 mp4（无窗口、不依赖 play --video）。

为什么单独写一个
----------------
`play --video` 依赖三样东西同时成立：tyro 的 bool flag 写法（`--video` vs
`--video True` 随 mjlab 版本变）、一个能自己往前跑的 viewer（无 DISPLAY 时
viser 会停在等待连接，永远录不满 video_length）、以及 VideoRecorder 的
step_trigger 恰好被 wrapper 触发。任何一环不对，表现都是「mp4 没生成」，
而报错藏在 play 的输出里、很难定位。

本脚本直接建 `ManagerBasedRlEnv(render_mode="rgb_array")`、自己跑
`env.step(policy(obs))` 并逐帧 `base.render()`，用流式 writer 写 mp4：
  * 不开任何 viewer，不需要 DISPLAY（MUJOCO_GL=egl 即可）；
  * 不解析 tyro 参数，不受 mjlab 版本差异影响；
  * 帧边渲染边写，内存恒定（攒 1000 帧 1280x720 要 2.7 GB，会 OOM）；
  * 每 100 步打印进度，渲染失败会明确报错而不是静默产出空文件。

接口照抄仓库里已在服务器跑通的 scripts/diagnose_policy.py。

用法（服务器、仓库根目录）
------------------------
    MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/python \
        scripts/export_squat_video.py --mode down

    # 指定 checkpoint / 改时长 / 改机位 / 加速
    MUJOCO_GL=egl .venv/bin/python scripts/export_squat_video.py \
        --mode hold-squat --steps 6000 --every 2 --out ~/hold.mp4

    # 完全自定义（不用 --mode）
    MUJOCO_GL=egl .venv/bin/python scripts/export_squat_video.py \
        --task Elf3-Squat-Up-Play --checkpoint <path>/model_4000.pt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# mode -> (任务 ID, 实验目录, 默认步数)
# 步数是控制步（50 Hz），不是秒：1000 步 = 20 s。
MODES = {
    "down":        ("Elf3-Squat-Down-Play",     "elf3_squat_down",        1000),
    "up":          ("Elf3-Squat-Up-Play",       "elf3_squat_up",          1000),
    "hold-squat":  ("Elf3-Squat-Hold-Play",     "elf3_squat_hold_squat",  6000),
    "hold-stand":  ("Elf3-Stand-Hold-Play",     "elf3_squat_hold_stand",  6000),
}


def _latest_checkpoint(exp: str) -> str:
    """找该实验目录下最新 run 里迭代数最大的完好 checkpoint。"""
    root = Path("logs") / "rsl_rl" / exp
    if not root.is_dir():
        raise FileNotFoundError(
            f"没有实验目录 {root}。先在仓库根目录跑训练，或用 --checkpoint 指定路径。")

    def _best_in(d: Path):
        pts = []
        for p in d.glob("model_*.pt"):
            m = re.search(r"model_(\d+)\.pt$", p.name)
            # 训练被 kill 时正在写盘的 checkpoint 会是空/截断文件，必须跳过
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


def _unwrap(env):
    seen = 0
    while hasattr(env, "unwrapped") and seen < 5:
        inner = env.unwrapped
        if inner is env:
            break
        env = inner
        seen += 1
    return env


def _bind_camera(base, env_i: int, dist: float, elev: float, azim: float):
    """把离屏相机设成跟拍机器人；设不了跟踪相机就返回 (cam, False) 逐帧改 lookat。

    照抄 diagnose_policy.py::_bind_camera_to_robot，但拿不到渲染器时只警告、
    不放弃录像（用默认机位至少能出片）。
    """
    try:
        import mujoco
    except ImportError:
        return None, True

    r = getattr(base, "_offline_renderer", None)
    cam = getattr(r, "_cam", None) if r is not None else None
    if cam is None:
        print("[warn] 拿不到离屏渲染器相机，用默认机位继续录（画面可能不是跟拍）")
        return None, True

    cam.distance, cam.elevation, cam.azimuth = float(dist), float(elev), float(azim)
    try:
        body_id = int(base.scene["robot"].indexing.root_body_id)
        cam.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING.value)
        cam.fixedcamid = -1
        cam.trackbodyid = body_id
        print(f"[info] 相机：跟拍 env{env_i} root body {body_id} "
              f"(dist={dist} elev={elev} azim={azim})")
        return cam, True
    except Exception as e:
        print(f"[warn] 跟踪相机设不上（{e!r}），退化为逐帧 lookat")
        cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE.value)
        cam.trackbodyid = -1
        return cam, False


def _root_xyz(base, env_i: int):
    try:
        pos = base.scene["robot"].data.root_pos_w
        return np.asarray(pos[env_i].detach().cpu().tolist(), dtype=float)
    except Exception:
        return None


class FrameSink:
    """流式写 mp4；拿不到 ffmpeg 时退化为攒帧，最后用 mediapy/imageio 一次写。"""

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = max(1, int(round(fps)))
        self.kind = None
        self.obj = None
        self.frames: list[np.ndarray] = []
        self.n = 0

    def _init(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        try:  # 首选：imageio-ffmpeg 逐帧写，内存恒定
            import imageio.v2 as iio
            self.obj = iio.get_writer(
                str(self.path), fps=self.fps, codec="libx264",
                quality=8, macro_block_size=1)
            self.kind = "imageio"
            print(f"[info] mp4 writer = imageio/ffmpeg（流式，{w}x{h} @ {self.fps} fps）")
            return
        except Exception as e:
            print(f"[warn] imageio 流式写不可用（{e!r}），试 opencv")
        try:  # 次选：opencv 自带 ffmpeg
            import cv2
            vw = cv2.VideoWriter(str(self.path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(self.fps), (w, h))
            if vw.isOpened():
                self.obj, self.kind = vw, "cv2"
                print(f"[info] mp4 writer = opencv（流式，{w}x{h} @ {self.fps} fps）")
                return
            vw.release()
        except Exception as e:
            print(f"[warn] opencv 不可用（{e!r}）")
        # 兜底：攒帧（会占内存，打印警告）
        self.kind = "list"
        print(f"[warn] 没有流式 mp4 writer，改为攒帧后一次写："
              f"{w}x{h} 每帧约 {frame.nbytes / 1024:.0f} KB，帧多时可能吃内存")

    def add(self, frame) -> None:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            print(f"[warn] 跳过异常帧 shape={arr.shape}")
            return
        if self.kind is None:
            self._init(arr)
        if self.kind == "imageio":
            self.obj.append_data(arr)
        elif self.kind == "cv2":
            import cv2
            self.obj.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        else:
            self.frames.append(arr)
        self.n += 1

    def close(self) -> bool:
        """返回是否真的产出了 mp4。"""
        if self.kind == "imageio":
            self.obj.close()
        elif self.kind == "cv2":
            self.obj.release()
        elif self.kind == "list" and self.frames:
            frames = np.stack(self.frames)
            ok = False
            try:
                import mediapy as media
                media.write_video(str(self.path), frames, fps=self.fps)
                ok = True
            except Exception as e:
                print(f"[warn] mediapy 写 mp4 失败（{e!r}），试 imageio")
            if not ok:
                try:
                    import imageio.v2 as iio
                    iio.mimwrite(str(self.path), frames, fps=self.fps,
                                 macro_block_size=1)
                    ok = True
                except Exception as e:
                    print(f"[warn] imageio 写 mp4 失败（{e!r}）")
            if not ok:
                npz = str(self.path) + ".npz"
                np.savez_compressed(npz, frames=frames)
                print(f"[warn] mp4 写不出（缺 ffmpeg？pip install imageio-ffmpeg）"
                      f"，已把 {self.n} 帧存到 {npz}")
                return False
        return self.n > 0 and self.path.exists() and self.path.stat().st_size > 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=sorted(MODES), default="down",
                    help="四个蹲起/保持任务，自动带出任务 ID、实验目录、默认步数")
    ap.add_argument("--task", default=None, help="覆盖 --mode 的任务 ID")
    ap.add_argument("--checkpoint", default=None,
                    help="不指定就自动找该实验目录下最新 run 里迭代数最大的完好 pt")
    ap.add_argument("--out", default=None, metavar="OUT.mp4")
    ap.add_argument("--steps", type=int, default=None, help="跑多少控制步（默认按 mode）")
    ap.add_argument("--every", type=int, default=1, help="每 N 步取一帧（2 = 半速采样省时间）")
    ap.add_argument("--fps", type=float, default=None,
                    help="输出帧率，默认 = 控制频率 / --every，保证时长真实")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--record-env", type=int, default=0, help="录哪个 env 的画面")
    ap.add_argument("--dist", type=float, default=3.0)
    ap.add_argument("--elev", type=float, default=-45.0)
    ap.add_argument("--azim", type=float, default=200.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--zero", action="store_true",
                    help="不加载策略、零动作（对照纯参考开环）")
    args = ap.parse_args(argv)

    task, exp, def_steps = MODES[args.mode]
    if args.task:
        task = args.task
    steps = int(args.steps if args.steps is not None else def_steps)
    every = max(1, int(args.every))

    # checkpoint
    ckpt = args.checkpoint or _latest_checkpoint(exp)
    if not args.zero:
        sz = os.path.getsize(ckpt) if os.path.exists(ckpt) else -1
        if sz < 4096:
            print(f"[ERR] checkpoint 只有 {sz} 字节，是训练中断时写坏的：{ckpt}")
            return 2

    if args.out:
        out = Path(os.path.expanduser(args.out))
    else:
        tag = os.path.basename(os.path.dirname(ckpt)) or "run"
        tag = re.sub(r"[^A-Za-z0-9._-]", "_", tag)
        out = Path.home() / f"squat_{args.mode.replace('-', '_')}_{tag}_{Path(ckpt).stem}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 建环境（接口照抄 diagnose_policy.py，服务器上已跑通）----
    import mjlab.tasks  # noqa: F401  触发任务注册
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        env_cfg = load_env_cfg(task, play=True)
        agent_cfg = load_rl_cfg(task)
    except KeyError:
        from mjlab.tasks.registry import list_tasks
        have = sorted(t for t in list_tasks() if "Squat" in t or "Stand" in t)
        print(f"[ERR] 任务 {task!r} 没注册。当前相关任务：{have or '（一个都没有）'}")
        print("      先跑：bash train/install_task.sh")
        return 2

    env_cfg.scene.num_envs = args.num_envs
    try:
        env_cfg.viewer.width = args.width
        env_cfg.viewer.height = args.height
    except Exception as e:
        print(f"[warn] 设离屏分辨率失败（{e!r}），用配置默认值")
    try:
        env_cfg.observations["actor"].enable_corruption = False
    except Exception:
        pass

    print(f"[info] 任务={task}  ckpt={'<zero>' if args.zero else ckpt}")
    print(f"[info] device={device}  env={args.num_envs}  steps={steps}  "
          f"每 {every} 步一帧  {args.width}x{args.height}")
    print(f"[info] 输出={out}")

    # render_mode 必须是 rgb_array，否则 base.render() 返回 None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = _unwrap(env)
    dt = float(getattr(base, "step_dt", 0.02) or 0.02)
    fps = args.fps if args.fps is not None else (1.0 / dt) / every
    print(f"[info] step_dt={dt:.4f}s -> 输出 {fps:.2f} fps，"
          f"预计时长 {steps / every / fps:.1f} s")

    if args.zero:
        shape = env.action_space.shape

        class _Zero:
            def __call__(self, obs):
                del obs
                return torch.zeros(shape, device=device)

        policy = _Zero()
    else:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        try:  # 与 diagnose_policy.py 一致（项目里在服务器上跑通的签名）
            runner.load(str(ckpt), load_cfg={"actor": True}, strict=True,
                        map_location=device)
        except Exception as e:  # 新版 rsl_rl / play.py 的标准签名
            print(f"[warn] load(load_cfg=...) 失败（{e!r}），退回标准签名")
            runner.load(str(ckpt), map_location=device)
        policy = runner.get_inference_policy(device=device)
        print("[info] 策略加载完成")

    cam, tracks = _bind_camera(base, args.record_env,
                               dist=args.dist, elev=args.elev, azim=args.azim)
    sink = FrameSink(out, fps)

    out_reset = env.reset()
    obs = out_reset[0] if isinstance(out_reset, tuple) else out_reset

    t0 = time.time()
    render_err = 0
    try:
        for i in range(steps):
            with torch.no_grad():
                actions = policy(obs)
            step_out = env.step(actions)
            obs = step_out[0]

            if i % every == 0:
                if cam is not None and not tracks:
                    xyz = _root_xyz(base, args.record_env)
                    if xyz is not None:
                        try:
                            cam.lookat[:] = xyz.tolist()
                        except Exception:
                            pass
                try:
                    fr = base.render()
                    if fr is not None:
                        sink.add(fr[args.record_env]
                                 if getattr(fr, "ndim", 0) == 4 else fr)
                except Exception as e:
                    render_err += 1
                    if render_err <= 2:
                        print(f"[warn] 第 {i} 步渲染失败：{e!r}")
                    if render_err >= 10:
                        print("[ERR] 连续渲染失败，提前停止。检查 MUJOCO_GL=egl 与显存")
                        break

            if (i + 1) % 100 == 0 or i + 1 == steps:
                el = time.time() - t0
                print(f"[info] {i + 1}/{steps} 步，已录 {sink.n} 帧，"
                      f"耗时 {el:.1f}s，预计剩余 {el / (i + 1) * (steps - i - 1):.0f}s",
                      flush=True)
    finally:
        ok = sink.close()
        try:
            env.close()
        except Exception:
            pass

    if not ok or sink.n == 0:
        print("[ERR] 一帧都没录到。检查：render_mode='rgb_array' 是否生效、"
              "MUJOCO_GL=egl、显存是否被训练占满。")
        return 3
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[OK] mp4 已导出：{out}  ({sink.n} 帧 @ {sink.fps} fps，"
          f"{sink.n / sink.fps:.1f} s，{size_mb:.1f} MB，"
          f"总耗时 {time.time() - t0:.1f}s)")
    if render_err:
        print(f"[warn] 有 {render_err} 步渲染失败被跳过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
