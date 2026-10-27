"""bake_motion.py —— 把 elf3 的 MuJoCo 仿真录制成网页可回放的运动数据

浏览器里的 MuJoCo WASM 不带策略（没有 onnxruntime），所以网页端要"看运动仿真"，
就得先在本地把仿真跑一遍、把每个控制步的 `qpos` 录下来，网页端只做插值回放。

录两种：
  * policy —— 真·运动仿真：`elf3_play.py` 的 MuJoCo + policy.onnx 闭环（含物理接触）。
  * ref    —— 参考动作：直接取 fight.npz 的 body_pos_w/body_quat_w/joint_pos，
              纯运动学装配（无物理），用来和策略仿真对照。

输出（写到 web/assets/）：
  motion_<out>.bin     float32 小端，(nframes, nq) 行优先
  motion_<out>.json    元数据（fps/nq/nframes/关节名/统计量）

用法：
  python web/tools/bake_motion.py                    # 整段 fight 动作（约 245 s）
  python web/tools/bake_motion.py --steps 1500       # 只录前 30 s
  python web/tools/bake_motion.py --only ref         # 只录参考动作
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent      # web/tools
WEB = HERE.parent                           # web/
ROOT = WEB.parent                           # 项目根
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import elf3_play as ep  # noqa: E402

ASSETS = WEB / "assets"
DECIMATION = 4          # elf3_play 里 timestep 0.005 × decimation 4 = 50 Hz
DT = 0.005 * DECIMATION


def save(name: str, frames: np.ndarray, meta: dict) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    bin_name = f"motion_{name}.bin"
    arr = np.ascontiguousarray(frames, dtype="<f4")
    (ASSETS / bin_name).write_bytes(arr.tobytes())
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = {
        **meta,
        "bin": bin_name,
        "generated": stamp,
        # 网页端拿它当缓存版本号：数据一重录，URL 就变
        "version": f"{stamp}-{arr.shape[0]}",
        "nframes": int(arr.shape[0]),
        "nq": int(arr.shape[1]),
        "duration": round(float(arr.shape[0]) / meta["fps"], 3),
        "qposFormat": "float32-le, row-major, (nframes, nq)",
    }
    (ASSETS / f"motion_{name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    size = (ASSETS / bin_name).stat().st_size
    print(f"[输出] {bin_name}: {arr.shape[0]} 帧 × {arr.shape[1]} = "
          f"{size / 1048576:.2f} MB，时长 {meta['duration']:.1f} s")
    print(f"[输出] motion_{name}.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fight", help="输出名前缀（默认 fight）")
    ap.add_argument("--steps", type=int, default=0, help="仿真步数（0 = 整段动作）")
    ap.add_argument("--joint-damping", choices=["xml", "zero"], default="xml")
    ap.add_argument("--only", choices=["both", "policy", "ref"], default="both")
    ap.add_argument("--stop-on", choices=["fall", "tracking", "never"], default="fall",
                    help="何时截断录制：fall=真摔倒（默认，适合观看）；"
                         "tracking=mjlab 严格跟踪判据；never=录满")
    ap.add_argument("--print-every", type=int, default=1000)
    args = ap.parse_args()

    t_start = time.time()

    params = ep.load_params_from_onnx(ROOT / "policy.onnx")
    motion = ep.Motion(ROOT / "fight.npz", params.joint_names, params.body_names)
    model = ep.build_model(params, scene="plane", joint_damping=args.joint_damping)
    jqadr = np.array([model.jnt_qposadr[ep._jid(model, n)] for n in params.joint_names])
    nq = int(model.nq)

    steps = args.steps if args.steps > 0 else motion.num_frames
    steps = min(steps, motion.num_frames)
    print(f"[信息] 模型 nq={nq} 关节={len(params.joint_names)} | 参考动作 {motion.num_frames} 帧 "
          f"({motion.duration:.1f} s @ {motion.fps:.0f} fps) | 本次录 {steps} 步")

    common = {
        "fps": motion.fps,
        "dt": DT,
        "model": "elf3.xml",
        "motion": "fight.npz",
        "jointNames": list(params.joint_names),
        "jointQposAdr": [int(a) for a in jqadr],
        "anchorBody": params.anchor_body,
        "bodyNames": list(params.body_names),
    }

    # ---------------------------------------------------------------- 参考动作
    if args.only in ("both", "ref"):
        ref = np.zeros((steps, nq), dtype=np.float64)
        ref[:, 0:3] = motion.body_pos_w[:steps, 0]
        ref[:, 3:7] = motion.body_quat_w[:steps, 0]
        ref[:, jqadr] = motion.joint_pos[:steps]
        save(f"{args.out}_ref", ref, {
            **common,
            "name": f"{args.out}_ref",
            "kind": "ref",
            "label": "参考动作（fight.npz 运动学）",
            "source": "fight.npz（重定向后的参考运动，无物理）",
        })

    # ---------------------------------------------------------------- 策略仿真
    if args.only in ("both", "policy"):
        player = ep.Elf3Player(model, params, motion,
                               policy_path=ROOT / "policy.onnx", policy="onnx")
        frames = np.zeros((steps + 1, nq), dtype=np.float64)
        player.reset_to_frame(0)
        frames[0] = player.data.qpos
        err_anchor, err_body, root_z, tilt_deg = [], [], [], []
        stopped = steps
        tracking_violations = 0
        for i in range(steps):
            player._frame = i                      # failed() 要读当前参考帧
            player.step(i)                         # 一个控制步 = 20 ms
            frames[i + 1] = player.data.qpos
            e = player.tracking_error(i)
            err_anchor.append(e["anchor_pos"])
            err_body.append(e["body_pos"])
            root_z.append(float(player.data.qpos[2]))
            tilt_deg.append(e["tilt_deg"])
            if player.failed():
                tracking_violations += 1
            if (i + 1) % args.print_every == 0 or i == steps - 1:
                el = time.time() - t_start
                print(f"  {i + 1:6d}/{steps} 步  t={(i + 1) * DT:6.1f}s  "
                      f"anchor误差={np.mean(err_anchor[-args.print_every:]):.3f} m  "
                      f"倾角={tilt_deg[-1]:5.1f}°  根高={player.data.qpos[2]:.2f}  "
                      f"  {el:.0f}s", flush=True)
            # 截断判据
            fell = player.data.qpos[2] < 0.55 or tilt_deg[-1] > 60.0
            if args.stop_on == "fall" and fell:
                stopped = i + 1
                print(f"  [警告] 第 {i + 1} 步机器人摔倒（根高 {player.data.qpos[2]:.2f} m / "
                      f"倾角 {tilt_deg[-1]:.0f}°），提前截断录制")
                break
            if args.stop_on == "tracking" and player.failed():
                stopped = i + 1
                print(f"  [警告] 第 {i + 1} 步触发 mjlab 跟踪终止条件，提前截断录制")
                break
        frames = frames[:stopped + 1]
        # 收尾修剪：末尾可能是「正在摔倒」的几帧，回退到最后一个姿态健康的时刻
        #（倾角 < 30°、根高 > 0.80 m），再留 0.5 s 余量，避免回放里出现倒地画面。
        healthy = [i for i, (t, z) in enumerate(zip(tilt_deg, root_z))
                   if t < 30.0 and z > 0.80]
        cut = (healthy[-1] + 1 - 25) if healthy else 0
        cut = max(1, min(cut, stopped))
        trimmed = len(frames) - (cut + 1)
        if trimmed > 0:
            print(f"  [修剪] 去掉末尾 {trimmed} 帧（含倒地过程），保留 {cut + 1} 帧")
        frames = frames[:cut + 1]
        errs = np.array(err_anchor[:cut])
        body_errs = np.array(err_body[:cut])
        tilt_arr = np.array(tilt_deg[:cut])
        rootz_arr = np.array(root_z[:cut])
        save(f"{args.out}_policy", frames, {
            **common,
            "name": f"{args.out}_policy",
            "kind": "policy",
            "label": "策略仿真（MuJoCo + policy.onnx）",
            "source": f"elf3_play.py 闭环仿真（joint_damping={args.joint_damping}）",
            "steps": int(cut),
            "truncated": bool(cut < steps),
            "stopOn": args.stop_on,
            "trackingViolationFrames": int(tracking_violations),
            "stats": {
                "anchorErrMean": round(float(errs.mean()), 4),
                "anchorErrP95": round(float(np.percentile(errs, 95)), 4),
                "anchorErrMax": round(float(errs.max()), 4),
                "bodyErrMean": round(float(body_errs.mean()), 4),
                "tiltMax": round(float(tilt_arr.max()), 2),
                "rootZMin": round(float(rootz_arr.min()), 4),
                "rootZMax": round(float(rootz_arr.max()), 4),
            },
        })

    print(f"[完成] 用时 {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
