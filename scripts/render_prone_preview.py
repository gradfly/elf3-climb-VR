"""俯卧爬行动作预览：分阶段核对表 + 两段 GIF（离屏渲染，不需要 GUI）。

    python scripts/render_prone_preview.py                 # 默认 fast 档
    python scripts/render_prone_preview.py --preset normal

依赖：mujoco + Pillow；相机/模型取自 launch 目录，复用 crab_replay.make_cam。
产物写到 logs/prone_crawl/：
    a1_sheet_p1.png / a1_sheet_p2.png   动作 1 十个阶段 × (3/4 视角, 正侧视角)
    a1_standup_final.gif                动作 1 全程（每 5 帧取 1 帧）
    a2_gait_final.gif                   动作 2 全程（每 2 帧取 1 帧）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = (ROOT / "sim" / "vendor" / "bxi_controller_ros2" / "src"
          / "bxi_example_py_elf3" / "launch")
DATA = LAUNCH.parent / "data" / "mujoco_simulation"
OUT = ROOT / "logs" / "prone_crawl"
sys.path.insert(0, str(LAUNCH))
import crab_replay as CR  # noqa: E402

# v9 四阶段：站立 -> 蹲下 -> 双手撑地 -> 双脚向后伸 -> 趴姿。
# **手撑地（0.44）必须早于脚后伸（0.44~0.74）**，这是 v9 与 v8 的根本差别：
# v8 是脚提前后撤 + 边前倾边伸臂 + 腿长时间静止，观感不协调。
# 帧号按 250 帧 / 5.00 s 的 ENTRY_STAGES 关键归一化时刻选取（帧 = u × 249）。
STAGES = [
    (0,   "① 站立"),
    (27,  "② 下蹲"),
    (60,  "② 蹲底·脚不动"),
    (85,  "③ 手前伸下探"),
    (110, "③ 双手撑地"),
    (135, "④ 双脚后伸 1/3"),
    (159, "④ 双脚后伸 2/3"),
    (184, "④ 后伸到位·平板"),
    (219, "⑤ 胸腹下沉"),
    (249, "⑤ 俯卧爬行位"),
]


def _font(size: int):
    """尽量用中文字体，找不到就退回 PIL 默认字体。"""
    for p in (r"C:\Windows\Fonts\msyh.ttc",
              "/System/Library/Fonts/PingFang.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["normal", "fast"], default="fast")
    args = ap.parse_args()
    suf = "_fast" if args.preset == "fast" else ""
    OUT.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(str((DATA / "elf3_prone.xml").resolve()))
    # 出生点 site 会被渲染成一块黑椭圆，透明掉
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    q_a = np.load(DATA / f"prone_standup{suf}_traj.npz", allow_pickle=True)["qpos"]

    # ---- 分阶段核对表（两张，各 5 阶段 × 2 视角）----
    W, H = 480, 320
    views = [(105.0, -30.0, "3/4"), (90.0, -8.0, "side")]
    r = mujoco.Renderer(m, H, W)
    for part in range(2):
        stages = STAGES[part * 5:(part + 1) * 5]
        sh = Image.new("RGB", (W * len(stages), H * len(views) + 48), "white")
        dr = ImageDraw.Draw(sh)
        for row, (az, el, tag) in enumerate(views):
            for col, (i, label) in enumerate(stages):
                i = min(i, len(q_a) - 1)
                d.qpos[:] = q_a[i]
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                cam = CR.make_cam(lookat=d.xpos[torso], dist=1.9, az=az, el=el)
                r.update_scene(d, camera=cam)
                sh.paste(Image.fromarray(r.render()), (col * W, 24 + row * (H + 24)))
                dr.text((col * W + 8, 4 + row * (H + 24)),
                        f"{label}   t={i*0.02:.2f}s  [{tag}]",
                        fill="black", font=_font(15))
        path = OUT / f"a1_sheet_p{part+1}.png"
        sh.save(path)
        print("->", path)

    # ---- GIF ----
    gr = mujoco.Renderer(m, 270, 480)
    for name, out, every in (
        (f"prone_standup{suf}_traj.npz", "a1_standup_final.gif", 5),
        (f"prone_gait{suf}_traj.npz", "a2_gait_final.gif", 2),
    ):
        z = np.load(DATA / name, allow_pickle=True)
        q, dt = z["qpos"], float(z["dt"])
        imgs = []
        for i in range(0, len(q), every):
            d.qpos[:] = q[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cam = CR.make_cam(lookat=d.xpos[torso], dist=1.9, az=105.0, el=-30.0)
            gr.update_scene(d, camera=cam)
            imgs.append(Image.fromarray(gr.render()).convert(
                "P", palette=Image.ADAPTIVE, colors=128))
        path = OUT / out
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=int(dt * every * 1000), loop=0, optimize=True)
        print("->", path, len(imgs), "frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
