"""render_salute_shift_cmp.py — 敬礼「右腕向右平移量」对比图。

一个动作参数（``wrist_right``）会明显改变姿态观感：手位每往外挪 5 cm，
肩就必须多外展、肘从「抬起」滑向「侧平举」。这个脚本把几档平移量并排
渲染成一张图，方便一眼挑参数，而不是盲调数字。

跑法（项目根目录）：
    python scripts/render_salute_shift_cmp.py
    python scripts/render_salute_shift_cmp.py --shifts 0 0.03 0.06 0.10
    python scripts/render_salute_shift_cmp.py --az 145 --out logs/xxx.png

不写 npz、不碰 DATA 目录：轨迹只在内存里生成。
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
LAUNCH = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
          / "bxi_example_py_elf3" / "launch")
sys.path.insert(0, str(LAUNCH))
sys.path.insert(0, str(REPO / "scripts"))

import salute as S                                  # noqa: E402
from action_lib import IK                           # noqa: E402
import render_parade_actions as R                   # noqa: E402


def font(size: int):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shifts", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10],
                    help="右腕沿机器人右侧的水平平移量（米）")
    ap.add_argument("--az", type=float, default=180.0, help="相机方位角（180=正面）")
    ap.add_argument("--fovy", type=float, default=26.0)
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=680)
    ap.add_argument("--out", default="logs/parade_actions/salute_wrist_shift_cmp.png")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(str(R.SCENE))
    d = mujoco.MjData(m)
    m.vis.global_.fovy = args.fovy
    ik = IK()
    tiles: list[tuple[str, np.ndarray]] = []
    for s in args.shifts:
        p = S.Params(wrist_right=s)
        qpos, info = S.build(p, ik)
        k = int(round(2.35 / p.dt))               # 敬礼保持段中央
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        cam = R.camera(args.az, 2.35, 1.30, elevation=-4.0)
        r = mujoco.Renderer(m, height=args.height, width=args.width)
        r.update_scene(d, cam)
        img = r.render().copy()
        r.close()
        lab = (f"右移 {s*100:.0f} cm   腕 y = {info['wrist_y_m']:+.3f} m   "
               f"距头中心 {info['wrist_to_head_mm']:.0f} mm   "
               f"左臂竖直度 {info['left_forearm_down']:.3f}")
        tiles.append((lab, img))
        print(f"  [{s*100:5.1f} cm] 腕 y={info['wrist_y_m']:+.3f}  "
              f"距头 {info['wrist_to_head_mm']:.0f} mm  肘角 "
              f"{info['salute_joints_deg']['r_elbow_y']:+.1f}°")

    W, H, pad, lab_h = args.width, args.height, 8, 30
    sheet = Image.new("RGB", (len(tiles) * (W + pad) + pad, H + lab_h + 2 * pad),
                      (247, 248, 250))
    dr = ImageDraw.Draw(sheet)
    f = font(17)
    for i, (lab, img) in enumerate(tiles):
        x = pad + i * (W + pad)
        dr.text((x + 4, 6), lab, fill=(24, 32, 48), font=f)
        sheet.paste(Image.fromarray(img), (x, lab_h + pad))
    out = (REPO / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\n[输出] {out}  {sheet.width}x{sheet.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
