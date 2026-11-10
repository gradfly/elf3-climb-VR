"""render_salute_palm_cmp.py — 敬礼「右腕滚转角（掌心朝向）」对比图。

敬礼的掌心朝向由 ``wrist_roll_deg``（腕绕前臂轴 r_wrist_x 的滚转）决定。
手 mesh 在 ``r_wrist_z_link`` 里的局部轴是 **+x = 四指方向、+z = 掌心法向**，
所以绕 +x 滚转时掌心在 y-z 平面上转：

* 0°    掌心朝下（−0.92）、手掌宽面正对观众 —— 看着就是"手心朝前"；
* +90°  掌心朝后（−0.97）、掌缘朝前、四指斜指太阳穴 —— 军礼的标准立掌；
* −90°  掌心反过来正朝前（+0.92）—— 要避免的方向。

脚本把几档滚转角 × 几个机位并排渲染，标题里带上实测的掌心法向 / 手指方向。

跑法（项目根目录）：
    python scripts/render_salute_palm_cmp.py
    python scripts/render_salute_palm_cmp.py --rolls 0 90
    python scripts/render_salute_palm_cmp.py --views 180 135 --out logs/xxx.png

不写 npz、不碰 DATA 目录：轨迹只在内存里生成。
"""
from __future__ import annotations

import argparse
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

# 方位角约定同 render_parade_actions：campos = lookat + d·(−cos az, −sin az, sin el)
VIEWS = [(180.0, -4.0, "正面"), (135.0, 12.0, "前 3/4"), (90.0, 0.0, "右侧")]


def font(size: int):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rolls", type=float, nargs="+", default=[0.0, 90.0, -90.0],
                    help="右腕绕前臂轴的滚转角（度）")
    ap.add_argument("--views", type=float, nargs="+", default=None,
                    help="只渲这几个方位角（180=正面 / 135=前 3/4 / 90=右侧）")
    ap.add_argument("--fovy", type=float, default=34.0)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--out", default="logs/parade_actions/salute_palm_roll_cmp.png")
    args = ap.parse_args()

    views = VIEWS if not args.views else [v for v in VIEWS if v[0] in args.views]
    if not views:
        raise SystemExit(f"--views 只能是 {[v[0] for v in VIEWS]}")

    m = mujoco.MjModel.from_xml_path(str(R.SCENE))
    d = mujoco.MjData(m)
    m.vis.global_.fovy = args.fovy
    ik = IK()
    rows: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
    for roll in args.rolls:
        p = S.Params(wrist_roll_deg=roll)
        qpos, info = S.build(p, ik)
        d.qpos[:] = qpos[int(info["salute_hold_frame"])]
        mujoco.mj_forward(m, d)
        hc = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_y_link")].copy()
        wp = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_wrist_y_link")].copy()
        tgt = (wp + np.array([0.0, 0.0, 0.03])) * 0.65 + hc * 0.35
        imgs = []
        for az, el, vl in views:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 0.78
            cam.azimuth = az
            cam.elevation = el
            cam.lookat[:] = tgt
            r = mujoco.Renderer(m, height=args.height, width=args.width)
            r.update_scene(d, cam)
            imgs.append((vl, r.render().copy()))
            r.close()
        pn, fd = info["palm_normal"], info["finger_dir"]
        lab = (f"滚 {roll:+.0f}°   掌心 ({pn[0]:+.2f},{pn[1]:+.2f},{pn[2]:+.2f})   "
               f"四指 ({fd[0]:+.2f},{fd[1]:+.2f},{fd[2]:+.2f})")
        rows.append((lab, imgs))
        print(f"  [{roll:+6.1f}°] 掌心 x(前) {pn[0]:+.2f}  z(下) {pn[2]:+.2f}   "
              f"四指 y(内) {fd[1]:+.2f}")

    W, H, pad, lab_h = args.width, args.height, 8, 30
    sheet = Image.new("RGB",
                      (len(views) * (W + pad) + pad,
                       len(rows) * (lab_h + H + pad) + pad), (247, 248, 250))
    dr = ImageDraw.Draw(sheet)
    f = font(19)
    for ri, (lab, imgs) in enumerate(rows):
        y = pad + ri * (lab_h + H + pad)
        dr.text((pad + 4, y + 4), lab, fill=(20, 30, 50), font=f)
        for ci, (vl, img) in enumerate(imgs):
            x = pad + ci * (W + pad)
            sheet.paste(Image.fromarray(img), (x, y + lab_h))
            dr.text((x + 6, y + lab_h + 3), vl, fill=(170, 60, 40), font=font(16))
    out = (REPO / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\n[输出] {out}  {sheet.width}x{sheet.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
