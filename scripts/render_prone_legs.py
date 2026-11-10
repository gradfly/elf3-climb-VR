"""动作 1 后半段关键帧大图（5 帧 × 3 视角），肉眼核对腿的姿态。

帧号对应 v9 四阶段的归一化时刻（250 帧）：0.24 蹲底 / 0.44 双手撑地 /
0.54 脚后伸 1/3 / 0.74 平板位 / 1.00 俯卧。重点看 0.44 之前脚有没有提前
后撤、0.44~0.74 腿有没有走成「蛙式」（两膝挤到中线、小腿外翘、脚离地）。

    python scripts/render_prone_legs.py            # 默认 fast 档
    python scripts/render_prone_legs.py normal     # normal 档
    python scripts/render_prone_legs.py fast --frames=32,75,120   # 自定义帧号

产物：logs/prone_crawl/a1_legs_big.png（3/4、正俯视、侧视各一行）。
数值验收只查穿地不查离地，改后半段时必须同时看这张图或 GIF。
"""
import sys
import numpy as np
import mujoco
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch"
DATA = LAUNCH.parent / "data/mujoco_simulation"
OUT = ROOT / "logs/prone_crawl"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(LAUNCH))
import crab_replay as CR

_argv = sys.argv[1:]
_pre = next((a for a in _argv if not a.startswith("--")), "fast")
_fr = next((a for a in _argv if a.startswith("--frames")), None)
if _fr:
    _frames = [int(v) for v in _fr.split("=", 1)[1].split(",")]
else:                                   # 默认取 v8 五个阶段关键帧
    _frames = [60, 110, 135, 184, 249]
TR = {"fast": "prone_standup_fast_traj.npz",
      "normal": "prone_standup_traj.npz"}.get(_pre, _pre)
z = np.load(DATA / TR, allow_pickle=True)
q, dt = z["qpos"], float(z["dt"])
mm = mujoco.MjModel.from_xml_path(str((DATA / "elf3_prone.xml").resolve()))
mm.site_rgba[:, 3] = 0.0
dd = mujoco.MjData(mm)
torso = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
W, H = 560, 400
r = mujoco.Renderer(mm, H, W)
sel = _frames
views = [(115.0, -35.0, "3/4"), (90.0, -75.0, "top"), (90.0, -6.0, "side")]
sh = Image.new("RGB", (W * len(sel), (H + 26) * len(views)), "white")
dr = ImageDraw.Draw(sh)
for row, (az, el, vt) in enumerate(views):
    for col, i in enumerate(sel):
        dd.qpos[:] = q[i]
        mujoco.mj_kinematics(mm, dd)
        mujoco.mj_comPos(mm, dd)
        cam = CR.make_cam(lookat=dd.xpos[torso], dist=1.75, az=az, el=el)
        r.update_scene(dd, camera=cam)
        y = row * (H + 26)
        sh.paste(Image.fromarray(r.render()), (col * W, y + 24))
        dr.text((col * W + 10, y + 5), f"f{i}  u={i / max(1, len(q) - 1):.2f}  [{vt}]", fill="black")
path = OUT / "a1_legs_big.png"
sh.save(path)
print("->", path)
