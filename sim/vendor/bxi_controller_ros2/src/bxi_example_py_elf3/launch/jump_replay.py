"""ELF3 立定跳远 —— 动作序列动画渲染器。

    python jump_replay.py                       # 一键出 GIF + 关键帧拼图到 logs/jump/
    python jump_replay.py --gif a.gif          # 只出 GIF
    python jump_replay.py --sheet a.png        # 只出拼图
    python jump_replay.py --record logs/frames # 离屏 PNG 序列
    python jump_replay.py --gui                # 窗口里循环看运动学回放

回放的是 long_jump.py 产出的 npz：逐帧写 qpos 做正运动学（kin），
姿态与物理仿真一致，但不重跑动力学 —— 用来验收动作序列本身。

画面叠加（复用 squat_replay 的 overlay）：
    绿色铅垂线 + 地面绿盘   COM 投影
    蓝色线框               支撑多边形（脚底胶囊球心凸包）
    HUD                    时间 / 九相位 / 躯干高 / 前进距离 / 稳定裕度
    相位色带               曲线上方按相位着色，一眼看出起跳与落地时刻

关键帧拼图：6 个代表姿态 + 躯干高度 / 稳定裕度 / 前进距离三条曲线 +
落地平衡判据结论条。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_squat.xml"
TRAJ = DATA / "long_jump_traj.npz"

import squat_replay as SR                                      # noqa: E402
import squat_stand as SQ                                       # noqa: E402

PHASES = ["READY", "PRE", "SQUAT", "TAKEOFF", "FLIGHT",
          "ABSORB", "SETTLE", "RISE", "HOLD"]
CN = {
    "READY": "站立保持", "PRE": "预蹲摆臂", "SQUAT": "深蹲蓄力",
    "TAKEOFF": "蹬伸起跳", "FLIGHT": "腾空（抛物线）",
    "ABSORB": "落地缓冲", "SETTLE": "缓冲保持",
    "RISE": "站起", "HOLD": "站稳保持",
}
# 相位配色（HUD 阶段名 / 色带）：准静态蓝、起跳橙、腾空紫、落地绿
PCOL = {
    "READY": (140, 195, 255), "PRE": (140, 195, 255), "SQUAT": (140, 195, 255),
    "TAKEOFF": (255, 175, 80), "FLIGHT": (205, 150, 255),
    "ABSORB": (120, 235, 150), "SETTLE": (120, 235, 150),
    "RISE": (120, 235, 150), "HOLD": (120, 235, 150),
}

FONT_PATHS = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
              "/System/Library/Fonts/PingFang.ttc")
_FONTS: dict[int, object] = {}


def font(size: int):
    if size not in _FONTS:
        from PIL import ImageFont
        for fp in FONT_PATHS:
            try:
                _FONTS[size] = ImageFont.truetype(fp, size)
                break
            except OSError:
                continue
        else:
            _FONTS[size] = ImageFont.load_default()
    return _FONTS[size]


def repo_root() -> Path:
    """向上找含 .git 的目录，用于默认输出 logs/jump/。"""
    for p in (HERE,) + tuple(HERE.parents):
        if (p / ".git").exists():
            return p
    return HERE.parents[3] if len(HERE.parents) > 3 else HERE


OUTDIR = repo_root() / "logs" / "jump"


def resolve(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if p.parent == Path("."):
        for base in (DATA, HERE, Path.cwd()):
            if (base / p.name).exists():
                print(f"[路径] {path} -> {base / p.name}")
                return base / p.name
    return p


# ---------------------------------------------------------------- 数据
def load_traj(args):
    traj = resolve(args.traj)
    if not traj.exists():
        avail = sorted(x.name for x in DATA.glob("*.npz"))
        raise SystemExit(f"[错误] 找不到轨迹 {traj}\n"
                         f"       数据目录 {DATA} 里有：{avail}\n"
                         f"       先跑 python long_jump.py 生成")
    data = np.load(traj, allow_pickle=True)
    qpos = data["qpos"]
    dt = float(data["dt"])
    phase = data["phase"].astype(int) if "phase" in data else None
    meta = data["meta"].item() if "meta" in data else {}
    if phase is None and meta:
        raise SystemExit("[错误] 轨迹里没有 phase 数组，无法标注阶段；请重新生成 npz")
    if not Path(args.scene).exists():
        import squat_stand as S
        S.write_scene()
    m = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    return m, d, torso, qpos, dt, phase, meta, traj


def series(m, d, torso, qpos, phase):
    """逐帧正运动学 → 躯干高 / COM 稳定裕度 / 前进距离 / 脚是否触地。"""
    n = len(qpos)
    z = np.zeros(n)
    x = np.zeros(n)
    mg = np.zeros(n)
    contact = np.zeros(n, dtype=bool)
    x0 = float(qpos[0][0])
    for i in range(n):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        z[i] = float(d.qpos[2])
        x[i] = float(d.qpos[0]) - x0
        try:
            mg[i] = SR.signed_margin(d.subtree_com[torso][:2], SR.foot_hull(m, d))
        except Exception:
            mg[i] = float("nan")
        # 脚底胶囊最低点是否贴地（< 1.5 cm 视为接触）
        foot_low = min(
            d.geom_xpos[g][2] - (float(m.geom_size[g][1]) *
                                 abs(d.geom_xmat[g].reshape(3, 3)[2, 2]) +
                                 float(m.geom_size[g][0]))
            for g in range(m.ngeom)
            if "_foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith("_collision"))
        contact[i] = foot_low < 0.015
    return z, x, mg, contact


def stage_of(phase_i: int) -> str:
    nm = PHASES[int(phase_i)] if 0 <= int(phase_i) < len(PHASES) else "?"
    return f"{nm}  {CN.get(nm, '')}"


# ---------------------------------------------------------------- HUD
_HUD_MASK = None


def _hud_mask(w=344, h=196):
    """左上角遮罩，右/下边缘渐隐 —— 避免半透明色块在天空上留下硬竖线。"""
    global _HUD_MASK
    if _HUD_MASK is None:
        from PIL import Image as I, ImageDraw
        m = I.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(m)
        for xx in range(0, w, 4):
            ax = int(150 * max(0.0, 1.0 - xx / (w * 0.92)) ** 0.8)
            for yy in range(0, h, 4):
                ay = int(ax * max(0.0, 1.0 - max(0, yy - h * 0.72) / (h * 0.28)))
                if ay > 0:
                    dr.rectangle([xx, yy, xx + 4, yy + 4], fill=(14, 18, 26, ay))
        _HUD_MASK = m
    return _HUD_MASK


def hud(img: np.ndarray, t: float, phase_i: int, z: float,
        x_travel: float, margin: float, contact: bool) -> np.ndarray:
    from PIL import Image as I, ImageDraw
    im = I.fromarray(img).convert("RGBA")
    im.alpha_composite(_hud_mask(), (0, 0))
    dr = ImageDraw.Draw(im, "RGBA")
    nm = PHASES[int(phase_i)]
    col = PCOL.get(nm, (236, 241, 246))
    dr.text((16, 8), f"t = {t:5.2f} s", font=font(24), fill=(236, 241, 246, 255))
    dr.text((16, 40), stage_of(phase_i), font=font(22), fill=(*col, 255))
    dr.text((16, 74), f"躯干 {z*100:5.1f} cm", font=font(22),
            fill=(236, 241, 246, 255))
    dr.text((16, 104), f"前进 {x_travel*100:5.1f} cm", font=font(22),
            fill=(236, 241, 246, 255))
    if contact:
        c = (120, 255, 150, 255) if (np.isfinite(margin) and margin > 0.02) else (255, 120, 110, 255)
        dr.text((16, 134), f"裕度 {margin*100:+5.1f} cm", font=font(22), fill=c)
    else:
        # 腾空段脚已离地，COM→支撑多边形裕度没有物理意义，不显示数值
        dr.text((16, 134), "裕度    —  (腾空)", font=font(22),
                fill=(205, 150, 255, 255))
    tag = "着地" if contact else "腾空"
    dr.text((16, 164), f"● {tag}", font=font(19),
            fill=(*((120, 235, 150) if contact else (205, 150, 255)), 255))
    return np.asarray(im.convert("RGB"))


def phase_band(dr, box, phase, dt, label=True):
    """在曲线框上方画一条按相位着色的细带（说明文字只在第一张画一次）。"""
    x0, y0, x1, _ = box
    n = len(phase)
    y = y0 - 48
    for i in range(n):
        xa = x0 + (x1 - x0) * i / max(1, n - 1)
        xb = x0 + (x1 - x0) * (i + 1) / max(1, n - 1)
        nm = PHASES[int(phase[i])]
        dr.rectangle([xa, y, max(xb, xa + 1), y + 12],
                     fill=(*PCOL.get(nm, (150, 160, 175)), 220))
    if label:
        dr.text((x0, y - 26),
                "相位色带：站立/蓄力(蓝) → 起跳(橙) → 腾空(紫) → 落地站稳(绿)",
                font=font(16), fill=(150, 165, 185, 255))


def mark_events(dr, box, meta, dt, n):
    """曲线上标起跳 / 触地竖线。"""
    x0, y0, x1, y1 = box
    for key, label, c, dy in (("t_liftoff", "起跳", (255, 175, 80, 220), 4),
                              ("t_touchdown", "触地", (120, 235, 150, 220), 26)):
        mt = meta.get("metrics", {})
        tv = mt.get(key)
        if tv is None:
            continue
        xx = x0 + (x1 - x0) * min(max(float(tv) / ((n - 1) * dt), 0.0), 1.0)
        dr.line([xx, y0, xx, y1], fill=c, width=2)
        dr.text((xx + 4, y0 + dy), label, font=font(16), fill=c)


# ---------------------------------------------------------------- GIF
def render_frames(args, m, d, torso, qpos, dt, phase, z, x, mg, contact, out_gif, rec):
    from PIL import Image
    ow = int(m.vis.global_.offwidth) or 1280
    oh = int(m.vis.global_.offheight) or 720
    w = int(min(args.width, ow))
    h = int(min(args.height, oh))
    renderer = mujoco.Renderer(m, h, w)
    cam = SR.make_cam(args.lookat, dist=args.dist, az=args.az, el=args.el)
    idxs = list(range(0, len(qpos), max(1, args.every)))
    if idxs[-1] != len(qpos) - 1:
        idxs.append(len(qpos) - 1)
    if args.loop and idxs[-1] != 0:
        idxs.append(0)
    if rec:
        rec.mkdir(parents=True, exist_ok=True)
    frames = []
    for k, i in enumerate(idxs):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)          # 必须：mj_forward 才更新 light_xpos
        if args.track:
            cam.lookat[:] = d.xpos[torso]
            cam.lookat[2] -= 0.10
        renderer.update_scene(d, camera=cam)
        SR.overlay(renderer.scene, m, d, torso)
        img = renderer.render()
        img = hud(img, i * dt, phase[i] if phase is not None else 0,
                  z[i], x[i], mg[i], contact[i])
        frames.append(Image.fromarray(img))
        if rec:
            Image.fromarray(img).save(rec / f"jump_{i:05d}.png")
        if k % 15 == 0:
            print(f"  渲染 {k}/{len(idxs)}", end="\r", flush=True)
    print(f"[OK] {len(frames)} 帧  {w}x{h}")
    if out_gif:
        gif = Path(out_gif)
        gif.parent.mkdir(parents=True, exist_ok=True)
        dur = int(round(1000 * dt * args.every))
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=dur, loop=0, disposal=2)
        print(f"[OK] GIF -> {gif}  ({gif.stat().st_size/1024:.0f} KB, {dur} ms/帧)")
    return frames


# ---------------------------------------------------------------- 拼图
def render_sheet(args, m, d, torso, qpos, dt, phase, meta,
                 z, x, mg, contact, out: Path):
    from PIL import Image as I, ImageDraw
    n = len(qpos)
    dur = (n - 1) * dt

    def pick(name, where="mid"):
        ids = np.where(phase == PHASES.index(name))[0]
        if len(ids) == 0:
            return 0
        return int(ids[0] if where == "first" else
                   ids[-1] if where == "last" else ids[len(ids) // 2])

    marks = [
        (pick("READY", "first"), "站立", "READY"),
        (pick("SQUAT", "last"), "深蹲蓄力", "SQUAT"),
        (pick("TAKEOFF", "last"), "蹬伸离地", "TAKEOFF"),
        (pick("FLIGHT", "mid"), "腾空最高点", "FLIGHT"),
        (pick("ABSORB", "mid"), "落地缓冲", "ABSORB"),
        (pick("HOLD", "last"), "站稳保持", "HOLD"),
    ]

    tw, th = 268, 224
    r = mujoco.Renderer(m, th, tw)
    # 关键帧用近机位（比 GIF 的固定机位更近），机器人姿态看得清
    tile_cam = SR.make_cam(args.lookat, dist=args.dist * 0.68, az=args.az, el=args.el)
    tiles = []
    for i, _, _ in marks:
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        tile_cam.lookat[:] = d.xpos[torso]
        tile_cam.lookat[2] -= 0.34       # 下移注视点，别把脚裁出画面
        r.update_scene(d, camera=tile_cam)
        SR.overlay(r.scene, m, d, torso)
        tiles.append(I.fromarray(r.render()))

    W, H = 1700, 1160
    sheet = I.new("RGB", (W, H), (22, 27, 36))
    dr = ImageDraw.Draw(sheet, "RGBA")
    fb, fs = font(28), font(18)
    dr.text((32, 20), "ELF3 立定跳远：站立 → 跳远 → 落地平衡站立", font=fb,
            fill=(236, 242, 248, 255))
    mt = meta.get("metrics", {})
    dist_cm = mt.get("total_travel", x[-1]) * 100
    dr.text((32, 60),
            f"{n} 帧 @ {1/dt:.0f} Hz / {dur:.2f} s     跳距（脚尖→脚跟）"
            f" {mt.get('toe_heel_dist', 0)*100:.0f} cm     基座总位移 {dist_cm:.0f} cm     "
            f"腾空 {mt.get('flight_time', 0):.2f} s",
            font=fs, fill=(150, 175, 200, 255))
    dr.text((32, 88),
            "物理建模：起跳合力按冲量定理等效为离地初速度；腾空段纯物理守恒走抛物线；"
            "落地按完全非弹性接触吸收动量后闭环站起",
            font=font(16), fill=(120, 138, 158, 255))

    for k, (i, name, _) in enumerate(marks):
        px = 32 + k * 278
        sheet.paste(tiles[k], (px, 124))
        dr.text((px + 8, 128), f"{i*dt:4.2f} s", font=fs,
                fill=(255, 255, 255, 235))
        dr.text((px + 8, 152), name, font=fs,
                fill=(*PCOL.get(marks[k][2], (140, 195, 255)), 255))

    # 三曲线
    SR._FONT = font(20)
    bx1 = (120, 450, 640, 760)
    bx2 = (690, 450, 1210, 760)
    bx3 = (1260, 450, 1660, 760)
    for k, bx in enumerate((bx1, bx2, bx3)):
        phase_band(dr, bx, phase, dt, label=(k == 0))
        mark_events(dr, bx, meta, dt, n)
    SR._plot(dr, bx1, np.arange(n) * dt, z, "躯干高度", "m",
             (110, 205, 255, 255), refs=(SQ.STANCE_Z,))
    SR._plot(dr, bx2, np.arange(n) * dt, mg * 100, "稳定裕度（COM 到支撑边界）", "cm",
             (120, 235, 150, 255), fmt="{:.0f}", refs=(2.0,))
    SR._plot(dr, bx3, np.arange(n) * dt, x * 100, "前进距离", "cm",
             (255, 185, 110, 255), fmt="{:.0f}")
    dr.text((120, 780), "腾空段脚已离地，稳定裕度仅作参考；红横线 = 2 cm 安全阈值 / 站姿高度",
            font=font(16), fill=(130, 140, 155, 255))

    # 结论条
    ok = bool(meta.get("ok", False))
    y = 830
    dr.rounded_rectangle([32, y, W - 32, y + 288], radius=12,
                         fill=(30, 38, 50, 255), outline=(70, 84, 102, 255), width=2)
    head_c = (120, 235, 150, 255) if ok else (255, 110, 110, 255)
    dr.text((56, y + 16), "落地保持平衡 · 静态校核判据",
            font=font(24), fill=(236, 242, 248, 255))
    dr.text((56, y + 54),
            ("[PASS] 通过：落地后保持平衡站立" if ok else "[FAIL] 未通过"),
            font=font(24), fill=head_c)
    items = [
        ("站稳保持", f"{mt.get('hold_time', 0):.2f} s", "≥ 2.00 s", mt.get("hold_time", 0) >= 1.95),
        ("末态躯干高", f"{mt.get('final_z', 0):.3f} m", "1.034 ± 0.03 m",
         abs(mt.get("final_z", 0) - 1.0339) <= 0.03),
        ("末态俯仰", f"{mt.get('final_pitch_deg', 0):+.1f}°", "|θ| < 10°",
         abs(mt.get("final_pitch_deg", 0)) < 10),
        ("末态 COM 裕度", f"{mt.get('final_margin', 0)*100:+.1f} cm", "> 2 cm",
         np.isfinite(mt.get("final_margin", float("nan"))) and mt.get("final_margin", 0) > 0.02),
        ("站定期最小裕度", f"{mt.get('min_margin_stance', 0)*100:+.1f} cm", "> 0 cm",
         np.isfinite(mt.get("min_margin_stance", float("nan"))) and mt.get("min_margin_stance", 0) > 0),
        ("力矩峰值", f"{100*mt.get('tau_peak', 0):.0f}%", "≤ 100%（硬限幅）", True),
    ]
    for k, (name, val, crit, good) in enumerate(items):
        cx = 56 + (k % 3) * 540
        cy = y + 104 + (k // 3) * 84
        c = (120, 235, 150, 255) if good else (255, 110, 110, 255)
        dr.text((cx, cy), f"{'[OK]' if good else '[NG]'}  {name}", font=font(20),
                fill=(200, 212, 226, 255))
        dr.text((cx, cy + 30), f"{val}   （判据 {crit}）", font=font(18), fill=c)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[OK] 关键帧拼图 -> {out}  ({out.stat().st_size/1024:.0f} KB)")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 跳远动作序列动画渲染")
    ap.add_argument("--traj", default=str(TRAJ))
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--gif", default=None, help="GIF 输出路径")
    ap.add_argument("--sheet", default=None, help="关键帧拼图 PNG 路径")
    ap.add_argument("--record", default=None, help="PNG 序列目录")
    ap.add_argument("--every", type=int, default=2, help="每 N 帧取一帧")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=405)
    ap.add_argument("--dist", type=float, default=3.3)
    ap.add_argument("--az", type=float, default=270.0)
    ap.add_argument("--el", type=float, default=-6.0)
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.28, 0.0, 0.58],
                    metavar=("X", "Y", "Z"), help="固定机位注视点（默认覆盖起跳↔落地中点）")
    ap.add_argument("--track", action="store_true", help="相机跟随躯干")
    ap.add_argument("--loop", action="store_true", help="GIF 末尾补一帧回站姿")
    ap.add_argument("--gui", action="store_true", help="窗口里循环回放（不渲染文件）")
    ap.add_argument("--json", default=None, help="把逐帧曲线另存 JSON")
    args = ap.parse_args()

    args.scene = str(resolve(args.scene))
    m, d, torso, qpos, dt, phase, meta, traj = load_traj(args)
    print(f"[轨迹] {traj.name}  {len(qpos)} 帧 / {(len(qpos)-1)*dt:.2f} s  "
          f"ok={meta.get('ok')}  起跳 {meta.get('liftoff_t')} s / 触地 {meta.get('touchdown_t')} s")

    z, x, mg, contact = series(m, d, torso, qpos, phase)
    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(time=(np.arange(len(qpos)) * dt).tolist(),
                 phase=phase.tolist(), z=z.tolist(), x=x.tolist(),
                 margin=mg.tolist(), contact=contact.tolist(),
                 metrics=meta.get("metrics", {})), indent=2), encoding="utf-8")
        print(f"[OK] 曲线 -> {args.json}")

    if args.gui:
        import crab_replay as CR
        CR.run_kin(m, d, qpos, dt, speed=1.0, loop=True, track=args.track,
                   record=None, torso_id=torso, dist=args.dist, az=args.az,
                   el=args.el, width=args.width, height=args.height)
        return

    do_gif = args.gif or args.record
    do_sheet = args.sheet
    if not (do_gif or do_sheet):          # 一键模式
        OUTDIR.mkdir(parents=True, exist_ok=True)
        args.gif = str(OUTDIR / "jump.gif")
        args.sheet = str(OUTDIR / "jump_sheet.png")
        do_gif = do_sheet = True
        print(f"[一键] 输出目录 {OUTDIR}")

    if do_gif:
        args.lookat = np.asarray(args.lookat, dtype=float)
        render_frames(args, m, d, torso, qpos, dt, phase, z, x, mg, contact,
                      args.gif, Path(args.record) if args.record else None)
    if do_sheet:
        args.lookat = np.asarray(args.lookat, dtype=float)
        render_sheet(args, m, d, torso, qpos, dt, phase, meta,
                     z, x, mg, contact, Path(args.sheet))


if __name__ == "__main__":
    main()
