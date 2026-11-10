"""蹲姿动作序列回放器 / 动画渲染器。

    python squat_replay.py                        # 窗口里看运动学回放
    python squat_replay.py --mode phys            # PD + 重力前馈跑真物理
    python squat_replay.py --gif out.gif          # 出 GIF
    python squat_replay.py --sheet out.png        # 一张图：关键帧 + 曲线
    python squat_replay.py --record logs/frames   # 离屏 PNG 序列

kin：逐帧写 qpos 做正运动学，纯摆姿势，用来验收动作序列本身（姿态、脚接触）。
phys：PD×8 + 重力前馈，跑 1 kHz 真物理。蹲姿是静态平衡的，所以这里**应该**站得住；
      站不住就说明轨迹或姿态有问题，脚本会打印判据。

渲染时叠三样辅助标记，让"重心在不在支撑面内"肉眼可见：
    绿色铅垂线 + 地面绿盘   COM 投影（静态时就是 ZMP）
    蓝色线框               支撑多边形（14 个脚底胶囊球心的凸包）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_squat.xml"
TRAJ = DATA / "squat_stand_traj.npz"

VCOL = {
    "com": (0.05, 0.90, 0.25, 1.00),
    "poly": (0.10, 0.50, 1.00, 1.00),
}


def resolve(path: str, kind: str = "轨迹") -> Path:
    """只给文件名时，按 数据目录 → 脚本目录 → 当前目录 依次找一遍。

    轨迹和场景都住在 `../data/mujoco_simulation/`，但用户习惯直接写
    `--traj squat_extreme_cycle_traj.npz`（照文档抄），以前会 FileNotFoundError。
    """
    p = Path(path)
    if p.exists():
        return p
    if p.parent == Path("."):                     # 纯文件名才兜底
        for base in (DATA, HERE, Path.cwd()):
            cand = base / p.name
            if cand.exists():
                print(f"[路径] {path} -> {cand}")
                return cand
    return p


# ---------------------------------------------------------------- 场景辅助
def foot_hull(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    GE = mujoco.mjtObj.mjOBJ_GEOM
    pts = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, GE, g) or ""
        if "_foot" not in nm or not nm.endswith("_collision"):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        half = float(m.geom_size[g][1])
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        pts.append((c + half * R[:, 2])[:2])
        pts.append((c - half * R[:, 2])[:2])
    pts = np.array(pts)
    # 单调链凸包
    p = np.unique(np.round(pts, 9), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in p[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
            up.pop()
        up.append(q)
    return np.array(lo[:-1] + up[:-1])


def _add_geom(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, size, pos, mat, rgba)
    scene.ngeom += 1


def overlay(scene, m, d, torso_id):
    """COM 铅垂线 + 地面投影盘 + 支撑多边形线框。"""
    com = d.subtree_com[torso_id].copy()
    # 铅垂线：COM → 地面（侧面机位时不会被脚完全挡住）
    if scene.ngeom + 1 < scene.maxgeom:
        mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.005, np.array([com[0], com[1], com[2]]),
                             np.array([com[0], com[1], 0.004]))
        scene.geoms[scene.ngeom].rgba[:] = VCOL["com"]
        scene.ngeom += 1
    # 地面扁盘 = ZMP/COM 投影
    _add_geom(scene, mujoco.mjtGeom.mjGEOM_CYLINDER,
              np.array([0.055, 0.004, 0.0]), np.array([com[0], com[1], 0.004]),
              np.eye(3).reshape(-1), VCOL["com"])
    hull = foot_hull(m, d)
    n = len(hull)
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        pa = np.array([a[0], a[1], 0.005])
        pb = np.array([b[0], b[1], 0.005])
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                             0.006, pa, pb)
        scene.geoms[scene.ngeom].rgba[:] = VCOL["poly"]
        scene.ngeom += 1


def signed_margin(c: np.ndarray, hull: np.ndarray) -> float:
    """点到支撑凸包边界的带符号距离（内部为正）。"""
    best = 1e9
    n = len(hull)
    for k in range(n):
        a, b = hull[k], hull[(k + 1) % n]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-9:
            continue
        nout = np.array([e[1], -e[0]]) / L
        best = min(best, float(np.dot(a - c, nout)))
    return best


_FONT = None


def hud(img: np.ndarray, t: float, z: float, margin: float, stage: str) -> np.ndarray:
    """左上角状态条：时间 / 阶段 / 躯干高 / 稳定裕度。"""
    global _FONT
    from PIL import Image as _Image, ImageDraw, ImageFont
    im = _Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    # 竖排左上角，不横贯整幅 —— 横条会把站姿的头顶切掉
    dr.rectangle([0, 0, 252, 152], fill=(14, 18, 26, 145))
    if _FONT is None:
        for fp in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                   "/System/Library/Fonts/PingFang.ttc"):
            try:
                _FONT = ImageFont.truetype(fp, 24)
                break
            except OSError:
                continue
        else:
            _FONT = ImageFont.load_default()
    dr.text((18, 10), f"t = {t:5.2f} s", font=_FONT, fill=(236, 241, 246, 255))
    dr.text((18, 42), stage, font=_FONT, fill=(140, 195, 255, 255))
    dr.text((18, 78), f"躯干 {z*100:5.1f} cm", font=_FONT, fill=(236, 241, 246, 255))
    col = (120, 255, 150, 255) if margin > 0.02 else (255, 110, 110, 255)
    dr.text((18, 110), f"裕度 {margin*100:+5.1f} cm", font=_FONT, fill=col)
    return np.asarray(im)


def make_cam(lookat, dist=1.9, az=270.0, el=-5.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.trackbodyid = -1
    cam.fixedcamid = -1
    cam.lookat[:] = lookat
    cam.distance = dist
    cam.azimuth = az
    cam.elevation = el
    return cam


# ---------------------------------------------------------------- 运动学
def run_kin(args, m, d, qpos, dt, meta):
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    n = len(qpos)
    if args.gif or args.record:
        from PIL import Image
        rec = Path(args.record) if args.record else None
        if rec:
            rec.mkdir(parents=True, exist_ok=True)
        ow = int(m.vis.global_.offwidth) or 1280
        oh = int(m.vis.global_.offheight) or 720
        w = int(min(args.width, ow))
        h = int(min(args.height, oh))
        renderer = mujoco.Renderer(m, h, w)
        cam = make_cam(args.lookat, dist=args.dist, az=args.az, el=args.el)
        idxs = list(range(0, n, max(1, args.every)))
        if args.loop and idxs[-1] != 0:
            idxs.append(0)                    # 循环时补一帧回站姿
        t_pre = float(meta.get("t_pre", 0.45))
        t_desc = float(meta.get("t_desc", 1.15))
        t_hold = float(meta.get("t_hold", 0.6))
        t_rise = float(meta.get("t_rise", 1.25))
        frames = []
        for k, i in enumerate(idxs):
            d.qpos[:] = qpos[i]
            # 必须 mj_forward：mj_kinematics 不更新 light_xpos，定位光会留在原点，
            # 整幅画面直接暗一半（实测 mean 67 vs 155）
            mujoco.mj_forward(m, d)
            if args.track:
                cam.lookat[:] = d.xpos[torso]
                cam.lookat[2] -= 0.10
            renderer.update_scene(d, camera=cam)
            overlay(renderer.scene, m, d, torso)
            img = renderer.render()
            tt = i * dt
            stage = ("站立 → 预蹲" if tt < t_pre else
                     "下蹲" if tt < t_pre + t_desc else
                     "蹲姿保持" if tt < t_pre + t_desc + t_hold else
                     "站起" if tt < t_pre + t_desc + t_hold + t_rise else "站立")
            com = d.subtree_com[torso]
            mg = signed_margin(com[:2], foot_hull(m, d))
            img = hud(img, tt, float(d.qpos[2]), mg, stage)
            frames.append(Image.fromarray(img))
            if rec:
                Image.fromarray(img).save(rec / f"squat_{i:04d}.png")
            if k % 10 == 0:
                print(f"  渲染 {k}/{len(idxs)}", end="\r", flush=True)
        print(f"[OK] {len(frames)} 帧  {w}x{h}")
        if args.gif:
            gif = Path(args.gif)
            gif.parent.mkdir(parents=True, exist_ok=True)
            dur = int(round(1000 * dt * args.every))
            frames[0].save(gif, save_all=True, append_images=frames[1:],
                           duration=dur, loop=0, disposal=2)
            print(f"[OK] GIF -> {gif}  ({gif.stat().st_size/1024:.0f} KB, "
                  f"{dur} ms/帧)")
        return 0

    # GUI
    import crab_replay as CR
    CR.run_kin(m, d, qpos, dt, speed=args.speed, loop=args.loop,
               track=args.track, record=None, torso_id=torso,
               dist=args.dist, az=args.az, el=args.el,
               width=args.width, height=args.height)
    return 0


# ---------------------------------------------------------------- 物理
def run_phys(args, m, d, qpos, dt, meta):
    """PD + 重力前馈跟踪，跑真物理。

    增益必须放大：mjlab 的 kp 是按 armature·(2π·10)² 算的**柔顺**增益，
    踝关节只有 33.5 N·m/rad —— 深蹲时重心在脚心前 5 cm，踝关节要扛
    424 N × 0.05 m ≈ 21 N·m，33.5 的刚度意味着 0.63 rad 稳态误差，腿会直接塌
    （实测 kp×1 时躯干从 0.78 m 掉到 0.11 m）。实机位置伺服的刚度远高于
    给 RL 留余量的训练参数，这里统一 ×8。

    ×8 之后显式积分只能用 1 kHz：500 Hz 发散、1 kHz 稳、×10 又发散，
    窗口很窄，所以倍率和频率都在命令行暴露成参数，别随手改。
    """
    from elf3_play import TrainParams
    P = TrainParams()
    names = list(P.joint_names)
    sf = args.kp_scale
    sd = args.kd_scale if args.kd_scale else args.kp_scale
    kp = {k: float(v) * sf for k, v in zip(names, P.stiffness)}
    kd = {k: float(v) * sd for k, v in zip(names, P.damping)}
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    jids = [(jn, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)) for jn in names]
    ank = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "l_ankle_x_link")

    step = 1.0 / args.sim_hz
    m.opt.timestep = step
    n_sub = max(1, int(round(dt / step)))
    n = len(qpos)
    nsteps = (n - 1) * n_sub

    # ---- 可选窗口：边算边显示。渲染按每 20 个物理步一次（1 kHz → 50 fps）
    v = None
    if args.gui:
        try:
            from mujoco import viewer as _mjv      # 别写 import mujoco.viewer：
            v = _mjv.launch_passive(m, d)           # 那会把 mujoco 变成函数局部名
            v.cam.lookat[:] = args.lookat
            v.cam.distance, v.cam.azimuth, v.cam.elevation = args.dist, args.az, args.el
            print("[窗口] 已打开：鼠标左键拖动=转视角，滚轮=缩放，关窗=退出")
        except Exception as e:
            print(f"[跳过] 打不开窗口（{e}），改为无窗口跑完")

    log = None
    rep = 0
    while True:
        rep += 1
        d.qpos[:] = qpos[0]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_forward(m, d)
        log = dict(t=[], z=[], comx=[], qerr=[], margin=[], footx=[])
        aborted = False
        for s in range(nsteps):
            i = min(s // n_sub, n - 1)
            q_des = qpos[i]
            d.qfrc_applied[:] = 0.0
            for jn, j in jids:
                if j < 0:
                    continue
                qa_, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
                d.qfrc_applied[va] = (d.qfrc_bias[va]
                                      + kp[jn] * (q_des[qa_] - d.qpos[qa_])
                                      - kd[jn] * d.qvel[va])
            mujoco.mj_step(m, d)
            if v is not None and s % 20 == 0:
                v.cam.lookat[:] = [float(d.qpos[0]), 0.0, float(d.qpos[2]) - 0.10]
                v.sync()
                if not v.is_running():
                    aborted = True
                    break
            if s % n_sub == 0:
                log["t"].append(s * step)
                log["z"].append(float(d.qpos[2]))
                log["comx"].append(float(d.subtree_com[torso][0]))
                log["qerr"].append(float(np.linalg.norm(
                    [d.qpos[m.jnt_qposadr[j]] - q_des[m.jnt_qposadr[j]]
                     for _, j in jids if j >= 0])))
                log["margin"].append(signed_margin(d.subtree_com[torso][:2], foot_hull(m, d)))
                log["footx"].append(float(d.xpos[ank][0]))
        if aborted or not (args.loop and v is not None):
            break
        print(f"  重播 #{rep} ...", end="\r", flush=True)
    if v is not None:
        v.close()
    if aborted:
        print(f"\n[退出] 窗口已关闭（循环 {rep} 次）；中断的那轮不参与统计")
        return 0

    t = np.array(log["t"]); z = np.array(log["z"]); mg = np.array(log["margin"])
    fx = np.array(log["footx"]); qe = np.array(log["qerr"])
    # 检查窗：显式给了就用，否则从 meta 的段时长推（不同版本 t_pre/t_desc 不一样，
    # 写死 1.6 s 会把下蹲过程也算进保持段，波动当然大）。
    # 关键：**上界必须封在保持段末尾** —— 循环轨迹里 4.4 s 之后是起立，
    # 敞口到 t[-1] 会把 0.52 m 的起身高度差算成"保持段波动"（522 mm）。
    seg = str(meta.get("segment", "")) if meta else ""
    t_pre = float(meta.get("t_pre", 0.0)) if meta else 0.0
    t_desc = float(meta.get("t_desc", 0.0)) if meta else 0.0
    t_hold = float(meta.get("t_hold", 0.0)) if meta else 0.0
    lo, hi = 0.0, float("inf")
    if args.hold_start is not None:
        lo = args.hold_start
    elif seg == "up":
        # 起立段没有"保持段"：要检查的是起点那个蹲姿，取最前 0.3 s（几乎静止）
        hi = min(0.30, t[-1] * 0.25)
    elif seg == "down":
        # 下蹲段只有末尾可能带保持帧；没带就看最后 0.3 s
        lo = t_pre + t_desc
        if t_hold <= 0.0:
            lo = max(t[-1] - 0.30, 0.0)
    elif t_hold > 0.0:
        lo = t_pre + t_desc + min(0.15, 0.3 * t_hold)
        hi = t_pre + t_desc + t_hold
    elif meta:
        lo = max(t[-1] - 0.5, 0.0)
    else:
        lo = t[-1] - 0.5
    hold = (t > lo) & (t <= hi)
    if not hold.any():                       # 兜底：段太短时用最后 25% 帧
        lo, hi = t[-1] * 0.75, float("inf")
        hold = t >= lo
        if not hold.any():
            hold[:] = True
    win = f"t>{lo:.2f}s" if hi == float("inf") else f"{lo:.2f}~{hi:.2f}s"
    print(f"[检查窗] {win}（{int(hold.sum())} 帧）")
    print(f"\n---- 物理回放（PD×{sf:g} + 重力前馈，{args.sim_hz} Hz）----")
    print(f"躯干高     起 {z[0]:.4f} → 末 {z[-1]:.4f} m   保持段波动 "
          f"{np.ptp(z[hold])*1000:.1f} mm")
    print(f"关节跟踪   平均 {np.degrees(qe.mean()):.2f}°   最大 {np.degrees(qe.max()):.2f}°")
    print(f"稳定裕度   最小 {mg.min()*100:.2f} cm   保持段 {mg[hold].mean()*100:.2f} cm")
    print(f"脚踝滑移   {np.ptp(fx)*1000:.2f} mm")
    ok = (np.ptp(z[hold]) < 0.02) and (mg.min() > 0.02) and (np.ptp(fx) < 0.02)
    print(f"[判定] {'站稳了：蹲姿静态平衡成立' if ok else '不稳定，检查姿态/增益'}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            z0=float(z[0]), z_end=float(z[-1]), z_hold_ptp=float(np.ptp(z[hold])),
            qerr_mean=float(np.degrees(qe.mean())), qerr_max=float(np.degrees(qe.max())),
            margin_min=float(mg.min()), margin_hold=float(mg[hold].mean()),
            foot_slip=float(np.ptp(fx)), stable=bool(ok)), indent=2), encoding="utf-8")
    return 0 if ok else 1


# ---------------------------------------------------------------- 关键帧拼图
def _plot(dr, box, xs, ys, title, ylab, color, fmt="{:.3f}", refs=()):
    """在 box=(x0,y0,x1,y1) 里画一条折线（含坐标轴/网格/参考线）。"""
    from PIL import ImageFont
    global _FONT
    f = _FONT or ImageFont.load_default()
    fs = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17) if _FONT else f
    x0, y0, x1, y1 = box
    lo, hi = float(min(ys)), float(max(ys))
    pad = (hi - lo) * 0.18 + 1e-6
    lo, hi = lo - pad, hi + pad
    for r in refs:
        lo, hi = min(lo, r), max(hi, r)
    dr.rectangle(box, outline=(150, 160, 172, 255), width=1)
    for k in range(5):
        gy = y1 - (y1 - y0) * k / 4
        dr.line([x0, gy, x1, gy], fill=(72, 82, 96, 255))
        dr.text((x0 - 62, gy - 9), fmt.format(lo + (hi - lo) * k / 4),
                font=fs, fill=(176, 186, 198, 255))
    for r in refs:
        gy = y1 - (y1 - y0) * (r - lo) / (hi - lo)
        dr.line([x0, gy, x1, gy], fill=(240, 110, 110, 200))
    pts = [(x0 + (x1 - x0) * i / max(1, len(ys) - 1),
            y1 - (y1 - y0) * (v - lo) / (hi - lo)) for i, v in enumerate(ys)]
    dr.line(pts, fill=color, width=3, joint="curve")
    dr.text((x0, y0 - 26), title, font=fs, fill=(230, 236, 244, 255))
    dr.text((x0, y0 - 6), ylab, font=fs, fill=(150, 160, 175, 255))


def render_sheet(args, m, d, qpos, dt, meta, out: Path):
    """一张图讲完：上排关键帧，下排躯干高度 / 稳定裕度曲线。"""
    from PIL import Image as I, ImageDraw, ImageFont
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    n = len(qpos)
    z = np.zeros(n)
    mg = np.zeros(n)
    for i in range(n):
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        z[i] = d.qpos[2]
        mg[i] = signed_margin(d.subtree_com[torso][:2], foot_hull(m, d))

    t_pre = float(meta.get("t_pre", 0.45))
    t_desc = float(meta.get("t_desc", 1.15))
    dur = (n - 1) * dt
    marks = [(0.0, "站立"), (t_pre, "预蹲"),
             (t_pre + t_desc * 0.55, "下蹲中"), (t_pre + t_desc, "蹲姿"),
             (dur, "蹲姿保持")]

    r = mujoco.Renderer(m, 396, 330)
    cam = make_cam([0.05, 0.0, 0.60], dist=2.62, az=270.0, el=-4.0)
    tiles = []
    for tt, _ in marks:
        i = min(int(round(tt / dt)), n - 1)
        d.qpos[:] = qpos[i]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        overlay(r.scene, m, d, torso)
        tiles.append(I.fromarray(r.render()))

    W, H = 1700, 1000
    sheet = I.new("RGB", (W, H), (22, 27, 36))
    dr = ImageDraw.Draw(sheet, "RGBA")
    fb = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
    fs = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    dr.text((32, 22), "ELF3 站立 → 蹲姿：动作序列与静态稳定校核", font=fb,
            fill=(236, 242, 248, 255))
    dr.text((32, 60), f"{n} 帧 @ {1/dt:.0f} Hz / {dur:.2f} s    "
                      f"蹲姿躯干 {z[-1]:.3f} m（下降 {z[0]-z[-1]:.3f} m）    "
                      f"稳定裕度 {mg[-1]*100:.1f} cm",
            font=fs, fill=(150, 175, 200, 255))
    for k, (tt, name) in enumerate(marks):
        x = 32 + k * 330
        sheet.paste(tiles[k], (x, 104))
        dr.text((x + 8, 108), f"{tt:4.2f} s", font=fs, fill=(255, 255, 255, 235))
        dr.text((x + 8, 132), name, font=fs, fill=(140, 195, 255, 235))
    bx = (120, 610, 830, 916)
    _plot(dr, bx, np.arange(n) * dt, z, "躯干高度", "m", (110, 205, 255, 255),
          refs=(z[0], z[-1]))
    _plot(dr, (950, 610, 1660, 916), np.arange(n) * dt, mg * 100, "稳定裕度（COM 到支撑边界）",
          "cm", (120, 235, 150, 255), fmt="{:.1f}", refs=(2.0,))
    dr.text((120, 938), "红线 = 站姿/蹲姿高度参考；裕度红线 = 2 cm 安全阈值",
            font=fs, fill=(130, 140, 155, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[OK] 关键帧拼图 -> {out}  ({out.stat().st_size/1024:.0f} KB)")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="ELF3 蹲姿序列回放 / 渲染")
    ap.add_argument("--traj", default=str(TRAJ))
    ap.add_argument("--scene", default=str(SCENE))
    ap.add_argument("--mode", choices=["kin", "phys"], default="kin")
    ap.add_argument("--gif", default=None, help="输出 GIF 路径")
    ap.add_argument("--record", default=None, help="输出 PNG 序列目录")
    ap.add_argument("--every", type=int, default=2, help="每 N 帧取一帧")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=486)
    ap.add_argument("--dist", type=float, default=2.2)
    ap.add_argument("--az", type=float, default=270.0)
    ap.add_argument("--el", type=float, default=-5.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lookat", type=float, nargs=3, default=[0.06, 0.0, 0.55],
                    metavar=("X", "Y", "Z"), help="固定机位的注视点")
    ap.add_argument("--track", action="store_true", help="相机跟随躯干")
    ap.add_argument("--loop", action="store_true",
                    help="kin: 循环播放；phys: 循环重播")
    ap.add_argument("--gui", action="store_true",
                    help="phys 模式开窗口边算边看（默认无窗口）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--sheet", default=None, help="输出关键帧拼图 PNG（一张图讲完）")
    ap.add_argument("--hold-start", type=float, default=None,
                    help="蹲姿保持段起点（默认由 meta 的 t_pre+t_desc 推出）")
    ap.add_argument("--kp-scale", type=float, default=8.0,
                    help="PD 刚度倍率（训练增益是给 RL 留余量的柔顺值，深蹲撑不住）")
    ap.add_argument("--kd-scale", type=float, default=None, help="阻尼倍率（默认同 kp）")
    ap.add_argument("--sim-hz", type=int, default=1000, help="物理步频")
    args = ap.parse_args()

    args.traj = str(resolve(args.traj, "轨迹"))
    args.scene = str(resolve(args.scene, "场景"))
    traj_path = Path(args.traj)
    if not traj_path.exists():
        avail = sorted(x.name for x in DATA.glob("*.npz"))
        raise SystemExit(f"[错误] 找不到轨迹 {args.traj}\n"
                         f"       数据目录 {DATA} 里有：{avail}\n"
                         f"       可用 --out 重新生成，或写全路径")

    data = np.load(traj_path, allow_pickle=True)
    qpos = data["qpos"]
    dt = float(data["dt"])
    meta = data["meta"].item() if "meta" in data else {}
    if meta:
        print(f"[轨迹] {len(qpos)} 帧 / {len(qpos)*dt:.2f} s   "
              f"蹲姿躯干 {meta.get('squat_z', 0):.3f} m  "
              f"裕度 {meta.get('squat_margin', 0)*100:.1f} cm")

    if not Path(args.scene).exists():
        import squat_stand as S
        S.write_scene()
    m = mujoco.MjModel.from_xml_path(str(Path(args.scene).resolve()))
    m.site_rgba[:, 3] = 0.0
    d = mujoco.MjData(m)

    if args.sheet:
        render_sheet(args, m, d, qpos, dt, meta, Path(args.sheet))
        return 0

    if args.mode == "phys":
        raise SystemExit(run_phys(args, m, d, qpos, dt, meta))
    return run_kin(args, m, d, qpos, dt, meta)


if __name__ == "__main__":
    raise SystemExit(main())
