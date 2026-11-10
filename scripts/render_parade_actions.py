"""render_parade_actions.py — 把「走正步 / 敬礼」npz 轨迹渲染成 mp4 + 关键帧拼图。

为什么不用 ``play --video``：那条链要同时满足 tyro bool flag 写法、viewer 能自己
step、VideoRecorder 的 step_trigger 被触发，任一不成立就静默不产文件（项目里
已经在 ``scripts/export_squat_video.py`` 踩过一次）。这里直接：

    自建 MuJoCo 模型 + ``mujoco.Renderer`` 离屏渲染
    -> 逐帧 ``d.qpos[:] = row`` / ``mj_forward`` / ``update_scene`` / ``render``
    -> imageio-ffmpeg 流式写 mp4（不在内存里攒帧，341 帧 640×480 也不过 300 MB，
       但攒帧容易 OOM，统一走流式）

不需要 DISPLAY、不需要 viser、不需要 viewer。

跑法（项目根目录）：
    python scripts/render_parade_actions.py                    # 三段全导
    python scripts/render_parade_actions.py --only salute      # 只导敬礼
    python scripts/render_parade_actions.py --no-video         # 只出拼图
    python scripts/render_parade_actions.py --out-dir outputs/parade
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DATA = (REPO / "sim" / "vendor" / "bxi_controller_ros2" / "src"
        / "bxi_example_py_elf3" / "data" / "mujoco_simulation")
SCENE = DATA / "elf3_squat.xml"           # nq=38，与 npz 的 qpos 完全对应

# name -> (npz 文件, 机位方位角, 跟随基座, 关键帧时刻 s, 中文标题)
# MuJoCo 自由相机 campos = lookat + d*(−cos az, −sin az, sin el)：
#   az=180 站在 +x（机器人的正前方），az=90 站在 −y（侧面，+x 朝画面右），
#   az=145 是正面偏右 3/4。之前敬礼用的 az=35 其实站在 −x，是背面。
ACTIONS: dict[str, dict] = {
    # 踢腿峰值相位 = duty + (1-duty)*0.5 = 0.75；左腿 t=0.9+1.2k，
    # 右腿 t=0.3+1.2k。前进版步态段从 1.0 s 开始（立正 0.5 + 过渡 0.5）。
    "goose_walk": dict(
        npz="goose_step_traj.npz", az=135.0, follow=True, fovy=33.0,
        keys=[(0.00, "立正"), (0.80, "起步过渡"), (1.33, "右腿踢直"),
              (1.93, "左腿踢直"), (2.53, "右腿踢直"), (3.60, "行进"),
              (5.20, "行进"), (6.80, "收势立正")],
        title="走正步 · 前进 6.82 s / 3.15 m（前倾 5°、腿绷直、支撑腿后倾 5°、小臂横胸，前 3/4）"),
    "goose_inplace": dict(
        npz="goose_step_inplace_traj.npz", az=135.0, follow=False, fovy=33.0,
        keys=[(0.00, "双脚落地"), (0.18, "起腿"), (0.33, "右腿踢直"),
              (0.48, "收腿"), (0.63, "落地换腿"), (0.93, "左腿踢直"),
              (1.53, "右腿踢直"), (2.13, "左腿踢直")],
        title="原地正步 · 4.80 s 循环（腿绷直、支撑腿后倾 5°、前摆小臂水平横胸，前 3/4）"),
    "salute": dict(
        npz="salute_traj.npz", az=145.0, follow=False, fovy=31.0,
        keys=[(0.00, "自然站姿"), (0.40, "立正收臂"), (0.90, "抬手中"),
              (1.40, "手到太阳穴"), (2.20, "敬礼保持"), (2.90, "礼毕放下"),
              (3.60, "收势"), (4.40, "归位")],
        title="敬礼 · 右手军礼 4.52 s 循环（左臂自然下垂、右腕向右 10 cm、腕滚 +90° 立掌手背朝前，正面偏右 3/4）"),
    # 正步 + 敬礼 连贯序列：正步 0~6.82 s / 过渡 6.82~7.80 s / 敬礼 7.82~12.30 s /
    # 定格到 12.82 s。follow=True，否则走完 3.15 m 机器人就出画了。
    "parade_seq": dict(
        npz="parade_sequence_traj.npz", az=135.0, follow=True, fovy=33.0,
        keys=[(0.00, "立正"), (1.33, "右腿踢直"), (3.60, "行进"),
              (6.80, "收步立正"), (7.30, "收臂过渡"), (8.60, "抬手中"),
              (9.40, "手到太阳穴"), (10.20, "敬礼保持"), (11.20, "礼毕放下"),
              (12.30, "定格立正")],
        title="正步 + 敬礼 · 连贯 12.82 s（走 3.15 m 收步 → 1 s 收臂立正 → 军礼 → 定格，前 3/4 跟随）"),
}

# 双机位细节图：同一帧同时给「侧视看腿」和「正面看小臂」。
# 踢腿峰值处，另一侧手臂正好摆到最前（正步是异侧摆臂），所以一张图能同时验收两件事。
DETAIL: dict[str, dict] = {
    "goose_walk": dict(npz="goose_step_traj.npz", follow=True, fovy=30.0),
    "goose_inplace": dict(npz="goose_step_inplace_traj.npz", follow=False,
                          fovy=30.0),
}


def font(size: int):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def camera(azimuth: float, distance: float, lookat_z: float,
           elevation: float = -6.0) -> mujoco.MjvCamera:
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    c.lookat[:] = (0.0, 0.0, lookat_z)
    return c


def _tight_boxes(m) -> list[tuple[int, np.ndarray]]:
    """每个几何体在自己局部坐标系下的 8 个包围盒角点。

    不用 ``m.geom_rbound``：那是包围**球**半径，头部网格 0.284 会把 zmax 抬到
    1.47 m（真实头顶 1.19），机器人等于被画小三分之一。网格直接顶点取 min/max，
    其余图元用 rbound 已经够紧。
    """
    out: list[tuple[int, np.ndarray]] = []
    G = mujoco.mjtGeom
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:          # 跳过 worldbody 的 `squat_ground`
            continue
        gt, sz = int(m.geom_type[g]), np.asarray(m.geom_size[g], float)
        if gt == G.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            v = np.asarray(m.mesh_vert[a:a + n], dtype=np.float64)
            lo, hi = v.min(axis=0), v.max(axis=0)
        elif gt == G.mjGEOM_BOX or gt == G.mjGEOM_ELLIPSOID:
            lo, hi = -sz[:3], sz[:3]
        elif gt == G.mjGEOM_SPHERE:
            lo = -np.full(3, sz[0]); hi = -lo
        elif gt == G.mjGEOM_CAPSULE:
            lo = np.array([-sz[0], -sz[0], -sz[1] - sz[0]])
            hi = -lo
        elif gt == G.mjGEOM_CYLINDER:
            lo = np.array([-sz[0], -sz[0], -sz[1]])
            hi = -lo
        else:                              # 兜底：包围球
            r = float(m.geom_rbound[g])
            lo, hi = np.full(3, -r), np.full(3, r)
        out.append((g, np.array([[x, y, z] for z in (lo[2], hi[2])
                                 for y in (lo[1], hi[1])
                                 for x in (lo[0], hi[0])])))
    return out


def auto_frame(m, d, qpos: np.ndarray, fovy_deg: float, aspect: float,
               follow: bool, fill: float = 0.88) -> tuple[float, float]:
    """扫一遍轨迹求「完整装下整个机器人」的 lookat_z / distance。

    手调 dist 会被垂直视场卡死：720x540 + fovy 45° 在 dist=1.75 时垂直可见
    只有 ±0.72 m，脚（z=0）必然被下沿切掉。这里直接按几何包围盒反解，
    follow=True 时把 x 方向位置折回基座坐标系，避免 3 m 行程把半径算大。
    """
    boxes = _tight_boxes(m)
    zmin, zmax, rmax = 1e9, -1e9, 0.0
    step = max(1, len(qpos) // 80)
    for row in qpos[::step]:
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        bx, by = (float(row[0]), float(row[1])) if follow else (0.0, 0.0)
        for g, cor in boxes:
            w = d.geom_xpos[g] + cor @ d.geom_xmat[g].reshape(3, 3).T
            zmin = min(zmin, float(w[:, 2].min()))
            zmax = max(zmax, float(w[:, 2].max()))
            rmax = max(rmax, float(np.hypot(w[:, 0] - bx, w[:, 1] - by).max()))
    if zmin > zmax:
        raise RuntimeError("轨迹为空")
    tan_h = math.tan(math.radians(fovy_deg) / 2.0)
    lookat_z = (zmin + zmax) / 2.0
    dist = max((zmax - zmin) / 2.0 / tan_h, rmax / (tan_h * aspect)) / fill
    return lookat_z, dist


def render_action(name: str, cfg: dict, m, d, out: Path,
                  width: int, height: int, want_video: bool,
                  want_sheet: bool, fps: int) -> list[Path]:
    z = np.load(DATA / cfg["npz"], allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=np.float64)
    dt = float(z["dt"])
    n = len(qpos)
    fovy = cfg.get("fovy", 34.0)
    m.vis.global_.fovy = fovy
    lut_z, dist = auto_frame(m, d, qpos, fovy, width / height, cfg["follow"])
    cam = camera(cfg["az"], dist, lut_z)
    print(f"  取景: fovy={fovy:.0f}° lookat_z={lut_z:.3f} dist={dist:.2f}")
    r = mujoco.Renderer(m, height=height, width=width)

    made: list[Path] = []
    writer = None
    video = out / f"{name}.mp4"
    if want_video:
        import imageio
        writer = imageio.get_writer(
            str(video), fps=fps, format="FFMPEG", codec="libx264",
            macro_block_size=None,
            output_params=["-crf", "20", "-movflags", "+faststart"])

    key_ids = {min(n - 1, int(round(t / dt))): lab for t, lab in cfg["keys"]}
    tiles: list[tuple[str, np.ndarray]] = []
    for k, row in enumerate(qpos):
        d.qpos[:] = row
        mujoco.mj_forward(m, d)
        if cfg["follow"]:
            cam.lookat[0] = float(row[0])       # 跟随基座，机器人始终居中
        r.update_scene(d, cam)
        img = r.render()
        if writer is not None:
            writer.append_data(img)
        if want_sheet and k in key_ids:
            tiles.append((f"{k*dt:.2f}s  {key_ids[k]}", img.copy()))
    if writer is not None:
        writer.close()
        made.append(video)
        print(f"  [mp4 ] {video}  {n} 帧 @{1/dt:.0f} Hz = {n*dt:.2f} s "
              f"({video.stat().st_size/1024:.0f} KB)")

    if want_sheet and tiles:
        cols = 4
        pad, lab_h = 8, 34
        rows = math.ceil(len(tiles) / cols)
        sheet = Image.new("RGB", (cols * (width + pad) + pad,
                                 rows * (height + lab_h + pad) + lab_h + pad),
                          (247, 248, 250))
        dr = ImageDraw.Draw(sheet)
        f_title, f_lab = font(24), font(18)
        dr.text((pad + 6, 6), cfg["title"], fill=(24, 32, 48), font=f_title)
        for i, (lab, img) in enumerate(tiles):
            x = pad + (i % cols) * (width + pad)
            y = lab_h + pad + (i // cols) * (height + lab_h + pad)
            dr.text((x + 4, y), lab, fill=(40, 48, 60), font=f_lab)
            sheet.paste(Image.fromarray(img), (x, y + lab_h))
        png = out / f"{name}_sheet.png"
        sheet.save(png)
        made.append(png)
        print(f"  [sheet] {png}  {len(tiles)} 格  {sheet.width}x{sheet.height}")
    return made


def _kick_frames(m, d, qpos: np.ndarray) -> dict[str, int]:
    """各腿「踢到最前」的帧号：脚踝相对基座的 x 前伸量取 argmax。"""
    out: dict[str, int] = {}
    for side in ("l", "r"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                f"{side}_ankle_y_link")
        best, best_v = 0, -1e9
        for k, row in enumerate(qpos):
            d.qpos[:] = row
            mujoco.mj_forward(m, d)
            v = float(d.xpos[bid][0] - row[0])
            if v > best_v:
                best, best_v = k, v
        out[side] = best
    return out


def render_detail(name: str, cfg: dict, m, d, out: Path,
                  width: int, height: int) -> Path | None:
    """2 帧 × 2 机位：左列侧视（az=90）看腿，右列正面（az=180）看小臂。"""
    z = np.load(DATA / cfg["npz"], allow_pickle=True)
    qpos = np.asarray(z["qpos"], dtype=np.float64)
    dt = float(z["dt"])
    fovy = cfg.get("fovy", 30.0)
    m.vis.global_.fovy = fovy
    lut_z, dist = auto_frame(m, d, qpos, fovy, width / height, cfg["follow"])
    kicks = _kick_frames(m, d, qpos)
    r = mujoco.Renderer(m, height=height, width=width)

    views = [(90.0, "侧视 az=90 · 支撑腿大腿后倾 / 摆动腿绷直"),
             (180.0, "正面 az=180 · 前摆小臂水平横在胸前")]
    tiles: list[tuple[str, np.ndarray]] = []
    for side, k in kicks.items():
        for az, cap in views:
            cam = camera(az, dist, lut_z)
            d.qpos[:] = qpos[k]
            mujoco.mj_forward(m, d)
            if cfg["follow"]:
                cam.lookat[0] = float(qpos[k][0])
            r.update_scene(d, cam)
            swing = "右" if side == "r" else "左"
            stance = "左" if side == "r" else "右"
            lab = (f"{k*dt:.2f}s {swing}腿踢直（摆动腿 {swing} / 支撑腿 {stance}）")
            tiles.append((f"{lab}\n{cap}", r.render().copy()))
    r.close()

    pad, lab_h, gap = 10, 52, 10
    cols = 2
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (width + pad) + pad,
                             rows * (height + lab_h + pad) + lab_h + pad),
                      (247, 248, 250))
    dr = ImageDraw.Draw(sheet)
    dr.text((pad + 6, 8), f"{name} · 双机位细节（左列侧视看腿、右列正面看臂）",
            fill=(24, 32, 48), font=font(24))
    for i, (lab, img) in enumerate(tiles):
        x = pad + (i % cols) * (width + pad)
        y = lab_h + pad + (i // cols) * (height + lab_h + gap)
        for j, line in enumerate(lab.split("\n")):
            dr.text((x + 4, y + j * 21), line, fill=(40, 48, 60), font=font(17))
        sheet.paste(Image.fromarray(img), (x, y + lab_h))
    png = out / f"{name}_detail.png"
    sheet.save(png)
    print(f"  [detail] {png}  {len(tiles)} 格  {sheet.width}x{sheet.height}")
    return png


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=[*ACTIONS, "all"], default="all")
    ap.add_argument("--out-dir", default="logs/parade_actions")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-sheet", action="store_true")
    ap.add_argument("--no-detail", action="store_true")
    ap.add_argument("--detail-width", type=int, default=560)
    ap.add_argument("--detail-height", type=int, default=660)
    args = ap.parse_args()

    out = (REPO / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    if m.nq != 38:
        raise SystemExit(f"[错误] 需要 nq=38 的场景，实际 {m.nq}")

    names = list(ACTIONS) if args.only == "all" else [args.only]
    print(f"场景 {SCENE.name}  nq={m.nq}  输出 -> {out}")
    for name in names:
        print(f"\n=== {name} ===")
        render_action(name, ACTIONS[name], m, d, out, args.width, args.height,
                      not args.no_video, not args.no_sheet, args.fps)
        if not args.no_detail and name in DETAIL:
            render_detail(name, DETAIL[name], m, d, out,
                          args.detail_width, args.detail_height)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
