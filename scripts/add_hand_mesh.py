#!/usr/bin/env python3
"""给 elf3 系列 MJCF 的腕部叠加刚性的手网格（零自由度变化）。

为什么是「叠加」而不是新建 body
-------------------------------
`l_wrist_z_link` / `r_wrist_z_link` 就是手臂链的末端 body，手上没有任何要做
运动学的自由度需求。MJCF 里「body 不写 joint」= 焊死在父 body 上，仍然会
多一个 body、多一段惯性合成；而直接把 `<geom mesh="l_hand"/>` 塞进现有
body，连 body 都不多一个，nq / nv / nu / 总质量 100% 不变：

    elf3.xml  nq 38 nv 37 nu 31 njnt 32 nbody 33 mass 43.2225
    加手后    同上（几何体是 density=0 的纯视觉 mesh，contype=0）

这一点很重要：训练侧 `env_cfg.py` 的 joint_pos / joint_vel / actions /
dof_pos_limits 全是 `(".*",)`，一旦 nv 变了观测维度就变，旧 checkpoint
直接作废。方案 A 不碰这些，旧策略、qpos 回放、bake 出来的动作全部照旧。

保留原腕部法兰
--------------
手 mesh 的包围盒是 x 0.026~0.119、y ±0.053、z −0.086~0.098，原腕壳是
x 0~0.053 的法兰环。两者重叠 2.7 cm —— 实测叠在一起渲染与「把腕壳换成手」
逐像素几乎一致（法兰被掌心完全包住），所以保留原行、只追加，git diff
是纯新增，回退只需删两行。

处理哪些文件
------------
1) sim 侧 5 份独立整份拷贝（互不 include，必须逐份加）：
   elf3 / elf3_crab / elf3_squat / elf3_squat_x / elf3_prone
   `elf3_robot_only.xml` **不要手改**，它由 train/assets/make_robot_xml.py 从
   elf3.xml 派生（还会把视觉 geom 挪到 group 2，避免 height_scan 打到自己的手）。
2) web 侧 `web/assets/elf3.xml`（WASM 回放用的另一套精简模型，nq=36 / nu=29）。
   它的 `meshes/` 里早就有 l_hand.STL / r_hand.STL，但 XML 从来没引用过；那对 STL
   和 sim 的手不是同一造型、坐标系也对不上，已换成 sim 那一对（脚本会校验 md5）。
   加完还要 `node web/tools/make-manifest.mjs` 重写 manifest.json（里面记了每个文件
   的字节数，前端拿来做进度条）。
   加完还要 `node web/tools/make-manifest.mjs` 重写 manifest.json（里面记了
   每个文件的字节数，前端拿来做进度条）。

用法：
    python scripts/add_hand_mesh.py            # 执行
    python scripts/add_hand_mesh.py --check    # 只检查，不写盘
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = (
    ROOT
    / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"
)
WEB = ROOT / "web" / "assets"

SIDES = ("l", "r")

# web 的手部 geom 不需要 pos/quat 补偿：`web/assets/meshes/l_hand.STL` 与
# `r_hand.STL` 已直接换用 sim 那一对（由 _check_web_assets 强校验 md5），
# 位姿由构造保证，不加任何变换。
#   踩过的坑：web 原来那对 STL 是**另一种造型**（sim 是握拳、web 那张更像张开的爪）。
#   MuJoCo 编译时会把顶点重定心并旋转到主惯性轴，偏移/旋转分别记在 mesh_pos /
#   mesh_quat，而主轴符号是任意的 —— 所以「只加 pos」能对上包围盒但朝向反 180°，
#   「pos+quat 解析解」残差仍有 11 mm、p95 22 mm。不是同一只手，任何刚体变换都
#   救不了，只能换资源。别再对着包围盒调 pos 了。
NN_TOL_MM = 8.0

# (xml 路径, 该文件手部 geom 需要的额外属性；空 dict = 手 STL 已与腕坐标系对齐)
SOURCES: list[tuple[Path, dict[str, str]]] = [
    (DATA / "elf3.xml", {}),
    (DATA / "elf3_crab.xml", {}),
    (DATA / "elf3_squat.xml", {}),
    (DATA / "elf3_squat_x.xml", {}),
    (DATA / "elf3_prone.xml", {}),
    (WEB / "elf3.xml", {}),
]


def _snapshot(xml_path: Path) -> tuple:
    """编译一次，取拓扑/质量指纹。include 与 meshdir 都相对 xml 自身目录解析。"""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    return (m.nq, m.nv, m.nu, m.njnt, m.nbody, round(float(m.body_mass.sum()), 6))


def _md5(p: Path) -> str:
    import hashlib

    return hashlib.md5(p.read_bytes()).hexdigest()


def _check_web_assets() -> list[str]:
    """web 端手 STL 必须与 sim 的逐字节相同，否则位姿一定对不上。"""
    errs = []
    for side in SIDES:
        a = DATA / "meshes" / f"{side}_hand.STL"
        b = WEB / "meshes" / f"{side}_hand.STL"
        if not b.exists():
            errs.append(f"缺 {b.relative_to(ROOT)}")
        elif _md5(a) != _md5(b):
            errs.append(
                f"{b.relative_to(ROOT)} 与 sim 的不同（{_md5(b)[:8]} vs {_md5(a)[:8]}），"
                f"请先 cp '{a}' '{b}'"
            )
    return errs


def _hand_verts(xml_path: Path) -> dict[str, "np.ndarray"]:
    """手网格顶点，换算到 wrist_z_link body 坐标系。

    注意 MuJoCo 编译期会把每个 mesh **重定心并旋转到主惯性轴**（偏移在 mesh_pos、
    旋转在 mesh_quat），所以只看 mesh_vert 会得到错位的结果 —— 必须补上这两个量。
    """
    import mujoco
    import numpy as np

    def R_of(q):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3)

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    out = {}
    for side in SIDES:
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, f"{side}_hand")
        if mid < 0:
            continue
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a : a + n] * np.array(m.mesh_scale[mid])
        v = v @ R_of(m.mesh_quat[mid]).T + m.mesh_pos[mid]      # -> geom 坐标系
        for g in range(m.ngeom):
            if m.geom_dataid[g] == mid:
                v = v @ R_of(m.geom_quat[g]).T + m.geom_pos[g]  # -> body 坐标系
                break
        out[side] = v
    return out


def _nn_gap_mm(ref: np.ndarray, got: np.ndarray, seed: int = 0) -> float:
    """两个点云的平均最近邻距离（mm）。

    参考集必须用**全量**：两边抽样的话，同一个顶点未必同时被抽中，会凭空多出几 mm
    的假差值。为什么不用包围盒：180° 翻转后包围盒仍然对称，照样"对得上"。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    b = got[rng.choice(len(got), min(400, len(got)), replace=False)]
    d = np.sqrt(((b[:, None, :] - ref[None, :, :]) ** 2).sum(-1)).min(1)
    return float(d.mean() * 1000)


def _hand_asset_line(anchor: str, side: str) -> str:
    """复制锚点行、只换名字，这样 content_type 之类的属性也跟着走。"""
    line = anchor.replace(f'name="{side}_wrist_z_link"', f'name="{side}_hand"')
    return line.replace(f'file="{side}_wrist_z_link.STL"', f'file="{side}_hand.STL"')


def _patch(txt: str, side: str, pos: str = "") -> tuple[str, bool]:
    """返回 (新文本, 是否改动)。锚点缺失直接抛异常，不静默跳过。"""
    if re.search(rf'<mesh\s+name="{side}_hand"', txt):
        return txt, False

    # 1) asset：在 <mesh name="{s}_wrist_z_link" .../> 之后插入手部 mesh
    asset_pat = re.compile(
        rf'^([ \t]*)<mesh\s+name="{side}_wrist_z_link"[^>]*/>[ \t]*$', re.MULTILINE
    )
    hits = asset_pat.findall(txt)
    if len(hits) != 1:
        raise RuntimeError(f'asset 锚点 <mesh name="{side}_wrist_z_link"> 命中 {len(hits)} 次，期望 1')

    def _asset_sub(m: re.Match) -> str:
        return m.group(0) + "\n" + m.group(1) + _hand_asset_line(m.group(0).strip(), side)

    txt = asset_pat.sub(_asset_sub, txt)

    # 2) body：在腕部视觉 geom 之后追加手部视觉 geom。
    #    直接复制锚点行的属性（class="visual" / rgba / type 等），只换 mesh 名，
    #    这样 sim 那份继承 childclass="elf3"、web 那份显式 class="visual" 都能对上。
    geom_pat = re.compile(
        rf'^([ \t]*)<geom[^>]*\bmesh="{side}_wrist_z_link"[^>]*/>[ \t]*$', re.MULTILINE
    )
    hits = geom_pat.findall(txt)
    if len(hits) != 1:
        raise RuntimeError(f'body 锚点 <geom mesh="{side}_wrist_z_link"> 命中 {len(hits)} 次，期望 1')

    def _geom_sub(m: re.Match) -> str:
        line = m.group(0).strip().replace(f'mesh="{side}_wrist_z_link"', f'mesh="{side}_hand"')
        if pos:
            line = line[:-2].rstrip() + pos + "/>"
        return m.group(0) + "\n" + m.group(1) + line

    txt = geom_pat.sub(_geom_sub, txt)
    return txt, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查不写盘")
    ap.add_argument("--verify", action="store_true", help="连已经加过手的文件也重新编译自检")
    args = ap.parse_args()

    if not DATA.is_dir():
        print(f"[ERR] 找不到模型目录：{DATA}")
        return 1

    # 参考位姿：sim 主模型的双手顶点，web 那份加完必须和它重合
    ref = _hand_verts(DATA / "elf3.xml")

    asset_errs = _check_web_assets()
    if asset_errs:
        print("[ERR ] web 手部网格资源不对：")
        for e in asset_errs:
            print(f"       {e}")
        return 1

    failed = 0
    touched_sim = touched_web = False

    for path, pos_map in SOURCES:
        rel = path.relative_to(ROOT).as_posix()
        if not path.exists():
            print(f"[SKIP] {rel} 不存在")
            continue

        before = _snapshot(path)
        raw = path.read_bytes()
        # 原文件是 LF，必须原样写回：read_text 会把换行 universal 成 "\n"，
        # 再用默认 newline 写盘会全篇变 CRLF（git 里看不出，但工作区脏了）
        newline = "\r\n" if b"\r\n" in raw else "\n"
        txt = raw.decode("utf-8")

        try:
            changed = []
            for side in SIDES:
                txt, ch = _patch(txt, side, pos_map.get(side, ""))
                changed.append(ch)
        except RuntimeError as e:
            print(f"[ERR ] {rel}: {e}")
            failed += 1
            continue

        if not any(changed):
            if not args.verify:
                print(f"[OK  ] {rel} 已有手部网格，跳过（幂等）")
                continue
            already = True
        else:
            already = False

        if args.check and not already:
            print(f"[DRY ] {rel} 需要加手（未写盘）")
            continue

        # 插入的行统一用 "\n"，这里再按原文件风格归一，避免半 CRLF 半 LF
        txt = txt.replace("\r\n", "\n")
        if newline == "\r\n":
            txt = txt.replace("\n", "\r\n")

        # 先写临时文件编译校验，通过了再覆盖，避免半截文件留在仓库里
        tmp = path.parent / f"_tmp_{path.name}"
        check_path = path
        if not already:
            tmp.write_text(txt, encoding="utf-8", newline=newline)
            check_path = tmp
        try:
            after = _snapshot(check_path)
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            print(f"[ERR ] {rel} 编译失败：{e}")
            failed += 1
            continue

        if after != before:
            tmp.unlink(missing_ok=True)
            print(f"[ERR ] {rel} 拓扑/质量变了：{before} -> {after}（本应是纯视觉改动）")
            failed += 1
            continue

        # 位姿自检：手顶点必须和 sim 的重合
        got = _hand_verts(check_path)
        tmp.unlink(missing_ok=True)
        bad = [f"{s} 手偏了 {g:.1f} mm" for s, v in got.items() if (g := _nn_gap_mm(ref[s], v)) > NN_TOL_MM]
        if bad:
            print(f"[ERR ] {rel} 手位姿不对：{'；'.join(bad)}（查 _check_web_assets / 手部 geom 属性）")
            failed += 1
            continue

        gap = max(_nn_gap_mm(ref[s], v) for s, v in got.items())
        if already:
            print(f"[OK  ] {rel} 已有手部网格，自检通过  nq/nv/nu={after[0]}/{after[1]}/{after[2]}  手位姿 {gap:.2f} mm ✔")
            continue

        path.write_text(txt, encoding="utf-8", newline=newline)
        touched_sim |= path.parent == DATA
        touched_web |= path.parent == WEB
        print(
            f"[OK  ] {rel} 已加手  nq/nv/nu={after[0]}/{after[1]}/{after[2]}  "
            f"mass={after[5]}（不变）  手位姿自检 {gap:.2f} mm ✔"
        )

    if failed:
        print(f"\n[FAIL] {failed} 个文件有问题，见上面 [ERR ]")
        return 1

    if touched_sim:
        print("\n下一步：python train/assets/make_robot_xml.py   # 重新派生 elf3_robot_only.xml")
    if touched_web:
        print("下一步：node web/tools/make-manifest.mjs         # 重写 assets/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
