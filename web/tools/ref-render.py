"""同姿态对照渲染：默认 qpos（全 0，基座 z=1.1），只画 group 2 视觉网格，正面视角。"""
import sys
from pathlib import Path
import numpy as np
import mujoco

CWD = Path(r"E:\mujoco\robot test")
xml = CWD / "elf3.xml"

model = mujoco.MjModel.from_xml_path(str(xml))
W = H = 700
model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), W)
model.vis.global_.offheight = max(int(model.vis.global_.offheight), H)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("qpos0 =", np.round(data.qpos, 4))
print("base z =", float(data.qpos[2]))

opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
for g in range(6):
    opt.geomgroup[g] = 0
opt.geomgroup[2] = 1
opt.geomgroup[0] = 1
opt.sitegroup[:] = 0
opt.flags[int(mujoco.mjtVisFlag.mjVIS_JOINT)] = 0
opt.flags[int(mujoco.mjtVisFlag.mjVIS_ACTUATOR)] = 0

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
cam.distance = 2.6
cam.lookat[:] = [0.0, 0.0, 0.78]
cam.azimuth = 90.0     # 正面
cam.elevation = -8.0

renderer = mujoco.Renderer(model, height=H, width=W)
renderer.update_scene(data, camera=cam, scene_option=opt)
img = renderer.render()
out = CWD / "render" / "cmp_front_zero.png"
out.parent.mkdir(exist_ok=True)
import imageio.v2 as imageio
imageio.imwrite(out, img)
print("已写出", out)

# 逐几何体世界 AABB（视觉组）
print()
print("gid  mesh                     Zmin     Zmax    Xmin    Xmax")
rows = []
for gi in range(model.ngeom):
    if int(model.geom_group[gi]) != 2 or int(model.geom_type[gi]) != 7:
        continue
    mid = int(model.geom_dataid[gi])
    adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    v = model.mesh_vert[adr:adr + num]
    R = data.geom_xmat[gi].reshape(3, 3)
    w = v @ R.T + data.geom_xpos[gi]
    rows.append((gi, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid), w[:, 2].min(), w[:, 2].max(), w[:, 0].min(), w[:, 0].max()))
rows.sort(key=lambda r: -r[3])
for r in rows:
    print(f"{r[0]:3d}  {r[1]:22s} {r[2]:8.4f} {r[3]:8.4f} {r[4]:8.4f} {r[5]:8.4f}")
allz = [r[2] for r in rows]
print("整体 Z 范围:", round(min(allz), 4), "→", round(max(r[3] for r in rows), 4))
