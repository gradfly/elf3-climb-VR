"""跨 fast/normal 两档联合扫描「双腿分开」关节空间路径的时间分配参数。

    python scripts/check_prone_exit.py

在当前落盘的两套 standup/gait 轨迹上，把 u≥0.68 的 8 个腿关节角按候选
(by, bz, bk) 重新收敛到各自 gait 首帧，逐帧复算接触穿地、中间段穿地、
自碰撞与脚/膝离地峰值，列出两档都达标且腿最贴地的参数组合。

改过 KNEEL_POSES / EXIT_TIMING_* / ENTRY_STAGES / gait 首帧后应重跑一次。
注意：这是 FK 探针估计，最终以 scripts/check_prone_crawl.py 的实测为准。
"""
import sys
import numpy as np
import mujoco
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / "sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch"
DATA = LAUNCH.parent / "data/mujoco_simulation"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(LAUNCH))
import check_prone_crawl as C

m = C.Kin(DATA / "elf3_prone.xml")
BO = mujoco.mjtObj.mjOBJ_BODY
bid = lambda n: mujoco.mj_name2id(m.m, BO, n)
floor_ids = {g for g in range(m.m.ngeom)
             if (mujoco.mj_id2name(m.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
             .startswith("ut_infinite_ground")}
LEG = [f"{s}_{k}_joint" for s in "lr"
       for k in ("hip_y", "hip_x", "hip_z", "knee_y")]


def sm(x):
    x = float(np.clip(x, 0, 1))
    return x * x * x * (x * (x * 6 - 15) + 10)


class Case:
    def __init__(self, pre):
        tr = {"fast": "prone_standup_fast_traj.npz",
              "normal": "prone_standup_traj.npz"}[pre]
        gt = {"fast": "prone_gait_fast_traj.npz",
              "normal": "prone_gait_traj.npz"}[pre]
        z = np.load(DATA / tr, allow_pickle=True)
        self.q, self.dt = z["qpos"], float(z["dt"])
        self.names = [str(x) for x in z["joint_names"]]
        self.IDX = np.array([self.names.index(n) for n in LEG])
        self.qe = np.load(DATA / gt, allow_pickle=True)["qpos"][0]
        self.qs = self.q[170][7:].copy()
        self.qe = self.qe[7:].copy()

    def make(self, by, bz, bk):
        qq = self.q.copy()
        for i in range(170, len(self.q)):
            s = (i - 170) / 79
            w = np.array([sm(s / by), sm(s / bz), sm(s / bz), sm(s / bk)] * 2)
            qq[i, 7 + self.IDX] = (1 - w) * self.qs[self.IDX] + w * self.qe[self.IDX]
        return qq

    def ev(self, qq):
        hand, foot, hip, knee, sc = [], [], [], [], []
        for i in range(len(self.q)):
            m.set(qq[i])
            mujoco.mj_forward(m.m, m.d)
            hand.append(min(m.min_z(m.hand_geoms[s]) for s in "lr") * 1000)
            foot.append(min(m.min_z(m.foot_geoms[s]) for s in "lr") * 1000)
            hip.append(min(m.body_min_z(bid(f"{s}_{a}_link"))
                           for s in "lr" for a in ("hip_y", "hip_x", "hip_z")) * 1000)
            knee.append(min(m.body_min_z(bid(f"{s}_knee_y_link")) for s in "lr") * 1000)
            sc.append(sum(1 for c in range(m.d.ncon)
                          if m.d.contact[c].geom1 not in floor_ids
                          and m.d.contact[c].geom2 not in floor_ids))
        con = np.minimum(hand, foot)
        mid = np.minimum(hip, knee)
        return dict(contact=-min(con), mid=-min(mid), sc=sum(1 for v in sc if v > 0),
                    foot_hi=max(foot[170:]), knee_hi=max(knee[170:]),
                    hip_hi=max(hip[170:]))


F, N = Case("fast"), Case("normal")
print(" by   bz   bk | fast: 接触/中间/脚高/膝高 | normal: 接触/中间/脚高/膝高 | 判定")
rows = []
for by in (0.55, 0.65, 0.70, 0.75, 0.85):
    for bz in (0.55, 0.65, 0.70, 0.75, 0.85):
        for bk in (0.55, 0.70, 0.85, 1.00):
            a, b = F.ev(F.make(by, bz, bk)), N.ev(N.make(by, bz, bk))
            ok = (a["contact"] <= 10 and a["mid"] <= 6 and a["sc"] == 0
                  and b["contact"] <= 10 and b["mid"] <= 6 and b["sc"] == 0)
            score = max(a["foot_hi"], b["foot_hi"]) + 0.4 * max(a["knee_hi"], b["knee_hi"])
            if ok:
                rows.append((score, by, bz, bk, a, b))
rows.sort(key=lambda x: x[0])
for s, by, bz, bk, a, b in rows[:15]:
    print(f"{by:4.2f}{bz:5.2f}{bk:5.2f} | "
          f"{a['contact']:6.1f}{a['mid']:7.1f}{a['foot_hi']:8.1f}{a['knee_hi']:7.1f} | "
          f"{b['contact']:8.1f}{b['mid']:7.1f}{b['foot_hi']:8.1f}{b['knee_hi']:7.1f}")
print("TOP3：", [(round(by, 2), round(bz, 2), round(bk, 2))
                for _, by, bz, bk, _, _ in rows[:3]])
