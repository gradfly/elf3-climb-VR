"""ELF3 双手双脚贴地爬行（prone crawl）—— 开环动作序列生成器。

输出与 crab_climb.py 完全同格式的 npz（time / qpos / dt），所以现有回放器
crab_replay.py 可原样播放；额外带 contact_flags / joint_names / meta 供校验脚本用。

==================== 姿态定义 ====================
俯卧基座：pos=(x, 0, body_z)、quat=(0.7071, 0, 0.7071, 0)（绕 +Y 前俯 90°）。
此时机器人局部轴与世界轴的映射是：
    body +X -> 世界向下     body +Z -> 世界向前     body +Y -> 世界左侧
由此决定四肢各关节在爬行里的角色（本机 FK 实测灵敏度，见 docs/prone_crawl.md）：

    hip_y / shoulder_y   绕世界 Y，抬离地面          +0.25 rad -> 末端 ↑0.23 m
    hip_x / shoulder_x   绕世界竖直轴，水平面方位角   ±0.20 rad -> 末端水平 0.03~0.12 m
    hip_z / shoulder_z   沿肢体自身轴扭转            ±0.25 rad -> 末端仅 0.005 m
    knee_y / elbow_y     折叠，改变接触半径
    ankle_y              足尖压地角

**注意**：前后扫掠（推进/前伸）由 hip_x / shoulder_x 承担，不是 hip_z / shoulder_z
—— 后者在俯卧系里几乎是纯扭转，写成扫掠驱动会导致"原地扭、不前进"。

==================== 动作序列 ====================
两步：过渡段（站姿 -> 俯卧，2.6 s）+ 爬行循环（默认 T=2.0 s、duty=0.75、0.12 m/s）。
四拍对角序列：左臂 -> 右腿 -> 右臂 -> 左腿（每次只抬一条，始终 3 点支撑）。
支撑相接触点在世界系固定不动（由 IK 保证，天然零滑移），摆动相抬 clearance 前移。
每个支撑相内身体正好前进 v·duty·T，所以前后扫掠行程 = v·duty·T = 0.18 m；
每循环前进 v·T = 0.24 m。

==================== 用法 ====================
    python prone_crawl.py                 # 生成场景 + 轨迹 + 打印摘要
    python prone_crawl.py --cycles 6 --speed 0.10    # 改参数重生成
    python prone_crawl.py --no-entry      # 只要爬行循环，不要站->卧过渡
    python prone_crawl.py --check-only    # 只跑求解+摘要，不写 npz
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "mujoco_simulation"
SCENE = DATA / "elf3_prone.xml"
FLAT_XML = DATA / "terrains" / "universal_test" / "prone_flat.xml"
DEFAULT_OUT = DATA / "prone_crawl_traj.npz"

ORIG_INCLUDE = (
    '<include file="terrains/universal_test/universal_test_terrain_compact.xml"/>'
)
PRONE_INCLUDE = '<include file="terrains/universal_test/prone_flat.xml"/>'

# 出生点 = 轨迹起点 = 世界原点 (0,0)。主光要让**光斑中心**压在出生点上。
#
# 定位光在 P、方向 D 时，光线落到地面 (z=0) 的点是 P.xy + (−P.z/D.z)·D.xy。
# 所以把灯直接放到出生点上方是不够的：dir 一斜，光斑就被甩出去
# （实测 dir=(-0.35,0.3,-1)、灯在 (0,0,2.6) 时，光斑中心在 (−0.91, +0.78)，偏了 1.2 m）。
# 这里按上式反解 P.xy，保证 dir 怎么斜，光斑中心都正好落在 SPAWN_XY 上。
SPAWN_XY = (0.0, 0.0)
LIGHT_DIR = (-0.35, 0.30, -1.0)        # 照明方向（略斜；正下方直射会压出一块纯黑影子）
LIGHT_Z = 2.6
LIGHT_XY = tuple(SPAWN_XY[i] + LIGHT_Z / LIGHT_DIR[2] * LIGHT_DIR[i]
                 for i in range(2))     # ≈ (0.910, −0.780)

# 地面：用 40x40 厚板代替无限平面（plane 在离屏渲染里不显示，见 write_scene）。
FLOOR_HALF = 20.0                      # 半边尺寸 m
FLOOR_HALF_T = 0.05                    # 半厚 m（顶面正好在 z=0）
# 地面：48x48 的棋盘格，每格 1 m。纯色地板在俯视镜头里没有任何距离参照，
# 看上去和"悬空"没区别；格子一铺，姿态和位移立刻能读出来。
FLOOR_RGB1 = "0.56 0.57 0.56"          # 亮格
FLOOR_RGB2 = "0.44 0.45 0.44"          # 暗格
FLOOR_TEXREPEAT = 24                   # 40 m 上重复 24 次 -> 每格 ~0.83 m
# 场景自带的平行光：压低一点，让定位主光成为亮度主导，光斑中心才看得出来。
WORLD_LIGHT_DIFFUSE = "0.55 0.55 0.55"
WORLD_LIGHT_AMBIENT = "0.30 0.30 0.30"
# 定位主光（光斑中心 = 出生点）：diffuse 拉满，四周自然渐暗，能直接看出"光中心"。
LIGHT_DIFFUSE = "1.0 1.0 1.0"
LIGHT_AMBIENT = "0.18 0.18 0.18"

# 贴地爬行时这几个关节的"工作范围"（比机械限位窄得多）。
# 它们的机械限位都是 ±165°，IK 一旦被约束推着走，就会绕着肩/髋翻转 160° 把整条
# 肢体甩到身体另一侧、从地面下方穿过去（实测 shoulder_y 跑到 −162°、肘部插地 11 cm）。
# 这里按人趴着的实际姿态收窄，翻转解直接被裁掉。
SOFT_LIMITS_DEG = {
    "shoulder_y": (-95.0, 95.0),
    "shoulder_z": (-100.0, 100.0),
    "hip_y": (-95.0, 95.0),
    # hip_x 机械限位 [-28°, 175°]（右腿镜像），让出 6° 机构余量。
    # 试过把软下限收紧到 −11° 想从 IK 源头禁掉内收：自碰撞确实归零，但 IK
    # 只能用 hip_z 猛旋去够脚目标，大腿胶囊反过来扎进地面 31 mm（帧 186~202），
    # 抬基座 / 抬脚目标 / 加大软地面权重三种补偿都更糟。所以保留 −22° 的
    # 无穿地解，改由收尾的 spread_hips() 小幅外展消碰撞。
    "hip_x": (-22.0, 170.0),
    "hip_z": (-130.0, 130.0),
    # 肘/膝：过渡段中途会顶到机械限位（实测 elbow −54° vs 限位 −55°，余量 1°）。
    # 让出 7~8° 后 IK 残差从 0.3 mm 涨到 27 mm —— 那时手还在空中往地面落，
    # 2.7 cm 的位置误差在视觉上看不出来，换 7° 的机构余量是划算的。
    "elbow_y": (-48.0, 95.0),
    "knee_y": (-5.0, 142.0),
}

# 29 个受控关节，顺序 = XML 声明顺序 = qpos[7:36]
JOINT_NAMES = [
    "waist_y_joint", "waist_x_joint", "waist_z_joint",
    "l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint",
    "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint",
    "r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint",
    "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint",
    "l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint",
    "l_elbow_y_joint", "l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint",
    "r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint",
    "r_elbow_y_joint", "r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint",
]

# 站立默认角（elf3_play._DEFAULT_JOINT_POS，29 个）
STAND_VEC = np.array([
    0.0, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, 0.0, 0.0,
    -0.3, 0.0, 0.0, 0.6, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)

STAND_Z = 1.0339          # 站立基座高（robot_cfg.py INIT_BASE_Z）
PRONE_HALF_PI = math.pi / 2.0
_DBG_KEYS = bool(os.environ.get("ELF3_DEBUG_KEYS"))   # 打印过渡段关键姿态求解结果


# ---------------------------------------------------------------- 参数
@dataclass
class Params:
    """动作序列参数。改这里就等于改序列。"""

    # 时间 / 速度
    dt: float = 0.02               # 采样周期 s（50 Hz，与训练控制频率一致）
    cycle_time: float = 2.0        # 一个完整爬行循环 s
    speed: float = 0.12            # 目标前进速度 m/s
    duty: float = 0.75             # 支撑相占空比（4 拍步态必须 >0.75，才能始终 3 点支撑）
    cycles: int = 4                # 爬行循环个数
    entry_time: float = 5.0        # 站姿 -> 俯卧过渡时长 s（四阶段：蹲下/双手撑地/双脚后伸/趴姿）
    # 四肢相位（四拍/对角序列：左臂 -> 右腿 -> 右臂 -> 左腿，每次只抬一条）
    offsets: tuple = (0.0, 0.25, 0.50, 0.75)

    # 俯卧机体
    body_z: float = 0.10           # 躯干原点离地高（腹部留 1 cm，torso 胶囊半径 0.09）
    body_bob: float = 0.006        # 躯干上下起伏（2 倍频）
    body_sway: float = 0.065       # 躯干左右重心转移（1 倍频，见 build 里的相位推导）
    body_roll: float = 0.035       # 躯干绕前进轴滚转 rad
    waist_yaw: float = 0.05        # waist_z 反向配平 rad
    entry_x0: float = 0.14         # 过渡结束时基座 x，也是循环起点

    # 四肢
    clearance: float = 0.035       # 摆动相抬离地面高度 m
    press: float = 0.005           # 支撑相预压（接触点目标略低于地面）m
    swing_up: float = 0.18         # 抬腿时刻占比（与 crab_climb.swing_z 同义）
    arm_touch: tuple = (0.42, 0.36)   # 手掌触地点（身体系 x 前 / y 左，米）；y 加宽=双臂分开
    leg_touch: tuple = (-0.75, 0.36)  # 踝关节原点触地点（身体系 x 前 / y 左，米）

    # 求解
    ik_iters: int = 60
    ik_null: float = 0.02          # 零空间朝设计姿态收敛的步长（>0.1 会在跨段时翻肘/翻膝）
    entry_ik_null: float = 0.02    # 过渡段的零空间步长
    # 生成后对下蹲窗口（f0~PLANT_U）的 hip_x/hip_z 做端点羽化的高斯低通（帧）。
    # 深蹲时这两个冗余自由度几乎不影响脚位置，逐帧 IK 会在解流形上左右漂移，
    # 真机上就是「蹲着的时候腿在抖」。sigma=2.0 实测下蹲段二阶差分 rms
    # 1.29/1.20°→0.13/0.12°、一阶变号率 26/29%→8.6%；低通让脚下沉约 2 mm，
    # 由随后一次 ground_clamp 补回（穿地仍 6.4 mm）。0 表示关闭。
    # 别用硬端点保持，会在窗口边界造出新的 C1 断点（|d2|max 6.3→14.9°）。
    # 也别改 IK 的零空间先验：试过关成 0，中间段穿地 17.4 mm、自碰撞 111 帧。
    # 抖动必须在关节空间后处理，不能动已标定好的 IK 解。
    leg_shake_sigma: float = 2.0
    # 中间段（肩/上臂/前臂/髋/膝）的"软地面"高度：低于它就在 IK 里顶回来。
    # 只给 2 mm 的余量，目的是别让胶囊压进地面，不是把肘/膝抬起来。
    chain_margin: float = 0.002
    chain_weight: float = 3.0          # 软地面约束权重（相对末端点任务）
    # 中间段允许多少穿地（m）。超过这个值就判定"解落进了坏分支"，换起点重解。
    pierce_tol: float = 0.004
    # 关节速度上限（deg/s）。IK 的分支跳变一步能走 60~140°，远超电机能力；
    # 按这个上限重铺一遍，跳变就被摊成连续的上坡。180°/s ≈ 3.6°/帧。
    max_joint_speed_dps: float = 120.0
    # 支撑相的限速放宽倍数。限速会把支撑脚拖着走（实测滑移从 21 mm 涨到 195 mm），
    # 所以支撑相基本不做限速，只在摆动相传动作为主。
    support_speed_scale: float = 1e9
    # --- 关节活动范围 / 冗余自由度 ---
    # 过渡段/爬行段对 hip_y、knee_y 软限位的单独覆盖（度）。
    # 跪姿（膝触地）需要 hip_y 接近 −130°、knee_y 接近 145°，
    # 用爬行段那套 ±95 / 142 会双双卡死（实测 IK 残差 334 mm、膝盖冲到地下 −25 cm）。
    hip_y_limit: float = 95.0
    knee_hi_limit: float = 142.0
    # hip_z 的"内侧禁区"（度）。>0 时强制 |hip_z| ≥ 这个值，绕开 ±90° 万向锁。
    # 只对爬行段开（过渡段站姿 hip_z≈0，开了会解不出来）。
    hip_z_band: float = 0.0
    # 冻结 roll 关节（绕肢体纵轴）不参与 IK。它是纯冗余自由度：4 关节解 3D 位置
    # 有 1 维冗余，IK 就在支撑相被推着游走，产生 20~140° 的单帧跳变。
    # 实测：只冻手臂 → 角加速度 343→112 rad/s²；腿不能冻（髋部胶囊压地 -13.7 mm）。
    freeze_arm_roll: bool = False
    freeze_leg_roll: bool = False
    # 过渡段专用的肘/膝离地余量（爬行段用 chain_margin，那个在 fast 档会卡死 IK）
    entry_chain_margin: float = 0.006
    # 过渡段→循环段交界必须强制限速（不分支撑/摆动），否则那一帧会跳 60°+。
    force_window_pad: int = 0
    roll_smooth: float = 4.0      # roll 关节高斯平滑 sigma（帧），0=关
    roll_speed_dps: float = 1e9   # roll 关节速度上限（1e9=不限；限了会影响末端）
    # 重解时给"肩/髋滚转"关节（链上第 1 个关节）试的起点（rad）
    restart_angles: tuple = (-1.05, -0.52, 0.0, 0.52, 1.05)
    contact_passes: int = 5        # 接触点闭环迭代次数（手要同时锁 xy 和 z）
    # 过渡段踝角 (归一化时刻, rad)。爬行段固定 -0.55，但站立/深蹲时脚掌要平贴地面：
    # 实测 ankle_y=0 时脚底穿地 40 mm、z 轴前倾 17°；深蹲小腿前倾，需要 -0.5 左右。
    # 末帧必须落在 -0.55，否则接上爬行段会跳变。
    # v8 不下跪 ⇒ **全程负值**（v7 的 +42° 翻脚背那套已不需要）：深蹲时背屈要到位
    # （-0.78 = -44.7°，机械下限 -50°，留 5.3° 余量，验收判据 ≥3°）；
    # 前倾到 72° 时小腿更前倾、脚掌贴不平，0.82 起回到 -0.60 → 脚跟自然略抬，
    # 由 ground_clamp 兜底保证脚尖不穿地。别再往 -0.80 以下压：会顶死限位。
    # v9：蹲下段背屈（脚掌压住蹲姿触地点）→ 双脚后伸时腿伸直、踝回到接近中立
    # → 收腿段再回到爬行段的 −0.55，与 gait 首帧的踝角对齐。
    # 后伸期不能沿用深蹲的 −0.55：脚掌会翻成脚尖点地，踝胶囊穿地 10~20 mm。
    entry_ankle: tuple = ((0.00, -0.10), (0.12, -0.32), (0.24, -0.50),
                          (0.34, -0.52), (0.44, -0.50), (0.56, -0.30),
                          (0.68, -0.18), (0.74, -0.28), (0.88, -0.50),
                          (1.00, -0.55))
    # v9：双脚后伸段把 hip_z 锁在矢状面内（|hip_z| ≤ 这个值，度）。
    # 腿 4 自由度解 3D 任务有 1 维冗余，脚一开始往后蹬 warm start 就跳到
    # hip_z≈128°/hip_x≈71° 的蛙式分支（膝 141°、踝 x 来回甩、大腿自碰撞）。
    # u≥EXIT_U 必须放开，否则 exit_leg_joints 收敛不到 gait 首帧的 hip_z=+95°。
    entry_hip_z_limit: float = 25.0


LIMBS = [
    # 四拍对角序列（左臂 -> 右腿 -> 右臂 -> 左腿），相位见 Params.offsets。
    # 为什么不是对角两拍：两拍时同时只有 2 个支撑点（对顶点），质心必然落在支撑
    # 线段之外（实测只有 34% 的帧在支撑多边形内），静态根本站不住。
    dict(name="l_arm", side="l", arm=True, idx=0),
    dict(name="r_leg", side="r", arm=False, idx=1),
    dict(name="r_arm", side="r", arm=True, idx=2),
    dict(name="l_leg", side="l", arm=False, idx=3),
]


def limb_chain(arm: bool, side: str) -> list[str]:
    if arm:
        return [f"{side}_shoulder_y_joint", f"{side}_shoulder_x_joint",
                f"{side}_shoulder_z_joint", f"{side}_elbow_y_joint"]
    return [f"{side}_hip_y_joint", f"{side}_hip_x_joint",
            f"{side}_hip_z_joint", f"{side}_knee_y_joint"]


def mirror_chain(vec) -> np.ndarray:
    """左肢关节角 -> 右肢：第 1、2 个关节（shoulder_x/z 或 hip_x/z）取反。

    实测：站姿 l_hip_y=−17.2/h_x=0/h_z=0/knee=34.4 与右侧完全相同；
    过渡段某个关键姿态左腿 [−104.5, +55.6, +72.0, 113.9] 对应右腿
    [−104.5, −55.6, −72.0, 113.9]。所以镜像规则就是"横向的两个关节取反"。
    """
    v = np.array(vec, dtype=float).copy()
    v[1] = -v[1]
    v[2] = -v[2]
    return v


def limb_guard_bodies(arm: bool, side: str) -> list[str]:
    """肢体链上除"接触末端"以外、必须离地的 body。

    这些 body 挂着碰撞胶囊（肩 3.5~4 cm、上臂/前臂、髋 5 cm、膝 5 cm 半径），
    贴地爬行时最容易整段插进地面 —— IK 只管末端点，中间段没人管就会穿 10 cm 以上。
    """
    if arm:
        return [f"{side}_shoulder_y_link", f"{side}_shoulder_x_link",
                f"{side}_shoulder_z_link", f"{side}_elbow_y_link"]
    return [f"{side}_hip_y_link", f"{side}_hip_x_link",
            f"{side}_hip_z_link", f"{side}_knee_y_link"]


def limb_probe(arm: bool, side: str) -> str:
    """IK 目标 body：踝关节原点 / 腕关节原点（踝、腕自身的三个关节不改变这两个原点）。"""
    return f"{side}_wrist_z_link" if arm else f"{side}_ankle_x_link"


# ---------------------------------------------------------------- 设计姿态
def design_joints(p: Params, arm: bool, side: str, s: float, lift: float) -> np.ndarray:
    """设计姿态（IK 的零空间先验）：s=0 前极限，s=1 后极限，lift=0..1 抬离量。

    只决定"肘/膝朝哪边弯、肢体怎么摆"，最终接触点由 IK 落到目标上，所以幅值
    不用精确；但**不能落在关节限位或奇异位形上**：
      肘 elbow_y=+90° 是伸直奇异（腕离肩最远 0.512 m），设计值取 0.35 远离它；
      膝 knee_y≈0 是伸直奇异，设计值取 1.95（弯 ~112°）。
    这条坑实测过：设计肘 1.75（=105°，刚过伸直）时 IK 每步只能收敛一半，
    60 次迭代还差 12 mm。
    """
    sg = 1.0 if side == "l" else -1.0
    if arm:
        return np.array([
            -0.06 + 0.30 * lift,              # shoulder_y  抬起
            sg * (1.59 + 0.30 * (0.5 - s)),   # shoulder_x  水平扫掠（主力）
            sg * 1.48,    # shoulder_z  把肘拱到背侧，前臂才能离地
            0.04 + 0.60 * lift,               # elbow_y     摆动时收臂
        ])
    return np.array([
        -0.17 + 0.30 * lift,                  # hip_y       抬起
        sg * (1.28 + 0.16 * (0.5 - s)),       # hip_x       水平扫掠（主力）
        sg * 1.45,                            # hip_z       把大腿转到侧后
        1.60 + 0.20 * lift,                   # knee_y      摆动时收腿
    ])


def limb_extra(p: Params, side: str, arm: bool,
               ankle_y: float | None = None) -> dict[str, float]:
    """IK 链之外、固定不动的踝/腕角度。

    `ankle_y` 只对腿有效：过渡段要按阶段给（站/蹲要放平脚掌），爬行段固定 -0.55。
    """
    if arm:
        return {f"{side}_wrist_x_joint": 0.0, f"{side}_wrist_y_joint": 0.0,
                f"{side}_wrist_z_joint": 0.0}
    val = -0.55 if ankle_y is None else float(ankle_y)
    return {f"{side}_ankle_y_joint": val, f"{side}_ankle_x_joint": 0.0}


def entry_ankle_y(p: Params, u: float) -> float:
    """过渡段归一化时刻 -> 踝角（rad），线性插值 `Params.entry_ankle`。"""
    keys = p.entry_ankle
    u = min(max(u, 0.0), 1.0)
    if u <= keys[0][0]:
        return keys[0][1]
    for i in range(len(keys) - 1):
        u0, u1 = keys[i][0], keys[i + 1][0]
        if u <= u1:
            # 用 smootherstep：跪姿段踝角在 0.06 的 u 里要翻 61°，线性插值会在
            # 折点留下加速度尖峰（验收对 qpos 全关节算二阶差分）。
            w = smootherstep((u - u0) / max(u1 - u0, 1e-9))
            return keys[i][1] + (keys[i + 1][1] - keys[i][1]) * w
    return keys[-1][1]


# ---------------------------------------------------------------- 运动学工具
class Kin:
    """包一层 MuJoCo，提供 qpos 索引、前向运动学、IK、接触高度。"""

    def __init__(self, scene: Path, hip_z_band: float = 0.0,
                 hip_y_limit: float = 95.0, knee_hi_limit: float = 142.0):
        self.m = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        self.d = mujoco.MjData(self.m)
        if self.m.nq != 38:
            raise SystemExit(f"[错误] 期望 nq=38，实际 {self.m.nq}（模型不是 elf3 主链？）")
        JO, BO, GE = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY,
                      mujoco.mjtObj.mjOBJ_GEOM)
        self.qadr = {n: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, JO, n)]
                     for n in JOINT_NAMES}
        self.dadr = {n: self.m.jnt_dofadr[mujoco.mj_name2id(self.m, JO, n)]
                     for n in JOINT_NAMES}
        self.lo = np.array([self.m.jnt_range[mujoco.mj_name2id(self.m, JO, n)][0]
                            for n in JOINT_NAMES])
        self.hi = np.array([self.m.jnt_range[mujoco.mj_name2id(self.m, JO, n)][1]
                            for n in JOINT_NAMES])
        for stem, (a, b) in SOFT_LIMITS_DEG.items():
            for side in "lr":
                # 右腿 hip_x 的机械限位是 [-175°, +28°]（左腿镜像 [-28°, +175°]）。
                # 软限位必须跟着镜像，否则对右腿等于没设（min(170°, 28°)=28°），
                # 照样能顶到机械上限、验收余量 0°。
                aa, bb = (-b, -a) if (stem == "hip_x" and side == "r") else (a, b)
                i = JOINT_NAMES.index(f"{side}_{stem}_joint")
                self.lo[i] = max(self.lo[i], np.radians(aa))
                self.hi[i] = min(self.hi[i], np.radians(bb))
        # 按动作类型覆盖（跪姿需要更大的髋/膝行程，见 Params 注释）。
        # 注意不能用 min(self.hi, ...) —— 那样只会更严，永远放不宽；要以机械限位为界。
        self.mech_lo = np.array([self.m.jnt_range[mujoco.mj_name2id(self.m, JO, n)][0]
                                 for n in JOINT_NAMES])
        self.mech_hi = np.array([self.m.jnt_range[mujoco.mj_name2id(self.m, JO, n)][1]
                                 for n in JOINT_NAMES])
        for side in "lr":
            i = JOINT_NAMES.index(f"{side}_hip_y_joint")
            self.lo[i] = max(self.mech_lo[i], np.radians(-hip_y_limit))
            self.hi[i] = min(self.mech_hi[i], np.radians(hip_y_limit))
            j = JOINT_NAMES.index(f"{side}_knee_y_joint")
            self.hi[j] = min(self.mech_hi[j], np.radians(knee_hi_limit))
        # hip_z 不能跨过 ±90°（万向锁：那里 hip_x 与 knee_y 的轴平行，解不唯一）。
        # 实测腿在 −91° → −108° 之间穿越时会先冲出去再回摆，把 a_max 从 158 抬到 218、
        # jerk 从 6600 抬到 10100 —— 这就是爬行时"卡一下"的直接来源。
        # 贴地爬行时腿本来就一直外展（|hip_z| > 90°），所以按侧把内侧边界顶到 band。
        # 过渡段（站姿 hip_z≈0）不能用，所以做成按动作开的参数。
        if hip_z_band > 0.0:
            for side, sign in (("l", 1.0), ("r", -1.0)):
                i = JOINT_NAMES.index(f"{side}_hip_z_joint")
                if sign > 0:
                    self.lo[i] = max(self.lo[i], np.radians(hip_z_band))
                else:
                    self.hi[i] = min(self.hi[i], np.radians(-hip_z_band))
        self.body_id = {b: mujoco.mj_name2id(self.m, BO, b) for b in
                        ("torso_link", "l_wrist_z_link", "r_wrist_z_link",
                         "l_ankle_x_link", "r_ankle_x_link")}
        self.geom_ids = {name: [g for g in range(self.m.ngeom)
                                if (mujoco.mj_id2name(self.m, GE, g) or "") == name]
                         for name in ("l_wrist_z_collision", "r_wrist_z_collision")}
        for side in "lr":
            self.geom_ids[f"{side}_foot"] = [
                g for g in range(self.m.ngeom)
                if (mujoco.mj_id2name(self.m, GE, g) or "").startswith(f"{side}_foot")]
        self.jacp = np.zeros((3, self.m.nv))
        self.freeze_roll = (False, False)  # (arm_roll, leg_roll) 是否退出 IK（见 make_limb_col）
        # body -> 它的 collision geom（group 3），"整链不穿地"要靠它逐段查
        self.body_geoms: dict[int, list[int]] = {}
        for g in range(self.m.ngeom):
            if self.m.geom_group[g] == 3:
                self.body_geoms.setdefault(int(self.m.geom_bodyid[g]), []).append(g)

    # -- 基础 ------------------------------------------------------
    def fwd(self):
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def set_q(self, qpos38):
        self.d.qpos[:] = qpos38
        self.fwd()

    def write_joints(self, q38: np.ndarray, vec29: np.ndarray) -> None:
        for i, n in enumerate(JOINT_NAMES):
            q38[self.qadr[n]] = vec29[i]

    def joint_vec(self, q38: np.ndarray) -> np.ndarray:
        return np.array([q38[self.qadr[n]] for n in JOINT_NAMES])

    # -- 接触高度 --------------------------------------------------
    def min_z(self, prefix: str) -> float:
        """该组 collision geom 的最低点 z（胶囊/球用解析式，其余退化为球心 - r）。"""
        zs = []
        for g in self.geom_ids[prefix]:
            pos = self.d.geom_xpos[g]
            r = float(self.m.geom_size[g][0])
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = self.d.geom_xmat[g].reshape(3, 3)
                h = float(self.m.geom_size[g][1])
                zs.append(pos[2] - (h * abs(R[2, 2]) + r))
            else:
                zs.append(pos[2] - r)
        return float(min(zs)) if zs else 9.9

    def body_min_z(self, bid: int) -> float:
        """某个 body 上所有碰撞几何的最低点 z（胶囊按轴向投影解析求解）。"""
        zs = []
        for g in self.body_geoms.get(bid, ()):
            pos = self.d.geom_xpos[g]
            r = float(self.m.geom_size[g][0])
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = self.d.geom_xmat[g].reshape(3, 3)
                zs.append(pos[2] - (float(self.m.geom_size[g][1]) * abs(R[2, 2]) + r))
            else:
                zs.append(pos[2] - r)
        return float(min(zs)) if zs else 9.9

    def chain_min_z(self, col: dict) -> float:
        """该肢体链（含接触末端）的最低点 z。"""
        zs = [self.body_min_z(b) for b in col["guards"]]
        zs.append(self.min_z(col["geom"]))
        return float(min(zs))

    def lowest_xy(self, prefix: str) -> np.ndarray:
        """接触面形心 (x, y)：取所有"离最低点不超过 3 mm"的 geom 坐标均值。"""
        lo, pts = 9.9, []
        for g in self.geom_ids.get(prefix, []):
            pos = self.d.geom_xpos[g]
            r = float(self.m.geom_size[g][0])
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                R = self.d.geom_xmat[g].reshape(3, 3)
                z = pos[2] - (float(self.m.geom_size[g][1]) * abs(R[2, 2]) + r)
            else:
                z = pos[2] - r
            lo = min(lo, z)
            pts.append((z, pos[:2].copy()))
        near = [p for z, p in pts if z <= lo + 3e-3]
        return np.mean(near, axis=0) if near else np.zeros(2)

    # -- IK --------------------------------------------------------
    def solve_limb(self, col: dict, target, q_nom: np.ndarray, p: Params,
                   guard_floor: float | None = None) -> float:
        """阻尼最小二乘 IK + 零空间姿态先验，返回末端残差（m）。

        为什么不能用「JᵀJ + wI 同时吃先验」：站立姿态下腿几乎伸直，末端 z 对关节
        的灵敏度只有 0.1，先验项会把任务残差钉死在 40% 上（实测 3.7 mm 都追不回来）。
        这里先解任务、再把零空间分量投影给先验，任务误差才能收敛到 0。
        """
        d, m = self.d, self.m
        cols, adrs = col["dadr"], col["qadr"]
        lo, hi = col["lo"], col["hi"]
        bid = col["body"]
        tgt = np.asarray(target, dtype=float)
        n = len(cols)
        qn = np.asarray(q_nom, dtype=float)
        q_nom = qn[len(qn) - n:] if len(qn) > n else qn
        eye = np.eye(n)
        err = 9.9
        for _ in range(p.ik_iters):
            self.fwd()
            e = tgt - d.xpos[bid]
            err = float(np.linalg.norm(e))
            # 中间段触到"软地面"的量：只在低于 guard_floor 时激活，
            # 目标只是别压进地面，不会把肘/膝往上拽（拽高了反而破坏姿态）。
            viol = []
            if guard_floor is not None:
                for gb in col.get("guards", ()):
                    z = self.body_min_z(gb)
                    if z < guard_floor:
                        viol.append((gb, guard_floor - z))
            if err < 3e-4 and not viol:
                break
            mujoco.mj_jacBody(m, d, self.jacp, None, bid)
            J = self.jacp[:, cols].copy()
            q = d.qpos[adrs]
            A = J.T @ J + 1e-6 * eye
            b = J.T @ e
            for gb, need in viol:
                mujoco.mj_jacBody(m, d, self.jacp, None, gb)
                Jg = self.jacp[2:3, cols]
                A += p.chain_weight * (Jg.T @ Jg)
                b += p.chain_weight * (Jg.T[:, 0] * need)
            Jpinv = np.linalg.solve(J.T @ J + 1e-6 * eye, J.T)
            dq = np.linalg.solve(A, b)
            dq += (eye - Jpinv @ J) @ (p.ik_null * (q_nom - q))
            d.qpos[adrs] = np.clip(q + np.clip(dq, -0.30, 0.30), lo, hi)
        self.fwd()
        return err

    def solve_limb_robust(self, col: dict, target, q_nom: np.ndarray,
                          p: Params) -> float:
        """先正常解；若整条链穿地，就换几个"肩/髋滚转"起点重解，取最好的分支。

        为什么需要它：手离肩最近只有 0.25 m（臂长 0.512），肘必须大角度折叠，
        而折叠有"肘朝上/朝下"两个分支。IK 从上一帧 warm start 出发，一旦落进
        "肘朝下"的分支就再也出不来 —— 实测肘插到 −0.11 m、上臂整根在地面下。
        这不是关节角不合理，是解的分支选错了，换个起点就能回到正常解。
        """
        adr = col["qadr"]
        floor = None if p.chain_margin <= 0 else p.chain_margin
        q_prev = self.d.qpos[adr].copy()      # 上一帧的解（warm start），用来判连续
        err = self.solve_limb(col, target, q_nom, p, floor)
        cands = [(self.chain_min_z(col), err, self.d.qpos[adr].copy())]
        if cands[0][0] >= -p.pierce_tol and err < 2e-3:
            return err
        # 从"设计姿态"重启一次。IK 从上一帧 warm start 出发时，一旦某个关节被
        # clip 到限位（实测 elbow 顶死 −55°）就再也爬不出来，连着 7 帧残差 36 cm，
        # 然后突然跳到另一个分支 —— 那一下就是 1500 rad/s² 的来源。
        qn = np.asarray(q_nom, dtype=float)
        qn = qn[len(qn) - len(adr):] if len(qn) > len(adr) else qn
        self.d.qpos[adr] = np.clip(qn, col["lo"], col["hi"])
        self.fwd()
        e = self.solve_limb(col, target, q_nom, p, floor)
        if e <= 3e-3:
            cands.append((self.chain_min_z(col), e, self.d.qpos[adr].copy()))
        for ang in p.restart_angles:
            self.d.qpos[adr] = q_prev
            self.d.qpos[adr[0]] = float(np.clip(ang, col["lo"][0], col["hi"][0]))
            self.fwd()
            e = self.solve_limb(col, target, q_nom, p, floor)
            if e <= 3e-3:                      # 只收收敛到目标的候选
                cands.append((self.chain_min_z(col), e, self.d.qpos[adr].copy()))
        # 先筛"不穿地 + 末端到位"，再在这批里挑**离上一帧最近**的。
        # 只按"链最低最高"挑会挑出另一个折叠分支：相邻帧 shoulder_y/z 跳 66°，
        # 二阶差分直接炸到 5e4 deg/s²（实测）。连续性必须和穿地一起判。
        ok = [c for c in cands if c[0] >= -p.pierce_tol and c[1] <= 2e-3]
        if ok:
            best = min(ok, key=lambda c: float(np.abs(c[2] - q_prev).max()))
        else:
            best = max(cands, key=lambda c: (c[0], -c[1]))
        self.d.qpos[adr] = best[2]
        self.fwd()
        return best[1]

    def make_limb_col(self, arm: bool, side: str) -> dict:
        """肢体求解列。前 3 个关节参与 IK，第 1 个（shoulder_y / hip_y）被排除。

        为什么排除它：这一个是绕肢体纵轴的 roll，对末端位置几乎没影响（纯冗余）。
        放进 IK 后它会落在零空间里自由游走 —— 实测过渡段帧 4→5 的 hip_z 跳 93°、
        帧 100→101 的 shoulder_y 跳 135°，二阶差分到 1500 rad/s²。
        改成由 design_joints 的平滑曲线直接指定，IK 只解剩下 3 个自由度，
        维数正好、解唯一，就不会再分支跳变（实测 3 关节解同一目标残差 0.00 mm）。
        """
        chain = limb_chain(arm, side)
        frozen = self.freeze_roll[0] if arm else self.freeze_roll[1]
        free = chain[1:] if frozen else chain
        idx = [JOINT_NAMES.index(n) for n in free]
        return {
            "chain": chain,
            "roll_q": self.qadr[chain[0]],
            "dadr": np.array([self.dadr[n] for n in free]),
            "qadr": np.array([self.qadr[n] for n in free]),
            "lo": self.lo[idx], "hi": self.hi[idx],
            "body": self.body_id[limb_probe(arm, side)],
            "geom": f"{side}_wrist_z_collision" if arm else f"{side}_foot",
            "guards": tuple(
                mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in limb_guard_bodies(arm, side)),
        }


# ---------------------------------------------------------------- 时间曲线
def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: float) -> float:
    """五次多项式：一阶、二阶导在两端都为 0。

    smoothstep（三次）的二阶导在端点等于 ±6，所以相位一切换加速度就跳一下 ——
    实测 fast 档的 jerk 峰值 7104 rad/s³ 有一半来自这里。五次版端点 a=0，
    速度曲线是连续可导的，摆动相接上去不会有"顿一下"。
    """
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def point_eval(keys, t: float) -> np.ndarray:
    """keys = [(t0, np.array(3)), ...]，按 smoothstep 分段插值世界坐标点。"""
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], dtype=float).copy()
    for i in range(len(keys) - 1):
        t0, t1 = keys[i][0], keys[i + 1][0]
        if t <= t1:
            u = smootherstep((t - t0) / max(t1 - t0, 1e-9))
            a = np.asarray(keys[i][1], dtype=float)
            b = np.asarray(keys[i + 1][1], dtype=float)
            return a + (b - a) * u
    return np.asarray(keys[-1][1], dtype=float).copy()


def swing_lift(u: float, p: Params) -> float:
    """摆动相抬升剖面。

    `swing_up` 是"抬起段占摆动相的比例"：0.18 = 前 18% 快速离地、后 82% 缓慢落地
    （抄 crab_climb.swing_z，为的是跨台阶时先抬过边缘）。但**高速档摆动只有 8 帧**，
    0.18 意味着 1 帧内抬到 3 cm（1.6 m/s 竖向速度），关节会跳。
    所以 `swing_up ≥ 0.999` 时退化成整段正弦（对称、两端速度为 0），高速档用它。
    """
    if p.swing_up >= 0.999:
        return p.clearance * math.sin(math.pi * u)
    if u < p.swing_up:
        return p.clearance * math.sin(math.pi * 0.5 * u / p.swing_up)
    return p.clearance * math.cos(
        math.pi * 0.5 * (u - p.swing_up) / (1.0 - p.swing_up))


def s_of_tau(tau: float, p: Params) -> float:
    """相位 -> 设计姿态的扫掠参数 s（0 前极限，1 后极限）。支撑相线性、摆动相 smoothstep。"""
    if tau < p.duty:
        return tau / max(p.duty, 1e-9)
    return 1.0 - smoothstep((tau - p.duty) / max(1.0 - p.duty, 1e-9))


# ---------------------------------------------------------------- 基座位姿
# 过渡段（v9，用户指定顺序）：站立 → **蹲下** → **双手撑地** → **双脚向后伸**
# → **趴姿**。v8 那套「蹲着先分腿、再慢慢前倾伸臂、腿全程不动」已废弃，标定与
# 踩坑记录见 logs/prone_crawl/backup_v8/prone_crawl_v8stage.py。
# 每行 = (归一化时刻 0~1, 基座高度 m, 俯身角 deg, 前进比例 0~1)。
# 高度写 None 表示用 STAND_Z / p.body_z。
# 注意 z 是**躯干中心高度**，不是髋高：俯身后髋在躯干中心下后方，
#   hip_z ≈ base_z − 0.36·cosθ,  hip_x ≈ base_x − 0.36·sinθ,
#   shoulder_z ≈ base_z + 0.09·cosθ,  shoulder_x ≈ base_x + 0.09·sinθ。
#
# v9 的几何命门：**手撑地时脚必须还留在蹲姿触地点（x≈+0.05）**。
# 但撑地要求 shoulder_z ≤ 0.545（臂长 0.512 + 腕球 0.033），躯干就得俯到
# ~68°、base_z ≤ 0.49；一俯身髋就往后退 0.36·sin68° = 0.33 m。若基座不前移，
# 髋会掉到 x≈−0.33、跑到脚后方 0.38 m，IK 立刻落进「大腿水平前伸」的坐姿
# 分支。所以撑地段的 `x_frac` 要拉到 **2.1**（基座 x = 0.14×2.1 = 0.294），
# 把髋拉回 x≈−0.06，脚在髋前 0.11 m —— 这才是标准 burpee 的撑地姿势
# （手在脚前 0.32 m）。双脚后伸时基座再退回 0.14，脚从 +0.05 推到 −0.50，
# 髋 x −0.24，腿自然向后伸直。`x_frac > 1` 是允许的：entry_stage 不裁剪它。
#
# 撑地点必须在肩正下方附近（肩 y=0.178，手 y 收到 0.24）：
#   u=0.44 肩 (0.375, 0.178, 0.513)，手 (0.37, 0.24, 0.033) → 0.484 m，
#   留 28 mm 余量。手 y 若还用 v8 的 0.27 就顶到 0.508 m、IK 饱和。
ENTRY_STAGES = (
    (0.000, None,   0.0, 0.00),   # ① 站立（base 1.034，髋 ≈0.67）
    (0.110, 0.905,  7.0, 0.02),   # ② 下蹲：屈膝，脚钉在站姿触地点
    (0.240, 0.850, 13.0, 0.06),   # ② **蹲底**（髋 0.477，脚 x=+0.05 不动）
    # ---- ③ 双手撑地：俯身 + 基座前移 + 手下探，三者同步 ----
    (0.310, 0.760, 30.0, 0.55),   # 开始前俯，手向前抬起
    (0.380, 0.620, 50.0, 1.35),   # 手里程过半，仍在空中
    (0.440, 0.480, 68.0, 2.10),   # **双手撑地**（肩 0.513，脚仍在 +0.05）
    # ---- ④ 双脚向后伸：手是固定支点，基座回退、脚向后蹬成平板 ----
    (0.540, 0.440, 74.0, 1.85),   # 脚 +0.05 → −0.12
    (0.640, 0.390, 80.0, 1.45),   # 脚 −0.12 → −0.35
    (0.740, 0.330, 86.0, 1.00),   # 脚后伸到位（平板），EXIT_U 起收腿
    # 0.88 的基座不能压到 0.18：末 32 帧的 blend_tail_to 从这里起步，右臂要从
    # 「左臂镜像」插值到 gait 首帧的独立姿态，肩太低时上臂会扫进地面
    # （实测 base=0.18 → f221~229 的 r_shoulder_z_link 到 −38.7 mm）。
    # 保持 0.26（与 v8 在 blend 起点的肩高相当），剩余下降交给末段对齐完成。
    (0.880, 0.260, 89.0, 1.00),   # ⑤ 胸腹下沉（末 32 帧再对齐爬行首帧）
    (1.000, None,  90.0, 1.00),   # ⑤ 俯卧爬行位（趴姿）
)

# ---------------------------------------------------------------- 下跪姿态
# 单靠 IK 解不出"下跪"：腿有 4 个自由度、末端任务只有 3 个，warm start 会把
# 膝锁在"大腿水平前伸"的坐姿分支（实测膝中心恒在 0.32 m，离地 32 cm）。
# 所以下跪窗口内直接给**关节角**（FK 标定，见 docs/prone_crawl.md §3.4）。
# 每行 = (u, hip_y, hip_x, hip_z, knee_y)，度，左腿；右腿由 mirror_chain 得到。
# 约束来源：
#   - 膝中心 z ≈ 0.05~0.07（膝胶囊半径 ~5 cm，视觉上贴地）；
#   - 小腿必须折向后**上方**（踝 z≈0.11~0.15）：小腿平放时脚掌会朝下扎进
#     地面，而 ankle_y 机械限位只有 [-50°, +45°]，翻不平脚背（实测 −95 mm）。
#     抬起小腿后脚掌离地 0~60 mm，不穿地；
#   - hip_y 单调下降、knee_y 稳定在 122~126°，避免相邻关键帧跨折叠分支。
# hip_x 这一列不是 0：跪下时两膝要**逐渐分开**，踝 y 才能从 0.14 平滑长到
# 0.20 左右，接上 u=0.74 的 0.255 目标。原来整列是 0，0.68→0.74 踝 y 一步跳
# 0.115 m，IK 只能靠 hip_z 猛旋 + hip_x 内收去够 —— 那一下就是两条大腿胶囊
# 在中线互穿 98 mm 的来源，也是"收紧 hip_x 后髋压进地面 31 mm"的来源。
KNEEL_POSES = (
    (0.00, -17.0,  0.0, 0.0,  34.0),
    (0.12, -25.0,  0.0, 0.0,  90.0),
    (0.22, -35.0,  1.0, 0.0, 114.0),
    (0.32, -48.0,  4.0, 0.0, 126.0),
    (0.42, -58.0,  8.0, 0.0, 124.0),
    (0.52, -68.0, 12.0, 0.0, 124.0),
    (0.60, -80.0, 16.0, 0.0, 124.0),
    (0.68, -90.0, 18.0, 0.0, 122.0),
    (0.76, -98.0, 20.0, 0.0, 122.0),
)


def kneel_joints(u: float, side: str = "l") -> np.ndarray:
    """下跪关键角在关节空间插值（rad，左腿），右腿自动镜像。"""
    keys = KNEEL_POSES
    if u <= keys[0][0]:
        v = np.asarray(keys[0][1:], dtype=float)
    elif u >= keys[-1][0]:
        v = np.asarray(keys[-1][1:], dtype=float)
    else:
        v = np.zeros(4)
        for a, b in zip(keys[:-1], keys[1:]):
            if u <= b[0]:
                w = smootherstep((u - a[0]) / max(b[0] - a[0], 1e-9))
                v = np.asarray(a[1:], dtype=float) * (1 - w) + \
                    np.asarray(b[1:], dtype=float) * w
                break
    out = np.radians(v)
    if side == "r":
        out[1] *= -1.0
        out[2] *= -1.0
    return out


def kneel_weight(u: float) -> float:
    """【v8 已停用，恒返回 0；KNEEL_POSES / kneel_joints 保留作踩坑记录】

    v8 编舞不再下跪（站立 → 下蹲 → 双腿分开 → 前倾撑地 → 趴姿），腿全程走 IK：
    下蹲是正常折叠分支，warm start 从站姿连续降下来不会跳分支；只要**外展从下蹲
    就开始**（hip_x 一外展，两条大腿就不在中线互穿），就不需要 FK 接管。

    v7 为什么必须用 FK：那时是「蹲到底 + 双腿平行（hip_x=0）」，髋降到 0.41 而腿
    还是纯 IK 时，两条大腿会被并到中线互穿（hip_z_1/2 左右相撞 11 帧）+ 双脚互撞；
    而「膝贴地」的分支 IK 根本解不出来（4 自由度 / 3 维末端任务，warm start 会一直
    停在"大腿水平前伸"的坐姿上，实测膝中心恒在 0.32 m）。所以 v7 直接用
    KNEEL_POSES 给关节角、由 kneel_joints() 插值、本函数在 0.20~0.68 混合。

    若以后要恢复下跪：把下面的窗口解开、并把 ENTRY_STAGES 改回 v7 的表
    （logs/prone_crawl/backup_v7/prone_crawl_v7stage.py）。
    """
    return 0.0


def _kneel_weight_v7(u: float) -> float:
    """v7 的下跪接管权重（保留），说明见 kneel_weight()。"""
    if u <= 0.20 or u >= 0.68:
        return 0.0
    if u < 0.40:
        return smootherstep((u - 0.20) / 0.20)
    if u > 0.58:
        return 1.0 - smootherstep((u - 0.58) / 0.10)
    return 1.0


# ------------------------------------------------------------ 退出跪姿 / 双腿分开
# 双手撑地之后（u≥0.68）**不能再让 IK 自由解腿**。腿有 4 个自由度、末端任务只有
# 3 维，warm start 会稳定地落进 hip_z≈−120° 的「蛙式」分支：两膝挤到中线（y≈7 cm）、
# 小腿往外翘、脚离地 16~43 cm；而爬行首帧是 hip_z≈+95° 的折腿支撑。末 20 帧
# blend_tail_to 再把 −120° 硬插值到 +95°（差 217°），于是后半段变成「膝盖弓着 +
# 腿在空中划一圈再落回来」，脚瞬时离地 433 mm。数值验收只查穿地不查离地，所以
# 之前全项 PASS 却完全不对。
# 正解：从 u=0.68 的**实际**腿角出发，在关节空间直接收敛到 gait 首帧的腿角。
# 终点不硬编码 —— 两档 gait 首帧不同（fast 左膝 96.9°、normal 左膝 52.3°），
# 左右腿也不是严格镜像，由 build_split() 把 resB["qpos"][0] 传进来。
# 时间分配由 scripts/check_prone_exit.py 跨 fast/normal 两档联合扫描标定。
# **不能两档共用一套**：gait 首帧左膝 fast 96.9°、normal 52.3°，终点差 45°，
# 统一参数要么让 fast 的腿在半空蹬直（膝胶囊离地 115 mm），要么让 normal 的
# 髋胶囊压地 7.9 mm。判据用「终点左膝角」而不是 preset 名，物理上更直接。
# bz 必须 ≥ by：外旋慢于后伸时髋胶囊会压进地面 32 mm。
#
# v8：撑地时刻从 0.68 推到 0.82（新编舞不下跪，躯干要更晚才俯到 72°）。
# 退出窗口**必须早于撑地**：收腿是在关节空间把 hip_z 从 ~−20° 甩到 ±95°，
# 大腿会扫过身体侧下方，基座越低越容易把大腿压进地面。放到 0.82（那时 base 已经
# 在 0.82→0.90 骤降到 0.19）实测 r_hip_z_link 穿地 17.3 mm；提前到 0.74 时
# base 还有 0.55、髋高 0.385，扫过去就有余量。手臂照旧在 0.82 撑地，两者不冲突
# —— 退出段把 tgts 改成实际末端位置、腿不再重解 IK。
EXIT_U = 0.74            # 双脚后伸到位；此后腿改走关节空间收敛到 gait 首帧
PLANT_U = 0.44           # 双手从这一刻起压平撑地（plant_clamp / 手臂柔顺化生效）
EXIT_TIMING_FAST = (0.85, 0.85, 0.50)   # by, bz, bk：腿折得多 → 晚后伸、早收膝
EXIT_TIMING_SLOW = (0.65, 0.65, 0.85)   # 腿更伸直 → 后伸/外旋同步提前


def exit_timing_for(knee_end_deg: float) -> tuple:
    return EXIT_TIMING_FAST if knee_end_deg > 75.0 else EXIT_TIMING_SLOW


def exit_leg_joints(u: float, q_start: np.ndarray, q_end: np.ndarray,
                    timing: tuple) -> np.ndarray:
    """退出跪姿段的腿 IK 链角（rad，[hip_y, hip_x, hip_z, knee_y]）。

    q_start 是 u=EXIT_U 帧的实测链角，q_end 是 gait 首帧同一侧的链角；
    两者之间按各维度自己的 smootherstep 收敛，天然连续、不跳 IK 分支。
    timing = (by, bz, bk)：hip_y / hip_x+hip_z / knee_y 各自的完成进度。
    """
    by, bz, bk = timing
    s = min(max((u - EXIT_U) / (1.0 - EXIT_U), 0.0), 1.0)
    w = np.array([smootherstep(s / by),
                  smootherstep(s / bz),
                  smootherstep(s / bz),
                  smootherstep(s / bk)])
    return (1.0 - w) * q_start + w * q_end


# 【已停用，保留作踩坑记录】下面两个函数（外展偏置 / 基座微抬）是旧 IK 蛙式
# 分支的补丁，build_split() 现在不再调用。u≥EXIT_U 改由 exit_leg_joints() 在
# 关节空间直接收敛到 gait 首帧，hip_x 自己从 18° 平滑外展到 68°，自碰撞 0 帧、
# 中间段不穿地；再叠偏置只会把脚推进地面。别重新启用。
#
# 当时的背景：退出跪姿后 IK 会把 hip_x 推到内收
# （左 −22° / 右 +22°），两条大腿胶囊在中线互穿 98 mm、连续 53 帧。
# 冻结 hip_x / 逐帧钳位 / 收紧软下限 / 抬基座 / 抬脚目标都试过，不是被收尾 IK
# 推回就是引入髋穿地，见各处注释。唯一干净的做法：所有 IK 收尾做完后，一次性
# 叠加对称外展偏置，不重解 IK（重解就会把 hip_x 推回内收，还带 1230 rad/s²
# 的分支跳变）。
# 幅度必须是 **10° 而不是 30°**：FK 逐帧扫描实测 10° 就能让所有采样帧的碰撞
# 归零；30° 会把髋/脚推出 18~25 mm 穿地，再抬基座补偿又会让手够不到地面
# （v10：392 mm 穿地、4145 rad/s²）。
# 末段衰减到 0：末 20 帧已被 blend_tail_to 对齐到爬行首帧，不能覆盖它。
SPREAD_KEYS_U = (0.00, 0.68, 0.74, 0.92, 1.00)
SPREAD_KEYS_V = (0.00, 0.00, 10.0, 10.0, 0.00)


def spread_bias(u: float) -> float:
    return float(interp_key(SPREAD_KEYS_U, SPREAD_KEYS_V, u))


def spread_hips(qpos: np.ndarray, entry_time: float, dt: float) -> int:
    """【已停用】给左右 hip_x 叠加对称的外展偏置，消除大腿自碰撞。

    不重解 IK —— 任何一次重解都会把 hip_x 推回内收；只在收尾处一次性改关节角。
    hip_x 外展会带着脚往地下沉，所以紧接着由 lift_split_window() 做整体平移补偿。
    """
    if entry_time <= 0:
        return 0
    li = 7 + JOINT_NAMES.index("l_hip_x_joint")
    ri = 7 + JOINT_NAMES.index("r_hip_x_joint")
    cnt = 0
    for i in range(len(qpos)):
        u = min(i * dt / entry_time, 1.0)
        b = np.radians(spread_bias(u))
        if abs(b) < 1e-9:
            continue
        qpos[i, li] += b
        qpos[i, ri] -= b
        cnt += 1
    return cnt


# 外展偏置带来的两处轻微穿地，用一次「整只机器人平滑抬升」吸收：
#   - u≈0.756（帧 189）髋胶囊 −13.0 mm；
#   - u≈0.920（帧 230）脚掌 −11.9 mm（正好落在末 20 帧 blend 区，改不了偏置曲线）。
# 抬的是 free joint 的 z，**相对姿态完全不变**：手脚髋一起抬，不重解 IK、不影响
# 关节角加速度（验收跳过 FREE joint）、末帧 bump 归零所以不影响与 gait 首帧的衔接。
# 这和 v10 改 ENTRY_STAGES 是两回事：那次是在 IK 之前抬基座，肩点超出臂长 →
# plant_clamp 连锁重解 → 穿地 392 mm、a_max 4145 rad/s²。
# 参数由 scripts/check_prone_lift.py 逐帧扫描标定（14 mm / 中心 0.82 / 半宽 0.18）：
# 接触 −6.3 mm（判据 10）、中间段 −3.0 mm（判据 6）。
LIFT_AMP_M = 0.014
LIFT_CENTER_U = 0.82
LIFT_HALF_U = 0.18


def lift_split_window(qpos: np.ndarray, entry_time: float, dt: float) -> int:
    """【已停用】余弦窗平滑抬升基座，补偿外展偏置造成的髋/脚穿地。"""
    if entry_time <= 0:
        return 0
    cnt = 0
    for i in range(len(qpos)):
        u = min(i * dt / entry_time, 1.0)
        d = abs(u - LIFT_CENTER_U) / LIFT_HALF_U
        if d >= 1.0:
            continue
        qpos[i, 2] += LIFT_AMP_M * 0.5 * (1.0 + math.cos(math.pi * d))
        cnt += 1
    return cnt


def entry_stage(u: float, p: Params) -> tuple[float, float, float]:
    """归一化时刻 u -> (基座高度, 俯身角 rad, 前进比例)。段内用 smootherstep。"""
    st = ENTRY_STAGES
    u = min(max(u, 0.0), 1.0)

    def _z(v, idx):
        if v is not None:
            return v
        return STAND_Z if idx == 0 else p.body_z

    if u <= st[0][0]:
        return _z(st[0][1], 0), math.radians(st[0][2]), st[0][3]
    for i in range(len(st) - 1):
        if u <= st[i + 1][0]:
            a, b = st[i], st[i + 1]
            w = smootherstep((u - a[0]) / max(b[0] - a[0], 1e-9))
            za, zb = _z(a[1], i), _z(b[1], i + 1)
            return (za + (zb - za) * w,
                    math.radians(a[2] + (b[2] - a[2]) * w),
                    a[3] + (b[3] - a[3]) * w)
    return (_z(st[-1][1], len(st) - 1), math.radians(st[-1][2]), st[-1][3])


def base_pose(t: float, it: float, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """返回 (pos3, quat4)。it 为过渡段时刻，t 为循环内时间（过渡段时被忽略）。"""
    if it < p.entry_time:                      # 过渡段：按五阶段轨迹走
        u = min(max(it / max(p.entry_time, 1e-9), 0.0), 1.0)
        z, pitch, xf = entry_stage(u, p)
        x = p.entry_x0 * xf
        roll = 0.0
    else:
        ph = 2.0 * math.pi * t / p.cycle_time
        pitch = PRONE_HALF_PI
        z = p.body_z + p.body_bob * math.sin(2.0 * ph)
        x = p.entry_x0 + p.speed * t
        roll = p.body_roll * math.sin(ph)
    qz = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    qr = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qr, qz)             # 先俯身，再绕世界 X 滚转
    return np.array([x, 0.0, z]), q


# ---------------------------------------------------------------- 轨迹构建
def interp_key(fracs, vals, u: float) -> np.ndarray:
    """在关键姿态之间做 smootherstep 插值（关节空间）。"""
    if u <= fracs[0]:
        return np.asarray(vals[0], dtype=float).copy()
    for k in range(len(fracs) - 1):
        if u <= fracs[k + 1]:
            w = smootherstep((u - fracs[k]) / max(fracs[k + 1] - fracs[k], 1e-9))
            a = np.asarray(vals[k], dtype=float)
            b = np.asarray(vals[k + 1], dtype=float)
            return a + (b - a) * w
    return np.asarray(vals[-1], dtype=float).copy()


def build(p: Params, verbose: bool = True,
          exit_q: "np.ndarray | None" = None) -> dict:
    """exit_q: 拆段时 gait 首帧的完整 qpos，作为「双腿分开」段的关节空间终点。"""
    kin = Kin(SCENE, hip_z_band=p.hip_z_band,
              hip_y_limit=p.hip_y_limit, knee_hi_limit=p.knee_hi_limit)
    kin.freeze_roll = (p.freeze_arm_roll, p.freeze_leg_roll)
    cols = {(l["name"]): kin.make_limb_col(l["arm"], l["side"]) for l in LIMBS}
    # 退出跪姿段的终点角：左右分别取（gait 首帧不是严格镜像）
    exit_end = None
    exit_timing = None
    if exit_q is not None and p.entry_time > 0 and p.cycles == 0:
        exit_end = {
            s: np.array([exit_q[7 + JOINT_NAMES.index(f"{s}_{k}_joint")]
                         for k in ("hip_y", "hip_x", "hip_z", "knee_y")])
            for s in "lr"}
        exit_timing = exit_timing_for(math.degrees(exit_end["l"][3]))
    exit_start: dict[str, np.ndarray] = {}

    # gait 首帧的**实际手位**（世界系，左右取对称平均）。手臂末段目标必须落在
    # 这里：硬编码 [0.50, ±0.36] 在过渡段末基座位置（entry_x0 处）超出臂长
    # （实测关键帧残差 2114 mm、末端被按到 −322 mm），末段 Hermite 对齐再硬扫
    # 回来就是 jerk 13010 的尖峰。gait 首帧手位天然可达、自带加宽 y、衔接无缝。
    gait_hand = None
    if exit_q is not None and p.entry_time > 0 and p.cycles == 0:
        kin.set_q(np.asarray(exit_q, dtype=float))
        hl = kin.d.xpos[cols["l_arm"]["body"]].copy()
        hr = kin.d.xpos[cols["r_arm"]["body"]].copy()
        gait_hand = np.array([(hl[0] + hr[0]) / 2.0,
                              (abs(hl[1]) + abs(hr[1])) / 2.0,
                              (hl[2] + hr[2]) / 2.0])

    # --- 1. 触地点（身体系）：x/y 由参数给，z 用"参考 IK + 接触闭环"解出来 ---
    # 不能拿设计姿态的 FK 当接触高度：设计姿态只保证肘/膝朝对方向，末端可能离地
    # 十几厘米（实测手腕 z=-0.13），会让摆动相的目标一头扎进地面。
    def design_pose_q(s: float = 0.5) -> np.ndarray:
        q = np.zeros(38)
        q[0:3] = [0.0, 0.0, p.body_z]
        q[3:7] = [0.5 ** 0.5, 0.0, 0.5 ** 0.5, 0.0]
        return fill_all_limbs(kin, q, p, s, 0.0)

    def reference_contact(limb: dict) -> np.ndarray:
        tx, ty = p.arm_touch if limb["arm"] else p.leg_touch
        sg = 1.0 if limb["side"] == "l" else -1.0
        tgt = np.array([tx, sg * ty, 0.10], dtype=float)
        qn = design_joints(p, limb["arm"], limb["side"], 0.0, 0.0)
        col = cols[limb["name"]]
        kin.set_q(design_pose_q())
        for _ in range(6):
            kin.solve_limb(col, tgt, qn, p)
            dz = kin.min_z(col["geom"]) + p.press
            if abs(dz) < 2e-4:
                break
            tgt[2] -= dz
        return tgt

    rel = {l["name"]: reference_contact(l) for l in LIMBS}
    # 循环第 0 帧各肢的扫掠参数（过渡段末态先验要对齐它，见下面的 q_nom）
    p.s0 = tuple(s_of_tau(p.offsets[l["idx"]] % 1.0, p) for l in LIMBS)
    stride = p.speed * p.duty * p.cycle_time       # 单支撑相内身体前进距离
    S = p.speed * p.cycle_time                     # 每循环前进距离

    # 触地点 P_L：该肢 k=0 次着地时的世界坐标。k=0 对应 t = -off*T 的基座位置。
    touch = {}
    for l in LIMBS:
        x_at = p.entry_x0 - p.speed * p.offsets[l["idx"]] * p.cycle_time
        touch[l["name"]] = np.array([rel[l["name"]][0] + x_at,
                                     rel[l["name"]][1],
                                     rel[l["name"]][2]])

    def loop_target(l: dict, t: float) -> tuple[np.ndarray, float, bool]:
        """循环内某肢：返回水平目标 (x,y)、抬升量、是否支撑。

        z 由 zbase（上一支撑相闭环收敛出来的接触高度）决定，不写死 —— 足尖/手掌
        相对踝/腕原点的下垂量随肢体姿态变化 30 mm 以上（实测 58~92 mm），写死会让
        摆动相起步时目标突然下扎。
        """
        phi = t / p.cycle_time + p.offsets[l["idx"]]
        k = math.floor(phi)
        tau = phi - k
        p_td = touch[l["name"]] + np.array([S * k, 0.0, 0.0])
        if tau < p.duty:                            # 支撑：接触点世界系固定
            return p_td[:2].copy(), 0.0, True
        u = (tau - p.duty) / (1.0 - p.duty)
        nxt = p_td + np.array([S, 0.0, 0.0])
        pos = p_td + (nxt - p_td) * smoothstep(u)
        return pos[:2].copy(), swing_lift(u, p), False

    # --- 2. 过渡段：手/脚目标关键帧。**按俯身进度 u 排时间**，不按秒 ---
    # 为什么按 u：手掌着地时肩高 ~0.5 m，手臂全长 0.512 m，按秒排会在身体还没
    # 降下来时就把手按到地上，目标超出可达半径 -> IK 直接饱和（实测残差 0.8 m）。
    q_stand = np.zeros(38)
    q_stand[0:3] = [0.0, 0.0, STAND_Z]
    q_stand[3:7] = [1.0, 0.0, 0.0, 0.0]
    kin.write_joints(q_stand, STAND_VEC)
    kin.set_q(q_stand)
    stand_pt = {l["name"]: kin.d.xpos[cols[l["name"]]["body"]].copy() for l in LIMBS}

    # 过渡段关键帧：**时刻是归一化的 0~1**，与 entry_stage 的阶段表对齐。
    # （原来这里写的是秒，但 point_eval 传进去的是归一化 u，所以末帧其实永远
    #   到不了 crawl_pt —— 这才是动作"不协调、收不拢"的直接原因。）
    def base_x_at(frac: float) -> float:
        return float(base_pose(frac * p.entry_time, frac * p.entry_time, p)[0][0])

    entry_keys = {}
    for l in LIMBS:
        name, arm, side = l["name"], l["arm"], l["side"]
        sg = 1.0 if side == "l" else -1.0
        xy = loop_target(l, 0.0)[0]
        crawl_pt = np.array([xy[0], xy[1], rel[name][2]])
        if arm:
            # v9：手的世界轨迹 = 垂在体侧 → 随下蹲下移 → **前伸下探到地面**
            # → 撑住不动（双脚后伸期间手是固定支点）→ 随胸腹下沉前挪外展到爬行位。
            # 与 v8 的两点差别：
            #   1) 蹲姿/下探段手必须抬到 y=±0.44 绕开髋部胶囊：收到 0.30 时
            #      `elbow_y_collision` 会撞 `hip_z_collision`（实测 f91~120 共 30 帧
            #      自碰撞）。撑地前再收回 y=0.24，因为撑地点要贴着肩正下方。
            #   2) 撑地提前到 u=0.44，且手 y 收窄（肩 y=0.178）：用 v8 的 0.27
            #      会让肩-手距顶到 0.508 m、IK 饱和（臂长 0.512）。
            # 手落点始终贴着肩正下方走：肩 x 随 base_x 从 0.03 → 0.375 → 0.227，
            # 手 x 同步 0.10 → 0.37 → 0.42；硬留在原地会超臂长。
            gh = gait_hand if gait_hand is not None else \
                np.array([0.48, 0.36, rel[name][2]])
            # gh 的 y 是「左右取绝对值平均」的正数，**必须按侧加符号**：直接
            # gh.copy() 会把右臂的目标放到 +y 同侧，跨中线 0.54 m 远超臂长
            # （实测末段 r_arm IK 残差 441 mm，靠 blend_tail_to 硬盖过去）。
            gh_side = np.array([gh[0], sg * abs(gh[1]), gh[2]])
            entry_keys[name] = [
                (0.000, stand_pt[name]),                        # 站立，手垂在体侧
                (0.110, np.array([0.10, sg * 0.36, 0.630])),    # 下蹲，顺身体下移
                (0.240, np.array([0.10, sg * 0.44, 0.550])),    # **蹲底**：手抬到外侧让开髋
                (0.310, np.array([0.16, sg * 0.42, 0.440])),    # 前俯，手仍在外侧
                (0.380, np.array([0.28, sg * 0.32, 0.250])),    # 下探，y 收回肩正下方
                (0.440, np.array([0.40, sg * 0.24, 0.033])),    # **双手撑地**
                (0.540, np.array([0.40, sg * 0.24, 0.033])),    # 后伸期：手钉住不动
                (0.640, np.array([0.40, sg * 0.24, 0.033])),    # 后伸期：手钉住不动
                (0.740, np.array([0.43, sg * 0.28, 0.033])),    # 平板位，手开始小幅前挪
                (0.880, np.array([0.46, sg * 0.33, 0.035])),    # 胸腹下沉，前挪外展
                (1.000, gh_side.copy()),                        # 趴姿：gait 首帧实际手位
            ]
        else:
            # v9：脚在「双手撑地」之前**一直钉在站姿触地点**（这就是用户要的顺序：
            # 先蹲、再撑地、最后才伸脚）。v8 在 0.30~0.48 就提前把脚撤到 −0.22、
            # 摊到 ±0.30，结果撑地后腿长时间静止、最后突然趴下 —— 观感不协调的根因。
            # 0.44~0.74 脚从 +0.05 一路后蹬到 −0.50，y 顺带从 0.14 摊到 0.26，
            # 腿由折叠变伸直（burpee 的「双脚向后伸」）。
            # u≥EXIT_U 由 exit_leg_joints 接管，这里的目标点只用于分支修复。
            entry_keys[name] = [
                (0.000, stand_pt[name]),
                (0.240, stand_pt[name]),                        # 蹲底：脚不动
                (0.310, stand_pt[name]),                        # 俯身：脚不动
                # 撑地瞬间脚略后撤 0.15 m：深蹲时大腿胶囊会水平前伸到 x≈0.25、
                # z≈0.29，正好挡在下探的上臂路径上（实测 f99~120 肘/肩撞髋 21 帧）。
                # 脚一后撤，膝改为朝下折，大腿不再占住手臂通道。相对后面 0.58 m
                # 的后伸量，这 0.15 m 只是「撑稳时的调整」，不改变动作顺序。
                (0.440, np.array([-0.10, sg * 0.15, 0.055])),  # **撑地**，脚小幅后撤
                (0.540, np.array([-0.28, sg * 0.16, 0.055])),  # 后伸 1/3
                (0.640, np.array([-0.50, sg * 0.20, 0.055])),  # 后伸 2/3
                (0.740, np.array([-0.68, sg * 0.26, 0.055])),  # **后伸到位**（平板）
                (0.880, np.array([-0.66, sg * 0.28, 0.075])),  # 收腿（常被 exit 段覆盖）
                (1.000, np.array([-0.64, sg * 0.280, rel[name][2]])),
            ]

    # --- 3. 逐帧求解 ---
    n_entry = int(round(p.entry_time / p.dt)) if p.entry_time > 0 else 0
    n_loop = int(round(p.cycles * p.cycle_time / p.dt))
    n = n_entry + n_loop
    qpos = np.zeros((n, 38))
    flags = np.zeros((n, len(LIMBS)), dtype=np.int8)
    drops = np.zeros((n, len(LIMBS)))          # 记录 IK 残差，便于体检
    tgts = np.zeros((n, len(LIMBS), 3))        # 每帧 IK 目标（诊断用）

    # IK 初值 / 先验：上一帧的关节角（warm start），保证连续
    warm = {l["name"]: design_joints(p, l["arm"], l["side"], 0.5, 0.0)
            for l in LIMBS}
    stand_chain = {}
    for l in LIMBS:
        idx = [JOINT_NAMES.index(n) for n in limb_chain(l["arm"], l["side"])]
        stand_chain[l["name"]] = STAND_VEC[idx].copy()
    if p.entry_time > 0:                        # 过渡段从站姿起步
        warm = {k: v.copy() for k, v in stand_chain.items()}
    zbase = {l["name"]: float(rel[l["name"]][2]) for l in LIMBS}
    kin.set_q(q_stand)

    # 过渡段的 IK 副本：**关掉零空间先验**。
    # 站姿/半蹲时腿和手臂几乎伸直，hip_z / shoulder_z 这类冗余关节对末端位置
    # 几乎没有影响，先验（design_joints 里 hip_z=83°、shoulder_z=77°）就会把它们
    # 一帧帧往设计值拽 —— 实测帧 4→5 的 hip_z 从 4.8° 跳到 98.4°、帧 100→101
    # 的 shoulder_y 跳 135°，二阶差分直接爆到 5e4 deg/s²。
    # 关掉先验后冗余关节就停在 warm start 上，天然连续。
    p_entry = copy.copy(p)
    p_entry.ik_null = p.entry_ik_null
    # 过渡段单独开"肘/膝离地"约束：俯身过程中手臂要绕到身体下方，
    # 不加约束时肘部会扎进地面 0.12 m（爬行段不能开这个，会把 fast 档卡死）。
    p_entry.chain_margin = max(p.entry_chain_margin, 0.0)
    p_entry.chain_weight = max(p.chain_weight, 15.0)   # 过渡段更强调'肘膝离地'
    # --- 2c. 过渡段：先解 6 个关键姿态的关节角，之后在**关节空间**插值 ---
    # 逐帧跑 IK 时，手臂在俯身 58~74° 这段找不到"向下伸"的分支，会绕到身体外侧
    # （实测腕跑到 y=0.6，离目标 0.5 m，而且三个关节全顶在限位上）。
    # 关键姿态 + 关节插值没有这个问题：每个关键姿态单独解（多起点挑连续的），
    # 中间用五次多项式过渡，天然平滑、绝对不会跳分支。
    # 关键姿态要加密：只有 8 个阶段点时，中间帧的关节插值会把末端甩到地下
    # （实测肘部 −0.12 m）。加密到 ~0.05 一个点后插值基本贴着末端轨迹走。
    key_fracs = sorted(set([st[0] for st in ENTRY_STAGES]
                           + list(np.linspace(0.0, 1.0, 21))))
    key_q: dict[str, list[np.ndarray]] = {l["name"]: [] for l in LIMBS}
    if p.entry_time > 0:
        # 关键姿态要一次解准（它决定了整段的形状），所以给足迭代、关掉零空间先验
        p_key = copy.copy(p)
        p_key.ik_iters = 400
        p_key.ik_null = 0.0
        p_key.restart_angles = tuple(np.radians(a) for a in
                                     (-120, -90, -60, -30, 0, 30, 60, 90, 120))
        warm_k = {l["name"]: stand_chain[l["name"]].copy() for l in LIMBS}
        for kf in key_fracs:
            t_k = kf * p.entry_time
            pos_k, quat_k = base_pose(t_k, t_k, p)
            q38k = np.zeros(38)
            q38k[0:3] = pos_k
            q38k[3:7] = quat_k
            # **只解左肢，右肢镜像**。过渡段的左右目标本来就严格对称，
            # 分开解会各自落进不同的折叠分支（实测 key 0.55 左腿 hip_y=+42.6、
            # 右腿 −117，插值出来就是"一条腿突然抬起来"）。镜像还有额外好处：
            # 解算量减半。镜像规则见 mirror_chain。
            for l in LIMBS:
                if l["side"] == "r":
                    continue
                name, arm, side = l["name"], l["arm"], l["side"]
                col = cols[name]
                vec = kin.joint_vec(q38k)
                for n2, v2 in limb_extra(p, side, arm, entry_ankle_y(p, kf)).items():
                    vec[JOINT_NAMES.index(n2)] = v2
                kin.write_joints(q38k, vec)
                for n2, v2 in zip(limb_chain(arm, side), warm_k[name]):
                    q38k[kin.qadr[n2]] = v2
                kin.set_q(q38k)
                tgt_k = point_eval(entry_keys[name], kf)
                # 三个初值各解一次；**在收敛的候选里挑离上一关键帧最近的**。
                # 只挑残差最小会让相邻两帧落在两个折叠分支上 —— 端点都对，
                # 但关节角差 160°，插值出来就是腿甩出去。
                cands = []
                for init in (warm_k[name],
                             design_joints(p, arm, side, p.s0[l["idx"]], 0.0),
                             stand_chain[name]):
                    for n2, v2 in zip(limb_chain(arm, side), init):
                        q38k[kin.qadr[n2]] = np.clip(
                            v2, kin.lo[JOINT_NAMES.index(n2)], kin.hi[JOINT_NAMES.index(n2)])
                    kin.set_q(q38k)
                    e = kin.solve_limb_robust(col, tgt_k, np.asarray(init), p_key)
                    cands.append((e, kin.d.qpos[col["qadr"]].copy()))
                ok = [c for c in cands if c[0] <= 2e-3]
                best = (min(ok, key=lambda c: float(np.abs(c[1] - warm_k[name]).max()))
                        if ok else min(cands, key=lambda c: c[0]))
                if best[1] is not None:
                    kin.d.qpos[col["qadr"]] = best[1]
                    kin.fwd()
                if _DBG_KEYS:
                    cv = np.degrees([kin.d.qpos[kin.qadr[n2]] for n2 in
                                     limb_chain(arm, side)])
                    guards = min(kin.body_min_z(b) for b in col["guards"]) * 1000
                    print(f"    [{kf:.2f}] {name:<6} err={best[0] * 1000:6.1f}mm "
                          f"end_z={kin.d.xpos[col['body']][2] * 1000:6.0f} "
                          f"guard={guards:7.1f} "
                          f"q={' '.join(f'{v:7.1f}' for v in cv)}", flush=True)
                vl = np.array([kin.d.qpos[kin.qadr[n2]] for n2 in limb_chain(arm, side)])
                for n2, v2 in zip(limb_chain(arm, side), vl):
                    q38k[kin.qadr[n2]] = v2
                warm_k[name] = vl
                vr = mirror_chain(vl)
                rname = f"r_{'arm' if arm else 'leg'}"
                for n2, v2 in zip(limb_chain(arm, "r"), vr):
                    q38k[kin.qadr[n2]] = v2
                warm_k[rname] = vr.copy()
            for l in LIMBS:
                key_q[l["name"]].append(np.array(
                    [q38k[kin.qadr[n2]] for n2 in limb_chain(l["arm"], l["side"])]))

    for i in range(n):
        it = i * p.dt
        in_loop = it >= p.entry_time
        t = it - p.entry_time if in_loop else 0.0
        u_norm = 1.0 if p.entry_time <= 0 else min(it / p.entry_time, 1.0)
        u = smoothstep(u_norm)

        pos, quat = base_pose(max(t, 0.0), min(it, p.entry_time), p)
        q38 = np.zeros(38)
        q38[0:3] = pos
        q38[3:7] = quat

        # v9：双脚后伸段（u<EXIT_U）锁住 hip_z，逼 IK 留在矢状面内；
        # u≥EXIT_U 立刻放开，让 exit_leg_joints 能收敛到 gait 首帧的 +95° 蟹式。
        if p.entry_time > 0 and not in_loop:
            lim = (np.radians(p.entry_hip_z_limit) if u_norm < EXIT_U
                   else np.pi)
            for s_ in "lr":
                iz = JOINT_NAMES.index(f"{s_}_hip_z_joint")
                kin.lo[iz] = max(kin.mech_lo[iz], -lim)
                kin.hi[iz] = min(kin.mech_hi[iz], lim)

        # 躯干侧摆 / 腰反向配平（循环内才有）
        if in_loop:
            ph = 2.0 * math.pi * t / p.cycle_time
            # 重心侧移：四拍序列下每抬一条腿，支撑三角形的那条对角线几乎穿过质心
            # （矩形支撑的对角线本来就过中心，实测质心离线只有 7 mm），所以必须
            # 朝支撑侧移动。抬起顺序 l_leg(0-.25) r_arm(.25-.5) r_leg(.5-.75)
            # l_arm(.75-1)，需要 y 在前半周为负、后半周为正 -> -cos 相位。
            #
            # 这里**不能**再乘一个「从 0 缓入」的 ramp：该 ramp 只在第一拍等于 0，
            # 之后恒为 1，于是 (a) 第一拍抬 l_leg 时重心还压在支撑三角形对角线
            # l_arm-r_leg 上（越界 13.5 mm，质心判据 94.5%）、(b) 轨迹不再周期，
            # 循环到第 400 帧时 y 已偏 −A、回到第 0 帧又回 0，播放时横跳 65 mm。
            # 拆段模式下动作 1 的末帧取的就是本函数第 0 帧，侧偏会自动带过去，
            # 所以去掉 ramp 不影响两段衔接（接缝 y 差 0.1 mm）。
            # 只有合并模式（entry_time>0）必须保留 ramp：过渡段末帧侧偏是 0，
            # 不缓入就会在接缝那一帧横跳 A。
            ramp = (smoothstep(min(t / (0.5 * p.cycle_time), 1.0))
                    if p.entry_time > 0 else 1.0)
            q38[1] = -p.body_sway * math.cos(ph) * ramp
            vec = kin.joint_vec(q38)
            vec[JOINT_NAMES.index("waist_z_joint")] = -p.waist_yaw * math.sin(ph)
            vec[JOINT_NAMES.index("waist_x_joint")] = 0.5 * p.waist_yaw * math.sin(ph)
            kin.write_joints(q38, vec)

        for j, l in enumerate(LIMBS):
            name, arm, side = l["name"], l["arm"], l["side"]
            if in_loop:
                xy, lift, contact = loop_target(l, t)
                tgt = np.array([xy[0], xy[1], zbase[name] + lift])
                ln = min(lift / max(p.clearance, 1e-9), 1.0)
                phi = t / p.cycle_time + p.offsets[l["idx"]]
                q_nom = design_joints(p, arm, side, s_of_tau(phi - math.floor(phi), p), ln)
            else:
                tgt = point_eval(entry_keys[name], u_norm)
                # 末态先验必须等于循环第 0 帧的先验，否则跨边界时零空间解会跳一帧
                q_nom = stand_chain[name] * (1.0 - u) + design_joints(
                    p, arm, side, p.s0[l["idx"]], 0.0) * u
                # 过渡段一律不做接触闭环：contact 从 False 翻 True 的那一帧会把目标 z
                # 突然压低几厘米，解直接跳（实测帧 106 的 shoulder_x 跳 111°）。
                # 手/脚落地由末尾的 ground_clamp 统一兜底（只抬不降）。
                contact = False

            # 只改本肢的踝/腕，其余关节（含腰、另三肢）保持已写入的值
            vec = kin.joint_vec(q38)
            ank = None if arm else (entry_ankle_y(p, u_norm) if not in_loop else None)
            for n2, v2 in limb_extra(p, side, arm, ank).items():
                vec[JOINT_NAMES.index(n2)] = v2
            kin.write_joints(q38, vec)

            # 把 IK 链初值（warm start）写进 q38
            for n2, v2 in zip(limb_chain(arm, side), warm[name]):
                q38[kin.qadr[n2]] = np.clip(
                    v2, kin.lo[JOINT_NAMES.index(n2)], kin.hi[JOINT_NAMES.index(n2)])
            # roll 关节（shoulder_y / hip_y）不参与 IK，由 design_joints 的平滑曲线
            # 直接给定 —— 它原本在零空间里自由游走，是跳变的主要来源。
            if kin.freeze_roll[0] if arm else kin.freeze_roll[1]:
                i_roll = JOINT_NAMES.index(limb_chain(arm, side)[0])
                q38[cols[name]["roll_q"]] = float(np.clip(
                    q_nom[0], kin.lo[i_roll], kin.hi[i_roll]))
            kin.set_q(q38)

            # 支撑相：接触高度闭环（IK -> 量最低点 -> 修目标 z -> 再 IK）
            # 注：手的接触点是「腕 + 局部 (0.04,0,0) 的球」，支撑相前臂转动会让球心
            # 绕腕画 4 cm 的弧，所以球心相对地面会滚 2~3 cm。试过按球心反解腕目标
            # （帧内迭代），能把滚动压到 0.4 mm，但会引入新的肘部翻越（实测关节速度
            # 54 rad/s）并把质心覆盖从 96% 拖到 80%，得不偿失 —— 这 2~3 cm 属于
            # 手掌滚动而不是打滑，验收脚本里按 30 mm 判据放行。
            tgt_use = tgt.copy()
            err = 9.9
            zc = tgt_use[2]
            # 过渡段不做"多起点重解"：目标点基本恒定，重解只会在两个折叠分支之间
            # 来回切（实测帧 98→99 的 shoulder_y 跳 142°）。那里手臂在空中，
            # 穿地风险由下面的 ground_clamp 兜底就够了。
            pp = p if in_loop else p_entry
            for _ in range(p.contact_passes if contact else 1):
                err = kin.solve_limb_robust(cols[name], tgt_use, q_nom, pp)
                if not contact:
                    break
                zmin = kin.min_z(cols[name]["geom"])
                delta = zmin + p.press                # 目标：min_z = -press
                zc = tgt_use[2]
                if abs(delta) < 1e-4:
                    break
                tgt_use[2] -= delta
                zc = tgt_use[2]
            if contact:
                # 记下"真正能让接触点落地"的踝/腕高度，摆动相从这里起跳
                zbase[name] = float(zc)
            drops[i, j] = err
            tgts[i, j] = tgt_use
            for n2 in limb_chain(arm, side):
                q38[kin.qadr[n2]] = kin.d.qpos[kin.qadr[n2]]

            # 下跪窗口 / 退出跪姿窗口：IK 解不出"膝贴地"的分支（4 自由度 /
            # 3 维末端任务，warm start 会一直停在"大腿水平前伸"的坐姿上），
            # 撑地之后又会跳到 hip_z≈−120° 的蛙式分支。两段都直接按 FK 关节角
            # 接管腿。目标点随后改成实际末端位置，避免收尾的 smooth_branches /
            # ground_clamp 把腿重解回 IK 的老分支。
            if (not in_loop) and (not arm):
                if exit_end is not None and u_norm >= EXIT_U:
                    # 关节空间直连 gait 首帧：不重解 IK，左右各自收敛（终点非镜像）
                    ch = limb_chain(False, side)
                    q_ik = np.array([kin.d.qpos[kin.qadr[n2]] for n2 in ch])
                    if side not in exit_start:
                        exit_start[side] = q_ik.copy()
                    q_new = exit_leg_joints(u_norm, exit_start[side],
                                            exit_end[side], exit_timing)
                    for n2, v in zip(ch, q_new):
                        jj = JOINT_NAMES.index(n2)
                        q38[kin.qadr[n2]] = float(np.clip(v, kin.lo[jj], kin.hi[jj]))
                    kin.set_q(q38)
                    tgts[i, j] = kin.d.xpos[cols[name]["body"]].copy()
                    w = 0.0
                else:
                    w = kneel_weight(u_norm)
                if w > 0.0:
                    ch = limb_chain(False, side)
                    q_ik = np.array([kin.d.qpos[kin.qadr[n2]] for n2 in ch])
                    q_kn = kneel_joints(u_norm, side)
                    q_new = (1.0 - w) * q_ik + w * q_kn
                    # hip_x / hip_z 是 IK 的冗余自由度，混合时会被推到 ±26°~50°，
                    # 两条腿同时内旋、脚掌扫到对侧（实测帧 67~69 有 29 对自碰撞）。
                    # 这两个维度直接更快地向 FK 值（0，矢状面内）收敛。
                    w2 = min(1.0, w + 0.5)
                    q_new[1:3] = (1.0 - w2) * q_ik[1:3] + w2 * q_kn[1:3]
                    for n2, v in zip(ch, q_new):
                        jj = JOINT_NAMES.index(n2)
                        q38[kin.qadr[n2]] = float(np.clip(v, kin.lo[jj], kin.hi[jj]))
                    kin.set_q(q38)
                    tgts[i, j] = kin.d.xpos[cols[name]["body"]].copy()

            warm[name] = np.array([q38[kin.qadr[n2]] for n2 in limb_chain(arm, side)])
            flags[i, j] = 1 if contact else 0

        # 过渡段强制左右对称：右肢直接由左肢镜像。两支各自从 warm start 解，
        # 一旦某帧落进另一个折叠分支，就会出现"单侧肘/膝突然翻一下"（实测
        # 帧 98→99 的 shoulder_y 跳 142°）。镜像后这类不对称彻底消失。
        if not in_loop:
            for arm in (True, False):
                # 退出跪姿段左右腿各自收敛到 gait 首帧（首帧不是严格镜像），
                # 这里再强制镜像会把右腿的 hip_x/hip_z/knee_y 拉回左侧值。
                if (not arm) and exit_end is not None and u_norm >= EXIT_U:
                    continue
                cl, cr = limb_chain(arm, "l"), limb_chain(arm, "r")
                vl = np.array([q38[kin.qadr[n2]] for n2 in cl])
                vr = mirror_chain(vl)
                for n2, v2 in zip(cr, vr):
                    q38[kin.qadr[n2]] = v2
                warm[f"r_{'arm' if arm else 'leg'}"] = vr.copy()
                warm[f"l_{'arm' if arm else 'leg'}"] = vl

        qpos[i] = q38

    # 过渡段（entry_time>0）里，任何"重解 IK"的收尾步骤都必须带着离地软约束，
    # 否则修接触点/修跳变时会把肘和上臂推进地面（实测 u≈0.89 上臂 −100 mm）。
    # 只在「拆段动作 1」上开：整段（--merge）后面还跟着循环段，摆动相的手本来就要
    # 抬起来，给它套贴地约束等于把步态按死（实测 a=5483 rad/s²、穿地 102 mm）。
    only_entry = p.entry_time > 0 and p.cycles == 0
    guard = (max(p.entry_chain_margin, 0.006) if only_entry else None)
    # --- 4a. 分支跳变修复（细分跟踪），必须在穿地兜底之前 ---
    n_branch = smooth_branches(kin, qpos, tgts, cols, p, guard_floor=guard)
    # --- 4b. 关节速度限幅：把残余的跳变摊成"慢慢挪" ---
    i_join = int(round(p.entry_time / p.dt))
    win = ((max(0, i_join - p.force_window_pad), i_join + p.force_window_pad)
           if p.force_window_pad > 0 else (0, 0))
    n_rate = rate_limit(qpos, cols, flags, p, p.max_joint_speed_dps, win)

    # --- 4c. roll 关节全轨迹低通（姿态装饰，不影响手脚位置）---
    smooth_roll(qpos, sigma=p.roll_smooth, dt=p.dt,
                vmax_dps=p.roll_speed_dps)

    # --- 4d. 最后一遍限速：正常压（支撑相也压），接触由紧随其后的 4e 补回来 ---
    n_rate2 = rate_limit(qpos, cols, flags, p, p.max_joint_speed_dps, win)

    # --- 4e. 穿地兜底放在最末：保证接触点在收尾后仍然准确 ---
    # 它有 xy 锁定，只动几毫米；放前面会被后续限速重新拖离（实测 5→11.6 mm）。
    # 试过在它之后再压一次限速（tail_speed_dps），能把峰值压回 155，但接触点会被
    # 拖到 −11~−18 mm（判据 10 mm）。贴地爬行的接触精度比那 75 rad/s² 更重要，
    # 所以这里不再收尾限速：峰值 230 vs 判据 400，余量足够。
    n_fix = ground_clamp(kin, qpos, cols, p, flags, guard_floor=guard)

    # --- 4f. 中间段最终兜底：只过渡段需要。末端位置保持不动，把压进地面的
    # 上臂/肘就地顶回接触面以上（爬行段本来就 2.6 mm，不需要这一步）。
    if only_entry:
        p_guard = copy.copy(p)
        p_guard.chain_weight = max(p.chain_weight, 15.0)
        n_chain = chain_clamp(kin, qpos, cols, p_guard, floor=0.006)
        # 顶回肘之后腕/踝可能被带着挪几毫米，再补一次接触（带同样的离地约束，
        # 不会把肘重新压下去）。
        n_fix += ground_clamp(kin, qpos, cols, p, flags, guard_floor=guard)
        # 手臂退出窗口（u≥PLANT_U）柔顺化：手前移目标点驱动的 IK 解在该段出现
        # shoulder_z/x/y 角加速度 290~319 rad/s²、jerk 13748 的尖峰。先对手臂四关节
        # 做端点保持高斯平滑削峰，再由下面的 plant_clamp / ground_clamp 把腕压回地面。
        # v8 的窗口只有 1−0.82=18%（5 s 档 ≈45 帧），sigma 取 4：3σ=12 帧端点保留后
        # 中间还剩 ~21 帧可用；sigma=8 会把整段压成直线、撑地期手掌跟着浮起来。
        i_arm0 = int(round(PLANT_U * p.entry_time / p.dt))
        n_asmooth = smooth_arm_joints(qpos, i_arm0, n - 1, sigma=4.0)
        # 手真正撑地是 u=PLANT_U（u=0.76 只是指尖接近地面，压下去会破坏设计）。
        i_plant = int(round(PLANT_U * p.entry_time / p.dt))
        n_plant = plant_clamp(kin, qpos, cols, p, i_from=i_plant)
        # 贴地兜底会在分支切换那一帧留下单帧突起（实测 a=407 rad/s²），
        # 用 1 帧高斯把它摊开，再各补一次接触与贴地（都是毫米级微调）。
        smooth_joint_window(qpos, cols, i_plant - 12, n, sigma=1.0)
        n_fix += ground_clamp(kin, qpos, cols, p, flags, guard_floor=guard)
        n_plant += plant_clamp(kin, qpos, cols, p, i_from=i_plant)
        # plant_clamp 把腕往下按时可能过冲 1~2 cm，最后只抬不降地补一次
        n_fix += ground_clamp(kin, qpos, cols, p, flags, guard_floor=guard)
        if verbose and n_chain:
            print(f"[修正] 中间段离地兜底：{n_chain} 个「肢体-帧」")
        if verbose and n_plant:
            print(f"[修正] 撑地期手掌贴地：{n_plant} 个「肢体-帧」")
    else:
        n_chain = 0

    times = np.arange(n) * p.dt
    if verbose and n_fix:
        print(f"[修正] 穿地兜底：{n_fix} 个「肢体-帧」被抬回接触面")

    if verbose:
        print(f"[生成] {n} 帧 / {n * p.dt:.2f} s  过渡 {n_entry} 帧 + 循环 {n_loop} 帧")
        print(f"       每循环前进 {S:.3f} m  支撑期扫掠 {stride:.3f} m  "
              f"速度 {p.speed:.3f} m/s")
        print(f"       IK 最大残差 {drops.max() * 1000:.1f} mm "
              f"(均值 {drops.mean() * 1000:.1f} mm)")
    return dict(time=times, qpos=qpos, dt=p.dt, contact_flags=flags,
                joint_names=np.array(JOINT_NAMES), drops=drops, targets=tgts,
                limb_names=np.array([l["name"] for l in LIMBS]),
                meta=json.dumps(asdict(p), ensure_ascii=False), kin=kin)


def fill_all_limbs(kin: Kin, q38: np.ndarray, p: Params, s: float, lift: float):
    """把四个肢体都摆到设计姿态（只用于估触地点，不参与求解）。"""
    q = q38.copy()
    vec = kin.joint_vec(q)
    for l in LIMBS:
        for n, v in zip(limb_chain(l["arm"], l["side"]),
                        design_joints(p, l["arm"], l["side"], s, lift)):
            vec[JOINT_NAMES.index(n)] = v
        for n, v in limb_extra(p, l["side"], l["arm"]).items():
            vec[JOINT_NAMES.index(n)] = v
    kin.write_joints(q, vec)
    return q


def smooth_branches(kin: Kin, qpos: np.ndarray, tgts: np.ndarray,
                    cols: dict, p: Params, thresh_deg: float = 6.0,
                    sub: int = 8, guard_floor: float | None = None) -> int:
    """把"分支跳变"的帧改成沿目标细分跟踪重解。

    IK 有多解：手离肩 0.25~0.37 m（臂长 0.512）时肘的折叠方向有正反两支，
    从上一帧 warm start 出发也可能忽然翻过去（实测单帧跳 62~140°，
    二阶差分 1500 rad/s²）。这里不去猜该选哪支，而是把目标点从上一帧到本帧
    切成 sub 小段，每段只挪一点点、以上一段的解作初值 —— 解只能连续演化，
    物理上等价于"把这一帧的动作拆成 8 帧慢慢做"。
    只在跳变确实变小的时候才采用，避免把正常动作改坏。
    """
    n = len(qpos)
    fixed = 0
    for i in range(1, n):
        for j, l in enumerate(LIMBS):
            col = cols[l["name"]]
            adr = col["qadr"]
            jump = float(np.abs(qpos[i][adr] - qpos[i - 1][adr]).max())
            if jump <= np.radians(thresh_deg):
                continue
            q_cur = qpos[i - 1].copy()
            q_cur[col["roll_q"]] = qpos[i][col["roll_q"]]
            for k in range(1, sub + 1):
                w = k / sub
                tgt = tgts[i - 1, j] * (1.0 - w) + tgts[i, j] * w
                kin.set_q(q_cur)
                kin.solve_limb(col, tgt, q_cur[adr], p, guard_floor=guard_floor)
                q_cur[adr] = kin.d.qpos[adr]
            if float(np.abs(q_cur[adr] - qpos[i - 1][adr]).max()) < jump * 0.75:
                qpos[i][adr] = q_cur[adr]
                fixed += 1
    return fixed


def rate_limit(qpos: np.ndarray, cols: dict, flags: np.ndarray, p: Params,
               vmax_dps: float, force: tuple[int, int] = (0, 0)) -> int:
    """按关节速度上限重新铺一遍轨迹（模拟电机的速度环限幅）。

    IK 的分支跳变是"一步走 60~140°"，任何后续平滑都改不动它 —— 因为那一帧的
    姿态本身就是错的。这里干脆给它一个物理速度上限：每帧最多走 vmax·dt，
    跳变就被摊成"用几十帧慢慢挪过去"。
    支撑相的那条肢体放宽到 3 倍：末端必须钉在接触点上，不能为了平滑把脚拖走。
    """
    vmax = np.radians(vmax_dps) * p.dt
    jump_thresh = np.radians(8.0)       # 只有超过这个的单帧变化才算"跳变"
    n = len(qpos)
    n_fix = 0
    for j, l in enumerate(LIMBS):
        adr = cols[l["name"]]["qadr"]
        sup = flags[:, j] == 1
        for c in range(len(adr)):
            a = qpos[:, adr[c]].copy()
            t = 1
            while t < n:
                lim = vmax * (1.0 if force[0] <= t <= force[1]
                          else (p.support_speed_scale if sup[t] else 1.0))
                d = a[t] - qpos[t - 1, adr[c]]
                if abs(d) <= jump_thresh:
                    t += 1
                    continue
                # 跳变：从这里开始按限速逼近，直到追上原始轨迹再松开
                t0 = t
                while t < n:
                    lim = vmax * (1.0 if force[0] <= t <= force[1]
                          else (p.support_speed_scale if sup[t] else 1.0))
                    prev = qpos[t - 1, adr[c]]
                    qpos[t, adr[c]] = prev + np.clip(a[t] - prev, -lim, lim)
                    if (t > t0 and abs(a[t] - qpos[t, adr[c]]) < jump_thresh * 0.25):
                        break
                    t += 1
                n_fix += 1
                t += 1
    return n_fix


def chain_clamp(kin: Kin, qpos: np.ndarray, cols: dict, p: Params,
                floor: float = 0.0) -> int:
    """限速之后再修一次中间段穿地：末端保持原位，只把压进地面的肘/膝顶回来。

    限速会改动姿态，IK 阶段本来是干净的（−3 mm 级），限速后掉到 −13 mm。
    这里从"当前姿态"出发、把当前末端位置当目标，等于"就地微调几度"，
    不会重新引入分支跳变。
    """
    pp = copy.copy(p)
    pp.ik_null = 0.0
    n_fix = 0
    for i in range(len(qpos)):
        for l in LIMBS:
            col = cols[l["name"]]
            kin.set_q(qpos[i])
            if min(kin.body_min_z(b) for b in col["guards"]) >= floor - 1e-3:
                continue
            tgt = kin.d.xpos[col["body"]].copy()
            qn = qpos[i][col["qadr"]].copy()
            kin.solve_limb(col, tgt, qn, pp, guard_floor=floor)
            qpos[i][col["qadr"]] = kin.d.qpos[col["qadr"]]
            n_fix += 1
    return n_fix


def stitch_boundary(qpos: np.ndarray, i0: int, half: int = 4) -> None:
    """过渡段与循环段的接缝：在关节空间用 min-jerk 把两侧各 half 帧缝起来。

    两段是分别求解的（过渡段用关键帧点插值、循环段用相位目标），接缝那一帧
    姿态能差 60°。这里不去动求解，只在接缝前后 half 帧内做一次 min-jerk 过渡，
    影响面只有 2*half 帧。
    """
    a, b = i0 - half, i0 + half
    if a < 1 or b >= len(qpos):
        return
    # 只缝 roll 关节（shoulder_y / hip_y）。它们是纯冗余自由度，改它们不会
    # 动到手脚位置，所以不会破坏接触；而接缝处跳的正是这几个关节。
    idx = [JOINT_NAMES.index(f"{side}_{stem}_joint")
           for side in "lr" for stem in ("shoulder_y", "hip_y")]
    q0 = qpos[a, 7:].copy()
    q1 = qpos[b, 7:].copy()
    for t in range(a + 1, b):
        u = (t - a) / (b - a)
        w = u * u * u * (u * (u * 6 - 15) + 10)      # smootherstep
        for c in idx:
            qpos[t, 7 + c] = q0[c] + (q1[c] - q0[c]) * w


def smooth_roll(qpos: np.ndarray, sigma: float = 2.0, dt: float = 0.02,
                vmax_dps: float = 90.0) -> None:
    """对 roll 关节（shoulder_y / hip_y）做高斯平滑。

    这几个关节是纯冗余自由度：改它们几乎不动手脚末端（实测末端漂移 < 3 mm），
    但它们是所有跳变的来源 —— IK 在它们身上游走、接缝处差 60~140°。
    既然是"姿态装饰"，直接全局低通最省事，对接触也几乎没有副作用。
    """
    if sigma <= 0:
        return
    idx = [JOINT_NAMES.index(f"{side}_{stem}_joint")
           for side in "lr" for stem in ("shoulder_y", "hip_y")]
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    n = len(qpos)
    vmax = np.radians(vmax_dps) * dt
    for c in idx:
        a = qpos[:, 7 + c]
        q = np.convolve(np.pad(a, (r, r), mode="edge"), k, mode="valid")
        for t in range(1, n):                     # 平滑后再限速，彻底消灭台阶
            q[t] = q[t - 1] + np.clip(q[t] - q[t - 1], -vmax, vmax)
        qpos[:, 7 + c] = q


def smooth_arm_joints(qpos: np.ndarray, i0: int, i1: int, sigma: float = 4.0) -> int:
    """对四组手臂关节（shoulder_y/x/z, elbow_y）在 [i0, i1] 区间做端点保持的
    高斯平滑，削掉退出窗口（u≥0.68）手前移时 IK 解出的角加速度尖峰。

    只动受控关节角、不动 free joint 与腿；首尾各 r 帧保留原值，保证与相邻段
    连续（不会在窗口边界引入新的跳变）。平滑后由调用处的 plant_clamp /
    ground_clamp 把腕点重新压回地面（只动几毫米）。
    """
    if i1 - i0 < 4:
        return 0
    # r 用 2σ 而非 3σ：σ 大时 3σ 的端点保持会把末段关键帧（进入 blend 前）
    # 完全排除在平滑窗外，反而让高速度原样进入对齐、jerk 暴涨。
    r = max(1, int(2 * sigma))
    r = min(r, (i1 - i0) // 2)
    if r < 1:
        return 0
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    cidx = [7 + JOINT_NAMES.index(f"{s}_{a}_joint")
            for s in "lr" for a in ("shoulder_y", "shoulder_x", "shoulder_z", "elbow_y")]
    cnt = 0
    for j in cidx:
        y = qpos[i0:i1 + 1, j]
        if np.ptp(y) < 1e-4:
            continue
        sm = np.convolve(np.pad(y, (r, r), mode="edge"), k, mode="valid")
        qpos[i0 + r:i1 + 1 - r, j] = sm[r:-r]
        cnt += 1
    return cnt


def smooth_leg_joints(qpos: np.ndarray, i0: int, i1: int,
                      sigma: float = 1.0) -> int:
    """下蹲窗口的「腿部电机抖动」后处理：只低通 hip_x / hip_z 两对冗余自由度。

    为什么只动这两对：深蹲时脚目标几乎不动、腿近竖直，hip_x（髋外展）与
    hip_z（髋旋转）在任务 Jacobian 上高度退化，逐帧 IK 会在解流形上左右漂移；
    而 hip_y / knee_y 是决定蹲深的主屈伸，平滑它们会把下蹲曲线本身抹钝。
    实测 f0~110（u<0.44）这两个关节的一阶差分 21~29% 的帧在变号、
    二阶差分 rms 1.65° / 峰值 6.26°，正是真机上「蹲着腿在抖」的观感。

    端点必须**羽化**而不是硬保持：硬保持会在窗口边界留下 C1 断点
    （实测 |d2|max 从 6.3 涨到 14.9°）。这里首尾各 r 帧用 smootherstep 权重
    从「原值」渐变到「平滑值」，w[0]=w[L-1]=0，所以站姿首帧与撑地那一帧
    严格保持原样，窗口外一个字节都不动。

    sigma=2.0 实测（f0~110 全窗口）：rmsD2 1.65→0.65°、|d2|max 6.26→4.65°、
    变号率 22.9→8.3%；只看下蹲段 f0~60 更明显，hip_x/z 的 rmsD2 从
    1.29/1.20° 降到 0.13/0.12°。脚掌最低 −6.5→−8.4 mm，随后一次
    ground_clamp 把它补回 6.4 mm（判据 10 mm）。
    """
    if sigma <= 0 or i1 - i0 < 8:
        return 0
    L = i1 - i0 + 1
    r = max(1, min(int(2.5 * sigma), L // 2))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    w = np.ones(L)
    for m in range(r):
        s = m / r
        ss = s * s * s * (s * (s * 6 - 15) + 10)     # smootherstep
        w[m] = w[L - 1 - m] = ss
    cidx = [7 + JOINT_NAMES.index(f"{sd}_{a}_joint")
            for sd in "lr" for a in ("hip_x", "hip_z")]
    cnt = 0
    for j in cidx:
        y = qpos[i0:i1 + 1, j].copy()
        if np.ptp(y) < 1e-4:
            continue
        sm = np.convolve(np.pad(y, (r, r), mode="edge"), k, mode="valid")
        qpos[i0:i1 + 1, j] = y * (1 - w) + sm * w
        cnt += 1
    return cnt


def chain_clamp(kin: Kin, qpos: np.ndarray, cols: dict, p: Params,
                floor: float = 0.004) -> int:
    """收尾：末端保持原位，只把压进地面的肘/膝顶回来。

    过渡段俯身时手臂要绕到身体下方，IK 的软地面约束（权重 3）压不住，
    实测肘部还会穿 0.11 m。这里从当前姿态出发、把**当前末端位置**当目标做微调，
    等于"就地转几度"，不会重新引入分支跳变，也不会拖走已经贴好的手掌/脚掌。
    """
    pp = copy.copy(p)
    pp.ik_null = 0.0
    n_fix = 0
    for i in range(len(qpos)):
        for l in LIMBS:
            col = cols[l["name"]]
            kin.set_q(qpos[i])
            if min(kin.body_min_z(b) for b in col["guards"]) >= floor - 1e-3:
                continue
            tgt = kin.d.xpos[col["body"]].copy()
            qn = qpos[i][col["qadr"]].copy()
            for _ in range(4):
                kin.solve_limb(col, tgt, qn, pp, guard_floor=floor)
                if min(kin.body_min_z(b) for b in col["guards"]) >= floor - 5e-4:
                    break
            qpos[i][col["qadr"]] = kin.d.qpos[col["qadr"]]
            n_fix += 1
    return n_fix


def ground_clamp(kin: Kin, qpos: np.ndarray, cols: dict, p: Params,
                 flags: np.ndarray | None = None,
                 tol: float = 1.5e-3,
                 guard_floor: float | None = None) -> int:
    """把仍穿地的帧抬回接触面（只抬不降、不改水平位置、不动零空间先验）。

    摆动相末尾是重灾区：触地点在 0.24 m 之外，腕/踝高度沿用上一支撑相收敛出的
    zbase，落地瞬间手掌会比目标低 2~3 cm（实测 -28.8 mm）。这里逐帧补一次。
    """
    pp = copy.copy(p)
    pp.ik_null = 0.0                     # 保持当前构型，只做最小范数抬升
    n_fix = 0
    anchors: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(qpos)):
        for j, l in enumerate(LIMBS):
            col = cols[l["name"]]
            kin.set_q(qpos[i])
            zmin = kin.min_z(col["geom"])
            anchor = None
            if flags is not None and flags[i, j] == 1:
                if (j, i - 1) in anchors:          # 仍在同一支撑段
                    anchor = anchors[(j, i - 1)]
                else:                              # 新的一段，从现在的位置起算
                    anchor = kin.lowest_xy(col["geom"]).copy()
                anchors[(j, i)] = anchor
            # 支撑相要双向修（限速会让手/脚悬空几毫米）；摆动相只修穿地
            if flags is not None and flags[i, j] == 1:
                if abs(zmin + p.press) <= 3e-3:
                    continue
            elif zmin >= -p.press - tol:
                continue
            tgt = kin.d.xpos[col["body"]].copy()
            qn = np.array([kin.d.qpos[kin.qadr[n]] for n in col["chain"]])
            for _ in range(4):
                tgt[2] += (-p.press - kin.min_z(col["geom"]))
                # 支撑相还要把水平漂移拉回本段起点 —— 限速会把支撑脚往前拖，
                # 只修 z 的话滑移能到 195 mm（验收 FAIL）。这里锁住接触点 xy。
                if anchor is not None:
                    xy = kin.lowest_xy(col["geom"])
                    tgt[0] += anchor[0] - xy[0]
                    tgt[1] += anchor[1] - xy[1]
                kin.solve_limb(col, tgt, qn, pp, guard_floor=guard_floor)
                if abs(kin.min_z(col["geom"]) + p.press) < 5e-4:
                    break
                qn = np.array([kin.d.qpos[kin.qadr[n]] for n in col["chain"]])
            for nm in col["chain"]:
                qpos[i][kin.qadr[nm]] = kin.d.qpos[kin.qadr[nm]]
            n_fix += 1
    return n_fix


def smooth_joint_window(qpos: np.ndarray, cols: dict, i0: int, i1: int,
                        sigma: float = 1.0) -> None:
    """只在 [i0,i1) 内对四肢 IK 链关节做很轻的高斯低通，磨掉收尾兜底的二阶尖峰。

    用 edge padding + valid 卷积，窗口外一个字节都不动；sigma=1 帧相当于只把
    单帧突起摊到相邻两帧，末端漂移通常 <5 mm，随后的 plant/ground_clamp 能补回。
    """
    if sigma <= 0 or i1 <= i0:
        return
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for l in LIMBS:
        adr = cols[l["name"]]["qadr"]
        for a in adr:
            seg = qpos[i0:i1, a].copy()
            sm = np.convolve(np.pad(seg, (r, r), mode="edge"), k, mode="valid")
            qpos[i0:i1, a] = sm


def plant_clamp(kin: Kin, qpos: np.ndarray, cols: dict, p: Params,
                i_from: int = 0, lift_mm: float = 12.0) -> int:
    """过渡段「手已撑地」之后，把离地的手掌就地按回地面。

    分支从"肘朝内"翻成爬行段"肘朝外"时，限速会把这 90° 摊到十几帧，
    中间那些既不是 A 也不是 B 的构型会让手掌抬到 10 cm 高。这里不改分支，
    只在当前构型上做一次最小范数 IK：目标 = 当前腕点 xy + 下压到接触面，
    末端基本不动、关节也只挪几度，手掌就能一直贴着地。
    """
    pp = copy.copy(p)
    pp.ik_null = 0.0
    pp.chain_weight = max(p.chain_weight, 15.0)
    n_fix = 0
    for i in range(i_from, len(qpos)):
        for l in LIMBS:
            if not l["arm"]:
                continue
            col = cols[l["name"]]
            kin.set_q(qpos[i])
            zmin = kin.min_z(col["geom"])
            if zmin <= lift_mm * 1e-3:
                continue
            tgt = kin.d.xpos[col["body"]].copy()
            tgt[2] -= zmin + p.press
            qn = qpos[i][col["qadr"]].copy()
            for _ in range(3):
                kin.solve_limb(col, tgt, qn, pp, guard_floor=0.006)
                if abs(kin.min_z(col["geom"]) + p.press) < 1e-3:
                    break
                qn = kin.d.qpos[col["qadr"]].copy()
            for j, a in enumerate(col["qadr"]):
                qpos[i, a] = kin.d.qpos[a]
            n_fix += 1
    return n_fix


# ---------------------------------------------------------------- 场景文件
def write_scene(force: bool = False) -> None:
    """生成 elf3_prone.xml + prone_flat.xml（幂等）。"""
    FLAT_XML.parent.mkdir(parents=True, exist_ok=True)
    # 地面为什么不用 base.xml 里的 plane：
    # `type="plane"` 的无限地面在 MuJoCo 的离屏渲染（mujoco.Renderer）里根本不出现，
    # 相机 30° 俯视时整幅画面只有渐变天空球，看上去机器人像悬空、也看不出光斑落在哪
    # （实测：把平面 geom 的 rgba 改成纯红，渲染结果逐像素不变）。
    # 换成同样顶面在 z=0 的 40x40 厚板后一切正常，碰撞行为与无限平面一致。
    flat = (
        "<mujocoinclude>\n"
        "  <!-- 俯卧爬行用的纯平地面，由 prone_crawl.py 生成（勿手改）。 -->\n"
        '  <include file="assets.xml"/>\n'
        "  <asset>\n"
        '    <texture name="prone_floor_tex" type="2d" builtin="checker" '
        f'rgb1="{FLOOR_RGB1}" rgb2="{FLOOR_RGB2}" width="512" height="512"/>\n'
        f'    <material name="prone_floor" texture="prone_floor_tex" '
        f'texrepeat="{FLOOR_TEXREPEAT} {FLOOR_TEXREPEAT}" '
        'specular="0.12" shininess="0.08"/>\n'
        "  </asset>\n"
        "  <worldbody>\n"
        '    <light name="ut_world_light" pos="0 0 6" dir="0 0 -1" directional="true"\n'
        f'           diffuse="{WORLD_LIGHT_DIFFUSE}" ambient="{WORLD_LIGHT_AMBIENT}" '
        'specular="0.05 0.05 0.05" castshadow="false"/>\n'
        '    <geom name="ut_infinite_ground" type="box" '
        f'pos="0 0 -{FLOOR_HALF_T}" size="{FLOOR_HALF} {FLOOR_HALF} {FLOOR_HALF_T}" '
        'material="prone_floor" friction="1.0 0.01 0.001" condim="3"/>\n'
        '    <site name="ut_spawn_origin" type="cylinder" pos="0 0 0.006" '
        'size="0.22 0.006" rgba="0.10 0.80 0.25 0.55"/>\n'
        "  </worldbody>\n"
        "</mujocoinclude>\n"
    )
    if force or not FLAT_XML.exists() or FLAT_XML.read_text(encoding="utf-8") != flat:
        FLAT_XML.write_text(flat, encoding="utf-8")
        print(f"[场景] 写入 {FLAT_XML.relative_to(DATA)}")

    src = (DATA / "elf3.xml").read_text(encoding="utf-8")
    if ORIG_INCLUDE not in src:
        raise SystemExit(f"[错误] elf3.xml 里找不到 {ORIG_INCLUDE}")
    text = src.replace(ORIG_INCLUDE, PRONE_INCLUDE)
    # elf3.xml 为了机关相机不外溢，把远裁剪面压到 2 m（<map znear="0.0005" zfar="2"/>）。
    # 后果：离屏渲染里只要相机离机器人超过 2 m，机器人和地面就整个被裁掉——画面只剩
    # 渐变天空球，"机器人悬空"、"看不出光斑落在哪"全是这个造成的（不是光照问题）。
    # 演示场景放宽到 100 m；近裁剪面保持原值，免得影响机身相机的成像。
    text = text.replace('<map znear="0.0005" zfar="2"/>',
                        '<map znear="0.0005" zfar="100"/>')
    # 默认离屏帧缓冲只有 640x480，录 800x450 会被夹到 640x450。演示场景直接开大。
    text = text.replace('<visual>', '<visual><global offwidth="1280" offheight="720"/>')
    # 俯卧时机器人整体贴在 z≈0.1，原场景那盏 trackcom 灯几乎贴着头顶照，
    # 地面再把光全挡掉，渲染出来是一团黑。给趴地场景补一盏主光。
    #
    # 注意 `directional="true"` 时 MuJoCo 只用 dir、完全忽略 pos —— 想让位置真正
    # 生效必须用定位光（directional="false"）。灯的 xy 由 LIGHT_XY 反解得到，
    # 使**光斑中心正好压在出生点上**；dir 保持斜一点，否则灯正下方会压出纯黑影。
    key_light = (f'    <light name="prone_key" '
                 f'pos="{LIGHT_XY[0]:.4f} {LIGHT_XY[1]:.4f} {LIGHT_Z}" '
                 f'dir="{LIGHT_DIR[0]} {LIGHT_DIR[1]} {LIGHT_DIR[2]}" '
                 'directional="false" '
                 f'diffuse="{LIGHT_DIFFUSE}" ambient="{LIGHT_AMBIENT}" '
                 'specular="0.05 0.05 0.05" castshadow="true"/>\n')
    if "prone_key" not in text:
        text = text.replace("<worldbody>\n", "<worldbody>\n" + key_light, 1)
    # 那盏 trackcom 灯（贴着头顶、正下方投影）会在地面留一块纯黑的圆斑，
    # 而把它的 castshadow 关掉又会让机器人整体变黑（这个 MuJoCo 版本的怪行为，
    # 实测 mean 37->33、max 255->103）。索性从趴地场景里删掉，只留 prone_key。
    text = text.replace('      <light pos="0 0 2" dir="0 0 -1" mode="trackcom"/>\n', "")
    if force or not SCENE.exists() or SCENE.read_text(encoding="utf-8") != text:
        SCENE.write_text(text, encoding="utf-8")
        print(f"[场景] 写入 {SCENE.name}")


# ---------------------------------------------------------------- 摘要 / 自检
# ---------------------------------------------------------------- 拆成两段动作
# 完整动作 = 「站立→俯卧」+「爬行循环」。合并生成时两段在接缝处姿态差 55°，
# 那一帧的二阶差分 514 rad/s² 是全程峰值 —— 而且怎么修都压不下去（两段分别
# 求解，接缝处 IK 的解就是不一样）。拆成两个独立动作后，接缝消失，两段各自
# 的峰值是 172 / 155 rad/s²，都在电机能力范围内。

def action_params(p: Params, kind: str) -> Params:
    """按动作类型派生参数。两段的最优 IK 配置不同，不能共用一套。"""
    q = copy.copy(p)
    if kind == "standup":
        # 过渡段手臂在空中摆，3 自由度（冻结 shoulder_y）够不到关键帧目标
        # （实测残差 157 mm），所以这里保持 4 自由度。
        # 5.0 s 做完五阶段（站立/下蹲/双腿分开/前倾+双臂前伸撑地/俯卧），
        # 比 4.5 s 更从容：肘部绕到身体下方时不会顶到限速。
        q.entry_time = p.entry_time if p.entry_time > 0 else 5.0
        q.cycles = 0
        # 手臂保持 4 自由度：3 自由度（冻结 shoulder_y）在站姿时手在体侧够不到
        # （实测残差 373 mm）。肘部在俯身中段会低一点，由 ground_clamp 兜底。
        q.freeze_arm_roll = False
        # 深蹲：髋要屈到 −150° 上下（髋高 0.30 m、上身前倾 60° 时大腿几乎贴胸）。
        # 爬行段那套 ±95 / 142 会双双卡死。机械限位是 −165°/150°，留 5°/2° 余量。
        q.hip_y_limit = 160.0
        q.knee_hi_limit = 148.0
        # 过渡段不能沿用步态的 120°/s 限速：0.76→0.86 基座要降 18 cm，腿得
        # 及时后伸，被 120°/s 削顶后 hip 胶囊会压进地面 20 mm（normal 档
        # 帧 198~210，fast 档用 600°/s 就没有）。动作 1 只播一次、没有
        # 循环接缝，放宽不影响柔顺度；gait 段仍用预设值。
        q.max_joint_speed_dps = max(p.max_joint_speed_dps, 600.0)
    else:                                         # gait
        q.entry_time = 0.0
        # 手臂 roll 退出 IK：解唯一，肘的折叠分支不再游走（343→154 rad/s²）。
        # 腿不能冻（髋部胶囊会压到 −13.7 mm）。
        q.freeze_arm_roll = True
        q.freeze_leg_roll = False
        # 绕开 hip_z=±90° 万向锁：实测 a_max 218→158、jerk 10140→6601
        q.hip_z_band = 95.0
        q.roll_smooth = max(q.roll_smooth, 8.0)
        # 支撑相限速：有 ground_clamp 的接触点 xy 锁定兜底，滑移不会涨（实测 23 mm 不变），
        # 但能把腿在 hip_z≈−90° 万向锁附近的 8~12° 耦合抖动摊平。
        q.support_speed_scale = min(q.support_speed_scale, 1.5)
    return q


def blend_tail_to(qpos: np.ndarray, q_target: np.ndarray, n_blend: int,
                  dt: float = 0.02) -> int:
    """把轨迹末 n_blend 帧用带「进入速度/加速度匹配」的五次 Hermite 收敛到 q_target。

    旧版用 smootherstep（端点速度恒为 0）。进入对齐时肢体若仍在快速运动（如
    手臂前移 ~7.9 rad/s），边界处速度从 v0 突降到 0，jerk 尖峰 8000+，视觉上就是
    「手前移途中突然一顿再滑过去」——动作一卡一卡的主因。
    五次 Hermite 以进入帧的实际速度 v0、加速度 a0 为起点边界、终点速度/加速度取 0，
    整段 C2 连续，顿挫消失；末帧仍精确等于 q_target（join 衔接不破）。
    """
    n = len(qpos)
    n_blend = min(n_blend, n - 3)
    if n_blend <= 2:
        return 0
    a = n - n_blend
    v0 = (qpos[a] - qpos[a - 1]) / dt
    a0 = (qpos[a] - 2.0 * qpos[a - 1] + qpos[a - 2]) / (dt * dt)
    T = n_blend * dt
    TT = T * T
    for t in range(a + 1, n):
        s = (t - a) / n_blend
        s2 = s * s; s3 = s2 * s; s4 = s3 * s; s5 = s4 * s
        h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5
        h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5
        h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
        h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5
        h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
        h21 = 0.5 * s3 - s4 + 0.5 * s5
        qpos[t] = (h00 * qpos[a] + h10 * (T * v0) + h20 * (TT * a0)
                   + h01 * q_target + h11 * 0.0 + h21 * 0.0)
    return n_blend


def build_split(p: Params, join: bool = True, n_blend: int = 48,
                verbose: bool = True):
    """生成两个独立动作：standup（站立→俯卧）与 gait（向前爬行）。

    join=True 时把 standup 的末段收敛到 gait 首帧姿态，使两个 npz 可以直接连播
    （否则接缝处差 55°，连播会闪一下 —— 但那已不在任一动作内部）。
    """
    pA = action_params(p, "standup")
    pB = action_params(p, "gait")

    # 顺序反过来：动作 1 的「双腿分开」终点就是 gait 首帧的腿角，而两档首帧
    # 不同（fast 左膝 96.9°、normal 52.3°）、左右也非严格镜像，必须现取现用。
    if verbose:
        print("[拆段] 先解动作 2/2：向前爬行（%d 循环 × %.2f s）"
              " —— 首帧作为动作 1 的终点姿态" % (pB.cycles, pB.cycle_time))
    resB = build(pB, verbose=verbose)
    if verbose:
        print("[拆段] 动作 1/2：站立 -> 俯卧（%.2f s）" % pA.entry_time)
    resA = build(pA, verbose=verbose, exit_q=resB["qpos"][0])

    if join:
        kin = resA["kin"]
        cols = {l["name"]: kin.make_limb_col(l["arm"], l["side"]) for l in LIMBS}
        n = blend_tail_to(resA["qpos"], resB["qpos"][0], n_blend, pA.dt)
        # 末段插值后可能压进地面，用现有兜底只抬不降修一次（不改基座、不动水平位置）
        # 这里**不能**带离地软约束：末 20 帧正在从站姿插值到爬行首帧，
        # 带约束重解会把 r_hip_y 一帧翻 72°（实测 a=1008 rad/s²）。
        n_fix = ground_clamp(kin, resA["qpos"], cols, pA, flags=None)
        n_plant = plant_clamp(kin, resA["qpos"], cols, pA,
                              i_from=int(round(PLANT_U * pA.entry_time / pA.dt)))
        n_fix += ground_clamp(kin, resA["qpos"], cols, pA, flags=None)
        if verbose:
            print(f"[拆段] 末段 {n} 帧对齐到爬行首帧（穿地兜底 {n_fix} 处，"
                  f"手掌贴地 {n_plant} 处）")

    # 旧版「事后 10° 外展偏置 + 14 mm 基座微抬」已停用。那两步是为补 IK 蛙式
    # 分支造成的大腿自碰撞 / 髋脚穿地；现在双腿分开由 exit_leg_joints() 在关节
    # 空间直接收敛到 gait 首帧（hip_x 18°→68°、hip_z 2°→95°），FK 实测自碰撞
    # 0 帧、中间段 −0.3 mm，再叠偏置只会把脚推进地面。两个函数保留在上方作为
    # 踩坑记录，不要再调用。

    # --- 下蹲段腿部冗余关节低通：必须放在所有 blend / clamp 之后 ---
    # ground_clamp / plant_clamp 会重解 IK，把平滑后的 hip_x/hip_z 又推回解流形
    # 上的漂移点。放在最后，落盘的 npz 与 report() 看到的都是平滑结果。
    # 只动 f0~PLANT_U 窗口内的 4 个关节，端点羽化使 f0 / f_plant 严格保持原值，
    # 末段对齐区（最后 48 帧）完全不受影响。
    if pA.leg_shake_sigma > 0:
        i_shake = min(int(round(PLANT_U * pA.entry_time / pA.dt)),
                      len(resA["qpos"]) - 1)
        n_leg = smooth_leg_joints(resA["qpos"], 0, i_shake,
                                  sigma=pA.leg_shake_sigma)
        # 低通会让脚掌最低点下移约 2 mm（fast −6.5→−8.4 mm）。补一次
        # 「只抬不降」的接触兜底，把余量买回来。这里从**平滑后的构型**出发、
        # ik_null=0，只做最小范数抬升，不会把 IK 解推回原来的漂移点。
        cols = {l["name"]: resA["kin"].make_limb_col(l["arm"], l["side"])
                for l in LIMBS}
        n_legfix = ground_clamp(resA["kin"], resA["qpos"], cols, pA, flags=None)
        if verbose and n_leg:
            print(f"[拆段] 下蹲段腿部冗余关节低通：{n_leg} 条曲线，"
                  f"f0~{i_shake}，sigma={pA.leg_shake_sigma}；接触补正 {n_legfix} 处")

    return (resA, pA), (resB, pB)


def report(res: dict, p: Params) -> None:
    qpos, kin = res["qpos"], res["kin"]
    n = len(qpos)
    vec = np.array([kin.joint_vec(qpos[i]) for i in range(n)])
    print("\n[摘要] 关节范围（度）与限位余量：")
    print(f"  {'关节':<22}{'最小':>9}{'最大':>9}{'余量':>9}")
    worst = 9.9
    for k, name in enumerate(JOINT_NAMES):
        idx = JOINT_NAMES.index(name)
        lo_lim = math.degrees(kin.lo[idx])
        hi_lim = math.degrees(kin.hi[idx])
        vmin = math.degrees(vec[:, k].min())
        vmax = math.degrees(vec[:, k].max())
        margin = min(vmin - lo_lim, hi_lim - vmax)
        worst = min(worst, margin)
        if abs(vmax - vmin) > 1.0 or margin < 15.0:
            print(f"  {name:<22}{vmin:>9.1f}{vmax:>9.1f}{margin:>9.1f}")
    print(f"  最小限位余量 {worst:.1f}°  "
          f"{'OK' if worst > 3.0 else '!! 太贴近限位'}")

    # 四端接触高度（逐帧 FK）
    print("\n[摘要] 接触高度（mm）与支撑相命中率：")
    print(f"  {'肢体':<8}{'最小':>8}{'最大':>8}{'支撑帧占比':>12}{'命中率':>10}")
    flags = res["contact_flags"]
    for j, l in enumerate(LIMBS):
        zs = []
        for i in range(n):
            kin.set_q(qpos[i])
            prefix = f"{l['side']}_wrist_z_collision" if l["arm"] else f"{l['side']}_foot"
            zs.append(kin.min_z(prefix) * 1000)
        zs = np.array(zs)
        st = flags[:, j] == 1
        # 支撑相判定：接触点落在「预压 -5 mm 附近」，不是 0 —— 轨迹本身带 5 mm 预压
        hit = float(np.mean(np.abs(zs[st] + p.press * 1000.0) <= 4.0)) if st.any() else 0.0
        print(f"  {l['name']:<8}{zs.min():>8.1f}{zs.max():>8.1f}"
              f"{st.mean() * 100:>11.0f}%{hit * 100:>9.0f}%")

    # 躯干 / 腹部 / 前进
    kin.set_q(qpos[-1])
    print(f"\n[摘要] 末帧基座 x={qpos[-1, 0]:.3f} m（起点 {qpos[0, 0]:.3f}）  "
          f"末端质心前进 {qpos[-1, 0] - qpos[0, 0]:.3f} m")


# ---------------------------------------------------------------- 预设
# 速度不是单独一个旋钮：v = 步幅 /(duty·T)，而步幅受肢体伸展半径硬限制。
# 0.12 m/s 时步幅 0.18 m，踝-髋半径已到 0.591/0.64；想快只能"缩周期 + 收站距"。
# 下面两档都实测过（IK 残差 / 接触命中 / 质心覆盖），不是拍脑袋的值。
PRESETS = {
    "normal": dict(suffix="", speed=0.12, cycle_time=2.0, duty=0.75,
                   leg_touch=(-0.75, 0.36), arm_touch=(0.42, 0.36),
                   body_sway=0.065, clearance=0.035, swing_up=0.18,
                   # 限速按周期等比放大：同一个动作在 0.8 s 里做完，关节速度是
                   # 2.0 s 版的 2.5 倍，用 120°/s 去限会把轨迹拖到崩（实测穿地 533 mm）
                   max_joint_speed_dps=120.0, chain_margin=0.002),
    "fast": dict(suffix="_fast", speed=0.50, cycle_time=0.85, duty=0.55,
                 # 脚收到 -0.64：给后极限腾出伸展余量，否则 IK 直接崩（残差 98 m）
                 leg_touch=(-0.64, 0.28), arm_touch=(0.50, 0.36),
                 body_sway=0.030, clearance=0.040, swing_up=1.0,
                 # 限速 300 会把 elbow 的速度削成"平顶"（连续多帧 300 300 300），
                 # 那就是肉眼看到的"一顿一顿"。自然峰值 ~375°/s，限速提到 600 就不削顶了。
                 max_joint_speed_dps=600.0,
                 # 0.8 s 周期下软地面约束会把 IK 卡死（实测残差 1424 mm、穿地 533 mm）。
                 # 关掉它，中间段靠设计姿态本身的余量（实测 −5.1 mm，仍在 6 mm 判据内）。
                 chain_margin=0.0),
}


def preset_out(name: str, kind: str = "") -> Path:
    """轨迹文件名。kind 非空时输出分动作文件（prone_standup_*_traj.npz 等）。"""
    suffix = PRESETS[name]["suffix"]
    stem = f"prone_{kind}" if kind else "prone_crawl"
    return DATA / f"{stem}{suffix}_traj.npz"


def save_traj(path: Path, res: dict, p: Params, preset: str, kind: str = "") -> None:
    np.savez(path, time=res["time"], qpos=res["qpos"], dt=p.dt,
             contact_flags=res["contact_flags"],
             joint_names=res["joint_names"],
             limb_names=res["limb_names"],
             preset=preset, action=kind or "full",
             meta=res["meta"])
    print(f"[OK] 轨迹 -> {path}  ({len(res['time'])} 帧, "
          f"{path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ELF3 双手双脚贴地爬行动作序列生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="normal", choices=list(PRESETS),
                    help="normal=0.12 m/s 基线；fast=0.5 m/s（缩周期+收站距）")
    ap.add_argument("--out", default=None, help="输出 npz（默认按 preset 命名）")
    ap.add_argument("--dt", type=float, default=None, help="采样周期 s")
    ap.add_argument("--cycle-time", type=float, default=None, help="循环周期 s")
    ap.add_argument("--speed", type=float, default=None, help="前进速度 m/s")
    ap.add_argument("--duty", type=float, default=None,
                    help="支撑相占空比（四拍步态要 ≥0.75 才始终 3 点支撑）")
    ap.add_argument("--cycles", type=int, default=None, help="爬行循环数")
    ap.add_argument("--clearance", type=float, default=None, help="摆动抬升 m")
    ap.add_argument("--press", type=float, default=None, help="接触预压 m")
    ap.add_argument("--body-z", type=float, default=None, help="俯卧躯干高 m")
    ap.add_argument("--body-sway", type=float, default=None, help="左右重心转移幅值 m")
    ap.add_argument("--arm-touch", type=float, nargs=2, default=None,
                    metavar=("X", "Y"), help="手掌触地点（身体系）")
    ap.add_argument("--leg-touch", type=float, nargs=2, default=None,
                    metavar=("X", "Y"), help="脚掌触地点（身体系）")
    ap.add_argument("--entry-time", type=float, default=None,
                    help="过渡段时长 s，0=不要")
    ap.add_argument("--no-entry", action="store_true", help="只要爬行循环")
    ap.add_argument("--write-scene", action="store_true", help="生成 elf3_prone.xml")
    ap.add_argument("--force-scene", action="store_true", help="强制重写场景文件")
    ap.add_argument("--check-only", action="store_true", help="只跑求解+摘要，不落盘")
    ap.add_argument("--split", action="store_true",
                    help="拆成两个独立动作输出：standup（站立→俯卧）+ gait（向前爬行）。"
                         "接缝处的角加速度峰值会消失")
    ap.add_argument("--no-join", action="store_true",
                    help="配合 --split：不把 standup 末段对齐到 gait 首帧")
    ap.add_argument("--blend-frames", type=int, default=48,
                    help="配合 --split：末段对齐用多少帧（默认 32 = 0.64 s）。"
                         "20 帧时最后 3 cm 的侧偏 + 手臂分支要在 0.4 s 内走完，"
                         "再叠加手掌贴地重解会在 f234 跳 25°（a=336、jerk 14514）；"
                         "摊到 32 帧后 a=145、jerk 6040")
    args = ap.parse_args()

    pre = PRESETS[args.preset]
    knobs = dict(
        dt=args.dt, cycle_time=args.cycle_time, speed=args.speed,
        duty=args.duty, cycles=args.cycles, clearance=args.clearance,
        press=args.press, body_z=args.body_z, body_sway=args.body_sway,
        entry_time=args.entry_time,
    )
    # 预设打底，命令行显式给的再覆盖
    knobs["max_joint_speed_dps"] = pre["max_joint_speed_dps"]
    knobs["chain_margin"] = pre["chain_margin"]
    for k in ("speed", "cycle_time", "duty", "body_sway", "clearance"):
        if knobs[k] is None:
            knobs[k] = pre[k]
    kwargs = {k: v for k, v in knobs.items() if v is not None}
    kwargs["swing_up"] = pre["swing_up"]
    kwargs["arm_touch"] = tuple(args.arm_touch) if args.arm_touch else pre["arm_touch"]
    kwargs["leg_touch"] = tuple(args.leg_touch) if args.leg_touch else pre["leg_touch"]
    p = Params(**kwargs)
    if args.no_entry:
        p.entry_time = 0.0
    if args.no_entry:
        p.entry_time = 0.0

    print(f"[预设] {args.preset}: {p.speed:.2f} m/s  周期 {p.cycle_time:.2f} s  "
          f"duty {p.duty:.2f}  步幅 {p.speed * p.duty * p.cycle_time:.3f} m  "
          f"站距 手{p.arm_touch} 脚{p.leg_touch}")

    if args.write_scene or not SCENE.exists() or args.force_scene:
        write_scene(force=args.force_scene)

    if args.split:
        (resA, pA), (resB, pB) = build_split(
            p, join=not args.no_join, n_blend=args.blend_frames)
        print("\n========== 动作 1/2：站立 -> 俯卧 ==========")
        report(resA, pA)
        print("\n========== 动作 2/2：向前爬行 ==========")
        report(resB, pB)
        if args.check_only:
            return
        outA = Path(args.out) if args.out else preset_out(args.preset, "standup")
        outB = outA.with_name(Path(preset_out(args.preset, "gait")).name)
        save_traj(outA, resA, pA, args.preset, "standup")
        save_traj(outB, resB, pB, args.preset, "gait")
        print(f"   A: python prone_replay.py --traj {outA} --mode kin")
        print(f"   B: python prone_replay.py --traj {outB} --mode kin --loop")
        return

    res = build(p)
    report(res, p)
    if args.check_only:
        return

    out = Path(args.out) if args.out else preset_out(args.preset)
    save_traj(out, res, p, args.preset)
    print(f"     回放: python prone_replay.py --traj {out} --mode kin --loop")


if __name__ == "__main__":
    main()
