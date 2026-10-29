# elf3-climb-VR

精灵3（elf3 / BX 29-31 自由度人形）机器人**爬楼梯**、**打拳**的 VR 沉浸式体验与本地 MuJoCo 仿真。

本仓库合并了两套工程：

- `sim/` —— 本地 MuJoCo 闭环爬楼梯仿真（Python，无需 ROS / 浏览器）
- `web/` —— WebXR 网页端（three.js + MuJoCo WASM），可在头显 / 浏览器里沉浸式回放「打拳」与「爬楼梯」动作

---

## 目录结构

```
elf3-climb-VR/
├── sim/                          # 本地 MuJoCo 仿真（爬楼梯主链路）
│   ├── vendor/
│   │   ├── elf3_check.py         # 14 项自检
│   │   ├── play.py
│   │   ├── policy.onnx           # motion-tracking 打拳策略
│   │   ├── fight.npz             # 打拳参考动作
│   │   ├── climb_stairs.py       # 深度视觉闭环爬楼梯 runner
│   │   ├── climb_normal_depth.mp4  # 55s 通关录像
│   │   ├── climb_traj_*.npz      # 回放轨迹
│   │   ├── run_climb.bat         # 一键启动（native 窗口）
│   │   ├── run_sim.bat
│   │   ├── 本地仿真跑通指南.md
│   │   └── bxi_controller_ros2/  # 第三方框架包（含 elf3.xml / 深度相机 / 地形 / 各 mod 的 ONNX）
│   └── ...
│       # 仿真入口脚本实际位于
│       #   vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch/
│       #   （elf3_play.py / view_mujoco.py / climb_stairs.py / e2e_depth_policy.py）
├── web/                          # WebXR 网页端（VR 沉浸式体验）
│   ├── index.html
│   ├── js/  (main.js / mjscene.js / vrnav.js / xr.js / motion.js)
│   ├── vendor/  (mujoco.wasm / three.module.js)
│   ├── assets/  (elf3.xml / 地形 scene_bx.xml + scene_crab.xml / 回放 motion_*.bin|json)
│   ├── tools/  (verify.mjs 自检 / bake_*.py 烘焙 / 证书生成)
│   └── start.bat / start_vr_motion.bat
└── docs/                         # 设计说明与云端部署手册
    ├── crab_rl_training.md       # RL 任务设计（奖励/课程/验收）
    └── gpu_server_deploy.md      # GPU 云主机部署与换机清单
```

> 注：`sim/vendor/bxi_controller_ros2` 是从 `bxirobotics/bxi_controller_ros2` 下载的第三方框架包；
> `com.bxi.normal_depth` 等模型权重来自厂商，**请按仓库可见性（Private）妥善管理，勿外传**。

---

## 1. 本地仿真（`sim/`）

依赖（隔离 venv）：`mujoco` + `onnxruntime` + `numpy` + `pillow`。

会爬楼梯的策略：`normal_depth.onnx` + `body_depth_camera`（躯干相机，俯角 70°）。
实测 55s 全程通关：10cm 低楼梯上下 + 15cm 高楼梯上下 + 壕沟堤岸，从未摔倒。

```bat
:: 一键启动（双击 run_climb.bat，自动 cd 到 launch 目录）
run_climb.bat normal_depth 55
:: 或指定策略与场景
run_sim.bat onnx stairs
```

关键说明：
- `.bat` 必须为纯 ASCII + CRLF（UTF-8 中文注释会被 cmd 按 GBK 截断导致秒关）。
- 深度图背景值 0（天空）需在渲染后替换为 1e3，否则被当成近障碍。
- PD 必须按模型时间步（0.002s）每个物理子步重算，50Hz PD 太软。

详见 `sim/vendor/本地仿真跑通指南.md`。

---

## 2. VR 网页端（`web/`）

- 运行：`start.bat`（本地静态服务器 + 浏览器）或 `start_vr_motion.bat`（VR 运动模式）。
- 在「显示 → 楼梯 / 障碍」里查看 universal_test 紧凑地形；
  「运动仿真」可选「打拳」(`motion_fight_*`)、「爬楼梯」(`motion_climb_policy`)
  或「螃蟹步横移上楼」(`motion_crab`) 回放。
- 自检验证：`node tools/verify.mjs http://localhost:8123/`（56 项）。

网页模型 `nq=36`（老版 elf3.xml），官方场景 `nq=38`（多 head_z/head_y），
前 36 个 qpos 关节顺序一致，故仿真闭环轨迹可直接回放。

---

## 3. 螃蟹步横移爬楼梯（一键启动）

侧身沿 +Y 横移上下楼，**脚本控制器**（四相位时序 + 闭式解种子 + DLS 数值 IK），
不依赖 ONNX 策略。台阶随机生成，级高 12–25 cm、进深 20–40 cm。

```bat
run_crab.bat                      :: 一键：生成地形 -> 规划轨迹 -> 弹回放窗口
run_crab.bat --preset easy        :: 简单地形（级高 12-16cm / 进深 30-40cm）
run_crab.bat --show-config        :: 只打印最终生效参数，不执行
run_crab.bat --stages gen,plan    :: 不弹窗，只出地形和轨迹
```

Git Bash / Linux 用 `./run_crab.sh`，参数完全相同。

**改参数**（优先级从低到高）：`crab_config.json`（长期生效）< `--preset` < `--set 组.键=值` < 快捷参数。

| 想改什么 | 怎么改 |
|---|---|
| 换一套台阶 | `--seed 7` |
| 级高范围 | `--rise-min 0.12 --rise-max 0.20` |
| 踏面进深 | `--tread-min 0.25 --tread-max 0.40` |
| 走慢一点更稳 | `--step-time 1.2` |
| 不弹回放窗口 | `--no-replay` |
| 存 PNG 序列 | `--record out/crab` |
| 物理回放（预期会摔） | `--mode phys` |

脚本分工（都在 `launch/` 下）：
`gen_crab_stairs.py` 随机台阶 → MJCF + 真值表 ｜ `crab_ik.py` 可行性回归 + IK
（`--grid` / `--tilt` / `--verify` 自检）｜ `crab_climb.py` 四相位轨迹
（`--check-only` 只体检）｜ `crab_replay.py` 回放（kin / phys / 离屏 PNG）｜
`run_crab.py` 上面的流水线。

体检基线（seed 42，13 级，最高 1.276 m）：1285 帧 / 25.7 s，位置误差 0.97 mm，
脚掌严格平贴，全程不超关节限位；但 **43% 的帧质心在支撑面外**（最长连续 0.24 s），
纯准静态站不住，这部分需要 RL 兜底。

**在网页里看**（`web/`）：面板「回放 → 脚本步态 · 螃蟹步横移上楼」，或直接
`http://localhost:8080/?motion=crab`。这一项用的是**另一套地形**（沿 +Y 上升），
`bake_crab.py` 会把它同步成 `web/assets/scene_crab.xml` —— 生成 + 规划 + 烘焙一条命令：

```bash
python web/tools/bake_crab.py                  # 全链约 20 s
python web/tools/bake_crab.py --seed 7         # 换一套台阶重录
```

---

## 备注

- 训练环境（GPU 云主机）的选型、依赖版本与换机验收：见 `docs/gpu_server_deploy.md`。
- 全部大文件（`.mp4` / `.onnx` / `.npz` / `.wasm` / `.STL`）均 < 100MB，未使用 Git LFS。
- 已排除：`.workbuddy/`、`.ros/` 日志、`__pycache__`、`*.log`、`*.rknn`。
