# 螃蟹步爬楼梯 · 强化学习训练（mjlab）

本目录把「侧向螃蟹步爬楼梯」接进 mjlab（MuJoCo Warp），训练后导出 ONNX 策略。
**先读第 0 节**，它说明现在能做什么、还差什么。

---

## 0. 现状与边界（务必先看）

| 项 | 状态 |
|---|---|
| 纯机器人资产 `elf3_robot_only.xml` | ✅ 已生成并**本地编译验证**（`nq=38, nv=37, nu=0, 32 关节, 0 个 plane`） |
| 29 关节增益 / 默认位 / action_scale | ✅ 直接取 `elf3_play.py` 真值常量 |
| 初始基座高度 | ✅ `1.0339`（取 `crab_climb.py:244`） |
| 侧向楼梯地形 | ✅ 用 mjlab 自带方形金字塔台阶（向任意水平方向都上楼） |
| 任务注册 / 环境配置 / PPO 配置 | ✅ 已写（基于 mjlab `main` 分支 API） |
| **首次跑通** | ⚠️ **需要在服务器上 smoke 一次**，按报错微调（mjlab 是 beta，字段可能变） |
| **训练出可用策略** | ⏳ 需要 GPU 实跑（4096 env，约 1–2 天） |
| **可部署 ONNX** | ⏳ 训练完成后由 `export_onnx.py` 产出 |

> 一句话：**代码链路已经写全，但"可部署文件"必须由一次真实训练产生**，不能凭空生成。

---

## 1. 为什么用金字塔台阶

mjlab 自带的 `BoxPyramidStairsTerrainCfg` 是「中心平台 + 四周逐级抬升」。
机器人站在中心、面朝 **+X**，速度指令只给 **+Y**（`lin_vel_x=(0,0)`、`lin_vel_y=(0.1,0.25)`），
于是它只能靠**侧向迈步**前进——这就是螃蟹步，不需要自定义地形生成器。

---

## 2. 安装（服务器上，仓库根目录）

```bash
cd ~/elf3
source .venv/bin/activate

# 2.1 生成纯机器人 XML（去地形 include、去 motor 执行器）
python train/assets/make_robot_xml.py
#   期望输出：nq=38 nv=37 nu=0 ... plane geom 数量 = 0

# 2.2 确认 mjlab 可用
python -c "import mjlab, mujoco_warp; print('mjlab ok')"

# 2.3 把任务包安装进 mjlab（它只自动导入 mjlab/tasks/ 下的子包）
bash train/install_task.sh
#   期望输出：四个任务齐全 ✔
#   ['Elf3-Crab-Stairs', 'Elf3-Crab-Stairs-Play', 'Elf3-Crab-Residual', 'Elf3-Crab-Residual-Play']
#   默认用**软链接**（site-packages/mjlab/tasks/crab -> 仓库目录），所以 git pull 后立刻生效；
#   若文件系统不支持软链接，用 bash train/install_task.sh --copy（那就每次 pull 后都要重跑）。
```

若 `install_task.sh` 报找不到 `mjlab`，说明当前 venv 没激活。

---

## 3. 冒烟（先别开 4096）

```bash
# 3.1 零动作 / 随机动作：只验 MDP 不炸
MUJOCO_GL=egl uv run play Elf3-Crab-Stairs-Play --agent zero
MUJOCO_GL=egl uv run play Elf3-Crab-Stairs-Play --agent random

# 3.2 小规模训练 5 轮：验 reward/观测/终止
MUJOCO_GL=egl uv run train Elf3-Crab-Stairs --env.scene.num-envs 256 --agent.max-iterations 5
```

三类常见首跑报错与处理：

| 报错 | 原因 | 处理 |
|---|---|---|
| `KeyError: 'robot/imu_ang_vel'` | 该 mjlab 版本不自动建 IMU 传感器 | 在 `env_cfg.py` 里改用 `mdp.base_ang_vel` / `mdp.base_lin_vel`，或给 `EntityCfg` 加 IMU |
| `invalid scale` / `actuator_names` 不匹配 | `ACTION_SCALE_MAP` 的键是关节名，需匹配执行器名 | 把 `actuator_names=(".*",)` 改成具体关节名元组 |
| `terrain_scan` 或 `foot_height_scan` 报错 | 传感器 frame/pattern 字段随版本变化 | 对照 `mjlab/tasks/velocity/velocity_env_cfg.py` 调整 |

---

## 4. 正式训练

```bash
MUJOCO_GL=egl uv run train Elf3-Crab-Stairs \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000 \
  --log-root logs/rsl_rl
```

- 并发：4090 用 4096；先用 256/1024 确认不炸再放大。
- 课程：地形已带 `curriculum=True`，从近平地逐级升到 rise 0.22。
- 日志：`logs/rsl_rl/elf3_crab/<时间戳>/`，每 50 iter 存 checkpoint。

### 4.1 续训（可选）

```bash
MUJOCO_GL=egl WANDB_MODE=disabled \
  .venv/bin/train Elf3-Crab-Stairs --env.scene.num-envs 1024 \
  --agent.max-iterations 3000 --agent.resume True
```

三个**实测**结论（与直觉相反，别按累计理解）：

- `--agent.max-iterations` 是**本次运行再跑多少轮**。从 800 轮的 checkpoint 续训写 2500，
  会再跑 2500 轮，**不是**跑到 2500 总共。
- checkpoint 命名是 `model_<本次轮数>.pt`（如 `model_2500.pt`），不是全局累计编号。
- **课程会重置**：速度课程回到 0.05–0.12、地形等级回到 0。所以尽量一次跑够
  （用 `tmux` / `screen` 挂着），少做多次续训，否则每次都要重新爬课程。

### 4.2 logger：wandb 不可用会自动退回 tensorboard

`rl_cfg.py` 里的 `_logger_name()` 会**先探测**再决定：调一次
`wandb.Settings(start_method="thread")`（rsl_rl 的 `WandbLogWriter` 就是这么调的），
抛异常就改用 `tensorboard`，并在启动日志里打一行 `[rl_cfg][warn]`。

服务器上的典型报错（不是项目代码问题，是 wandb 与 rsl_rl 版本不匹配）：

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for Settings
start_method  Extra inputs are not permitted
```

新版 wandb 的 `Settings` 改成 pydantic 模型且 `extra=forbid`，`start_method` 已移除，
而 rsl_rl 里写死了这个参数。**不要去改 venv 里的 rsl_rl**（升级会被覆盖）。

- 用 tensorboard 时指标写在 `logs/rsl_rl/elf3_crab/<时间戳>/`，
  看曲线：`tensorboard --logdir logs/rsl_rl/elf3_crab --port 6006`（配 SSH 端口转发）。
- 想强制用 wandb：`--agent.logger wandb`（必须先把 wandb 降到兼容版本）。

---

## 5. 导出可部署文件

```bash
python train/export_onnx.py \
  --task Elf3-Crab-Stairs \
  --checkpoint logs/rsl_rl/elf3_crab/<run>/model_3000.pt \
  --out logs/export/elf3_crab_policy.onnx
```

导出后用现有链路验证与回放：

```bash
# 物理模式回放（看会不会摔）
python launch/crab_replay.py --mode phys
# 烘焙到网页回放（下拉选「脚本步态 · 螃蟹步横移上楼」）
python web/tools/bake_crab.py
```

> 导出的 ONNX 可以直接在 MuJoCo 里闭环回放（不依赖 mjlab / GPU）：
>
> ```bash
> python crab_replay.py --policy onnx \
>   --onnx train/checkpoints/elf3_crab_policy.onnx \
>   --cmd-vy 0.25 --steps 2000 --record out/policy
> ```
>
> 观测按训练时的 290 维布局拼装（含 187 维 height_scan），执行器换成 mjlab 同款
> `<position>`。`--record` 出 PNG 序列，`--no-gui` 无窗口跑完。

---

## 6. 验收指标

| 指标 | 目标 |
|---|---|
| 质心在支撑面内帧占比 | 43% → **>85%** |
| 完整楼梯成功率 | >80%（100 次随机地形） |
| 落脚点误差 | 踏面中心 ±3 cm |
| 泛化 | 换 seed/尺寸后成功率下降 <10% |

---

## 7. 文件说明

| 文件 | 作用 |
|---|---|
| `assets/make_robot_xml.py` | 从 `elf3.xml` 派生纯机器人 XML（已本地验证） |
| `mjlab_tasks/crab/robot_cfg.py` | `EntityCfg` + 29 个位置执行器（真值增益） |
| `mjlab_tasks/crab/terrain_cfg.py` | 金字塔台阶地形（4 个难度） |
| `mjlab_tasks/crab/env_cfg.py` | 观测/动作/指令/奖励/终止/课程 |
| `mjlab_tasks/crab/rl_cfg.py` | PPO 超参 |
| `mjlab_tasks/crab/__init__.py` | 注册 `Elf3-Crab-Stairs` |
| `install_task.sh` | 复制任务包进 mjlab 并自检注册 |
| `export_onnx.py` | checkpoint → ONNX |

---

## 8. 后续可做（按优先级）

1. **残差动作**：把 `q_ref(t)`（`crab_climb.py` 的四相位轨迹）作为基准，策略只输出 Δq。
   ✅ **已实现**，见第 9 节。
2. **落脚点奖励**：用 `terrain_scan` 查踏面中心，惩罚踩边。
3. **侧向专门塑形**：直接奖励 world +Y 位移，而不只靠速度跟踪。
   ✅ 已实现 `forward_progress_y`。
4. ONNX 回放 ✅ 已实现：`--policy onnx`（290 维）与 `--policy onnx-residual`（393 维）。

---

## 9. 残差任务 `Elf3-Crab-Residual`（第 1 条已实现）

旧任务 `Elf3-Crab-Stairs` 保持不动，可继续用 `model_9297.pt` 对比。
新任务把 `crab_climb.py` 的步态离线做成参考表，策略只输出限幅残差。

### 9.1 生成参考表（纯 CPU，约 90 s）

```bash
python train/assets/build_ref_tables.py
# 48 组（rise 0–0.22 步长 0.02 × tread 0.24/0.30/0.36/0.40）
# 全部通过后写出 train/assets/ref_tables/crab_ref_v1.npz（约 0.2 MB）
```

自检项：IK 误差、帧间二阶差（不是一阶——抬腿快是正常钟形加速）、
`dy_cycle ≈ tread`、限位余量、以及 rise 0.12 vs 0.22 的敏感性 ≈0.37 rad。

单独验证选表逻辑（不需要 mjlab）：

```bash
python train/mjlab_tasks/crab/ref_table.py
```

### 9.2 安装与训练

```bash
bash train/install_task.sh          # 会注册 4 个任务
MUJOCO_GL=egl WANDB_MODE=disabled \
  .venv/bin/train Elf3-Crab-Residual --env.scene.num-envs 1024 \
  --agent.max-iterations 1000
```

### 9.3 与旧任务的区别

| 项 | Elf3-Crab-Stairs | Elf3-Crab-Residual |
|---|---|---|
| 动作 | 绝对关节位置（默认站姿 offset） | **q_ref(相位) + 限幅残差** |
| 观测 | 290（height_scan 1.6×1.0） | **393（height_scan 1.2×2.0 + 相位/参考）** |
| 侧向感知 | ±0.5 m | **±1.0 m** |
| 奖励 | 速度跟踪为主 | 速度 + **世界 +Y 进度** + **参考跟踪** + 转身/横漂惩罚 |
| 指令 | 0.15–0.35 | **0.15–0.24**（与参考步速 0.18 匹配） |

⚠️ 观测与动作语义都变了，**旧 checkpoint 不能加载**。

### 9.4 首跑怎么判断对不对（**已更正，别再按老判据**）

**先纠正一个错误预期**：`--agent zero` 时机器人摔倒是**必然**的，不是配置错。
实测（本地 MuJoCo，`build_model` 同款 PD）：

| 姿势 | 结果 |
|---|---|
| 保持默认站姿不动 | **1.24 s 摔倒**，执行器力矩仅 15 N·m（上限 150）|
| 参考表任取一帧保持不动 | 1.6–1.8 s 摔倒 |
| 零残差回放整周期 | 1.3–2.4 s 摔倒 |

原因是 `ankle_x` 的 `kp` 只有 10 N·m/rad、力矩上限 15 N·m，纯 PD **不具备平衡能力**；
基线任务能站住也是靠策略输出动作，不是靠 PD。所以「零残差能走」从来不是有效判据。

**现在的设计用 α 课程绕开这个问题**：

```
q_ref = q_stand + α · (q_gait(phase) − q_stand)      α: 0 → 1，1500 iter 线性
```

* α=0 时参考 = 默认站姿 → 与已验证可训练的基线**完全同难度**；
* α→1 时参考渐变为脚本步态，策略有足够时间先把平衡学会；
* `phase_shift=33/90`：phase=0 落在**双支撑起点**（表帧 33），不会一开场就单腿支撑；
* play 配置 `alpha_fixed=1.0`（满幅），训练配置 `-1`（走课程）。

**有效的检查顺序**：

1. `play Elf3-Crab-Residual-Play --agent zero` 只验证「能建环境、跑 1–2 s 不报错」，
   **不要**把摔倒当失败。
2. 训练看三条曲线：`Episode_Reward/ref_tracking` 应缓慢上升、
   `Episode_Reward/forward_progress_y` > 0、`Curriculum/terrain_levels/mean` 开始涨。
3. 1500 iter（α 到 1）之后再判断步态像不像螃蟹步。

### 9.5 本地回放残差策略（393 维 ONNX）

```bash
python train/export_onnx.py --task Elf3-Crab-Residual \
  --checkpoint logs/rsl_rl/elf3_crab/<run>/model_XXXX.pt \
  --out train/checkpoints/elf3_crab_residual.onnx

python sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch/crab_replay.py \
  --policy onnx-residual --onnx train/checkpoints/elf3_crab_residual.onnx \
  --cmd-vy 0.18 --steps 2000 --no-gui
```

`--policy onnx-residual` 与 `--policy onnx` 的区别：

| | 基线（onnx） | 残差（onnx-residual） |
|---|---|---|
| 观测 | 290 维，height_scan 1.6×1.0 = 187 | **393 维**，height_scan 1.2×2.0 = **273** |
| 额外观测 | — | `joint_pos_ref_err(12)`、`phase sin/cos(2)`、`ref_meta(3)` |
| 动作 | `default + scale·a` | **`q_ref(rise,tread,phase) + clip(scale·a, ±clip_rad)`** |
| 地形 | 无关 | 用 273 条射线实时估 `rise/tread`，EMA 平滑 |

残差限幅：腿 ±0.30 rad、腰 ±0.15、臂 ±0.15；相位速率 = `clip(|cmd_y|/v_ref, 0.6, 1.6)`，
前 25 步冻结（与训练一致）。`--spawn-z` 默认 `1.0339`，和训练 reset 一致。

> 找不到参考表时会提示设 `ELF3_REPO=<仓库根>`；表本身在
> `train/assets/ref_tables/crab_ref_v1.npz`。

---

## 10. 怎么确认「是不是螃蟹步」

### 10.1 先说一个坑：跟拍相机会骗人

`ViewerConfig.origin_type` 默认 `AUTO`（跟随机器人）。跟拍时机器人永远在画面中央、
台阶跟着画面一起退，**肉眼完全看不出它到底走了没有**。2026-10-29 一段 16 s 视频
就是这么被误读成「一次侧步都没有」，而同一份 checkpoint 无界面实测 `v_y=0.126 m/s`。

所以 `Elf3-Crab-Residual-Play` 已默认改成 **WORLD 固定机位**（`env_cfg_residual._retune_viewer`），
台阶在画面里不动，机器人相对它平移才看得出来。

### 10.2 录视频

**推荐用诊断脚本录**（`--record`）——它会自动把相机对准 env 的真实出生点：

```bash
cd ~/elf3 && git pull --ff-only

MUJOCO_GL=egl .venv/bin/python scripts/diagnose_policy.py \
  --task Elf3-Crab-Residual-Play \
  --checkpoint logs/rsl_rl/elf3_crab/<run>/model_XXXX.pt \
  --num-envs 1 --steps 1000 --terrain-level 5 \
  --record /tmp/walk.mp4
```

一次性拿到**视频 + 全部数字**（`v_y` / yaw 速率 / `rise_est` / 基座 z），
不用再单独跑一遍 play。可调项：`--record-offset "0,0.9,0.35"`（注视点相对出生点）、
`--record-dist 6`、`--record-elev -15`、`--record-azim 90`。

用 `play --video` 也可以，但要记住两个默认值坑：

```bash
MUJOCO_GL=egl .venv/bin/play Elf3-Crab-Residual-Play \
  --checkpoint-file logs/rsl_rl/elf3_crab/<run>/model_XXXX.pt \
  --num-envs 1 \
  --video --video-length 1000 --video-width 1280 --video-height 720
```

- `--checkpoint-file` 是**必须**的（不是 `--checkpoint`）；启动时必须看到
  `[INFO]: Loading checkpoint: ...`，没有这行就是没加载。
- 默认 `--video-length 200` 只录 **4 秒**，太短，至少给 1000（=20 s）。
- 默认分辨率只有 320×240，必须显式放大，否则看不清脚。
- 视频落在 `logs/rsl_rl/elf3_crab/<run>/videos/play/rl-video-step-0.mp4`。
- ⚠️ `play` 的固定机位只对**单 env、出生点在原点**有效：mjlab 的
  `OffscreenRenderer` 直接用 `camera.lookat = cfg.lookat`，**不加 env_origins**，
  而多 env 场景里 env0 可能在世界 x=27.8 处 —— 那时画面里是一片空地。
  这种就用上面的 `--record`。

### 10.3 画面看什么（按顺序）

| # | 看什么 | 是螃蟹步 | 不是 |
|---|---|---|---|
| 1 | 机器人相对**台阶**的位置 | 持续往侧向挪 | 站在原地晃 |
| 2 | 身体朝向（躯干正面/肩线） | 始终朝同一侧，不转身 | 转过身去正面朝前走 |
| 3 | 落脚 | 两脚交替抬起、踩上高一级踏面 | 双脚不离地蹭、或一直劈叉 |
| 4 | 基座高度 | 随台阶一级级升高 | 一直往下掉（在下楼） |

第 2 条是定义性的：**螃蟹步 = 朝向不变 + 侧向平移**。转身走 = 普通行走，不算。

### 10.4 固定机位不对时这样调（不用改代码）

```bash
ELF3_VIEWER_ORIGIN=world  .venv/bin/play ...   # 固定（默认，看位移）
ELF3_VIEWER_ORIGIN=asset  .venv/bin/play ...   # 跟拍（看脚下细节）
ELF3_VIEWER_LOOKAT="0,0.8,0.3"  ELF3_VIEWER_DIST=6  ELF3_VIEWER_AZIM=90 \
  ELF3_VIEWER_ELEV=-18  .venv/bin/play ...
```

画面里看不到机器人 ⇒ 出生点不在原点，先用诊断脚本打印 `root=(x, y, z)`，
再把它填进 `ELF3_VIEWER_LOOKAT`。

### 10.5 别只看视频，用数字定案

```bash
.venv/bin/python scripts/diagnose_policy.py --task Elf3-Crab-Residual-Play \
  --checkpoint logs/rsl_rl/elf3_crab/<run>/model_XXXX.pt \
  --num-envs 8 --steps 1000 --terrain-level 5
```

判据（三条同时满足才算螃蟹步）：

| 指标 | 螃蟹步 | 反例 |
|---|---|---|
| `yaw 速率 \|ω_z\|` | **< 0.15 rad/s** | 大 = 在自转，不是螃蟹步 |
| `跟踪比 v_y/cmd` | > 0.8 | ~0.5 = 挪得动走不动 |
| `基座 z` Δ | 随台阶升高 | 持续为负 = 在下楼 |

另外两个「链路有没有坏」的守门指标：

| 指标 | 正常 | 异常含义 |
|---|---|---|
| `地形估计链路` | `OK` | 断了 ⇒ `rise` 恒 0，参考步态锁平地档 |
| `rise_est` | 与实际台阶一致（等级 5–6 约 0.15） | **顶在 0.220** = 被机器人自身遮挡骗了；**恒 0** = 取数/地形问题 |

---

## 11. height_scan 为什么不能打到机器人自己

`terrain_scan` 用 `include_geom_groups=(0,)`，**只对 geom group 0 打射线**，地形在 group 0。
但机器人 XML 的 `<default class="elf3">` 里视觉 mesh 也是 `group="0"`
（碰撞胶囊是 `group="3"`），于是从躯干往下打的射线会打到自己身上：

| 射线偏移 | 命中 body | 距离 |
|---|---|---|
| `dy=+0.1` | `l_hip_y_link` | **0.330 m** ← 与实测 `h=33.2 cm` 完全对上 |
| `dy=+0.2` | `l_shoulder_z_link` | 0.026 m |

`exclude_parent_body=True` 只能排掉 `torso_link`（根 body）自己的网格，
**排不掉子 body（腰、腿、手臂）**。后果就是 `rise_est` 被顶到 clamp 上限 0.22 m，
而实地台阶只有 0.15 m，参考步态选成最难档 → 步幅过大 → 走得动才怪。

**修法**：`make_robot_xml.py` 把视觉 mesh 挪到 **group 2**
（viewer 默认显示 group 0/1/2，外观不变；group 3 是碰撞体不动）。
`install_task.sh` 会自检 group 0 为空。

```bash
python train/assets/make_robot_xml.py     # 重新生成 elf3_robot_only.xml
bash train/install_task.sh                # 期望：geom group 分布 {2: 32, 3: 37}，group 0 为空 ✔
```

> ⚠️ 改 group 后 `height_scan` 的数值会变（不再含自身遮挡），观测分布随之改变，
> **必须重新训练**，旧 checkpoint 不要再用。


