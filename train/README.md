# 螃蟹步爬楼梯 · 强化学习训练（mjlab）

本目录把「侧向螃蟹步爬楼梯」接进 mjlab（MuJoCo Warp），训练后导出 ONNX 策略。  
**先读第 0 节**，它说明现在能做什么、还差什么。

---

## 0. 现状与边界（务必先看）

| 项                            | 状态                                                       |
| ---------------------------- | -------------------------------------------------------- |
| 纯机器人资产 `elf3_robot_only.xml` | ✅ 已生成并**本地编译验证**（`nq=38, nv=37, nu=0, 32 关节, 0 个 plane`） |
| 29 关节增益 / 默认位 / action_scale | ✅ 直接取 `elf3_play.py` 真值常量                                |
| 初始基座高度                       | ✅ `1.0339`（取 `crab_climb.py:244`）                        |
| 侧向楼梯地形                       | ✅ 用 mjlab 自带方形金字塔台阶（向任意水平方向都上楼）                          |
| 任务注册 / 环境配置 / PPO 配置         | ✅ 已写（基于 mjlab `main` 分支 API）                             |
| **首次跑通**                     | ⚠️ **需要在服务器上 smoke 一次**，按报错微调（mjlab 是 beta，字段可能变）        |
| **训练出可用策略**                  | ⏳ 需要 GPU 实跑（4096 env，约 1–2 天）                            |
| **可部署 ONNX**                 | ⏳ 训练完成后由 `export_onnx.py` 产出                             |

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

| 报错                                     | 原因                                | 处理                                                                              |
| -------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------- |
| `KeyError: 'robot/imu_ang_vel'`        | 该 mjlab 版本不自动建 IMU 传感器            | 在 `env_cfg.py` 里改用 `mdp.base_ang_vel` / `mdp.base_lin_vel`，或给 `EntityCfg` 加 IMU |
| `invalid scale` / `actuator_names` 不匹配 | `ACTION_SCALE_MAP` 的键是关节名，需匹配执行器名 | 把 `actuator_names=(".*",)` 改成具体关节名元组                                            |
| `terrain_scan` 或 `foot_height_scan` 报错 | 传感器 frame/pattern 字段随版本变化         | 对照 `mjlab/tasks/velocity/velocity_env_cfg.py` 调整                                |

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

### 4.1 挂着跑（tmux）与日志

训练动辄 1–2 小时，SSH 一断（合盖 / 网络抖动）前台进程就吃 SIGHUP 被杀。  
所以放进 tmux，并把文本输出 `tee` 一份：

```bash
tmux new -s crab                       # 已存在会报 duplicate session，改用 tmux a -t crab
# ---- 进入 tmux 之后 ----
cd ~/elf3
MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/train Elf3-Crab-Residual \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 8000 \
  2>&1 | tee ~/train_v2.log
# ---- Ctrl+B 松手再按 D 退出（训练继续）；tmux a -t crab 回去 ----
```

要点：

- `> file 2>&1`（不是 `2>&1 > file`，顺序反了 stderr 还会打在屏幕上）会把 stdout  
  |  |  |  
  | --- | --- |  
  全导走，**屏幕一片空白，容易误以为卡死**。用 `| tee` 才能同时看和存。
- `PYTHONUNBUFFERED=1`：否则 `tee` 要等缓冲区满才输出，看着像卡住。
- 日志别放 `/tmp`（重启会清空），放家目录。
- **tensorboard 曲线和 checkpoint 本来就会落盘**（`logs/rsl_rl/elf3_crab/<时间戳>/`），  
  与重定向无关。`tee` 只是额外把**文本**（`[warn]` 行、traceback）留下来。
- tmux 不能扛服务器重启。要扛重启得用 `systemd-run --user` 或 `nohup` + `disown`。

### 4.2 续训（可选）

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

### 4.3 logger：wandb 不可用会自动退回 tensorboard

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

| 指标         | 目标                   |
| ---------- | -------------------- |
| 质心在支撑面内帧占比 | 43% → **>85%**       |
| 完整楼梯成功率    | >80%（100 次随机地形）      |
| 落脚点误差      | 踏面中心 ±3 cm           |
| 泛化         | 换 seed/尺寸后成功率下降 <10% |

---

## 7. 文件说明

| 文件                                | 作用                             |
| --------------------------------- | ------------------------------ |
| `assets/make_robot_xml.py`        | 从 `elf3.xml` 派生纯机器人 XML（已本地验证） |
| `mjlab_tasks/crab/robot_cfg.py`   | `EntityCfg` + 29 个位置执行器（真值增益）  |
| `mjlab_tasks/crab/terrain_cfg.py` | 金字塔台阶地形（4 个难度）                 |
| `mjlab_tasks/crab/env_cfg.py`     | 观测/动作/指令/奖励/终止/课程              |
| `mjlab_tasks/crab/rl_cfg.py`      | PPO 超参                         |
| `mjlab_tasks/crab/__init__.py`    | 注册 `Elf3-Crab-Stairs`          |
| `install_task.sh`                 | 复制任务包进 mjlab 并自检注册             |
| `export_onnx.py`                  | checkpoint → ONNX              |

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

| 项    | Elf3-Crab-Stairs         | Elf3-Crab-Residual                     |
| ---- | ------------------------ | -------------------------------------- |
| 动作   | 绝对关节位置（默认站姿 offset）      | **q_ref(相位) + 限幅残差**                   |
| 观测   | 290（height_scan 1.6×1.0） | **393（height_scan 1.2×2.0 + 相位/参考）**   |
| 侧向感知 | ±0.5 m                   | **±1.0 m**                             |
| 奖励   | 速度跟踪为主                   | 速度 + **世界 +Y 进度** + **参考跟踪** + 转身/横漂惩罚 |
| 指令   | 0.15–0.35                | **0.15–0.24**（与参考步速 0.18 匹配）           |

⚠️ 观测与动作语义都变了，**旧 checkpoint 不能加载**。

### 9.4 首跑怎么判断对不对（**已更正，别再按老判据**）

**先纠正一个错误预期**：`--agent zero` 时机器人摔倒是**必然**的，不是配置错。  
实测（本地 MuJoCo，`build_model` 同款 PD）：

| 姿势          | 结果                                  |
| ----------- | ----------------------------------- |
| 保持默认站姿不动    | **1.24 s 摔倒**，执行器力矩仅 15 N·m（上限 150） |
| 参考表任取一帧保持不动 | 1.6–1.8 s 摔倒                        |
| 零残差回放整周期    | 1.3–2.4 s 摔倒                        |

原因是 `ankle_x` 的 `kp` 只有 10 N·m/rad、力矩上限 15 N·m，纯 PD **不具备平衡能力**；  
基线任务能站住也是靠策略输出动作，不是靠 PD。所以「零残差能走」从来不是有效判据。

**现在的设计用 α 课程绕开这个问题**：

```
q_ref = q_stand + α · (q_gait(phase) − q_stand)      α: 0 → 1，1500 iter 线性
```

- α=0 时参考 = 默认站姿 → 与已验证可训练的基线**完全同难度**；
- α→1 时参考渐变为脚本步态，策略有足够时间先把平衡学会；
- `phase_shift=33/90`：phase=0 落在**双支撑起点**（表帧 33），不会一开场就单腿支撑；
- play 配置 `alpha_fixed` 默认 1.0（满幅）。**看某一版 checkpoint 的真实水平时  
  要覆盖回训练当时的 α**：`ELF3_ALPHA=<Metrics/ref_alpha>`，详见 10.2.0；  
  训练配置是 `-1`（走课程）。

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

|      | 基线（onnx）                        | 残差（onnx-residual）                                        |
| ---- | ------------------------------- | -------------------------------------------------------- |
| 观测   | 290 维，height_scan 1.6×1.0 = 187 | **393 维**，height_scan 1.2×2.0 = **273**                  |
| 额外观测 | —                               | `joint_pos_ref_err(12)`、`phase sin/cos(2)`、`ref_meta(3)` |
| 动作   | `default + scale·a`             | **`q_ref(rise,tread,phase) + clip(scale·a, ±clip_rad)`** |
| 地形   | 无关                              | 用 273 条射线实时估 `rise/tread`，EMA 平滑                         |

残差限幅：腿 ±0.30 rad、腰 ±0.15、臂 ±0.15；相位速率 = `clip(abs(cmd_y)/v_ref, 0.6, 1.6)`，
前 25 步冻结（与训练一致）。`--spawn-z` 默认 `1.0339`，和训练 reset 一致。

> 找不到参考表时会提示设 `ELF3_REPO=<仓库根>`；表本身在  
> `train/assets/ref_tables/crab_ref_v1.npz`。

---

## 10. 怎么确认「是不是螃蟹步」

### 10.1 先说一个坑：跟拍相机会骗人

`ViewerConfig.origin_type` 默认 `AUTO`（跟随机器人）。跟拍时机器人永远在画面中央、  
台阶跟着画面一起退，**肉眼完全看不出它到底走了没有**。2026-10-29 一段 16 s 视频  
就是这么被误读成「一次侧步都没有」，而同一份 checkpoint 无界面实测 `v_y=0.126 m/s`。

> 2026-10-29 晚：按使用习惯把 `Elf3-Crab-Residual-Play` 的默认机位改回了 **AUTO（跟拍）**，
> 方便看脚下细节。**上面那个坑也就跟着回来了** —— 想看「到底有没有侧移」时，  
> 显式加 `ELF3_VIEWER_ORIGIN=world` 切成固定机位（台阶在画面里不动，机器人相对它平移），
> 或者干脆别看画面、看诊断脚本的 `v_y`。改动的取舍写在 `_retune_viewer` 的 docstring 里。

### 10.2 看当前权重的动画

三种方式，按「想不想拿到数字」和「要不要实时交互」选。

#### 10.2.0 先找 checkpoint，再对齐 α

```bash
cd ~/elf3 && git pull --ff-only

RUN=$(ls -td logs/rsl_rl/elf3_crab/*/ | head -1)   # 按 mtime 取正在写的那个
CKPT=$(ls -t ${RUN}model_*.pt | head -1)           # save_interval=50，取最新的
echo "$CKPT"
```

⚠️ **最大的坑是 α，不是 checkpoint**。`Elf3-Crab-Residual-Play` 的 `alpha_fixed`  
默认写死 **1.0**（满幅脚本步态），而训练是走 α 课程的 —— 790 轮时  
`Metrics/ref_alpha` 才 **0.527**。直接回放等于「满幅参考 + 为半幅参考训出来的残差」，  
残差补不上那个差额，画面会明显比训练时差很多，很容易被误读成「策略退步」。

所以看「这一版 checkpoint 的真实水平」时，一律用 `ELF3_ALPHA` 对齐：

```bash
ELF3_ALPHA=0.527 ...          # 取日志里同轮次的 Metrics/ref_alpha
```


不设则按 1.0。**看最终成品时反而应该不设**（那时 α 课程也早满幅了）。

#### 10.2.1 方式 A：诊断脚本录 mp4（推荐）

一条命令同时拿到**视频 + 全部数字**（`v_y` / yaw 速率 / `rise_est` / 基座 z），  
不用另外跑一遍 play。相机会自动对准 env 的真实出生点：

```bash
MUJOCO_GL=egl ELF3_ALPHA=0.527 .venv/bin/python scripts/diagnose_policy.py \
  --task Elf3-Crab-Residual-Play \
  --checkpoint "$CKPT" \
  --num-envs 4 --steps 1000 \
  --terrain-level 6 \
  --record /tmp/crab_walk.mp4
```

- `--steps 1000` = 20 s。少于 200 看不到一个完整步态周期。
- `--terrain-level` 要挑**和当前训练水平相当**的：看日志 `Curriculum/terrain_levels/mean`。  
  注意**列（地形类型）是随机的**，env 0 可能落在 flat 列、画面里根本没有台阶 ——  
  那就换 `--record-env 1` / `2` / `3` 再录一条。
- 可调：`--record-offset "0,0.9,0.35"`（注视点相对出生点）、`--record-dist 6`、  
  `--record-elev -15`、`--record-azim 90`。

下载到本地：`scp ubuntu22:/tmp/crab_walk.mp4 .`（别留在 `/tmp`，重启会清）。

#### 10.2.2 方式 B：viser 实时看

想看某个动作的细节、或者想边看边调相机时用它：

```bash
MUJOCO_GL=egl ELF3_ALPHA=0.527 .venv/bin/play Elf3-Crab-Residual-Play \
  --checkpoint-file "$CKPT" --num-envs 1
```

启动后终端会打印一个 viser 地址，本地做端口转发才能打开（Windows 用 MobaXterm  
的隧道，或 VSCode Remote 的 Ports 面板）。

- `--checkpoint-file` 是**必须**的（不是 `--checkpoint`），启动时必须看到  
  `[INFO]: Loading checkpoint: ...`，没有这行就是没加载上。
- 训练同时开 play 会多占一份显存；报 OOM 就先 `Ctrl+B D` 挂起训练再录。

#### 10.2.3 方式 C：`play --video`（无窗口，落地在 run 目录）

```bash
MUJOCO_GL=egl ELF3_ALPHA=1 .venv/bin/play Elf3-Crab-Residual-Play \
  --checkpoint-file "$CKPT" \
  --num-envs 1 \
  --video --video-length 1000 --video-width 1280 --video-height 720
```

- 默认 `--video-length 200` 只录 **4 秒**，太短，至少给 1000（=20 s）。
- 默认分辨率只有 320×240，必须显式放大，否则看不清脚。
- 视频落在 `logs/rsl_rl/elf3_crab/<run>/videos/play/rl-video-step-0.mp4`。
- ⚠️ `play` 的固定机位只对**单 env、出生点在原点**有效：mjlab 的  
  `OffscreenRenderer` 直接用 `camera.lookat = cfg.lookat`，**不加 env_origins**，  
  而多 env 场景里 env0 可能在世界 x=27.8 处 —— 那时画面里是一片空地。  
  这种就用 10.2.1 的 `--record`。

### 10.3 画面看什么（按顺序）

| # | 看什么            | 是螃蟹步           | 不是           |
| - | -------------- | -------------- | ------------ |
| 1 | 机器人相对**台阶**的位置 | 持续往侧向挪         | 站在原地晃        |
| 2 | 身体朝向（躯干正面/肩线）  | 始终朝同一侧，不转身     | 转过身去正面朝前走    |
| 3 | 落脚             | 两脚交替抬起、踩上高一级踏面 | 双脚不离地蹭、或一直劈叉 |
| 4 | 基座高度           | 随台阶一级级升高       | 一直往下掉（在下楼）   |

第 2 条是定义性的：**螃蟹步 = 朝向不变 + 侧向平移**。转身走 = 普通行走，不算。

### 10.4 固定机位不对时这样调（不用改代码）

```bash
ELF3_VIEWER_ORIGIN=world  .venv/bin/play ...   # 固定机位（看位移）
ELF3_VIEWER_ORIGIN=auto   .venv/bin/play ...   # 跟拍（默认，看脚下细节）
ELF3_VIEWER_ORIGIN=asset  .venv/bin/play ...   # 跟资产根，比 auto 更近
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

| 指标               | 螃蟹步              | 反例            |
| ---------------- | ---------------- | ------------- |
| `yaw 速率 \|ω_z\|` | **< 0.15 rad/s** | 大 = 在自转，不是螃蟹步 |
| `跟踪比 v_y/cmd`    | > 0.8            | ~0.5 = 挪得动走不动 |
| `基座 z` Δ         | 随台阶升高            | 持续为负 = 在下楼    |

另外两个「链路有没有坏」的守门指标：

| 指标         | 正常                                                                                                        | 异常含义                                                |
| ---------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `地形估计链路`   | `OK`                                                                                                      | 断了 ⇒ `rise` 恒 0，参考步态锁平地档                            |
| `rise_est` | **只当「估计器有没有坏」的守门指标**：既不恒 0、也不顶 0.220。它会被「跨过脚下那级」影响而变小，**不能当难度指标** —— 难度看 `Curriculum/terrain_levels/mean` | **顶在 0.220** = 被假边骗了（自身遮挡 / 摔倒姿态）；**恒 0** = 取数或地形问题 |

---

## 11. height_scan 为什么不能打到机器人自己

`terrain_scan` 用 `include_geom_groups=(0,)`，**只对 geom group 0 打射线**，地形在 group 0。  
但机器人 XML 的 `<default class="elf3">` 里视觉 mesh 也是 `group="0"`  
（碰撞胶囊是 `group="3"`），于是从躯干往下打的射线会打到自己身上：

| 射线偏移      | 命中 body             | 距离                                 |
| --------- | ------------------- | ---------------------------------- |
| `dy=+0.1` | `l_hip_y_link`      | **0.330 m** ← 与实测 `h=33.2 cm` 完全对上 |
| `dy=+0.2` | `l_shoulder_z_link` | 0.026 m                            |

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

---

## 12. 两个「绕圈刷分」的奖励漏洞（2026-10-30）

6000 轮跑完时日志一片向好（reward 20.75、ep_len 11.8 s、v_y 0.130 m/s），
但拿同一份 checkpoint 在 play 里**无扰动**跑 20 秒，实测对不上：

| 指标 | 训练日志 | play 实测 |
|---|---|---|
| v_y | 0.130 m/s | **0.026 m/s**（净位移 0.53 m）|
| 累计偏航 | 看不出来 | **367° / 20 s** |

瞬时速度中位数有 0.174 m/s，20 秒本该走 3.5 米，实际只挪了 0.53 米 —— **在绕圈**。
两个奖励项叠加造出了这个局部最优：

**① `forward_progress_y` 只收正向**

```python
r = torch.minimum(torch.clamp(v, min=0.0), vmax)     # 旧
r = torch.where(v >= 0.0, v, 0.3 * v)                # 新：倒退扣 30%
```

绕圈时 `dy` 时正时负，负的被 `min=0.0` 抹掉、正的照收，**净位移为 0 也能拿满分**。
全额负分又会让「原地站住」变成最优，所以打折而不是全罚。

**② `yaw_drift` 用 `sin(yaw)²`**

`sin(yaw)²` 的周期是 π，**原地绕一整圈回到原点就「无罪」**。实测转了 367°，
这条只报 0.018（等价净偏航 7.5°），等于完全没罚。改成 `(Δyaw/dt)²`，
帧间差先用 `atan2(sin Δ, cos Δ)` wrap 到 [-π, π]，避免 ±π 处跳变。

### 12.1 怎么早发现这类问题

只看训练日志发现不了 —— 那两项都「自洽地虚高」。必须做一次无扰动 play，
并看**净位移**：

```bash
.venv/bin/python scripts/diagnose_policy.py --task Elf3-Crab-Residual-Play \
  --checkpoint <ckpt> --num-envs 4 --steps 1000 --terrain-level 6
```

| 该看 | 别只看 |
|---|---|
| `净 Δy 均值` / 时长 | `v_y (存活期) mean` |
| `yaw 累计漂移` | `yaw_drift` 的奖励数值 |
| 两者对比 | `Mean reward` |

判据：20 秒的 `净 Δy` 应该接近 `cmd_y × 20`（命令 0.195 时约 3.9 m）。
**净位移明显偏小、而瞬时 v_y 中位数正常 → 就是在绕圈。**

> 顺带一个诊断脚本的坑：`--terrain-level` 强制等级后打印的 `基座 z` 会是负值
> （`env_origins` 的 z 与初始位姿没对齐），别拿它判断「有没有爬上台阶」。
