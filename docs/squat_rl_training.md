# 蹲下 / 起立 · 跟踪运动策略训练（mjlab）

给 `squat_extreme_down_traj.npz`（站 → 极限蹲，2.46 s）和
`squat_extreme_up_traj.npz`（极限蹲 → 站，2.02 s）各训一个跟踪策略。
两个任务共用一套代码，只差参考轨迹。

```
Elf3-Squat-Down        训练任务（下蹲）
Elf3-Squat-Down-Play   回放/评测
Elf3-Squat-Up          训练任务（起立）
Elf3-Squat-Up-Play     回放/评测
```

---

## 1. 思路：时间轴 + 残差，不是行为克隆

参考轨迹是开环 IK 解出来的，物理上并不完全可行（PD 跟踪误差、足底接触抖动都在
里面）。所以动作不是 `q_target = q_ref`，而是

```
q_target(t) = q_ref(t) + scale · clip(Δq, ±0.35 rad 腿 / ±0.15 腰 / ±0.25 臂)
```

策略只学「怎么在跟着参考动的同时别摔」。这也正是脚本版蹲姿能站住的前提：
极限蹲的静态裕度是 9.65 cm，但开环 PD 追下来躯干会掉 1.2 cm、保持段还有 1 mm 抖动，
残差就是吃掉这些差额的。

| | 值 | 说明 |
|---|---|---|
| 控制频率 | 50 Hz | `timestep 0.002 × decimation 10`，与轨迹 `dt=0.02` 一致 |
| 参考推进 | 1 帧 / env step | `TrajTrackActionCfg.advance`，所以不会「演快/演慢」 |
| 观测（actor） | 128 维 | 见下表 |
| 动作 | 29 维残差 | 与 crab 任务同一套 `ACTION_SCALE_MAP` |
| 回合长度 | 参考时长 + **保持 5 s** + 0.5 s | 参考播完后帧号 clamp 在末帧，必须原地站住 5 s；由 `time_out` 结束 |

actor 观测：`base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) +
joint_pos(29) + joint_vel(29) + actions(29) + traj_joint_err(29) +
traj_height_err(1) + traj_progress_sincos(2)`。

奖励（每步 × dt = 0.02）：

| 项 | 权重 | 作用 |
|---|---|---|
| `traj_track` | +1.5 | 29 关节 exp 核，σ 0.35 rad |
| `leg_track` | +2.0 | 只算 12 腿，σ 0.30 rad（站不站得住由腿决定） |
| `height_track` | +2.0 | 基座高度 vs 参考，σ 5 cm（**蹲多深 = 任务目标本身**） |
| `up_track` | +1.0 | 躯干朝向 vs 参考，σ 0.25（极限蹲是 18.5° 前倾，**不能用 upright**） |
| `symmetry` | +1.0 | **左右镜像差额** exp 核，σ 0.08 rad；参考严格对称，这条杀掉不对称解 |
| `alive` | +0.5 | 抵消「早摔早结束少挨罚」 |
| `base_anchor` | −40 | 基座水平位移平方（参考全程 x=y=0）；偏 5 cm ≈ −0.1/步 |
| `yaw_anchor` | −5 | 基座偏航角平方；偏 10° ≈ −0.15/步 |
| `arm_leg_contact` | −0.6 | 手臂-腿接触（传感器 `arm_leg_contact`）；`ARM_GUARD=0` 可关 |
| `joint_pos_limits` | −1.0 | 关节限位 |
| `action_rate_l2` | −0.02 | 抖动 |
| `is_terminated` | −100 | 摔一次约 −2 分（单回合满分约 60） |
| `squat_probe` | 1e-9 | 只写 `Metrics/*`，不参与学习 |

终止：`time_out` / `fell_over`(55°) / `base_too_low`（down 0.28 m、up 0.30 m）/
`drifted`(水平 1.2 m) / `torso_contact`(20 N)，以及 `traj_finished`——**只在
`HOLD=0` 时保留**；有保持段时它必须去掉，否则回合在参考播完就结束，保持段白加。

> ⚠️ 观测与动作语义都和 `Elf3-Crab-*` 不同，**不能加载爬楼的 checkpoint**。

### 1.1 v2：不对称 / 手臂压腿 / 站久了会倒——三个病灶与对策

第一版 `Elf3-Squat-Down` 训 4000 轮（1.31 亿步）后回放看到的三个问题，根因不是一个：

| 现象 | 根因 | 对策 |
|---|---|---|
| 蹲下后身体不对称（骨盆扭转、单臂外撇） | 跟踪奖励按 29 维**平均**，左右互换的等价解拿一样的分；不对称还能借力 | 新增 `symmetry`（左右镜像差额 exp 核，σ 0.08 rad） |
| 手臂压在大腿上 | 接触在奖励里是**免费**的（除了躯干触地没有任何接触惩罚），策略就把手臂搁腿上省力 | 新增手臂-腿接触传感器 + `arm_leg_contact` 惩罚；臂残差限幅 0.25 → **0.12 rad** |
| 保持久了慢慢歪、最后倒 | **回合只有 2.46 s**，参考播完就结束；策略从没学过长时间站住，慢漂在奖励里还是中性的 | 回合加 **5 s 保持段**（参考 clamp 在末帧）；新增 `base_anchor` / `yaw_anchor` 压漂移 |

三条改动互相独立，缺一条都会复发：只加保持段 → 策略用不对称姿势硬撑；只加对称 →
没有长时间保持的经验，照样会漂。所以这一版要**从头训**（别 resume 旧的
`model_4000.pt`，它已经收敛到不对称解上了）。

新指标（都在 `Metrics/*`）：

| 指标 | 期望 | 含义 |
|---|---|---|
| `sym_err_rad` | < 0.05 | 左右关节差额 RMS；> 0.1 就是还在不对称 |
| `base_xy` | < 0.03 m | 基座水平漂移；涨到 0.1 以上说明在慢慢挪 |
| `base_yaw_deg` | < 3° | 身体扭转 |
| `arm_leg_touch` | ≈ 0 | 手臂-腿接触比例；恒为 0 且画面还压腿 → 传感器没生效（见下） |

`arm_leg_touch` 恒为 0 有两种可能：真的没碰，或者**传感器读不到**。后者会在启动时
打印 `[squat][warn] 读不到接触传感器 ...`；如果连环境都建不起来（本版 mjlab 不支持
自碰撞接触检测），用 `ARM_GUARD=0 bash scripts/run_squat_train.sh down` 关掉这一项再训，
其余改动照旧生效。

---

## 2. 服务器上执行（VNC 里逐条跑）

以下假设仓库在 `~/elf3`、venv 在 `~/elf3/.venv`（与 `train/README.md` 一致）。

**懒人版**（一条命令走完：自检注册 → zero 冒烟 → 5 轮小规模 → 正式训练）：

```bash
cd ~/elf3 && git pull
bash scripts/run_squat_train.sh smoke     # 先冒烟，别急着开 1024
bash scripts/run_squat_train.sh all       # 冒烟通过后顺序训 down、up
# 可调：NUM_ENVS=768 ITERS=8000 HOLD=8 bash scripts/run_squat_train.sh all
#   HOLD   = 参考播完后还要站住几秒（默认 5，0 = 回到「播完即结束」的旧行为）
#   ARM_GUARD = 0 关掉手臂-腿接触检测（本版 mjlab 不支持自碰撞传感器时用）
```

v2 的改动（保持段 / 对称 / 接触）与上一版**不兼容**，是**从头训**，不要 resume
旧的 `model_4000.pt`：它已经收敛到不对称解上，续训只会接着走。

下面是把脚本拆开的手工版，出问题时逐条排查用。

### 2.1 拉代码 + 装任务

```bash
cd ~/elf3
git pull
source .venv/bin/activate

# 参考表（仓库里已有 train/assets/ref_tables/squat_ref_v1.npz，可跳过；
# 想从脚本轨迹重新烘焙就跑这一条，需要 data/mujoco_simulation 下的两份 down/up npz）
python train/assets/build_squat_ref.py

# 装任务（软链接，pull 完自动生效；这里重跑是为了自检）
bash train/install_task.sh
```

期望结尾：

```
已注册: [... 'Elf3-Squat-Down', 'Elf3-Squat-Down-Play', 'Elf3-Squat-Up', 'Elf3-Squat-Up-Play']
8 个任务齐全 ✔
geom group 分布 {...}，group 0 为空 ✔
```

### 2.2 冒烟（先别开 1024）

```bash
cd ~/elf3
# ① 零动作：参考轨迹单独能不能撑住（残差=0，纯开环）
MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/play Elf3-Squat-Down-Play --agent zero

# ② 小规模训练 5 轮：验奖励/终止/维度
MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/train Elf3-Squat-Down \
  --env.scene.num-envs 256 --agent.max-iterations 5
```

`play --agent zero` 是**关键一步**：它把残差乘 0，只让 PD 追参考轨迹。
如果连它都在 1 s 内塌掉，说明是参考 / 初始姿态 / 增益的问题，不是策略的问题。

### 2.3 正式训练

两卡/单卡都先看一眼显存：

```bash
nvidia-smi
```

单卡顺序跑（推荐，便于对比日志）：

```bash
cd ~/elf3
MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/train Elf3-Squat-Down \
  --env.scene.num-envs 1024 --agent.max-iterations 4000 \
  2>&1 | tee ~/train_squat_down.log

MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/train Elf3-Squat-Up \
  --env.scene.num-envs 1024 --agent.max-iterations 4000 \
  2>&1 | tee ~/train_squat_up.log
```

显存够（≥ 2×8 GB 余量）就两个终端并行，各降到 768 env：

```bash
MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/train Elf3-Squat-Up \
  --env.scene.num-envs 768 --agent.max-iterations 4000 2>&1 | tee ~/train_squat_up.log
```

- 日志别放 `/tmp`（重启会清空），放家目录。
- 必须带 `PYTHONUNBUFFERED=1` + `tee`，否则管道模式下日志滞后几分钟，
  且 traceback 看不到。**`tee` 后面不要再接 `tail -N`**，见 §2.3.1。
- 曲线与 checkpoint 落在 `logs/rsl_rl/elf3_squat_down/` 和 `logs/rsl_rl/elf3_squat_up/`
  （两边实验名分开，别混），
  看曲线：`tensorboard --logdir logs/rsl_rl --port 6006`。

### 2.3.1 训练中「每轮指标」要实时可见

**症状**：跑 `bash scripts/run_squat_train.sh down` 时屏幕长时间不动，
几千轮训练结束后才「唰」地冒出几十行指标，中间完全看不到进度。

**根因**：旧版脚本把训练输出接成了

```bash
.venv/bin/train ... 2>&1 | tee ~/train_squat_down.log | tail -40   # ❌
```

`tail -N` 必须等 stdin 到 EOF 才能确定「最后 N 行」是哪几行，所以训练不结束
它一行都不吐。**`tee` 那一步其实一直在实时写日志**，只是终端被 tail 挡住了。
同理，`grep` / `awk` 接在管道里也会各自带一层块缓冲。

**修法**（2026-11-09）：管道末端只留 `tee`，想要末尾摘要就训完再 `tail` 日志文件：

```bash
.venv/bin/train ... 2>&1 | tee ~/train_squat_down.log               # ✔
```

脚本里 smoke 与正式训练统一走 `run_logged()`，两种模式都实时输出。

**正在跑的那次不用等改代码**，另开一个终端直接看日志文件即可：

```bash
tail -F ~/train_squat_down.log
# 只看每轮关键指标；grep 必须带 --line-buffered，否则又多一层缓冲
tail -F ~/train_squat_down.log | grep --line-buffered -E \
  'Learning iteration|Mean reward|Mean episode length|Metrics/'
```

**终端还是不动时，按这个顺序排查**：

| 检查 | 命令 | 期望 |
|---|---|---|
| 日志文件在不在长 | 隔几秒看两次 `ls -l --time-style=+%T ~/train_squat_down.log` | mtime 持续更新 |
| 日志里有没有逐轮行 | `grep -c -E 'iteration\|Iteration\|Metrics/' ~/train_squat_down.log` | 随轮数增长 |
| Python 是否 unbuffered | 启动命令里有没有 `PYTHONUNBUFFERED=1` | 脚本已 export，裸命令要自己加 |
| 管道里有没有缓冲环节 | `tail -N`（无 `-f`）、无 `--line-buffered` 的 `grep`、无 `fflush` 的 `awk` | 都不该有 |
| 是不是 `\r` 覆盖式进度条 | `wc -l` 很小但文件很大 | 是 → `CR_FIX=1 bash scripts/run_squat_train.sh down` |

`CR_FIX=1` 会在 `tee` 前用 `sed -u 's/\r/\n/g'` 把回车转成换行，让每轮变成
独立一行（旧版 rsl_rl / tqdm 用 `\r` 原地覆盖打印进度时需要）。`sed -u` 是
无缓冲的，不会引入新的延迟。

---

### 2.4 看训练有没有真的在学

三条必看指标（都在日志里，`Metrics/traj_*` 由 `squat_probe` 写）：

```bash
# 进度：应该从 0 一路涨到 1（= 能把整条轨迹播完）
grep -o "Metrics/traj_progress: [0-9.]*" ~/train_squat_down.log | tail -5

# 跟踪误差（弧度，腿）：< 0.25 rad 算跟上，> 0.6 基本没在跟
grep -o "Metrics/traj_err_rms_leg: [0-9.]*" ~/train_squat_down.log | tail -5

# 保持段能不能站住（v2 新增）：sym_err_rad / base_xy / base_yaw_deg / arm_leg_touch
grep -o "Metrics/sym_err_rad: [0-9.]*"  ~/train_squat_down.log | tail -5
grep -o "Metrics/base_xy: [0-9.]*"      ~/train_squat_down.log | tail -5
grep -o "Metrics/base_yaw_deg: [0-9.]*" ~/train_squat_down.log | tail -5
```

判据（前 300 轮之内应该出现）：

| 指标 | 好 | 坏 |
|---|---|---|
| `traj_progress` 均值 | → 0.9 以上 | 停在 0.3~0.4（中途塌了） |
| `traj_err_rms_leg` | < 0.25 rad | 恒 0.6 以上（没在跟，只在保命） |
| `sym_err_rad` | < 0.05 | > 0.1（还是不对称） |
| `base_xy` | < 0.03 m | > 0.1（在慢慢挪，会倒） |
| `base_yaw_deg` | < 3° | > 8°（身体在扭） |
| `arm_leg_touch` | ≈ 0 | 恒 0.3+（手臂压腿） |

> `traj_done_frac` 在 v2 里不再适合当判据：参考播完后还有 5 s 保持段，
> 这段时间它恒为 1，均值被稀释成 ~0.6，看不出好坏。

### 2.5 回放（训练完）

一条命令，自动找该任务目录下**最新的 checkpoint**：

```bash
bash scripts/run_squat_train.sh play down    # = play Elf3-Squat-Down-Play + 最新 model_*.pt
bash scripts/run_squat_train.sh play up
# 想指定某一份：
CKPT=logs/rsl_rl/elf3_squat_down/<run>/model_1000.pt \
  bash scripts/run_squat_train.sh play down
```

它等价于（down / up 的实验名是分开的，找文件不会混）：

```bash
MUJOCO_GL=egl .venv/bin/play Elf3-Squat-Down-Play \
  --checkpoint-file logs/rsl_rl/elf3_squat_down/<run>/model_4000.pt
```

启动后终端会打印 viser 地址，本地端口转发后打开即可（见 `train/README.md` §10.2）。

#### 回放里「蹲下去又站起来」是 `traj_finished` 触发的重置，不是策略在起身

参考表里 `down` 只有 124 帧（1.034 → 0.523 m，单调下降），`up` 也只有 102 帧，
两份 npz 都是单程的。所以回放里看到的「起身」只可能来自**回合结束后的 reset**：

- 训练时 `traj_finished`（参考播完）是必要终止项，回合该在 2.46 s 结束；
- 回放时它意味着每 2.46 s 重置一次，机器人瞬间回到初始站姿再蹲一遍 ——
  看起来就是「蹲下 → 站起 → 蹲下」的无限循环。

已在 `squat_env_cfg(play=True)` 里去掉这条（只影响回放配置），播完停在末帧、
一直保持极限蹲。启动日志会打印：

```
[squat][play] 已移除 traj_finished 终止：播完保持末帧，不重置
```

如果看到这行却仍在循环，检查是不是跑成了 `Elf3-Squat-Down`（训练配置）而不是
`Elf3-Squat-Down-Play`：`play` 读的是 `play_env_cfg`。兜底手段是 `--no-terminations`
（mjlab 自带，直接清空所有终止项）。

#### 判断「不好看」是参考的问题还是策略的问题

```bash
# 纯参考开环：残差乘 0，只让 PD 追轨迹。画面 = 轨迹本身能有多好看
MUJOCO_GL=egl .venv/bin/play Elf3-Squat-Down-Play --agent zero
```

对比着看：`--agent zero` 也难看 → 参考轨迹/物理参数问题（改 `squat_extreme.py`）；
只有策略难看 → 训练/奖励问题（加长训练、调权重、放松残差限幅）。
想留档或发给别人看，就把回放导出成 mp4，见 §2.6。
注意 `--video` 默认只录 200 步（4 s）/ 320×240；play 已经去掉 `traj_finished`、
回放是「下蹲 → 一直保持末帧」，4 s 连 5 s 保持段都录不完，必须显式放大。


---

### 2.6 回放导出 mp4

底层用的是 mjlab `play` 自带的 `VideoRecorder`：**不用自己写渲染循环**，
也不需要交互式窗口。两种用法——脚本一条命令（推荐），或等价的裸 `play --video`。

#### 2.6.1 一条命令

```bash
cd ~/elf3
bash scripts/run_squat_train.sh video down          # 下蹲，默认 1000 步 = 20 s
bash scripts/run_squat_train.sh video up            # 起立
bash scripts/run_squat_train.sh video hold-squat   # 保持任务默认 6000 步 = 120 s
bash scripts/run_squat_train.sh video hold-stand

# 指定 checkpoint / 改分辨率 / 改时长
CKPT=logs/rsl_rl/elf3_squat_down/<run>/model_4000.pt \
VIDEO_STEPS=2000 VIDEO_W=1920 VIDEO_H=1080 \
  bash scripts/run_squat_train.sh video down
```

脚本依次做：自动找该实验目录下最新的 `model_*.pt` → 起 `play --video` →
轮询 mp4 是否写完 → **收掉不会自己退出的 viewer** → 把固定文件名的产物复制成
`~/squat_<任务>_<run>_<迭代>.mp4`（重录不会互相覆盖）。

#### 2.6.2 等价的裸命令（具体代码）

```bash
cd ~/elf3
CKPT=logs/rsl_rl/elf3_squat_down/<run>/model_4000.pt

MUJOCO_GL=egl PYTHONUNBUFFERED=1 .venv/bin/play Elf3-Squat-Down-Play \
  --checkpoint-file "$CKPT" \
  --num-envs 1 \
  --video --video-length 1000 \
  --video-width 1280 --video-height 720
```

四个任务只换任务 ID 和 checkpoint 目录：

```text
down        Elf3-Squat-Down-Play     logs/rsl_rl/elf3_squat_down/
up          Elf3-Squat-Up-Play       logs/rsl_rl/elf3_squat_up/
hold-squat  Elf3-Squat-Hold-Play     logs/rsl_rl/elf3_squat_hold_squat/
hold-stand  Elf3-Stand-Hold-Play     logs/rsl_rl/elf3_squat_hold_stand/
```

录满 `--video-length` 步后 mp4 已经落盘，但 **viewer 不会自动退出**，此时 `Ctrl-C`
即可（脚本模式自动做了这步）。原始产物：

```text
logs/rsl_rl/elf3_squat_down/<run>/videos/play/rl-video-step-0.mp4
```

文件名是框架写死的，重录会覆盖，建议立刻改名留档：

```bash
cp logs/rsl_rl/elf3_squat_down/<run>/videos/play/rl-video-step-0.mp4 \
   ~/squat_down_model_4000.mp4
```

#### 2.6.3 参数与坑

| 参数 | 框架默认 | 这里给 | 为什么 |
|---|---|---|---|
| `--video-length` | **200 步** | 1000（down/up）/ 6000（hold） | 单位是**步数**不是秒；控制 50 Hz，步数 / 50 = 秒。200 步只有 4 s |
| `--video-width` / `--video-height` | **320×240** | 1280×720 | 默认分辨率看不清脚和手臂-腿干涉 |
| `--num-envs` | 配置里的值 | **1** | `VideoRecorder` 只录 env 0；给 1 最省心，也省显存 |
| `--checkpoint-file` | 无 | **必须给** | 不是 `--checkpoint`；`--agent zero/random` 时 `play.py` 会直接禁用录像 |
| `MUJOCO_GL=egl` | — | 必须 | 无 DISPLAY 也能离屏渲染 |

- play 配置里 `traj_finished` 已移除、`episode_length_s=1e6`，回放是「下蹲 → 一直保持」，
  所以视频长度**完全由 `--video-length` 决定**，跟回合长度无关。
- 保持任务要判断「久站会不会倒」，§3.3 要求看 60~120 s，对应 `--video-length 3000~6000`。
- 录制时别同时跑训练：play 会多占一份显存，OOM 会直接挂。
- 若报 `--video` 相关解析错（GitHub main 版 `play.py` 带 `tyro.conf.FlagConversionOff`，
  要求显式布尔值），用 `VIDEO_BOOL=1 bash scripts/run_squat_train.sh video down`，
  或把裸命令里的 `--video` 改成 `--video True`。
- 爬楼任务另有 `scripts/diagnose_policy.py --record`（`train/README.md` §10.2.1，跟拍机位、
  可自定义俯仰角），但那个脚本要传 `--terrain-level` / `ELF3_ALPHA`，是爬楼专用的，
  **squat 任务别用**。

#### 2.6.4 后处理 / 取回本地

```bash
# 核对时长、帧数、分辨率（确认「步数 / 50 = 秒」在你的版本里成立）
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate:format=duration,size \
  -of default=nw=1 ~/squat_down_*.mp4

# 转成浏览器 / 微信都能直接播的 H.264（moviepy 出的 mp4 偶尔被播放器挑）
ffmpeg -y -i in.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart out.mp4

# 下蹲 + 起立拼成一条完整蹲起循环（两段需同分辨率、同帧率）
printf "file '%s'\nfile '%s'\n" "$PWD/squat_down.mp4" "$PWD/squat_up.mp4" > /tmp/list.txt
ffmpeg -y -f concat -safe 0 -i /tmp/list.txt -c copy squat_cycle.mp4

# 0.5 倍慢放，看落脚和手臂-腿干涉
ffmpeg -y -i in.mp4 -filter:v "setpts=2*PTS" -an slow.mp4

# 下载到本机（别留在 /tmp，重启会清）
scp ubuntu22:~/squat_down_*.mp4 .
```

---

## 3. 蹲姿 / 站姿「保持」独立任务（hold）

§1.1 的 5 s 保持段只是「顺便学一下」：跟踪段占 62% 的步数，一个网络要兼顾两件事。
想让「长时间保持不动」成为第一目标，就得让**整个回合都是保持段** —— 于是拆成两个
独立任务，各自训一个网络：

| 任务 | 出生姿态 | 回合 | 参考 | 目标 |
|---|---|---|---|---|
| `Elf3-Squat-Hold` | 极限蹲（z 0.523 m，膝 149°） | 20 s | down 表**末帧冻结** | 蹲住不动 |
| `Elf3-Stand-Hold` | 站姿（z 1.034 m） | 20 s | down 表**首帧冻结** | 站住不动 |

**不新增参考表**：直接复用 `squat_ref_v1.npz` 的 down 表，把帧号冻结
（`TrajTrackActionCfg.freeze_frame=-1` / `0`）。冻结后 `traj_track / height_track /
up_track / symmetry` 全部自动退化成「保持这个位形」——奖励代码一行没改，
也就不存在两份奖励实现慢慢漂移的问题。

### 3.1 与跟踪任务的差异

| 维度 | 跟踪（down / up） | 保持（hold） | 为什么 |
|---|---|---|---|
| 回合 | 2.46 s 跟踪 + 5 s 保持 | **20 s 全程保持** | 保持占比 100% 才能成为第一目标 |
| 扰动 | 1.0~2.5 s / ±0.15 m/s | **0.8~2.0 s / ±0.25 m/s** | 不推的话参考位形本身准静态稳定，什么都不做也能满分 |
| 初始状态 | 基座 ±1 cm，关节不扰动 | 基座 ±2 cm / ±4.6°，**关节 ±0.05 rad** | 否则策略只记住「初始就是这个」，一推就崩 |
| 额外奖励 | — | `quiet`（关节速度 RMS，σ 0.8 rad/s，w 0.7）；`action_rate_l2` -0.02 → **-0.05** | 保持的敌人不是姿势错，是抖 |
| 漂移终止 | 1.2 m | **0.5 m** | 保持任务里挪半米就是错的 |
| 塌陷终止 | z < 0.28 | 蹲 **0.35** / 站 **0.90** | 站姿塌 13 cm 就是跪了 |
| PPO | 32 步 / 4000 轮 | **64 步 / 3000 轮** | 回合 1000 步，32 步只覆盖 3%，采不到完整恢复 |

### 3.2 跑

```bash
# 先冒烟（会额外跑两个 hold 任务的 5 轮）
bash scripts/run_squat_train.sh smoke

# 只训两个保持任务（先 squat 后 stand）
bash scripts/run_squat_train.sh hold

# 单独一个
bash scripts/run_squat_train.sh hold-squat
HOLD_EPISODE=40 bash scripts/run_squat_train.sh hold-stand    # 回合拉到 40 s

# 回放（自动找该实验目录下迭代数最大的 model_*.pt）
bash scripts/run_squat_train.sh play hold-squat
bash scripts/run_squat_train.sh play hold-stand

# 导出 mp4（hold 默认录 6000 步 = 120 s，够判断「久站会不会倒」）
bash scripts/run_squat_train.sh video hold-squat
bash scripts/run_squat_train.sh video hold-stand
```

四个任务四份实验目录，互不覆盖：

```
logs/rsl_rl/elf3_squat_down/          跟踪：站 -> 蹲
logs/rsl_rl/elf3_squat_up/            跟踪：蹲 -> 站
logs/rsl_rl/elf3_squat_hold_squat/    保持：极限蹲
logs/rsl_rl/elf3_squat_hold_stand/    保持：站姿
```

启动日志里要能看到这两行，否则配置没生效（`ELF3_HOLD_EPISODE` 是 import 时读的）：

```
[hold] squat（极限蹲保持）：20.0 s/回合，参考冻结在 down 第 123/123 帧，...
[traj] down: T=124 帧 ... | 冻结在第 123 帧（不推进，参考=静态位形，z=0.523 m）
```

### 3.3 判断「真的稳」还是「看着稳」

保持任务唯一的产出就是稳定性，所以判据看**趋势**，不看某一轮的值：

| 指标 | 稳 | 不稳 |
|---|---|---|
| `Metrics/base_xy` | 收敛到常数（< 0.03 m） | **单调上升** —— 慢漂 |
| `Metrics/base_yaw_deg` | 收敛（< 3°） | 单调涨 —— 身体在拧 |
| `Metrics/joint_speed` | < 0.5 rad/s | 长时间 > 1.5 —— 一直在抖 |
| `Metrics/traj_z` | 稳在参考 ±2 cm | 缓慢下沉 —— 腿在软 |
| `Mean episode length` | ≈ 1000 步（跑满 20 s） | 卡在 300~600 步 —— 5~12 s 就倒 |

`Mean episode length` 是最省事的一条：跑满 1000 步 = 整回合没摔。

回放时**别只看 10 s**。`Elf3-Squat-Hold-Play` 的回合是 1e6 s 且没有扰动，
让它自己跑 60~120 s 再判断：收敛就是稳，单向下滑就是没学透 —— 那时候该做的是
加大 `quiet` / `base_anchor` 权重或拉长 `HOLD_EPISODE` 重训，不是继续加迭代。

---

## 4. 常见报错 → 处理

| 报错 | 原因 | 处理 |
|---|---|---|
| `KeyError: 'Elf3-Squat-Down'` | 任务没注册（pull 了但没装/装的是旧副本） | `bash train/install_task.sh` |
| `KeyError: 'down/q'` / `找不到 squat_ref_v1.npz` | 参考表没进仓库 | `python train/assets/build_squat_ref.py`（需要 down/up 两份 npz） |
| `拿不到任何平地地形实现` | 这版 mjlab 既没有 `flat` 也没有 `pyramid_stairs_inv` | 把 traceback 发我，改 `squat_env_cfg._flat_terrain()` 的第三条分支 |
| `关节名对不上` 警告 | 机器人 XML 的关节顺序变了 | 重跑 `build_squat_ref.py`，再看动作项打印的 target 顺序 |
| `控制周期 ... != 轨迹 dt` 警告 | 改了 `decimation` / `timestep` | 保持 0.002×10，或把 `TrajTrackActionCfg.advance` 设成 2 |
| 训练指标全 0 / `traj_progress` 不动 | 探针没跑到 | 确认 `squat_probe` 的权重是 1e-9 而不是 0（0 会被 RewardManager 跳过） |
| `play --agent zero` 1 s 内就摔 | 参考本身站不住 | 先别训，把 log 发我：多半是初始姿态或摩擦随机化的问题 |

---

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `train/assets/build_squat_ref.py` | 把 down/up 轨迹整理成 `squat_ref_v1.npz`（29 关节 + 高度 + 朝向） |
| `train/assets/ref_tables/squat_ref_v1.npz` | 成品参考表（14 KB，入库） |
| `train/mjlab_tasks/crab/traj_ref.py` | 参考表加载 + 进程内缓存 |
| `train/mjlab_tasks/crab/traj_action.py` | `TrajTrackAction`：时间轴 + 残差动作项 |
| `train/mjlab_tasks/crab/squat_mdp.py` | 观测/奖励/终止项（含 hold 用的 `quiet` / `joint_speed`） |
| `train/mjlab_tasks/crab/squat_env_cfg.py` | 两个**跟踪**任务的环境配置（平地、无速度指令） |
| `train/mjlab_tasks/crab/squat_hold_cfg.py` | 两个**保持**任务的环境配置（参考帧冻结、20 s 回合、强扰动） |
| `train/mjlab_tasks/crab/rl_cfg.py` | `elf3_squat_ppo_runner_cfg(which)`，四个实验名分开 |
| `train/mjlab_tasks/crab/__init__.py` | 注册 8 个任务（4 任务 × 训练/回放） |
| `train/install_task.sh` | 自检清单加到 8 个任务 |
| `scripts/run_squat_train.sh` | 冒烟 + 训练 + 回放 + 导出 mp4（`smoke` / `all` / `down` / `up` / `hold` / `hold-squat` / `hold-stand` / `play` / `video`） |

参考轨迹的来源见 `docs/squat_extreme.md`（`launch/split_traj.py` 从循环轨迹切出
down / up 两段，接缝帧一致）。
