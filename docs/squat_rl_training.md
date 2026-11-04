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
| 回合长度 | 参考时长 + 0.5 s | 正常在参考播完时以 `traj_finished` 结束（记为 time_out，不算摔） |

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
| `alive` | +0.5 | 抵消「早摔早结束少挨罚」 |
| `joint_pos_limits` | −1.0 | 关节限位 |
| `action_rate_l2` | −0.01 | 抖动 |
| `is_terminated` | −100 | 摔一次约 −2 分（单回合满分约 60） |
| `squat_probe` | 1e-9 | 只写 `Metrics/traj_*`，不参与学习 |

终止：`time_out` / `fell_over`(55°) / `torso_contact`(20 N) / `base_too_low`
（down 0.28 m、up 0.30 m）/ `drifted`(水平 1.2 m) / `traj_finished`。

> ⚠️ 观测与动作语义都和 `Elf3-Crab-*` 不同，**不能加载爬楼的 checkpoint**。

---

## 2. 服务器上执行（VNC 里逐条跑）

以下假设仓库在 `~/elf3`、venv 在 `~/elf3/.venv`（与 `train/README.md` 一致）。

**懒人版**（一条命令走完：自检注册 → zero 冒烟 → 5 轮小规模 → 正式训练）：

```bash
cd ~/elf3 && git pull
bash scripts/run_squat_train.sh smoke     # 先冒烟，别急着开 1024
bash scripts/run_squat_train.sh all       # 冒烟通过后顺序训 down、up
# 可调：NUM_ENVS=768 ITERS=6000 bash scripts/run_squat_train.sh all
```

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
  且 traceback 看不到。
- 曲线与 checkpoint 落在 `logs/rsl_rl/elf3_squat/<时间戳>/`，
  看曲线：`tensorboard --logdir logs/rsl_rl/elf3_squat --port 6006`。

### 2.4 看训练有没有真的在学

三条必看指标（都在日志里，`Metrics/traj_*` 由 `squat_probe` 写）：

```bash
# 进度：应该从 0 一路涨到 1（= 能把整条轨迹播完）
grep -o "Metrics/traj_progress: [0-9.]*" ~/train_squat_down.log | tail -5

# 跟踪误差（弧度，腿）：< 0.25 rad 算跟上，> 0.6 基本没在跟
grep -o "Metrics/traj_err_rms_leg: [0-9.]*" ~/train_squat_down.log | tail -5

# 回合长度 / 完成率
grep -o "Metrics/traj_done_frac: [0-9.]*" ~/train_squat_down.log | tail -5
```

判据（前 300 轮之内应该出现）：

| 指标 | 好 | 坏 |
|---|---|---|
| `traj_progress` 均值 | → 0.9 以上 | 停在 0.3~0.4（中途塌了） |
| `traj_err_rms_leg` | < 0.25 rad | 恒 0.6 以上（没在跟，只在保命） |
| `traj_done_frac` | > 0.7 | < 0.2 |

### 2.5 回放（训练完）

```bash
MUJOCO_GL=egl .venv/bin/play Elf3-Squat-Down-Play \
  --checkpoint-file logs/rsl_rl/elf3_squat/<run>/model_4000.pt
```

启动后终端会打印 viser 地址，本地端口转发后打开即可（见 `train/README.md` §10.2）。

---

## 3. 常见报错 → 处理

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

## 4. 文件清单

| 文件 | 作用 |
|---|---|
| `train/assets/build_squat_ref.py` | 把 down/up 轨迹整理成 `squat_ref_v1.npz`（29 关节 + 高度 + 朝向） |
| `train/assets/ref_tables/squat_ref_v1.npz` | 成品参考表（14 KB，入库） |
| `train/mjlab_tasks/crab/traj_ref.py` | 参考表加载 + 进程内缓存 |
| `train/mjlab_tasks/crab/traj_action.py` | `TrajTrackAction`：时间轴 + 残差动作项 |
| `train/mjlab_tasks/crab/squat_mdp.py` | 观测/奖励/终止项 |
| `train/mjlab_tasks/crab/squat_env_cfg.py` | 两个任务的环境配置（平地、无速度指令） |
| `train/mjlab_tasks/crab/rl_cfg.py` | `elf3_squat_ppo_runner_cfg()` |
| `train/mjlab_tasks/crab/__init__.py` | 注册 4 个任务 |
| `train/install_task.sh` | 自检清单加到 8 个任务 |
| `scripts/run_squat_train.sh` | 冒烟 + 训练一条命令（`smoke` / `all` / `down` / `up`） |

参考轨迹的来源见 `docs/squat_extreme.md`（`launch/split_traj.py` 从循环轨迹切出
down / up 两段，接缝帧一致）。
