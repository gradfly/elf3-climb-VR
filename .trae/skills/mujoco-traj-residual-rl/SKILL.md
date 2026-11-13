---
name: mujoco-traj-residual-rl
description: 把 MuJoCo 仿真动作 npz 变成可训练的残差跟踪任务。用于生成/拆分参考动作 npz、导出 GIF、烘焙参考表、注册残差 RL 任务与训练脚本。不要用于纯策略调参或与 elf3 仓库无关的 MuJoCo 项目。
---

# MuJoCo 动作 → 残差 RL 训练链路

本仓库（elf3-climb-VR）里「给新动作接上 RL 训练」固定是四段链路。缺任何一段都会
在训练时报观测维度/关节错位错误，所以按顺序走，每段都要看输出的自检行。

```
仿真动作(38维 qpos)  →  动作 npz + GIF  →  训练参考表(29关节+高度+朝向+xy)  →  残差跟踪任务  →  PPO
```

## 0. 先确认数据约定

- 导出 npz 的 `qpos` 是 **38 维**：`0:3 xyz | 3:7 quat(wxyz) | 7:36 29 个可控关节 | 36:38 被动`。
- 训练侧只用 `7:36`；基座 `0:7` 在物理仿真里由积分器生成，**不能直接赋值**。
- 关节顺序必须等于 `train/mjlab_tasks/crab/robot_cfg.py` 的 `JOINT_NAMES`，顺序错了不会报错，只会训不出来。
- 控制频率 50 Hz（`dt=0.02`），与 `TrajTrackAction` 的 `advance=1` 对齐。

## 1. 生成/拆分动作 npz + GIF

已有素材在 `sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation/`：
`prone_standup_traj.npz`（站立→趴）、`prone_gait_traj.npz`（趴姿爬行，4 循环）。

```bash
python scripts/build_prone_parts.py --cycles 2 --freeze-s 1.0
python scripts/build_prone_parts.py --preset fast --cycles 2 --freeze-s 1.0
python scripts/build_prone_parts.py --no-gif        # 只出 npz
```

产出 `logs/export/prone_{stand2crawl,crawl,crawl2stand}{,_fast}.npz|.gif`。

写新动作脚本时守这几条：

1. **每个动作末尾冻结 N 帧**（重复末帧），默认 1 s = 50 帧。冻结帧直接进参考表，训练时
   「做完再站住 1 s」是参考的一部分，不需要额外的 hold 逻辑。
2. **基座 x 必须跨段连续**。原地动作不能套用另一段的 x：A 段末帧 x 应等于 B 段首帧 x；
   做法是给整段 `qpos[:, 0] += dx`。x 不连续 = 物理上瞬移，策略永远学不会。
3. **段间过渡只插值关节维**。基座位置/四元数保持前一段末帧，否则会把机器人从前进位置
   拉回起点（线性插值 38 维全量是最常见的错）。
4. `contact_flags` 原样拷贝即可，但**冻结帧/过渡段的 flags 不可信**，不要拿它当观测。
5. 必须有自查输出：每段帧数、z 起止、x 起止、段间 x 差（应为 0.000000）。

## 2. 烘焙训练参考表

```bash
python train/assets/build_prone_ref.py                 # → prone_ref_v1.npz
python train/assets/build_prone_ref.py --preset fast   # → prone_ref_v1_fast.npz
```

参考表每个 key 存 `q[29] / z / xy / up / pitch / time / dt / names / meta`。

- 必须存 **`xy`**：爬行这类会前进的动作，没有水平参考就只能用「钉在原点」的奖励，
  等于惩罚正确行为。
- 烘焙时**校验关节名**，不一致直接退出，别让错误流到训练阶段。
- 蹲起表没有 `xy`，加载器要允许缺键（`TrajRef.xy` 可选，None 表示原地动作）。

## 3. 注册残差跟踪任务

照 `train/mjlab_tasks/crab/squat_env_cfg.py` 的模式写
新的 `*_env_cfg.py`，并在 `train/mjlab_tasks/crab/__init__.py` 用 try/except 注册（缺参考表时
只打印 `[xxx][ERR]`，不拖垮整个包导入）。

**不能照搬 squat 的项**（趴姿会每帧误杀/误罚）：

| squat 项 | 趴姿为什么错 | 改成 |
|---|---|---|
| `base_too_low` z<0.28 | 正常趴姿 z≈0.10 | `traj_z_far` 偏离参考 >0.35 m |
| `bad_orientation` 倾角>55° | 趴姿躯干本来就 90° | 用 `up_track` 跟参考朝向，不做角度终止 |
| `base_anchor` 钉在原点 | 爬行要前进 0.5 m | `xy_track`（对参考 xy 的 exp 核，σ≈0.10 m） |

其它约定：

- 地形改平地、`max_init_terrain_level=0`；不用射线/足底/躯干接触就把 `scene.sensors` 清空。
- 出生姿态设成参考第 0 帧（`init_state.pos` 用参考 `xy/z`，`joint_pos` 用参考 `q[0]`）。
- 观测里必须有 `traj_xy_err`，否则策略只能盲积速度，长回合必漂。
- 动作限幅按关节组给：腿 0.40 / 腰 0.20 / 手臂 0.45（撑起身体靠手臂，给太小起不来）。
- 奖励骨架：`traj_track 1.5(σ0.35)`、`leg_track 2.0(σ0.30)`、`height_track 1.5(σ0.06)`、
  `up_track 1.5(σ0.25)`、`xy_track 1.0(σ0.10)`、`alive 0.5`、`is_terminated -100`、
  探针权重 `1e-9`（只写 `Metrics/*`）。
- 回合长度 = 参考时长（已含冻结尾）+ 0.5 s；`traj_finished` 用 `time_out=True`。
- PPO：回合 250~300 步时 `num_steps_per_env=48`，`max_iterations` 给 6000 起。

## 4. 训练脚本

照 `scripts/run_prone_train.sh` 写，必须包含：

1. **训练前自检任务注册**（列 `list_tasks()` 里新任务的 ID，缺了就退出并指向参考表路径）。
2. **`export ELF3_<TASK>_PRESET` 必须在任何 python 之前**：任务 cfg 在 import 时构建，
   之后改环境变量无效。
3. `MUJOCO_GL=egl`、`PYTHONUNBUFFERED=1`。
4. `stdout | tee 日志`，**不要接 `tail`**（tail 等 EOF，训练全程无输出）。
5. 模式齐全：`smoke`（zero 策略 + 5 轮 × 256 env）、单任务、`all`、`play <which>`（自动找
   最新 `model_*.pt`）。
6. 每个实验独立目录 `logs/rsl_rl/elf3_prone_<which>`，三段 checkpoint 互不通用。

```bash
bash scripts/run_prone_train.sh smoke     # 必须先跑，确认自检行出现
bash scripts/run_prone_train.sh all
bash scripts/run_prone_train.sh play crawl
```

## 5. 验收清单

训练前（smoke 输出里必须能看到）：

- `[prone] <which>: 出生 (x=..., y=..., z=...) m`，且 z 与参考起始高度一致；
- `[prone] <which>: 参考 A s（含冻结尾） -> 回合 B s`，`B ≈ A + 0.5`；
- 参考表三段的 `xy` 起止与动作 npz 一致（先用 `load_traj` 单独读一遍最快）。

训练中：

- `Metrics/prone_err_rms_leg` 单调下降 → 关节相位在跟上；
- `Metrics/prone_xy_err` 下降但 `err_rms_all` 上升 → `xy_track` 权重调低（先 1.0 → 0.5）；
- `Metrics/prone_z` 在趴姿段应贴近 0.10，站立段贴近 1.03，中途掉到 0 → 策略放弃了撑起。

## 常见错误

- 关节顺序/数量不对：训练不报错但永远学不会 → 烘焙脚本里加名字断言。
- 用了 `--preset fast` 生成动作，却用 normal 参考表训练（或反过来）→ 帧数/速度不匹配。
- 段间 x 不连续、或过渡段全量插值把 x 拉回原点 → gif 里看起来没问题，物理里瞬移。
- 忘了 `--freeze-s`：参考提前结束，策略学会「动作没做完就倒下」。
