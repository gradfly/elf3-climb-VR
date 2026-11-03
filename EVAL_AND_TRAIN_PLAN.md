# ELF3 螃蟹步爬楼 · 模型评测方案与继续训练优化方案

- 目标模型：`model_7047.pt`（iter=7047）
- 静态体检已在本机完成，结果见文末；完整数值评测需在有 mjlab + GPU 的训练服务器上跑。
- 配套脚本：`scripts/eval_policy.py`（本文档所有命令都针对它）。

---

## 一、评测方案

### 1.1 为什么需要标准化评测

仓库里已有 [diagnose_policy.py](file:///e:/mujoco/elf3-climb-VR/scripts/diagnose_policy.py)，它面向"现场排障"（打印一次运行里每个分量的原始量）。判断"这个模型好还是坏、比上一版强还是弱"，需要一套**固定协议 + 可比较的分数卡**——这就是 `eval_policy.py`。

### 1.2 评测协议（协议内所有参数都可用 CLI 覆盖）

| 项 | 取值 | 说明 |
|---|---|---|
| 任务 | Elf3-Crab-Residual-Play | 无观测噪声、无随机地形重排 |
| 地形 | 逐档固定：默认 0/2/4/6/8 | 移除 play 的 randomize_terrain，直接写 `terrain_levels` 并同步 `env_origins` |
| 指令 | 固定 (0, 0.18, 0)，不重采样 | 逐帧写死 `vel_command_b` |
| 回合 | 20 s；每 env 默认 2 个完整回合 | 覆盖 play 的 1e9 秒"永不超时" |
| 扰动 | 默认无；`--push` 可选抗扰评测 | 与训练配置区分开 |
| 并行 | 默认 32 env | 可调 |

### 1.3 分数卡（每档地形一张）

| 指标 | 含义 | 硬门槛 |
|---|---|---|
| fall_rate | 摔倒率 | ≤ 15% |
| reach_rate | 未摔且净 Δy ≥ 0.7·cmd·T | 参考 |
| climb_rate | 未摔且 Δy ≥ 2.0 m 且 Δz ≥ 0.10 m | L≥4 时 ≥ 80% |
| track_ratio | 净 Δy / (cmd × 回合时长) | ≥ 0.75 |
| yaw_rate | 存活期 \|ω_z\| 均值 | ≤ 0.15 rad/s |
| vlon_bias | 机身系纵向速度（螃蟹步应≈0） | ≤ 0.05 m/s |
| sat_frac | 残差动作顶上限幅的比例 | ≤ 60% |
| ref_track / ref_rms | 12 腿关节对参考步态的跟踪（exp 核，σ=0.35） | 诊断 |
| support_proxy | 根节点投影落在支撑基底附近的帧占比 | 诊断（近似，见脚本头注释） |
| clearance / slip | 摆动脚离地高度、着地脚滑动 | 诊断 |

> `support_proxy` 用两脚 site 的 xy 做代理（双脚半径 0.18 m / 单脚 0.13 m），是近似量，不参与 PASS/FAIL，只用于区分"迈不开步"和"站不稳"。

### 1.4 运行命令

```bash
# 0) 静态体检（不需要 mjlab/GPU，本机已跑通）
python scripts/eval_policy.py --checkpoint model_7047.pt --audit-only

# 1) 验收（服务器，仓库根目录）
MUJOCO_GL=egl .venv/bin/python scripts/eval_policy.py \
    --checkpoint logs/rsl_rl/elf3_crab/<run>/model_7047.pt \
    --num-envs 32 --levels 0,2,4,6,8 --cmds 0.18

# 2) 新旧对比（同一套初始条件，先旧后新）
MUJOCO_GL=egl .venv/bin/python scripts/eval_policy.py \
    --checkpoint old.pt new.pt --num-envs 32 --levels 0,2,4,6,8

# 3) 抗扰评测（可选）
MUJOCO_GL=egl .venv/bin/python scripts/eval_policy.py \
    --checkpoint ckpt.pt --push --push-scale 0.5
```

JSON 默认写到 `logs/eval/<checkpoint名>.json`，含每个条件汇总与每条 episode 原始记录，可直接做趋势/回归库。

### 1.5 注意

- 若回放时 `ref_alpha < 0.95`（终端会打 `!!` 告警），必须用训练当时的 Metrics/ref_alpha 通过 `--alpha` 对齐，否则口径与成品不符。
- 只替换地形等级会留下旧的 `env_origins` 出生点，脚本已同时写 `terrain_levels` 和 `env_origins`，所以档位是"真的钉住"。

---

## 二、当前模型静态体检结果（本机已验证）

```
[audit] model_7047.pt
[audit]   iter=7047  obs=393  act=29  hidden=[512, 256, 128]  size=8901 KB
[audit]   obs_normalizer: count=173285376.0  |mean|=0.214  var=[0.000, 2.777]
```

- 结构与任务完全匹配（393 维观测 = 残差任务，29 维动作），可正常加载。
- 归一化样本 1.7 亿，均值/方差已充分收敛，无异常。
- 缺陷：**checkpoint 里没有保存训练曲线**（无法从文件本身判断是否收敛/过拟合），必须用上面的数值评测补上这一环。

---

## 三、继续训练优化方案

### 3.1 先复盘：repo 已记录的已知问题（决定"继续训练"前必须看）

读 [env_cfg_residual.py](file:///e:/mujoco/elf3-climb-VR/train/mjlab_tasks/crab/env_cfg_residual.py#L456-L461) 的注释，目前有**已确诊**的失败模式：

1. **推挤过重（最重要的坑）**：push 每 3~6 s 一次 ±0.2 m/s，而步速只有 0.17 m/s，相当于每次都被"速度翻倍"级撞击。从 7047 续训 413 轮的实测：训练 ep_len 48 → 512 步（+968%），但无扰动 eval 反而退步——16 env 里摔倒次数 3 → 20、跟踪比 1.70 → 0.81、前进速度 0.305 → 0.138 m/s。策略被优化成"抗推专家"而不是"走得好"。
2. **α 课程语义**：续训 checkpoint 时如果不设环境变量，ramp 从 0 重来会把参考打回站姿；`ELF3_ALPHA=<固定值>` 又永远到不了 1.0（实测 4049 轮全锁 0.70，爬 22 cm 台阶不够用）。正确做法是 `ELF3_ALPHA_START=<训练当时的 ref_alpha>` 继续线性升到 1.0。
3. **level 0 不是平地**：四档地形的低端已对齐到 0，但 25% env 开局仍站在 8 cm 台阶上、25% 在 5 cm 上（见 [terrain_cfg.py](file:///e:/mujoco/elf3-climb-VR/train/mjlab_tasks/crab/terrain_cfg.py#L43-L59) 注释）——"所有 env 在 level 0"和"rise_est=0.14"可以同时成立。
4. 残差限幅 0.40 → 0.55 后仍可能饱和（旧注释记录 62%），续训需监控 sat_frac。

### 3.2 三步走（每次只动一件事，先回归再叠加）

**第 0 步（必做，先拿基线）**：用第一节命令对 `model_7047.pt` 跑一次数值评测，得到分数卡。它同时是"继续训练"的**验收基线**——任何新 run 的结论都对着这张卡说。

**第 1 步：把训练分布拉回测试分布**
```bash
ELF3_ALPHA=1.0 ELF3_PUSH_SCALE=0.5 MUJOCO_GL=egl WANDB_MODE=disabled \
  .venv/bin/train Elf3-Crab-Residual --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run <7047 所在的 run> \
  --agent.load-checkpoint model_7047.pt \
  --agent.max-iterations 800 2>&1 | tee ~/crab_continue_s1.log
```
- 从 7047 续训，把 push 压到 0.5，先让策略回到"会走"水平。
- `--agent.max-iterations` 是**本次再跑多少轮**，不是累计轮数；新 run 的 checkpoint 从 `model_1.pt` 重新编号。
- 验证：用第 0 步同协议评测，`track_ratio` 回到 0.75 以上、`fall_rate` ≤ 15%、`sat_frac` 明显下降。

**第 2 步：确认 α，再一档一档加压**
```bash
# 2a. 先查 tensorboard 的 Metrics/ref_alpha（7047 的 α 到底是多少）：
#     α=1.0      -> 用 ELF3_ALPHA=1.0 固定住（7047 > 默认 ramp 1500 轮，通常是这档）
#     α=0.7~0.99 -> 用 ELF3_ALPHA_START=0.70 继续线性升到 1.0
ELF3_ALPHA=1.0 ELF3_PUSH_SCALE=0.75 ...后续同上...
# 2b. push 再回到 1.0；每一档都先跑无扰动评测，通过后再叠加
```
- 验证：`Metrics/ref_alpha` 与 `Metrics/rise_est` 接近 1.0；`climb_rate`（L≥4）≥ 80%；`track_ratio` 不随 push 加大而跌破 0.75。

**第 3 步：只解决"站上台阶后的末端稳定性"**
- 平台末端（平台宽度 0.52~0.80 m）站稳后再迈步是关键瓶颈。若 2b 后 climb_rate 仍不足：
  - 用 `ELF3_REWARD_OFF` 做单项 A/B（repo 已支持），先关 `track_angular_velocity`（只鼓励转圈、不鼓励前进）或 `foot_slip`，找出哪个项在"搬石头砸自己脚"。
  - 给 `upright`/`lin_vel_z` 权重微调前，先确认 2a/2b 已到位，否则容易把"没站稳"误诊成"站不稳"。

### 3.3 配置开关（续训常用）

| 环境变量 | 作用 |
|---|---|
| `ELF3_PUSH_SCALE` | push 力度倍率（0.5 / 0.75 / 1.0 阶梯加回） |
| `ELF3_ALPHA_START` | α 课程起点（= 训练当时的 ref_alpha，续训必设） |
| `ELF3_RAMP_ITERS` | α 课程总轮数（默认 1500） |
| `ELF3_REWARD_OFF` | 单项奖励 A/B（诊断用） |

### 3.4 风险与验证口径

- **不要用训练曲线当验收**：7047 续训 413 轮"训练一路向好"但 eval 全面退步，训练曲线对分布偏移不敏感。一切结论以第一节的无扰动评测为准，push 版只作为鲁棒性补充。
- 每个 run 至少保留：分数卡 JSON（`logs/eval/`）+ 对应的评测命令（写进 run 备注），保证"哪个数字是怎么来的"可追溯。
