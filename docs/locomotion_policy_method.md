# Elf3 螃蟹步爬楼 —— 当前运动策略技术解析

> 本文描述**当前实际在跑**的运动策略实现（残差 RL 路线），以
> `train/mjlab_tasks/crab/*.py` 与 `sim/.../launch/crab_climb.py` 的代码为准。
> 历史设计稿见 `docs/crab_rl_training.md`（部分数字已被实现取代）；
> 开环脚本步态的网页动画链路见 `docs/crab_gait_pipeline.md`。
>
> **一句话概括**：不是从零学步态的 end-to-end RL，而是
> **「离线准静态 IK 参考步态表 + 在线地形参数估计 + PPO 残差修正」**
> 的分层残差架构——策略网络只输出对参考关节角的修正量 Δq，
> 步态的"骨架"由脚本控制器离线解出。

---

## 1. 方法总览

```
离线（一次性）                          在线（每步 50 Hz）
┌──────────────────────────┐           ┌─────────────────────────────────────┐
│ crab_climb.py            │           │ ① terrain_scan 射线高度场            │
│ 四相位准静态 IK 解算      │  枚举     │    [B, 21×13]  → 剖面/中值滤波/边缘  │
│ LOAD→SWING→LAND→SETTLE   │ ────────► │ ② RefTable.estimate()               │
└──────────────────────────┘  (rise,   │    rise_est(级高) tread_est(踏面)    │
┌──────────────────────────┐   tread)  │ ③ RefTable.select()                 │
│ build_ref_tables.py      │           │    按相位插值出 q_ref[29]            │
│ 枚举 14×4 组参数，        │           │ ④ 策略网络 π(obs) → Δq              │
│ 存 crab_ref_v1.npz       │           │    q_cmd = q_ref + scale·clip(Δq)   │
│ q[R,T,90帧,29关节]        │           │ ⑤ MuJoCo PD 跟踪 q_cmd              │
└──────────────────────────┘           └─────────────────────────────────────┘
```

| 层 | 职责 | 代码 |
|---|---|---|
| 参考步态生成 | 把"怎么迈步"离线解成参数化关节角表 | `launch/crab_climb.py` + `crab_ik.py` → `assets/build_ref_tables.py` |
| 地形感知与选表 | 从高度扫描估计 (rise, tread)，插值出当前相位参考 | `crab/ref_table.py`（RefTable） |
| 残差动作映射 | q_cmd = q_ref + scale·clip(Δq)，管理相位/α 课程/限幅 | `crab/residual_action.py`（ResidualGaitAction） |
| 环境与奖励 | 观测重组、奖励重排、课程、终止、域随机化 | `crab/env_cfg.py`（基线）+ `env_cfg_residual.py`（增量）+ `mdp_extra.py` |
| 训练算法 | PPO（rsl_rl） | `crab/rl_cfg.py` |
| 地形 | 四列倒置金字塔台阶 + 时间型课程 | `crab/terrain_cfg.py` |

任务 ID：`Elf3-Crab-Residual`（训练）/ `Elf3-Crab-Residual-Play`（回放评测）。
⚠️ 观测/动作语义与旧任务 `Elf3-Crab-Stairs` 不同，checkpoint 不通用。

---

## 2. 机器人与任务建模

| 项 | 值 |
|---|---|
| 机器人 | Elf3 人形，**29 个可控关节**（腰 3 + 腿 12 + 臂 14；head_z/head_y 被动） |
| 控制频率 | 50 Hz（`timestep=0.002 × decimation=10`），MuJoCo Warp |
| 回合长度 | 20 s（1000 步） |
| 朝向约定 | 机器人**面朝 +X**，台阶沿 **+Y** 逐级抬升——侧向横移上楼即"螃蟹步" |
| 速度指令 | `lin_vel_x=(0,0)`（禁前后）、`lin_vel_y=(0.14,0.20)`（只许 +Y 侧移）、`ang_vel_z=0`（训练不指令转向） |
| 仿真参数 | `iterations=20`、`tolerance=1e-6`（float32 下可达，避免求解器每步跑满） |

机器人 XML 由 `train/assets/make_robot_xml.py` 生成：去地形 include、去 29 个 motor
外的执行器、**视觉 mesh 挪到 geom group 2**（否则向下射线会打到自己的腿——见 §9）。

---

## 3. 参考步态层：脚本控制器 → 参数化参考表

### 3.1 四相位准静态步态（`crab_climb.py`）

每个"步"分四相位：**LOAD**（重心平移到支撑脚正上方）→ **SWING**（摆动脚抬到
max(z0,z1)+5.5cm，横移过踏面边界）→ **LAND**（落到目标踏面）→ **SETTLE**
（重心回到双脚中点）。每步 0.9 s @ 50 Hz = 45 帧，一个周期（先 L 后 R 两步）90 帧。

关键技术点：

- **GATHER/SPLIT 站姿序列**：每个落脚面给 L/R 各一个固定落点（中心 ∓off），
  相邻转换只动一只脚，保证始终有清晰支撑多边形；
- **骨盆侧倾解级高**：25 cm 级高只需 waist_x 侧倾 10.4°（限位 15°），
  破解"平脚 + 踝限位 ±20°"的瓶颈；
- **DLS 数值 IK**：闭式解只作种子，每帧收敛到 <1 mm，全程检查关节限位与
  质心-支撑面裕度；重心时序在 p∈(0.32,0.78) 区间钉在支撑脚正上方；
- **回退梯度 LADDER**：IK 不收敛时按 (重心转移比例 lam, 摆动脚姿态权重) 逐级放松。

### 3.2 参数化参考表（`build_ref_tables.py` → `crab_ref_v1.npz`）

脚本轨迹只依赖 (rise, tread) 两个标量，故离线枚举：

- rise：0.00 ~ 0.26 m，步长 0.02 → **14 档**
- tread：{0.24, 0.30, 0.36, 0.40} m → **4 档**
- 每组取第 2、3 步的稳态两步（跳过起步），存 `q[14, 4, 90, 29]`，
  外加 `root_y/root_z/swing/dy_cycle/v_ref`，循环滑动平均（k=5）抹单帧尖峰。

质量门槛：IK 误差 < 1 mm、一阶差分 < 0.40 rad/帧、二阶差分 < 0.12 rad、
周期净位移 ≈ tread（±25%）。该表（207 KB）**入库**，是训练/回放共用的唯一参考源。

---

## 4. 地形感知与参考选择（`ref_table.py`）

### 4.1 高度扫描

`terrain_scan`：从躯干向下打射线，yaw 对齐网格 **21(y) × 13(x)**、分辨率 0.1 m、
覆盖前后 ±0.6 / 左右 ±1.0 m（螃蟹步加宽侧向），`include_geom_groups=(0,)` 只打地形。
`scan_to_heights()` 把射线距离换算成"躯干高出地形"的高度场 [B, 273]。

### 4.2 (rise, tread) 估计（`RefTable.estimate()`）

1. 取中央 3 列 x 的**中值**得 y 向剖面（抗机器人自腿遮挡的单格凹陷）；
2. 3 点中值滤波 → 上升边缘 `dh = h[iy] − h[iy+1]`（地形升高 ⇒ height 变小）；
3. 检测带 |y| ∈ (0.15, 1.00)，边缘须满足 `edge_thr=0.03 < dh < dh_max=0.30`
   （参考表最大档才 0.22，更大的"台阶"必是遮挡/异常）；
4. rise = 各边缘 dh 的**高斯加权均值**（σ=0.5 m，离机器人越近权重越大），
   两侧都认（可能倒着接近）；tread 只看 +Y 半侧，取**相邻边缘间距均值**；
5. **EMA 平滑**：rise α=0.5、tread α=0.3；长期测不到边时 rise 以 0.9/步回落到 0；
   测不到边（平地）时 tread 用**per-env 兜底值**——按 `terrain_origins` 最近邻反查
   该 env 所在**地形列**的真实踏面（0.40/0.40/0.30/0.24），而非全局标量
   （四列踏面各不相同，全局值最多对齐一列）。每 5 步刷新一次。

### 4.3 查表插值（`RefTable.select()`）

对 (rise, tread, phase) 三维做双线性 + 相位线性插值，返回 `q_ref[29]` 和该档参考
步速 `v_ref`。纯 numpy/torch、不依赖 mjlab，训练与回放（crab_replay）共用同一份代码。

---

## 5. 残差动作项（`residual_action.py`，核心）

```
q_cmd = q_ref(rise, tread, phase) + α课程 · ... + scale · clip(Δq, ±clip_rad)
        └── 时变 offset，每步由参考表给出        └── 策略输出，即"残差"
```

继承 mjlab `JointPositionAction`，只重写 `process_actions` / `reset`：

1. **相位是 per-env 状态量**：`phase += (v_cmd / v_ref) / 90`，速率夹在 [0.6, 1.8]——
   指令快则步频快，但几何上限是 `可达速度 = tread × 1.8/1.8 = tread`；
   开局 **hold 25 步**（0.5 s）相位冻结，站稳再走；相位平移 33/90 使 phase=0
   落在**双支撑**帧（避免一出生就单腿支撑），初始相位随机 {0, 0.5}；
2. **α 课程（参考幅度）**：`q_ref = q_stand + α·(q_gait − q_stand)`，α 从 0 线性升到 1，
   默认 1500 iteration × 24 步。α=0 时参考退化为默认站姿——与已验证的基线任务
   （动作全零 = 站姿）难度一致，早期训练不失稳。续训必须用 `ELF3_ALPHA=1.0` 钉住
   或 `ELF3_ALPHA_START` 接上（否则 ramp 从 0 重来，参考打回站姿，策略当场崩）；
3. **残差限幅按关节组**：腿 ±0.55 rad、腰 ±0.20 rad、臂 ±0.20 rad（可用
   `ELF3_LEG_CLIP` 等环境变量覆盖）。0.40 时代 76% 输出顶死限幅、梯度恒 0，
   放宽到 0.55 解除天花板；限幅同时是动作映射，可用旧 checkpoint 做零样本对拍；
4. **诊断开关**：`ELF3_RESIDUAL_SCALE=0` 纯参考开环走（拆分"参考自带偏置"与
   "残差学出偏置"）；`ELF3_RESET_RAMP=25` 每回合开局站姿→参考软启动；
5. 内部量全部暴露为只读属性（ref_joint_pos / phase / rise_est / scan_ok...），
   供观测项与奖励项取用，保证**观测、奖励、offset 三者口径一致**。

---

## 6. 观测空间（actor 393 维，critic 更大）

| 项 | 维度 | 说明 |
|---|---|---|
| base_lin_vel / base_ang_vel | 3+3 | 躯干线/角速度（传感器 `robot/torso_lin_vel` 等） |
| projected_gravity | 3 | 重力投影 |
| joint_pos | 29 | 相对默认站姿（带噪 ±0.01） |
| **joint_pos_ref_err** | 12 | 12 条腿相对 q_ref 的偏差（残差任务核心观测） |
| joint_vel | 29 | 关节速度 |
| actions | 29 | 上一动作 |
| command | 3 | 速度指令 |
| **phase** | 2 | sin/cos(2π·phase) |
| **ref_meta** | 3 | (rise×5, tread×2.5, v_ref×5) 归一化地形档位 |
| **height_scan** | 273 | 21×13 高度网格（×1/5.0 归一化，带噪 ±0.1） |

critic 在 actor 基础上：joint_pos 换**无偏**版本、加 foot_height / foot_air_time /
foot_contact / foot_contact_forces（特权信息）以及**完整 29 维 ref_joint_pos**。
所有噪声只加在 actor 上（asymmetric actor-critic）。

---

## 7. 奖励函数（残差版重排后）

设计主线：**速度项只管"别停别超速"，步态质量全权交给 ref_tracking + 抬脚正激励，
方向约束单列**。所有数值都经实跑标定，代码注释里保留了每次改动的实测依据。

| 项 | 权重 | 作用与关键设计 |
|---|---|---|
| `forward_progress_y` | **+5.0** | 替换内置瞬时速度 exp 核。参考步态 COM 是脉冲式（峰值 1.6 m/s、26% 帧倒退），瞬时核会奖励"匀速蹭走"。改为 **EMA(tau=1.8s=一个步态周期) 平均推进 ÷ 指令**，夹 [0,1]：跟参考 0.85 / 蹭走 0.70 / 站住 0。原地踏步 env 打五折防 value baseline 被抬高 |
| `ref_tracking` | **+2.0** | 12 腿关节跟踪 exp 核（σ=0.30 rad）——唯一逼出"真正侧步"的项。用 exp 而非 L2，避免退化成行为克隆 |
| `swing_clearance` | **+2.0** | 脚底离地高度/target **平方**后按脚平均。破"抬脚无收益、失衡 −5 分"的不对称稳态（贴地蹭）。平方是关键：线性版被"贴地快蹭"刷分（air_time 涨、peak_height 反降）。带 max_height 截断 + **竖直姿态门控**（up_z>0.6）防摔倒扑腾刷分 |
| `air_time` | +1.0 | 门槛 threshold_min 0.05 s（原 0.12 让这条恒为 0） |
| `upright` | +0.5 | 从 1.0 压到 0.5：原值 0.92/步属"白送分"，策略优先生存不前进 |
| `foot_swing_height` / `foot_clearance` | −1.0 / −2.0 | target 统一压到 0.15 m（原 0.18 与 α<1 时的参考 5~12 cm 直接打架） |
| `yaw_drift` | **−4.0** | 帧间偏航角速度平方，**死区 0.08 rad/s**。原 sin²(yaw) 版本绕圈"无罪"（20 s 转 367° 只罚 0.018）；纯平方版在 0.09 rad/s 处梯度塌缩。权重按 `w ≈ 1853/ep_len` 重标 |
| `lateral_bias` | −20 | 机身系**纵向**速度 EMA(1.8s) 偏置平方，死区 0.03——治步态自带 X 向漂移（实测三档都是 +0.10~0.13 m/s）。⚠️ −80 会把策略压垮成"别动"（ref_tracking 0.98→0.30），疼但不主导才有效 |
| `lateral_drift` | −0.2 | (x−x₀)² 位移惩罚（保留，量级已由 lateral_bias 覆盖） |
| `action_rate_l2` | −0.006 | 从 −0.1 一路压下来：原值 −2.89/回合是前进奖励的 2.2 倍，等于奖励"别动" |
| `foot_slip` | −0.2 | 打滑惩罚（基线保留） |
| `dof_pos_limits` | −1.0 | 关节限位（基线保留） |
| `is_terminated` | **−250** | 摔倒一次性惩罚。−5 形同虚设（×dt 只有 0.1 分）；−400 把策略逼成"站着不动"（摔率 78% 时走路期望收益转负）→ −250 平衡点 |
| `ref_probe` | 1e-9 | 探针：把 rise_est/tread_est/scan_ok/ref_alpha 写进 Metrics（权重为 0 时 mjlab 连 func 都不调，故用 1e-9） |

A/B 开关：`ELF3_REWARD_OFF=项名,项名` 一次关一批做归因（实测曾出现"训练指标向好、
eval 反而退步"的静默换目标，靠它定位）。

---

## 8. 课程与训练算法

### 8.1 课程（三重）

1. **地形课程**（自研 `terrain_levels_time`，替换内置 `terrain_levels_vel`）：
   内置判据"走够 `|cmd|×剩余时间`"导致**摔得越早要求越高**，全部 env 钉死 level 0
   （曾因此 12000 轮无效）。新判据：**升级 = 活到 25%（5 s）且离出生点 ≥ 0.90 m**
   （碗底平台半宽 flat 1.00 / easy 0.80 / mid 0.40 / hard 0.62，0.90 等于"至少踏上
   第一级"）；**降级 = 活不到 40%（8 s）**（0.10 时代存在 [2s,5s] 死带冻死课程）；
2. **指令速度课程**：lin_vel_y 分三段 (0.10,0.14) → (0.13,0.17) → (0.14,0.20)，
   上界与"步态几何可达速度 = tread"对齐，避免出现物理上追不上的指令；
3. **参考幅度 α 课程**：见 §5。

初始地形等级钉 0、四档 `step_height_range` 低端全对齐 0——保证有"先在平地站稳"
的阶段（曾实测 level 0 八个 env 里 7 个出生就在 6~14 cm 台阶上）。

### 8.2 PPO 超参（`rl_cfg.py`）

| 项 | 值 | 备注 |
|---|---|---|
| 网络 | MLP 512-256-128，ELU，obs 归一化 | actor/critic 同构 |
| 分布 | 高斯，std_type=scalar，init_std=1.0 | |
| entropy_coef | **0.0002** | ⚠️ mjlab 奖励按 per-step×dt(0.02) 缩放，每步任务奖励仅 ~0.03；29 维高斯 H≈37，coef=0.0015 时熵项 0.056/步 = 任务奖励 1.5 倍 → 优化器在**最大化随机性**（std 与熵单调升、ep_len 单调降）。按量纲重标到 0.0002 |
| lr | 1e-3，adaptive（desired_kl=0.01） | |
| γ / λ / clip | 0.99 / 0.95 / 0.2 | |
| 采样 | 24 步/env × mini-batch 4 × 5 epochs | |
| logger | 启动时探测 wandb 兼容性，失败自动退 tensorboard（rsl_rl 写死 `start_method="thread"` 与新版 wandb 冲突） | |

---

## 9. 安全终止、域随机化与地形

**终止**：躯干/腰/头触地力 > 20 N 即终止（`fell_over` 70° 挡不住"跪着蹭上去"
这类局部最优）；倾角 > 70°；超时；出地形边界。

**域随机化**：推撞从每 1~3 s ±0.3 降到每 3~6 s ±0.2（爬台阶容错小一个量级；
可 `ELF3_PUSH_SCALE` 缩放——实测训练分布比测试暴力一个量级时，策略会变成
"抗推专家"而非"走得好"）；脚底摩擦 0.4~1.2、编码器偏置 ±0.015 rad、
躯干质心偏移 ±2.5~3 cm、出生位姿 ±0.2 m / yaw ±0.15 rad。

**地形**：`pyramid_stairs_inv` **倒置**金字塔（碗底出生，向外走全是上台阶；
正金字塔 origin 在最高平台，曾因此白训 12000 轮）。四列：flat(0~2cm/踏面0.40)、
easy(0~10cm/0.40)、mid(0~16cm/0.30)、hard(0~25cm/0.24)。踏面取 0.1 m 整数倍
以对齐射线栅格，使 tread 估计不在档位间跳变。

---

## 10. 已踩过的坑（诊断指纹速查）

| 指纹 | 根因 | 修复 |
|---|---|---|
| rise_est 恒 0、tread_est 恒等于兜底值 | height_scan 取数 `frame_pos[:,:,2:3] − hit[...,:2]` 广播成 [B,B,N]，被维度守卫静默丢弃；B=1 时恰好正确 | 统一走 `scan_to_heights()`，两边先 view 成 [B,F,N] |
| rise/tread 顶到 clamp 上限 | 射线打到机器人自己的腿（视觉 mesh 在 geom group 0） | mesh 挪 group 2；estimate 里中值滤波 + dh_max=0.30 丢弃异常 |
| 全员站一会就摔、 rise_est 恒 0 | 正金字塔出生在最高平台，往外全是下楼梯 | 换 `pyramid_stairs_inv` |
| 课程冻结在 level 0 | 内置课程"摔得越早要求越高" | 自研时间型课程 |
| ep_len 稳定在 2~4 s 不动 | 课程死带 [2s,5s]：不降级也升不了 | demote_frac 0.10→0.40 |
| 策略贴地蹭、peak_height 1.6 cm | 抬脚在奖励里零收益、失衡 −5 分 | swing_clearance 正激励（平方+门控） |
| std/熵单调升、ep_len 单调降 | entropy_coef 相对任务奖励量纲过大 | 0.0015→0.0002 |
| 续训后当场崩 | α ramp 从 0 重来，参考打回站姿 | `ELF3_ALPHA=1.0` / `ELF3_ALPHA_START` |
| play 漂移数字不可信 | play 配置指令范围被放宽 | 评测必须 `--freeze-command` |

诊断工具链：`diagnose_policy.py`（--alpha/--trace/--dump-scan-all/--freeze-command）、
`analyze_trace.py`（漂移 A/B/C 分类）、`check_ref_vs_terrain.py`（参考档 vs 实际地形）、
`ref_table.py` 自测（`python -m ...ref_table` 覆盖遮挡/miss/兜底等回归用例）。

---

## 11. 文件索引

```
train/mjlab_tasks/crab/
├── env_cfg.py            基线环境（传感器/观测/奖励/事件/终止，400 行）
├── env_cfg_residual.py   残差版增量修改（动作/观测重组/奖励重排/课程/安全）
├── residual_action.py    残差动作项：q_cmd = q_ref + scale·clip(Δq)
├── ref_table.py          参考表运行时：读表/估地形/插值 + 自测
├── mdp_extra.py          自定义观测项与奖励项（含大量实测标定注释）
├── terrain_cfg.py        四列倒置金字塔地形
├── rl_cfg.py             PPO 超参
└── robot_cfg.py          机器人资产/增益/scale

train/assets/
├── build_ref_tables.py   离线生成 crab_ref_v1.npz（14 rise × 4 tread）
├── make_robot_xml.py     生成 elf3_robot_only.xml（含 geom group 修复）
└── ref_tables/crab_ref_v1.npz   参考表（入库）

sim/.../launch/
├── crab_climb.py         四相位准静态 IK 脚本步态（参考表的源头）
├── crab_ik.py            DLS 数值 IK / 可行性窗口
└── crab_replay.py        用同一份 RefTable 做开环/闭环回放

scripts/
├── diagnose_policy.py    策略诊断（α/扫描/轨迹落盘）
├── analyze_trace.py      漂移分类（A/B/C）
└── check_ref_vs_terrain.py  参考步态档位 vs 实际地形核对
```
