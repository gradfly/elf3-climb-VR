# 重训 squat_hold：长期稳定保持蹲姿（已落码，训练机执行即可）

> 背景：本地 MuJoCo 闭环实测（2026-11-19，见 `docs/squat_onnx_sim_verify.md` §8）证明
> 现版 `squat_hold.onnx` 是**慢发散**而非稳态保持：每秒约 +2.2° 前倾，hold 5-10 s
> 裕度归零必倒。根因：训练柔顺 kp（踝 33.5 N·m/rad）下参考蹲姿物理上撑不住
> （开环跟踪 q_ref 同样滑到前倾 42°），而训练终止条件（倾角 55°、塌陷 0.35 m）
> 松到允许策略"漂着活完 20 s 回合"——策略学到了顺应漂移，而不是主动补偿。
>
> 2026-11-19 二次需求：重训时 **obs 与真机完全对齐**（去掉 base_lin_vel 与
> traj_height_err），并**增加外力触碰扰动**（模拟真人推搡/碰撞仍能保持平衡）。
> 全部改动已写入 `train/mjlab_tasks/crab/squat_hold_cfg.py`，训练机拉代码后
> 直接按 §三 训练，无需再改配置。

## 一、改动点清单（全部已实现在 `train/mjlab_tasks/crab/squat_hold_cfg.py`）

策略有足够能力补偿稳态误差（29 维残差、腿 clip 0.35 rad ≫ 柔顺 kp 稳态误差 ~0.1 rad），
缺的只是**训练压力**。改动方向：把"漂移路径"在训练里变成早死，让真保持成为唯一高分解。

### 1. 真机观测对齐（obs 130 → 126 维，本次新增需求）

去掉真机没有的两个观测块，训练输入分布 = 真机能提供的全部信息，
不再依赖部署侧"填 0"的近似（本地闭环实测：deploy 填 0 明显比 train 真值差）：

- `base_lin_vel`（3 维）：真机无 odom，编码器/IMU 给不出世界系线速度；
- `traj_height_err`（1 维）：真机无基座高度测量，部署只能填 0，而训练噪声
  仅 ±5 mm —— 填 0 是分布外输入。

actor 观测 126 维 = `base_ang_vel(3) + projected_gravity(3) + joint_pos(31)
+ joint_vel(31) + actions(29) + traj_joint_err(29)`，全部可由真机传感器提供。

- **critic（特权观测，不上真机）保留 `traj_height_err`**：真值高度误差帮价值
  函数判断"蹲没蹲到位"；`base_lin_vel` 在保持任务里真值 ≈0、噪声 ±0.3 纯属
  垃圾信息，一并去掉。
- 导出链零改动：`train/export_onnx.py` 的 obs_dim 与 metadata
  observation_names 自动跟随（新 hold 应为 6 项、obs[1,126]）。
- 部署侧零改动：`squat_residual_policy.py` 已改为按 metadata 的
  observation_names **驱动拼块**，同时兼容旧 130 维（legacy，两项填 0）与
  新 126 维契约 —— 换 ONNX 文件即可，无需改代码。

### 2. 收紧终止（`_build_terminations`）

```python
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(30.0)},  # 55 → 30
    )
    # KINDS["squat"]["min_z"]: 0.35 → 0.48（蹲姿 z=0.523，掉 4 cm 即终止）
```

`KINDS` 里已同步：`"squat": dict(frame=-1, min_z=0.48, ...)`。
30° = 蹲姿固有前倾 18.5° + 12° 余量：允许姿态波动，不允许持续前倾漂移。

### 3. 强化姿态奖励（`squat_hold_env_cfg` 内 `_build_rewards(...)` 之后叠加，已落码）

```python
    cfg.rewards["up_track"].weight = 3.0        # 原 1.0
    cfg.rewards["up_track"].params["sigma"] = 0.15   # 原 0.25
    cfg.rewards["height_track"].weight = 4.0    # 原 2.0
    cfg.rewards["quiet"].params["sigma"] = 0.4  # 原 0.8：更严格的"别抖"
```

### 4. 外力触碰扰动 `touch_robot`（本次新增需求，已落码 `_add_touch_event`）

用 mjlab 的 `mdp.apply_body_impulse`（类事件，mode="step"，自管
冷却→触发→持续→清零生命周期）在 `torso_link` 上施加随机方向的**持续真实力**：

| 参数 | 值 | 说明 |
|---|---|---|
| `force_range` | ±35 N/分量（世界系） | 典型水平合幅 ~26 N |
| `torque_range` | ±5 N·m | 附加小扭矩 |
| `duration_s` | 0.2~0.5 s | 一次触碰时长 |
| `cooldown_s` | 1.5~3.0 s | 触碰间隔 |
| `body_point_offset` | (0, 0, 0.12) m | 力作用点上移 12 cm（胸口/后背高度），offset×force 产生倾覆力矩 —— 模拟"推在身上"而非"推在质心上" |

量级标定（整机 43.2 kg、足底 μ≈1.0、双足打滑极限 ~420 N）：每分量 ±35 N
→ 冲量 5~13 N·s，与既有 `push_robot` 速度冲击 ±0.25 m/s（≈10.8 N·s）同量级。
两类扰动互补且都保留：速度冲击是"被撞一下"（无质量感），持续力是
"被人推一把"（有倾覆力矩）；力含竖直分量，顺带覆盖"按压肩膀/托举"类扰动。
40 s 回合约挨 13~27 次触碰 + 14~50 次速度冲击。

⚠️ **mjlab 版本要求**：`apply_body_impulse` 是 mjlab main 分支才有的类事件。
`_add_touch_event` 里做了显式检测，缺失时直接报错并提示升级命令
（`bash scripts/bootstrap_gpu.sh`），不做静默降级。训练机若报此错，
先升级 mjlab 再训。

### 5. 回合加长（让慢漂移在训练里显现后果）

```bash
ELF3_HOLD_EPISODE=40 bash scripts/run_squat_train.sh hold-squat
# 环境变量已现成支持（squat_hold_cfg.py EPISODE_S）；20 s 时漂移压线能活完，
# 40 s 时必然早死 → 配合收紧的终止条件形成正确梯度。
```

### 6. 不动的部分

- **执行器 kp/kd 保持训练值**（与 down/up 一致）：残差语义、导出 metadata 的
  kp/kd 自动一致（Plan B 的 ×N 增益方案仅作备选，Plan A 训不出再启用）。
- `push_robot`（0.8~2.0 s / ±0.25 m/s）、初始扰动（关节 ±0.05 rad、基座
  ±2 cm / ±4.6°）保持不变。
- `TrajTrackActionCfg`（freeze=-1、clip_rad 腿 0.35/腰 0.15/臂 0.12）不变。
- `hold-stand`（Elf3-Stand-Hold）同一份 cfg 构建，终止/奖励收紧同样生效
  （站姿本就该竖直，30° 限制更不碍事）；本次只重训 hold-squat。

## 二、训练与导出命令（训练机）

```bash
# 0. 若报 "当前 mjlab 没有 mdp.apply_body_impulse"：先升级 mjlab
bash scripts/bootstrap_gpu.sh

# 1. 冒烟（可选，确认 126 维观测与 touch 事件就位）
bash scripts/run_squat_train.sh smoke

# 2. 训练（40 s 回合）
ELF3_HOLD_EPISODE=40 bash scripts/run_squat_train.sh hold-squat

# 3. 训练中检查：bin/play 回放看 20+ s 是否仍稳在参考蹲姿（tilt ~18.5° 不漂）

# 4. 导出（SQUAT_MODES 里 hold-squat = freeze -1）
python train/export_onnx.py --squat hold-squat
#    导出后核对：obs_dim=126、metadata observation_names 应为
#    base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,traj_joint_err（6 项）
```

## 三、验收标准（本地 Windows，无需训练机）

```bash
# 1. 单策略长期保持：60 s diag 看 tilt/z 走势
python launch/verify_squat_sim.py --policy hold --obs-mode train --diag-every 30
#    通过线：60 s 内 tilt < 30°、z > 0.48、稳定裕度 > +2 cm、无发散趋势
#    注：新 hold 无 base_lin_vel/traj_height_err 观测，train 模式自动等于 deploy
#    （脚本会提示"真机对齐版"），这项近似检查自然消失。

# 2. 全链闭环：down → hold → up 全程不摔、up 站直
python launch/verify_squat_sim.py --obs-mode train

# 3. 适配层数值回归（自动识别 130/126 两版契约）
python mods/com.bxi.squat_actions/verify_policy.py
```

三件全过后：`mods/com.bxi.squat_actions/plugin.py` 里把
`hold_mode="open_loop"` 改回 `hold_mode="onnx"`（新 ONNX 覆盖
`assets/squat_hold.onnx`，替换后 colcon build），真机吊架首验。

## 四、过渡期安排（重训完成前）

部署侧 hold 已默认 `hold_mode="open_loop"`（固定参考蹲姿 + kp×8，
state.py `_OPEN_LOOP_GAIN`）：真机可安全长期保持（本地实测 60 s 零漂移、
省电 11%）、up 从其末态直接完成 97%。重训完成且本地验收通过后切回
onnx 模式获得 RL 抗扰能力（新 hold 训练时已含触碰扰动，抗推搡）；两者可
随时用 plugin.py 一行切换对比。
