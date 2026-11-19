# 重训 squat_hold：长期稳定保持蹲姿（配置方案）

> 背景：本地 MuJoCo 闭环实测（2026-11-19，见 `docs/squat_onnx_sim_verify.md` §8）证明
> 现版 `squat_hold.onnx` 是**慢发散**而非稳态保持：每秒约 +2.2° 前倾，hold 5-10 s
> 裕度归零必倒。根因：训练柔顺 kp（踝 33.5 N·m/rad）下参考蹲姿物理上撑不住
> （开环跟踪 q_ref 同样滑到前倾 42°），而训练终止条件（倾角 55°、塌陷 0.35 m）
> 松到允许策略"漂着活完 20 s 回合"——策略学到了顺应漂移，而不是主动补偿。
> 本文档给出训练侧改动方案，在训练机执行。

## 一、改动点（全部在 `train/mjlab_tasks/crab/squat_hold_cfg.py`）

策略有足够能力补偿稳态误差（29 维残差、腿 clip 0.35 rad ≫ 柔顺 kp 稳态误差 ~0.1 rad），
缺的只是**训练压力**。改动方向：把"漂移路径"在训练里变成早死，让真保持成为唯一高分解。

### 1. 收紧终止（`_build_terminations`，squat_hold_cfg.py:102-121）

```python
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(30.0)},  # 55 → 30
    )
    # KINDS["squat"]["min_z"]: 0.35 → 0.48（蹲姿 z=0.523，掉 4 cm 即终止）
    cfg.terminations["base_too_low"] = TerminationTermCfg(
        func=S.base_too_low, params={"min_z": KINDS[kind]["min_z"]},
    )
```

`KINDS`（:56-59）里同步改：`"squat": dict(frame=-1, min_z=0.48, ...)`。
30° = 蹲姿固有前倾 18.5° + 12° 余量：允许姿态波动，不允许持续前倾漂移。

### 2. 强化姿态奖励（`squat_hold_env_cfg` 内 `_build_rewards(...)` 之后叠加）

在 squat_hold_cfg.py:177-182 附近追加覆盖：

```python
    # 保持任务专属：姿态权重加大、带宽收紧 —— 逼策略把稳态误差主动补掉
    # 而不是顺应前倾（原 up_track 1.0/σ0.25、height_track 2.0/σ0.05）。
    cfg.rewards["up_track"].weight = 3.0
    cfg.rewards["up_track"].params["sigma"] = 0.15
    cfg.rewards["height_track"].weight = 4.0
    cfg.rewards["quiet"].params["sigma"] = 0.4     # 原 0.8：更严格的"别抖"
```

### 3. 回合加长（让慢漂移在训练里显现后果）

```bash
ELF3_HOLD_EPISODE=40 bash scripts/run_squat_train.sh hold
# 环境变量已现成支持（squat_hold_cfg.py:63）；20 s 时漂移压线能活完，
# 40 s 时必然早死 → 配合收紧的终止条件形成正确梯度。
```

### 4. 不动的部分

- **执行器 kp/kd 保持训练值**（与 down/up 一致）：obs 定义、残差语义都不变，
  导出的 metadata 自动一致，适配层零改动。
- push 扰动（0.8~2.0 s / ±0.25 m/s）、初始扰动（±0.05 rad）保持不变——抗扰能力要保住。
- `_build_observations` / `TrajTrackActionCfg`（freeze=-1、clip_rad）不变。

### 备选 Plan B（激进，若上述收敛太慢）

hold 任务独立把执行器 stiffness/damping ×4~×8（hold cfg 覆盖 robot actuator 参数，
不影响 down/up）。物理直接撑得住参考蹲姿，训练更容易；代价：hold 的 metadata kp
变为 ×N 值（适配层自动跟随，自洽），但与 down/up 的动力学断开。**只有 Plan A
（收紧终止）训不出再走这条**。

## 二、训练与导出命令（训练机）

```bash
# 1. 训练（先用现有脚本；确认 scripts/run_squat_train.sh 存在，否则查 train/README.md）
ELF3_HOLD_EPISODE=40 bash scripts/run_squat_train.sh hold

# 2. 训练中检查：bin/play 回放看 20+ s 是否仍稳在参考蹲姿（tilt ~18.5° 不漂）

# 3. 导出（SQUAT_MODES 里 hold-squat = freeze -1，见 train/export_onnx.py:49-54）
python train/export_onnx.py --squat hold-squat
```

## 三、验收标准（本地 Windows，无需训练机）

```bash
# 1. 单策略长期保持：60 帧 diag 看 tilt/z 走势，再改脚本跑 60 s 长保持
python launch/verify_squat_sim.py --policy hold --obs-mode train --diag-every 30
#    通过线：60 s 内 tilt < 30°、z > 0.48、稳定裕度 > +2 cm、无发散趋势

# 2. 全链闭环：down → hold → up 全程不摔、up 站直
python launch/verify_squat_sim.py --obs-mode train

# 3. deploy 近似（真机观测）复验
python launch/verify_squat_sim.py --obs-mode deploy

# 4. 适配层数值回归
python mods/com.bxi.squat_actions/verify_policy.py
```

四项全过后：`mods/com.bxi.squat_actions/plugin.py` 里把
`hold_mode="open_loop"` 改回 `hold_mode="onnx"`（新 ONNX 覆盖
`assets/squat_hold.onnx`，替换后 colcon build），真机吊架首验。

## 四、过渡期安排（重训完成前）

部署侧 hold 已默认 `hold_mode="open_loop"`（固定参考蹲姿 + kp×8，
state.py `_OPEN_LOOP_GAIN`）：真机可安全长期保持（本地实测 60 s 零漂移、
省电 11%）、up 从其末态直接完成 97%。重训完成且本地验收通过后切回
onnx 模式获得 RL 抗扰能力；两者可随时用 plugin.py 一行切换对比。
