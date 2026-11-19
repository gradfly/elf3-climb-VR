# 三份自训蹲起 ONNX 的本机 MuJoCo 闭环验证

> 实测日期：2026-11-19 · 本机 Windows · mujoco 3.17.0 / onnxruntime 1.32.0 / numpy 2.4.4
> 验证脚本：`sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch/verify_squat_sim.py`
> 验证录像（本地生成）：`.../render/squat_verify/squat_full.gif`

## 1. 目的与结论

在真机部署前，先在本机 MuJoCo 里**真正闭环**跑三份自训蹲起 ONNX（`squat_down/up/hold.onnx`），
判断策略本身是否成立、真机填 0 近似观测是否可行、是否需要回云服务器重训。

**结论（一句话）**：`down` 闭环成立、`hold` 能蹲住但缓慢下沉、`up` 能站到约 0.99 m（目标 1.034 m）
但末段失稳；失稳根因是**参考轨迹末段的动力学 + 训练柔顺 kp**，**不是 ONNX 策略本身**，也不是
关节角跟踪不准（跟踪误差全程仅 ~11°）。真机常用的填 0 近似观测会让 down/up 明显变差。
**暂不需要重训**，但上真机前建议按 §6 处理。

## 2. 验证方法（关键一致性措施）

闭环链路：从 `mujoco.MjData` 构造 `InferenceFrame` → `SquatResidualPolicy.step()`
（obs 拼装 + ONNX 推理 + 残差后处理，完全复用部署适配层）→ `d.ctrl = q_target` → `mj_step`。

| 项目 | 做法 | 为什么重要 |
| --- | --- | --- |
| 模型 | `elf3_squat.xml`（平地），经 `elf3_play.build_model` 换成 mjlab 同款内置 `<position kp kd>` 执行器，kp/kd/action_scale/default 全取 ONNX metadata，EFFORT=4·scale·kp，implicitfast 积分 | 原 XML 的执行器是占位 `<motor gear=1>`；不换成立器就撑不住重力。显式软件 PD 实测要么撑不住、要么 QACC 发散，内建 position 才稳定 |
| 关节布局 | 29 可控关节顺序 == `ELF3_POLICY_JOINTS` == 执行器 0:29（已逐名核对） | 顺序错则输出全错 |
| 初始姿态 | 用源轨迹 `squat_extreme_{down,up}_traj.npz` 的**完整 38 维 qpos**（根节点 xyz + quat + 29 关节 + 2 头关节） | 蹲姿躯干前倾 18.5°、根节点四元数 y=+0.1607、COM 裕度 +9.65cm；用 `q_ref+z` 拼姿态会因四元数符号/根节点位置错误导致开局就倒（实测踩坑） |
| 摔倒判据 | 躯干 up 向量与竖直夹角 > 60°，或躯干 z 低于当前参考帧 0.15 m | 蹲姿目标 z≈0.523 m，不能用固定 z<0.5 m 误判 |
| 观测模式 | `--obs-mode train`（喂仿真 `torso_lin_vel`/基座高度真值）与 `--obs-mode deploy`（与真机一致，这两项填 0）对照 | 直接回答「真机填 0 近似会不会毁掉策略」 |

## 3. 实测数据

全流程 `DOWN(124帧) → HOLD(60帧) → UP(102帧)`，控制 50 Hz、物理子步按模型 timestep。

### 3.1 train 模式（仿真真值，检验策略本身）

| 阶段 | 结果 | 躯干 z | q 跟踪误差 mean/max | 说明 |
| --- | --- | --- | --- | --- |
| squat_down | ✅ 完成 124 帧 | 1.034 → 0.523 m（-51.1 cm） | 12.0° / 16.3° | 末值精确命中参考 0.523，深蹲姿态正常（见 GIF） |
| squat_hold | ⚠️ 完成 60 帧不摔 | 0.523 → 0.441 m（下沉 8.2 cm） | 22.1° / 26.7° | 蹲住但缓慢下沉，柔顺 kp 在深蹲下撑不稳 |
| squat_up | ❌ k93 摔倒 | 0.523 → 0.877 m（+35.4 cm） | 11.1° / 16.7° | k70 曾到 1.017 m（目标 1.034，完成 ~97%），随后失稳后仰 |

up 逐步诊断（每 10 步）：

```
k=  0 z=0.523 z_ref=0.523 tilt=18.6° qerr= 1.7° 裕度=+9.7cm
k= 30 z=0.667 z_ref=0.659 tilt= 5.0° qerr=11.0° 裕度=+6.1cm
k= 50 z=0.857 z_ref=0.882 tilt= 2.5° qerr= 9.2° 裕度=+0.9cm
k= 70 z=1.017 z_ref=1.021 tilt= 5.8° qerr=10.0° 裕度=-8.5cm   ← 已到站姿高度
k= 90 z=0.915 z_ref=1.034 tilt=29.6° qerr=13.5° 裕度=-30.6cm  ← COM 掉出支撑面，后仰
```

关键：q 误差全程 < 17°，说明**关节角在准确跟随参考**；失稳是质心前移超出支撑面
（裕度从 +9.7 cm 降到 -8.5 cm）导致的动力学问题，不是策略跟踪错误。

### 3.2 deploy 模式（真机填 0 近似，检验可部署性）

| 阶段 | 结果 | 躯干 z | q 跟踪误差 mean/max |
| --- | --- | --- | --- |
| squat_down | ❌ k67 摔倒 | 1.034 → 0.536 m（-49.8 cm） | 20.7° / 28.2° |
| squat_hold | ⚠️ 完成 60 帧 | 0.523 → 0.434 m（下沉 9.0 cm） | 28.9° / 36.3° |
| squat_up | ❌ k54 摔倒 | 0.523 → 0.753 m（+23.0 cm） | 12.3° / 21.5° |

**对比**：填 0 近似使 down 跟踪误差从 12° 升到 20.7°、up 从 k93 提前到 k54 摔倒。
说明 `base_lin_vel` 与基座高度这两项在真机取 0 会系统性拉低策略表现。

## 4. 失稳根因定位（关键对照实验）

为区分「ONNX 策略问题」还是「参考轨迹/物理问题」，做了开环对照：
**完全不跑 ONNX，直接把参考轨迹 q_ref 喂给执行器**（`d.ctrl = q_ref`）：

```
开环 squat_extreme_down_traj: z 1.034→0.479（末帧目标 0.523），不摔倒
开环 squat_extreme_up_traj  : z 0.523→0.092，倾角摔倒 @k34（末帧目标 1.034）
```

即**纯参考轨迹的开环起立比 ONNX 闭环更早倒**（k34 vs k93）——ONNX 闭环反而显著更稳。
再叠加两点：

- kp 倍率 1.0/1.5/2.0/3.0 摔倒点几乎不变（k65~77），排除增益不足是主因；
- 训练配置 `squat_env_cfg.py:204` 有 `base_anchor` 奖励（weight -40）专门约束基座，
  该稳定机制在纯回放里没有。

结论：**up 末段失稳是参考轨迹末段在柔顺 PD 下的固有动力学特性，ONNX 策略已尽力补偿**，
不属于策略训练失败。

## 5. 是否需要回云服务器重训？

**当前不需要。** 理由与触发条件：

- down 闭环成立、up 能完成 ~97% 起立且跟踪误差小 —— 策略学到了正确的动作映射；
- 失稳来自参考轨迹末段 + 柔顺 kp 的物理特性，重训同样的参考轨迹不会消除它；
- 真机位置伺服刚度通常远高于训练用的柔顺 kp（见 `squat_replay.py:249-260` 的 ×8 说明），
  真机上 up 末段反而可能更稳 —— 必须在真机实测后才能定论。

**触发重训的条件**（满足任一再考虑）：
1. 真机（正常刚度）下 up 仍在站直末段持续后仰摔倒；
2. 真机填 0 近似导致 down 也站不住 / 跟踪误差 > 25°；
3. 希望去掉 `base_lin_vel` 和 `traj_height_err` 两个真机缺失观测——那就必须删观测重训
   （改 `train/mjlab_tasks/crab/squat_env_cfg.py` 的 `_build_observations`，再用
   `export_onnx.py --squat` 重新导出）。

## 6. 上真机前的建议

1. **优先保留真值可用的观测**：真机若有任何基座速度估计（足端里程计/IMU 积分）或基座高度
   （腿部正运动学），应尽量填上，不要一律置 0 —— deploy 模式数据表明填 0 代价明显。
2. **up 末段做状态兜底**：起立接近完成（进度 > 0.9）时，若检测到姿态后仰/COM 越界，
   切到一个短时站姿保持策略或直接切走路态 `com.bxi.basic_actions/normal`，不要让 up 策略
   独自扛最后的站稳阶段。
3. **首跑降刚度**：按 `docs/squat_hw_deploy.md` 用 `gain_scale=0.3` + 吊架；但注意本仿真显示
   kp 倍率对 up 失稳点影响很小，降刚度主要是安全目的。
4. **先在仿真复现 safer 配置**：可在本脚本基础上试「up 后 20% 帧切 stand 参考」等改法，
   无需重训即可验证兜底策略。

## 7. 复现命令

在 `sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/` 下：

```bash
# 全流程（默认 deploy 真机近似观测；--policy all 为连续模式，段间保留末态
# 并仿真 dual_running_blend 切换，--segmented 可回退每段 reset 的旧模式）
python launch/verify_squat_sim.py
# 仿真真值观测（检验策略本身）
python launch/verify_squat_sim.py --obs-mode train
# 单策略 + 逐步诊断
python launch/verify_squat_sim.py --policy up --obs-mode train --diag-every 10
# up 提前完成扫参（end_frame_trim 等价）
python launch/verify_squat_sim.py --policy up --obs-mode train --up-trim 8
# 开原生窗口看 / 生成 GIF
python launch/verify_squat_sim.py --policy down --viewer native
python launch/verify_squat_sim.py --obs-mode train --gif render/squat_verify/squat_full.gif --render-every 8
```

需要 `numpy / mujoco / onnxruntime / pyyaml`（本机已装）；脚本不依赖 ROS2 / colcon。

## 8. 第二轮诊断与修复（2026-11-19 下午，P0-P2）

### 8.1 P0 修复：traj_height_err 填法（适配层 1 行，已修）

训练侧事实（`squat_env_cfg.py:131-172`）：`base_lin_vel` 噪声 UNoise(±0.3) → 真机填 0
落在噪声包络内可容忍；**`traj_height_err` 噪声仅 UNoise(±0.005)** → 适配层原填
`-z_ref`（蹲起时 −1.034~−0.523）是深度分布外、且落在训练 `base_too_low` 终止区，
策略误判"已摔倒"输出异常。改为填 0（"贴合参考"近似，err=0 落训练噪声附近）。

**复验（分段模式）**：deploy 模式从「down k67 摔倒 / 误差 20.7°、up k54 摔倒」
改善为「down 完成 124 帧 / 误差 12.4° / 末值 0.525 精确命中；up k92 摔倒 / 11.8°」
——与 train 模式（12.0° / k93）几乎重合。**deploy 退化几乎全部来自这一个错误填法，
真机填 0 近似方案成立**（`base_lin_vel` 填 0 的代价可忽略）。

### 8.2 P1 修复：down→hold 切换瞬态（state.py，已修）

训练时 down 的 5 s 末帧保持段由**同一个 down 网络**跑 freeze 帧
（squat_env_cfg.py:351-357），部署却切到独立 hold 网络——两网络残差不连续；
原 `end_transition="soft_switch"`（0.02 s）把目标角突变直接打进电机。
连续模式实测切换瞬间 tilt 突跳 +10°（18.6°→28.7°）。已改 `dual_running_blend`
（0.3 s smootherstep，与 up→normal 同款）；blend 仿真复验：**切换后前 5 帧
z 纹丝不动（0.523→0.521），突降完全消除**。

### 8.3 hold 慢发散（关键新发现，推翻 §3 的"准稳态"判断）

加长观察窗后发现 hold **不是稳态**：每秒约 +2.2° 持续前倾——
`2s:49.3°/+4.9cm → 6s:58.9°/-0.3cm → 12s:64.7°/-5.0cm`（裕度 6 s 归零、
之后静态已出支撑面）。§3 记录的"稳在 43-45°"只是 1.2 s 观察窗的错觉。
**hold 5-10 s 必倒，无法长期保持**；hold >3 s 后开环纠姿也拉不回
（裕度≈0 时高增益猛拉反而失稳，hold 6s→纠姿→up 摔倒@26）。

根因链：训练柔顺 kp 下参考蹲姿物理撑不住（开环跟踪 q_ref 同样滑到 42°）
→ 训练终止条件太松（倾角 55°/塌陷 0.35 m，漂移 20 s 压线能活完回合）
→ 策略学到顺应漂移而非主动补偿。**开环参考 + kp×8 则实测 60 s 零漂移**
（z 0.513 / tilt 24.5° / 裕度 +7.3 cm 恒定），×4 不足（50 帧滑到 49°）。

能耗（静态保持力矩 qfrc_bias，铜损 Στ²）：参考蹲姿 526.8 → ONNX hold 前倾态
319.0（0.61×，确实更省但会倒）→ 开环×8 稳态 466.8（0.89×，稳且省）。

### 8.4 最终决策与部署侧落地

hold 需求是**长期保持蹲姿** → 选**重训 hold 长期根治**（方案见
`docs/squat_hold_retrain.md`：收紧终止 55°→30°、min_z 0.35→0.48、
姿态奖励加权、回合 20→40 s，不动执行器/obs）。过渡期部署侧
`SquatHoldState` 已加 `hold_mode` 参数，**默认 "open_loop"**（固定参考蹲姿 +
kp/kd=训练×8）：重训完成前真机即有安全的长期保持（60 s 零漂移、省电 11%），
up 从其末态直接完成 97%（1.029 m / tilt 5.5°）；重训验收通过后一行切回
"onnx" 获得 RL 抗扰能力。`squat_hold.onnx` 保留在 assets 备用。

up 已加 `end_frame_trim=8` 保险：末段失稳流形（从源蹲姿起步 k93 后仰摔倒）
实测 trim 点姿态很稳（frame 94: z 1.027 / tilt 6.3° / 裕度 +10 cm），
从开环 hold 末态起步则能完整跑满 102 帧——trim=8 两者通用。

### 8.5 重训判据修订

- **base_lin_vel / traj_height_err 观测**：不再需要为此重训（P0 修复后
  deploy≈train，真机填 0 近似成立）。
- **hold**：按 `docs/squat_hold_retrain.md` 重训（长期保持的唯一根治路径）。
- **up 末段后仰**：仍无需重训（参考轨迹末段动力学问题，trim=8 兜底；
  若真机 gain=1.0 实测仍失稳再议）。

