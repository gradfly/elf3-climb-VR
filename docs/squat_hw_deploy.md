# 三份自训蹲起 ONNX 真机操作指南（SOP）

适用范围：把 `sim/vendor/onnx/` 下自训的 `squat_down.onnx` / `squat_up.onnx` / `squat_hold.onnx`
通过新 Mod `com.bxi.squat_actions` 接入 bxi_controller_ros2，在 ELF3 真机上做蹲起测试。
只涉及这三个蹲起策略；`fight.onnx`（打拳）、`elf3_crab_residual.onnx`（残差爬楼）不在本指南范围。

前提：真机已装好 ROS2 环境（`/opt/bxi/bxi_ros2_pkg`、`hardware_elf3`、遥控器、`/opt/bxi/robot_config.yaml`
均可用），所以**不用重跑**上游 `script/README.md` 的 `usr_pro.sh` 整包部署（udev / IMU / 自启服务已就位）。

---

## 0. 状态机与按键速览

```
normal（走路） ──btn_10=3──> squat_down（2.46s）──完成自动──> squat_hold
      ↑                                                          │
      └────────────────完成自动──────── squat_up（2.02s）<──btn_10=4──┘
```

- `squat_hold` **只有 `btn_10=4`（起立）一个动作出口**，其他动作键全被屏蔽；
  但 `btn_2`（零力矩急停）和 IMU 失联保护仍可随时打断。
- **hold 默认开环保持**（`hold_mode="open_loop"`，2026-11-19）：不跑 ONNX，
  固定参考蹲姿 + kp×8，可**长期保持**（实测 60 s 零漂移）；现版 squat_hold.onnx
  是慢发散（hold 5-10 s 必倒），仅作短过渡备用，详见 §2 与验证报告 §8.3。
- `down→hold`、`up→normal` 是代码驱动（`state.py` 里 `ctx.request_state`），
  **不是** `mod.yaml` route——route 字段不支持"动作完成"事件；
  `down→hold`、`up→normal` 均带 0.3 s `dual_running_blend` 平滑（切换目标角插值，
  消除两网络残差突跳）。
- up 提前 8 帧（`end_frame_trim=8`）切走路态：末段失稳流形的兜底，实测
  trim 点姿态已站直（z 1.027 m / tilt 6.3°）。
- `btn_10` 的 value 3/4 与厂家已用的 2/5/6/7/8 不冲突。
- `mod.yaml` 里 states 的 `confirm: true` 只是 UI 元数据，框架不消费它，
  遥控按键触发**不需要**二次确认。
- 切换目标是 `com.bxi.basic_actions/normal`（走路，AMP 策略），**不是** `normal_run`：
  `model_normal.onnx` 其实属于 `normal_run`（2 m/s 奔跑），蹲起完直接奔跑有安全风险。

---

## 1. 部署到真机

### 1.1 同步代码（二选一）

真机工程目录通常是 `/opt/bxi/bxi_rl_controller_ros2_example`，以实际为准，可用
`ros2 pkg prefix bxi_example_py_elf3` 定位。

方式 A — git（真机工程是本仓库 clone 时）：

```bash
cd /opt/bxi/bxi_rl_controller_ros2_example
git pull
```

方式 B — scp（真机工程是独立包时，只拷新 Mod）：

```bash
# 在开发机执行
scp -r sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/mods/com.bxi.squat_actions \
  user@robot:/opt/bxi/bxi_rl_controller_ros2_example/src/bxi_example_py_elf3/mods/
```

### 1.2 编译

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
cd /opt/bxi/bxi_rl_controller_ros2_example
colcon build --packages-select bxi_example_py_elf3
source ./install/setup.bash
```

三份模型是 `.onnx`，`ModelSpec.portable_onnx()` 会自动回退到 onnxruntime，**无需**转 RKNN。

验收：`install/bxi_example_py_elf3/share/bxi_example_py_elf3/mods/com.bxi.squat_actions/assets/`
下应有 `squat_down.onnx`、`squat_up.onnx`、`squat_hold.onnx`、`squat_ref_v1.npz`。

### 1.3 自检（可选，真机有 Python 依赖时）

```bash
python src/bxi_example_py_elf3/mods/com.bxi.squat_actions/verify_policy.py
```

应输出三份 ONNX 加载成功、obs 维度 132 / 130(legacy)|126(真机对齐版)、帧数
124/102、hold 锁帧、后处理与直接 ORT 推理逐帧一致（`<1e-3`）。
需 `onnxruntime`、`pyyaml`。适配层按 metadata 的 observation_names 自动识别
hold 的新旧契约（2026-11-19 重训 hold 为 126 维真机对齐版，旧 130 维仍兼容）。

---

## 2. 首跑增益说明（2026-11-19 修订）

真机框架**没有**全局 `kp_scale` 参数（`kp_scale` 只存在于本地仿真脚本）。
增益用适配层自带的 `gain_scale`（构造参数，默认 1.0，kp/kd 统一乘这个系数）。

**蹲起策略保持默认 `gain_scale=1.0`（训练原值），不要降到 0.3**：

- RL 策略是在训练 kp 下学的蹲起，偏离训练增益 = 策略分布外；
- 本地仿真（`docs/squat_onnx_sim_verify.md`）显示训练柔顺 kp 本身在深蹲下
  支撑就偏软（hold 下沉、up 末段失稳），再降到 0.3 支撑更软、更易下沉；
- 安全靠**吊架 + btn_2 零力矩急停 + 命令丢失 >100 ms 失能**保证，不靠降增益。

若首跑想更保守，可用 `gain_scale=0.7` 试探后尽快回 1.0（仿真未验证 0.7，
真机自行判断）；`0.3` 已实测不可行，勿用。改完 `colcon build`。

### hold 段的增益是独立的 ×8（勿改）

`squat_hold` 状态默认 `hold_mode="open_loop"`（state.py）：不跑 ONNX，固定下发
参考蹲姿 + kp/kd=训练×8（`_OPEN_LOOP_GAIN=8.0`）。这是实测唯一能**长期保持**
蹲姿的配置（60 s 零漂移；现版 squat_hold.onnx 是慢发散、hold 5-10 s 必倒，
详见 verify 报告 §8.3）。`gain_scale` 不影响 hold 的开环增益；如需调整改
state.py 的 `_OPEN_LOOP_GAIN` 常量。重训 hold 完成（`docs/squat_hold_retrain.md`）
并本地验收后，可把 plugin.py 的 `hold_mode` 改回 `"onnx"` 切换对比。

---

## 3. 启动

```bash
ros2 launch bxi_example_py_elf3 example_launch_demo_hw.py \
  robot_config_file:=/opt/bxi/robot_config.yaml enable_head:=auto
```

启动会自动完成（**不要手动 reset**）：

1. 取单实例锁 `/tmp/bxi_example_hw.lock`（重复启动会报错，停掉旧实例即可）；
2. 等 IMU 数据（超时 0.1 s 触发锁定保护）；
3. 两阶段 reset：`step 1 → 等 1 秒 → step 2`；
4. 停在初始态 `com.bxi.basic_actions/zero_torque`。

看到日志 `robot reset step 2 requested` 即 reset 完成。
注意：真机 reset 后**不会**自动切到 `normal`（只有仿真会自动 `pd_brake→normal`），必须手动进。

---

## 4. 进入待机走路态

在 `zero_torque` 下按遥控器 **`btn_1 = 1`** 进入 `com.bxi.basic_actions/normal`（走路）。

- **切换前摇杆回中**：`normal` 是 AMP 策略，要求速度指令非 None，摇杆偏了会直接走。
- 确认：`ros2 topic echo /hardware/state_machine_info`（String 类型，发布当前状态）。

---

## 5. 执行蹲起

1. **`btn_10 = 3`** → `squat_down`（约 2.46 s）→ 完成**自动**切 `squat_hold`
2. `squat_hold` 期间**只有 `btn_10 = 4` 有效**，其他动作键无效
3. **`btn_10 = 4`** → `squat_up`（约 2.02 s）→ 完成**自动**切回 `normal`（走路）

按键落地前提：遥控器固件能否让 `btn_10` 物理输出 value 3/4 **不在本仓库**。
上机前先 `ros2 topic echo /motion_commands` 按一遍手柄，确认能看到 `btn_10` 输出 3 和 4。
发不出就得改映射或换按键（改 `mod.yaml` 的 `events` 段 slot/value，并重新 build）。

---

## 6. 观察与验证

- `ros2 topic echo /hardware/state_machine_info`：核对状态按 `down→hold→up→normal` 流转。
- launch 窗口是 debug 级日志，关注 `request_state` / transition 记录。
- 看蹲起幅度是否到位、有无抖动或过冲；记录 `|q−q_ref| > clip_rad` 的帧比例
  （`clip_rad`：腿 0.35 / 腰 0.15 / 臂 0.12 rad）。

---

## 7. 安全与异常

| 情况 | 处理 |
|---|---|
| 需要急停 | 按 `btn_2` → `zero_torque`（down/hold/up 三态都保留了这条 route，hold 也有效） |
| IMU 失联 | >0.1 s 由守护进程 `force=True` 直接打断，绕过 routes，任何态都能切走 |
| 命令丢失 | >100 ms 电机失能 |
| 动作异常 | 走软急停（冻结目标 + 阻尼下电），**不要照搬训练 reset**（关节目标会瞬间跳几十度） |
| hold 卡帧 / 输出非法 | 看门狗强制切零力矩（onnx 模式） |
| hold 姿态漂移 | 实际关节偏离固定目标 > 0.5 rad（被推/踩空/打滑）→ 看门狗切零力矩（open_loop 模式） |

---

## 8. 收尾

1. 按 `btn_2` 回 `zero_torque`，关闭电机电源，再处理吊架。
2. 若首跑用了试探性 `gain_scale<1.0`，稳定后改回 `1.0` 重新 `colcon build`。

---

## 9. 是否需要回云服务器重训

**按 2026-11-19 仿真结论修订（`docs/squat_onnx_sim_verify.md` §8.5）**：

- **hold（唯一确定要重训的）**：现版 ONNX 慢发散（每秒 +2.2° 前倾，5-10 s 必倒），
  无法长期保持；过渡期 hold 走开环×8（默认，长期稳+省电 11%），根治按
  **`docs/squat_hold_retrain.md`** 重训。2026-11-19 重训方案已落到
  `squat_hold_cfg.py`：收紧终止 55°→30°、min_z 0.35→0.48、姿态奖励加权、
  回合 40 s；**obs 去掉 base_lin_vel/traj_height_err（130→126 维，与真机完全
  一致，部署侧不再需要填 0 近似）**；新增外力触碰扰动 touch_robot
  （±35 N 持续 0.2~0.5 s，模拟真人推搡/碰撞）。不动执行器 kp/kd 与导出链。
- **down / up：不用重训**。真机缺失观测的填 0 近似已被 P0 修复证实成立
  （`traj_height_err` 填 0 而非 `-z_ref`；`base_lin_vel` 填 0 在训练噪声 ±0.3 内），
  deploy 模式仿真与真值模式几乎重合（down 完成/12.4°，up k92/11.8°）。
  up 末段失稳属参考轨迹动力学问题，重训同一参考不解决，已用 `end_frame_trim=8`
  兜底（trim 点姿态很稳：z 1.027/tilt 6.3°/裕度 +10 cm）。

真机近似处理（按模型区分）：
  - **新 hold（126 维）**：训练侧已去掉真机缺失项，**无任何近似**，obs 全部
    来自真机传感器（IMU 角速度/重力投影/关节编码器/上一步动作/参考误差）。
  - **down / up（132 维）与 legacy 130 维 hold**：无全局 odom → `base_lin_vel`
    填 0；无基座高度 → `traj_height_err` **填 0（=假设贴合参考，勿填 -z_ref）**；
    无头关节编码器 → 填 metadata 默认值 0。
三份模型 `history_length` 全为 0，无历史帧。
观测分块：down/up = `3+3+3+31+31+29+29+1+2` = 132；legacy hold = 130；
新 hold = `3+3+31+31+29+29` = 126（无 base_lin_vel / traj_height_err）。
适配层按 metadata 的 observation_names 自动拼块，无需按模型改代码。
归一化已烘焙进 ONNX 图，**部署侧绝不能再归一化**。
输出的 29 维是**残差**：`q_target = q_ref(frame) + ACTION_SCALE * clip(a, ±clip_rad/scale)`，
不能当绝对关节角下发。

---

## 10. 相关文档

- 训练与导出：`docs/squat_rl_training.md`
- 契约核对与改造方案：`C:/Users/xx.mu/.workbuddy/plans/toasty-pulse-babbage-Po725Dx9.md`（本地计划文件，不在仓库）
- 适配层自检脚本：`mods/com.bxi.squat_actions/verify_policy.py`
