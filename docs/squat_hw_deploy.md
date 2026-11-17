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
- `down→hold`、`up→normal` 是代码驱动（`state.py` 里 `ctx.request_state`），
  **不是** `mod.yaml` route——route 字段不支持"动作完成"事件。
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

应输出三份 ONNX 加载成功、obs 维度 132/130、帧数 124/102、hold 锁帧、
后处理与直接 ORT 推理逐帧一致（`<1e-3`）。需 `onnxruntime`、`pyyaml`。

---

## 2. 首跑降刚度（强烈建议）

真机框架**没有**全局 `kp_scale` 参数（`kp_scale` 只存在于本地仿真脚本）。
降刚度用适配层自带的 `gain_scale`（构造参数，默认 1.0，kp/kd 统一乘这个系数）：

编辑 `src/bxi_example_py_elf3/mods/com.bxi.squat_actions/plugin.py` 的 `_make_loader`：

```python
return SquatResidualPolicy(
    str(context.asset(f"assets/squat_{mode}.onnx")),
    str(context.asset("assets/squat_ref_v1.npz")),
    mode=mode,
    gain_scale=0.3,      # ← 首跑加这行；稳定后改回 1.0 再 build
)
```

改完**重新 colcon build**。首跑必须上吊架。

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
| hold 卡帧 | 看门狗超时会自行切零力矩 |

---

## 8. 收尾

1. 按 `btn_2` 回 `zero_torque`，关闭电机电源，再处理吊架。
2. 稳定后把 `plugin.py` 的 `gain_scale` 改回 `1.0`，重新 `colcon build`。

---

## 9. 是否需要回云服务器重训

**默认不用重训**。只有真机/仿真对拍中 `|q−q_ref| > clip_rad` 的帧比例持续 >5%，
或反复站不稳，才升级为训练议题：改 `train/mjlab_tasks/crab/squat_env_cfg.py` 的观测定义
（删掉真机拿不到的 `base_lin_vel`，必要时连 `traj_height_err` 一起删），
再用 `train/export_onnx.py --squat down/up/hold-*` 重新导出。

真机近似处理：无全局 odom → `base_lin_vel` 填 0；无基座高度 → `z_meas` 由腿部正运动学估或填 0；
无头关节编码器 → 填 metadata 默认值 0。三份模型 `history_length` 全为 0，无历史帧。
观测分块：down/up = `3+3+3+31+31+29+29+1+2` = 132；hold 去掉末 2 维 = 130。
归一化已烘焙进 ONNX 图，**部署侧绝不能再归一化**。
输出的 29 维是**残差**：`q_target = q_ref(frame) + ACTION_SCALE * clip(a, ±clip_rad/scale)`，
不能当绝对关节角下发。

---

## 10. 相关文档

- 训练与导出：`docs/squat_rl_training.md`
- 契约核对与改造方案：`C:/Users/xx.mu/.workbuddy/plans/toasty-pulse-babbage-Po725Dx9.md`（本地计划文件，不在仓库）
- 适配层自检脚本：`mods/com.bxi.squat_actions/verify_policy.py`
