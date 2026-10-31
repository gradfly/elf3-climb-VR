# 侧身爬楼梯稳定性量化测量执行清单（X 向漂移）

> 第一轮只做**量化**：不改训练配置、不动 checkpoint，只跑评估 + 给诊断脚本加一个 `--trace` 开关。
> 产出是「一张基线表 + 一句结论（漂移属于哪一类）」，之后才决定改奖励还是改观测。
> 依据：诊断脚本已能打印净 Δx / 偏航 / 速度比，但**缺少逐步轨迹**，无法区分「偏航耦合」和「步态不对称」。

---

## 0. 先说结论口径

| 术语 | 定义 | 为什么用它 |
|---|---|---|
| 漂移量 | `\|x − x0\|`（x0 = 本回合出生点 x） | 世界系，不受偏航后 body 系旋转影响 |
| **漂移比** | `\|Δx\| / Δy`（米/米） | 与速度无关，能跨 checkpoint/地形比较；判据见 §4 |
| 朝向误差 | `wrap(yaw)`（世界系，目标 0） | 地形轴对齐世界，螃蟹步要求朝向不变 |
| 偏航速率 | `ω_z`（rad/s） | 区分「慢慢偏」和「抖」 |

---

## 1. 前置确认（5 分钟）

在 **GPU 服务器**仓库根目录执行（本机没装 mjlab，`import mjlab` 会超时）。

```bash
# 1) 训练任务到底是哪个（Residual 是带漂移奖励的主线；Stairs 是旧基线）
.venv/bin/python -c "
import mjlab.tasks
from mjlab.tasks.registry import list_tasks
print([t for t in list_tasks() if 'Crab' in t])"

# 2) 找最新 checkpoint（按时间排序，取最后一个）
ls -t logs/rsl_rl/elf3_crab/*/model_*.pt | head -5

# 3) 读该 run 的 ref_alpha（α 课程是否已满幅；play 要按它对齐，否则残差与参考错位）
.venv/bin/python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea=EventAccumulator('logs/rsl_rl/elf3_crab/<run>'); ea.Reload()
print([e.value for e in ea.Scalars('Metrics/ref_alpha')][-3:])"
```

记录：`TASK=______`，`CKPT=______`，`ALPHA=______`。
`ALPHA < 0.95` 时，评估必须带 `ELF3_ALPHA=<ALPHA>`，否则测的是"满幅参考 + 半幅残差"，数字没意义。

---

## 2. 基线采集 B0~B3（无代码改动）

同一份 checkpoint、同一 seed，只改地形等级。**每个 run 只跑 20 s 仿真**，很快。

```bash
MUJOCO_GL=egl ELF3_ALPHA=<ALPHA> .venv/bin/python scripts/diagnose_policy.py \
  --task Elf3-Crab-Residual-Play \
  --checkpoint <CKPT> --num-envs 16 --steps 1000 --terrain-level 5
```

| run | `--terrain-level` | 目的 | 若这里就漂说明 |
|---|---|---|---|
| B0 | 5 | 主判据（标准难度） | — |
| B1 | 3 | 简单档 | 主因是初始偏航/步态不对称，不是台阶几何 |
| B2 | 7 | 难档 | 漂移随难度放大 → 与残差饱和/落脚有关 |
| B3 | 5 + `--record /tmp/b3.mp4` | 看形态 | 整条斜 = 偏航；每步蛇形 = 不对称 |

> 坑：`--terrain-level` 会把 `基座 z` 打印成负值（`env_origins` 未对齐），**不要拿它判断爬升**。
> 坑：`play` 配置会 `pop("push_robot")`，所以上面测到的都是**无扰动**漂移；抗扰漂移见 §5。

### 2.1 从输出里抄哪些数（脚本原字段名）

| 字段（脚本输出） | 填到 §6 表里 | 判据 |
|---|---|---|
| `净 Δx 均值` / 时长 | `dx_rate` | ≤ 0.01 m/s |
| `净 Δy 均值` / 时长 | `dy_rate` | ≥ 0.7 × cmd_y |
| `yaw 速率 \|ω_z\|` | `yaw_rate` | ≤ 0.08 rad/s |
| `yaw 累计漂移` | `yaw_travel` | ≤ 15° / 20 s |
| `跟踪比 v_y/cmd` | `track_ratio` | ≥ 0.70 |
| `ep_len median` | `ep_len` | ≥ 15 s（level ≤ 5） |
| `残差饱和比例` | `sat` | ≤ 40% |
| `净 Δy 最差/最好` | `dy_range` | 看 env 之间离散度 |

---

## 3. 加一条轨迹诊断补丁（只改 `scripts/diagnose_policy.py`）

**为什么必须加**：现有输出只有均值，无法回答"这次漂移是 yaw 带出来的，还是腿不对称"。
补丁只做记录，不碰训练代码、不碰 checkpoint。

改动点（4 处）：

1. `argparse` 增加：
   ```python
   ap.add_argument("--trace", default=None, metavar="OUT.npz", help="逐步轨迹落盘")
   ap.add_argument("--push", action="store_true",
                   help="保留扰动（play 默认把 push_robot pop 掉了）")
   ```
2. 环境构造前（`env = ManagerBasedRlEnv(...)` 之前）处理 `--push`：
   ```python
   if args.push:
       from mjlab.managers.event_manager import EventTermCfg
       from mjlab.tasks.velocity import mdp as vm
       env_cfg.events["push_robot"] = EventTermCfg(
           func=vm.push_by_setting_velocity, mode="interval",
           interval_range_s=(3.0, 6.0),
           params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2),
                                      "z": (-0.1, 0.1), "roll": (-0.15, 0.15),
                                      "pitch": (-0.15, 0.15), "yaw": (-0.1, 0.1)}})
   ```
3. 主循环里（`net_dx += ...` 附近）累计逐 env 轨迹；`x0` 在 `dones` 那一步重置：
   ```python
   if args.trace:
       x0 = torch.where(dones_b, p[:, 0], x0)          # p 已是新出生点
       rec.append({
         "dx": (p[:, 0] - x0), "dy": net_dy.clone(),
         "yaw": _root_yaw(base), "wz": _root_yaw_rate(base),
         "vx": d[:, 0], "vy": d[:, 1], "alive": alive,
       })
   ```
4. 收尾 `np.savez_compressed(args.trace, **{k: torch.stack(v).cpu().numpy()
   for k, v in rec.items()})`，并打印逐 env 的 `end dx` / `mean |dx|` / `漂移比`。

> 兼容性：`_root_yaw` / `_root_yaw_rate` 已处理"字段名缺失返回 None"，补丁里沿用同一套 `try/except`。

---

## 4. 轨迹分析（判定三类漂移之一）

```bash
.venv/bin/python - <<'PY'
import numpy as np
d = np.load('/tmp/trace_b0.npz')
a, dx, yaw, vx, vy = d['alive'], d['dx'], np.degrees(d['yaw']), d['vx'], d['vy']
# ① 漂移比：每爬 1 m 横偏多少 m（跳过热启动前 60 步，避开 reset 抖动）
m = a & (np.arange(a.shape[0])[:, None] > 60)
ratio = np.abs(dx[m]) / np.clip(vy[m], 1e-3, None)
print("漂移比 |dx|/dy   : mean %.3f  p90 %.3f  (m/m)" % (ratio.mean(), np.percentile(ratio, 90)))
# ② 朝向安静段（|yaw|<3°）里的平均 vx：这一步隔离出"步态不对称"
q = m & (np.abs(yaw) < 3.0)
print("安静段 vx        : mean %+.4f m/s  |vx|均值 %.4f  (样本 %d)"
      % (vx[q].mean(), np.abs(vx[q]).mean(), q.sum()))
# ③ 偏航活跃段
b = m & (np.abs(yaw) >= 3.0)
print("偏航段 vx        : mean %+.4f m/s  (样本 %d)" % (vx[b].mean(), b.sum()))
# ④ 逐步相关：dx 的变化与 yaw 的变化同向吗
ddx = np.diff(dx, axis=0); dyaw = np.diff(yaw, axis=0)
mm = m[1:] & m[:-1]
print("corr(Δdx, Δyaw)  : %.3f" % np.corrcoef(ddx[mm], dyaw[mm])[0, 1])
PY
```

| 观察 | 判定 | 下一轮主攻（写在结论里） |
|---|---|---|
| `corr(Δdx, Δyaw)` 显著 >0.3，且安静段 `vx ≈ 0` | **A. 偏航耦合** | P2 补 `heading_sincos` + `yaw_hold`；这是观测问题，调权重治不好 |
| `corr` 接近 0，安静段 `\|vx\|` 仍 ≥ 0.02 且同号 | **B. 步态/残差不对称** | 先查残差限幅与左右腿对称性，再加 `lin_vel_x_zero` |
| 安静段与偏航段都正常，只有 `end dx` 偶发大 | **C. 扰动恢复失败** | 先跑 §5 扰动测试，再做 push 课程 |

---

## 5. 扰动恢复测试 B4（可辨识是"走歪"还是"被打歪"）

```bash
MUJOCO_GL=egl ELF3_ALPHA=<ALPHA> .venv/bin/python scripts/diagnose_policy.py \
  --task Elf3-Crab-Residual-Play --checkpoint <CKPT> \
  --num-envs 16 --steps 1000 --terrain-level 5 --push --trace /tmp/trace_b4.npz
```

看的是**恢复时间**：每次推挤后，`vx` 回到 \|vx\|<0.02 需要多少秒；以及恢复后 `x0` 偏移是否被"带回"。
判据：`恢复时间 > 2 s` 或 `x 偏移单向累积` → 策略没有横向纠偏能力（支撑 §4 的 A/C 判定）。

---

## 6. 基线表（执行后填写，作为后续所有改动的对照）

| run | terrain | dx_rate (m/s) | 漂移比 (m/m) | yaw_rate | yaw_travel | track_ratio | ep_len | sat | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 5 | | | | | | | | |
| B1 | 3 | | | | | | | | |
| B2 | 7 | | | | | | | | |
| B3 | 5（录像） | | | | | | | | |
| B4 | 5 + push | | | | | | | | |

交付物：上表 + 一句结论（A / B / C）+ 一段录像（`ELF3_VIEWER_ORIGIN=world` 固定机位看整条轨迹）。

---

## 7. 决策门（量化完成后再动代码）

| 结果 | 下一步 |
|---|---|
| `dx_rate ≤ 0.01` 且 `漂移比 ≤ 0.08` | 已达到验收线，**不改**；只把指标加进训练日志防回归 |
| 判定 A（偏航耦合） | 直接走观测改造（`heading_sincos` + `x_offset` + 降噪），必须重训 |
| 判定 B（步态不对称） | 先做奖励侧（`lin_vel_x_zero` + 走廊惩罚），可续训验证 1000~1500 iter |
| 判定 C（扰动恢复） | 先做扰动课程 + 恢复奖励，最后再补观测 |
| `track_ratio < 0.65` 或 `ep_len < 10 s` | 策略本身还没走稳，**漂移不是首要矛盾**，先修推进/步态 |

> 本清单到此为止。奖励公式、观测项、课程设计的具体改法不在本轮，等 §7 的判定出来再单独出方案，
> 避免"先改一堆再解释不通"。

---

## 8. 注意事项（都是踩过的坑）

1. **别用训练日志的 `Mean reward` 判断好坏**：`forward_progress_y` 与 `yaw_drift` 都出现过"自洽地虚高"（README §12）。
2. **α 必须对齐**：`ELF3_ALPHA` 是"固定"，续训要用 `ELF3_ALPHA_START`，两者别混。
3. **录像机位**：`diagnose_policy.py` 的录像相机是**跟拍**，看不出位移；要判断"整条斜"用
   `ELF3_VIEWER_ORIGIN=world` 的 play，或直接看 §4 的 `漂移比`。
4. **多 env 播种**：16 env 的地形类型（flat/easy/mid/hard 四列）是终身随机分配的，均值会被类型差异污染；
   B0~B3 至少跑 16 env，必要时按 `terrain_types` 分组再看。
5. **`--terrain-level` 会禁用课程**（脚本内部 `_disable_curriculum`），这是刻意的，别去掉。
6. **本轮不改 `train/mjlab_tasks/crab/*`**：一旦改了奖励，基线就不可比了。
