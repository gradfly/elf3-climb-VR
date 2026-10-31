# 侧身爬楼梯：X 向漂移量化执行清单 v2

> v1（上一版）的方法链里有 **3 处会让结论直接失效**的错误，本版已修。
> 配套代码已经落在仓库里（`scripts/diagnose_policy.py` + 新增 `scripts/analyze_trace.py`），
> **不需要再手工改任何 .py**。下面每一段都是可直接复制执行的命令。
>
> 分工不变：本轮只做量化，不改 `train/mjlab_tasks/crab/*`、不动 checkpoint。

---

## 0. 评审：v1 哪些能用、哪些不能用

### 0.1 成立的部分（保留）

| 结论 | 评价 |
|---|---|
| 先量化、再动奖励 | 对。三类漂移治法相反，盲改必白跑 |
| A 偏航耦合 / B 步态不对称 / C 扰动失能 的分类 | 对，对症的治法也对 |
| `--terrain-level` 会禁课程、play 会 `pop("push_robot")`、跟拍录像看不出位移 | 都是真的坑 |
| α 必须与 checkpoint 对齐（`ELF3_ALPHA`） | 对 |

### 0.2 致命 H1｜在 Play 任务上测漂移，量到的全是指令

证据（`train/mjlab_tasks/crab/env_cfg.py`，`if play:` 分支 L408–421）：

```python
cfg.episode_length_s = int(1e9)                       # 永不超时
cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.1)   # ← 允许前后走指令
```

再加 `env_cfg_residual.py:210`：

```python
twist.ranges.ang_vel_z = (0.0, 0.1 if play else 0.0)  # ← 允许转向指令
```

（这两处放宽是为了 viser 滑块 `min=0.1`；训练时是 `(0,0)` / `(0,0)`。）

于是 `--task Elf3-Crab-Residual-Play` 建出来的环境里，90% 的 env 被要求：

- 以平均 **+0.05 m/s** 向 +X 走（`rel_standing_envs=0.1`，其余在 `[0,0.1]` 均匀采样）
- 以平均 **+0.05 rad/s** 偏航
- 每 3~8 s 换一次指令（`resampling_time_range=(3.0, 8.0)`）

20 s 下来**期望净 Δx ≈ +1.0 m、偏航 ≈ 57°**。而 v1 的判据是 `dx_rate ≤ 0.01 m/s`、
`yaw ≤ 15°/20s` —— 在 v1 的测法下**永远不可能满足**。

更要紧的是：测到的"漂移"里指令成分和策略缺陷是混在一起的，**分不开**。
也就是说，v1 观察到的"X 向漂移"现象本身可能就是假的。**这一条不修，后面全白跑。**

> 修法：评估前把指令**结构性地**钉成 `(0, v_y, 0)` —— 不重采样、不抽站立 env、不转向。
> 已做成 `--freeze-command`。

### 0.3 致命 H2｜漂移比的量纲是错的

v1 §4：

```python
ratio = np.abs(dx[m]) / np.clip(vy[m], 1e-3, None)
```

`dx` 是米，`vy` 是 m/s，除出来是**秒**，不是 m/m。

还有第二层更隐蔽的：脚本里 `dx` 每回合归零（`x0` 在 reset 刷新），
而 `dy` 是**跨回合累计**的 `net_dy`。两个基准不同，摔得越多除得越小。

> 修法：新增 `dy_ep`（本回合内 y 位移），用 `|dx| / |dy_ep|`，同基准。

### 0.4 致命 H3｜判别量选错了

v1 用 `corr(Δdx, Δyaw)` 判 A 类。但偏航耦合的物理形式是 `vx ≈ -v · sin(yaw)`，
这是 **vx 与 yaw 的水平相关**；做一阶差分恰好把这种慢变共模消掉，剩下全是噪声。

用合成数据验一下（yaw 随机游走、`vx = -0.17·sin(yaw)`，即**完美耦合**）：

| 统计量 | 值 | 会得出什么结论 |
|---|---|---|
| `corr(Δdx, Δyaw)`（v1 用这个） | **−0.04** | "没有耦合" → 误判成 B |
| `corr(vx, yaw_rel)`（v2 用这个） | **−0.80** | 强耦合 → 正确判 A |

即 v1 的判据会把 A 类判成 B 类，**治疗方向正好相反**（A 是观测缺失，改奖励治不好）。
反过来，B 类（vx 恒定、yaw 只是噪声）在 v1 的 `corr` 上也是 ≈0，两类根本分不开。

> 修法：主判别量换成 `corr(vx, yaw_rel)`；`corr(Δdx,Δyaw)` 只作为参考打印。

### 0.5 会让数字对不上、但不会让结论反向的 4 处

| # | v1 做法 | 为什么不行 | v2 |
|---|---|---|---|
| 1 | 朝向误差 = `wrap(yaw)`，目标 0 | `reset_root_state_uniform` 的 yaw 是 **±0.15 rad（±8.6°）随机的**，等于把出生朝向算成漂移 | 记 `yaw − yaw0` |
| 2 | 冷启动屏蔽 `idx > 60` | 绝对步号。摔倒重开后新回合的前 60 步照样计入 | 按 `ep_t`（回合内步数） |
| 3 | 判据 `ep_len ≥ 15 s` | play 里 `episode_length_s=1e9` 永不超时；不摔就是 `episodes: 0 / 16`，表里空着 | 改判"摔倒次数 = 0" |
| 4 | B4 要算"恢复时间"，但没记推挤时刻 | 3~6 s 推一次、方向随机，事后无法把某段 dx 归到某次推上 | 记 `push` 标记（wrap 住事件函数） |

### 0.6 三处小问题

1. v1 §3「加 `--trace` 补丁」是**既成事实**（commit `7ccf5e6` 已加）。照 v1 再做一遍只会改坏。
2. v1 §7「判定 B → 加 `lin_vel_x_zero` + 走廊惩罚」：`lateral_drift`（`(x−x0)²`, −0.2）和
   `yaw_drift`（`(Δyaw/dt)²`, −1.5）**早就在奖励里了**。不是"新增"，是"现有项太弱"——
   0.2 m 偏移时 `(0.2)²×(−0.2)×0.02 = −0.00016/步`，一整个回合才 −0.003 分，等于没有。
3. v1 引的「README §12」不存在（README 只有 141 行）。结论本身对，引用失效。

---

## 1. 口径（v2 定义）

| 术语 | 定义 | 判据 |
|---|---|---|
| 漂移比 | 每回合内 `\|dx\| / \|dy_ep\|`（米/米） | ≤ 0.08 m/m |
| `dx_rate` | `\|dx\| / 时长`，**仅在冻结指令下有意义** | ≤ 0.01 m/s |
| 朝向误差 | `yaw − yaw0`（世界系，相对出生朝向） | ≤ 15° / 20 s |
| 主判别量 | `corr(vx, yaw_rel)` | > 0.5 → A；< 0.3 → B |
| `track_ratio` | `mean(vy) / cmd_y` | ≥ 0.70 |
| 摔倒 | episode 计数（play 不超时，所以只有摔） | = 0 |
| `sat` | 残差饱和比例 | ≤ 40% |

---

## 2. 前置（5 分钟，GPU 服务器仓库根目录）

变量一次赋好，后面全部复用。

```bash
cd ~/elf3
TASK=Elf3-Crab-Residual-Play

# 最新 checkpoint 及其 run 目录（按修改时间倒序取第一个）
CKPT=$(ls -t logs/rsl_rl/elf3_crab/*/model_*.pt | head -1)
RUN=$(dirname "$CKPT")

# 该 run 的 ref_alpha（取最后一条；判断 α 课程是否已满幅）
ALPHA=$(.venv/bin/python - "$RUN" <<'PY'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator(sys.argv[1]); ea.Reload()
tag = "Metrics/ref_alpha"
if tag not in ea.Tags()["scalars"]:
    print("nan")
else:
    print(f"{[e.value for e in ea.Scalars(tag)][-1]:.3f}")
PY
)

# 任务是否注册（输出为空 = venv 里那份任务包是旧的，先跑 bash train/install_task.sh）
.venv/bin/python -c "import mjlab.tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Crab' in t])"

# 变量落定 + 门槛检查
printf 'TASK=%s\nRUN=%s\nCKPT=%s\nALPHA=%s\n' "$TASK" "$RUN" "$CKPT" "$ALPHA"
[ -n "$CKPT" ] || { echo "[ERR] 没找到 checkpoint"; exit 1; }
if awk -v a="$ALPHA" 'BEGIN{exit !(a<0.95)}'; then
  echo "[warn] ALPHA<0.95：评估必须带 ELF3_ALPHA=$ALPHA，否则测的是「满幅参考 + 半幅残差」"
fi
export MUJOCO_GL=egl
export ELF3_ALPHA="$ALPHA"
```

`ALPHA=nan` 说明这个 run 没写 `Metrics/ref_alpha`（`ref_probe` 没跑），
那就退回 `ELF3_ALPHA=1.0` 并在结论里注明"α 未核实"。

---

## 3. 预检 P0：`--terrain-level` 到底有没有改难度（必做，约 1 分钟）

v1 拿 `--terrain-level 3/5/7` 当"简单/标准/难"三档。但 play 里
`terrain_generator.curriculum = False`（`env_cfg.py:421`），**行是否仍是难度梯度取决于 mjlab 版本**。
先花一分钟验证，否则 B0/B1/B2 可能量的是同一批地形。

**先确认服务器上的代码是新的**——旧版脚本没有 `--freeze-command`，argparse 会直接报错退出：

```bash
cd ~/elf3
git log --oneline -1
git pull --ff-only                     # 没更新过就拉一次
.venv/bin/python scripts/diagnose_policy.py --help | grep -q freeze-command \
  && echo "[OK] 脚本是新的" \
  || echo "[ERR] 这份脚本没有 --freeze-command：仓库没更新，先 git pull"
```

> `TASK` / `CKPT` 是**每个 shell 会话独立**的。新开窗口要重跑 §2，或直接用下面的兜底赋值。

```bash
mkdir -p /tmp/elf3_drift
[ -n "$TASK" ] || TASK=Elf3-Crab-Residual-Play
[ -n "$CKPT" ] || CKPT=$(ls -t logs/rsl_rl/elf3_crab/*/model_*.pt | head -1)
printf 'TASK=%s\nCKPT=%s\n' "$TASK" "$CKPT"

for lvl in 3 5 7; do
  echo "=== terrain-level $lvl ==="
  .venv/bin/python scripts/diagnose_policy.py \
    --task "$TASK" --checkpoint "$CKPT" \
    --num-envs 32 --steps 400 --terrain-level $lvl --freeze-command 0.17 \
    > /tmp/elf3_drift/pre_p0_$lvl.log 2>&1
  echo "exit=$?"
  grep -E "rise 列|terrain_levels|指令 cmd" /tmp/elf3_drift/pre_p0_$lvl.log \
    || { echo "[没有匹配行，日志尾部如下]"; tail -15 /tmp/elf3_drift/pre_p0_$lvl.log; }
done
```

> **为什么先落盘再 grep**：直接 `2>&1 | grep` 会把 traceback 和 argparse 报错一起吃掉，
> 表现就是"每档只打印了标题、一行数据都没有"。落盘后失败时会自动把日志尾部打出来。

**判读**：看**每一列自己**的 `rise 列X` 是否随 level 单调上升。

```
rise 列0 n=8 : 0.010 m ...      ← flat 列，范围只有 0~2 cm，基本不动是正常的
rise 列1 n=8 : 0.041 m ...      ← easy 列
rise 列2 n=8 : 0.087 m ...      ← mid 列
rise 列3 n=8 : 0.126 m ...      ← hard 列，8~22 cm，梯度最明显
```

- `rise 列3` 在 level 3/5/7 上单调上升（约 0.12 → 0.16 → 0.19）→ **P0 通过**，继续 §4。
- 某一列三档几乎相同 → `--terrain-level` 在本版本上是空操作，**B0~B2 无意义**。
  改走训练配置（`terrain_generator.curriculum=True`，行确实是难度梯度）：
  **在 §4/§5 的每条命令后面加 `--train-env`**，其余完全不变。

> **别用全 env 的 `rise_est.mean()` 判断**：`terrain_types` 是创建时随机分配、终身不变的，
> 而各列的高度范围差一个量级（flat 0~2 cm、hard 8~22 cm）。32 个 env 里多抽到几个 hard，
> 均值就能差出 ±0.04 —— 比 level 3→7 本该有的梯度还大。8 个 env 时这个噪声直接盖过信号，
> 会得出"level 无效"的错误结论。**必须按列比。**

---

## 4. 基线 B0~B3（一条命令跑完，每个约 20 s 仿真）

```bash
mkdir -p /tmp/elf3_drift
CMD=0.17                    # 训练指令区间 (0.14, 0.20) 的中点

for spec in "b1 3" "b0 5" "b2 7"; do
  name=${spec%% *}; lvl=${spec##* }
  echo "########## $name  (terrain-level $lvl)"
  .venv/bin/python scripts/diagnose_policy.py \
    --task "$TASK" --checkpoint "$CKPT" \
    --num-envs 16 --steps 1000 --terrain-level $lvl \
    --freeze-command $CMD \
    --trace /tmp/elf3_drift/trace_$name.npz \
    2>&1 | tee /tmp/elf3_drift/$name.log | tail -32
done
```

**先看这一行**，它决定这批数据能不能用：

```
指令 cmd       : x +0.000 / y +0.170 / wz +0.000  ✅ 已冻结（Δx 不含指令成分）
```

出现 `❌ 没冻住` 就停下先修这条，否则所有 Δx 都是指令驱动的。

B3 只录一段看形态（跟拍机位，**看不出位移**，只看"整条斜"还是"每步蛇形"）：

```bash
.venv/bin/python scripts/diagnose_policy.py \
  --task "$TASK" --checkpoint "$CKPT" --num-envs 1 --steps 1000 \
  --terrain-level 5 --freeze-command $CMD \
  --record /tmp/elf3_drift/b3.mp4 --record-elev -35 --record-dist 3.5
```

---

## 5. B4 扰动恢复（区分"走歪"和"被打歪"）

```bash
.venv/bin/python scripts/diagnose_policy.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num-envs 16 --steps 1000 --terrain-level 5 \
  --freeze-command $CMD --push \
  --trace /tmp/elf3_drift/trace_b4.npz \
  2>&1 | tee /tmp/elf3_drift/b4.log | tail -32
```

`--push` 把 play 里被 pop 掉的 `push_robot` 补回来（3~6 s 一次、x/y ±0.2 m/s、yaw ±0.1 rad/s，
与 `_retune_safety` 参数一致），并**记录每次推挤发生在哪一步**——没有这个时间戳，
"恢复时间"只能靠猜。

---

## 6. 分析（判定 A / B / C）

```bash
# 三档基线
for n in b0 b1 b2; do
  echo "########################## $n"
  .venv/bin/python scripts/analyze_trace.py /tmp/elf3_drift/trace_$n.npz
done

# 扰动档（多一份恢复时间报告）
echo "########################## b4"
.venv/bin/python scripts/analyze_trace.py /tmp/elf3_drift/trace_b4.npz --push
```

要抄进 §7 表格的几个数：

| 输出行 | 含义 | 判据 |
|---|---|---|
| `漂移比 \|dx\|/Δy` | 每爬 1 m 横偏多少 m | ≤ 0.08 |
| `安静段 vx` | 朝向没歪时的横向速度（分离步态项） | \|vx\| < 0.02 |
| `c(vx, yaw_rel)` | **主判别量** | >0.5 → A，<0.3 → B |
| `c(Δdx, Δyaw)` | 参考量（v1 用过，差分会消掉共模） | 别看它下结论 |
| `c(速度方向,朝向)` | ≈1 说明策略在 heading 系里追指令 | 佐证 A |
| `[push] 耗时 / Δx 同号占比` | 恢复能力 | >2 s 或同号 >75% → 没有横向纠偏 |

---

## 7. 基线表（执行后填写）

| run | terrain | 漂移比 (m/m) | 安静段 vx | c(vx,yaw) | track_ratio | 摔倒 | sat | 结论 |
|---|---|---|---|---|---|---|---|---|
| B0 | 5 | | | | | | | |
| B1 | 3 | | | | | | | |
| B2 | 7 | | | | | | | |
| B3 | 5（录像） | — | — | — | — | | — | 整条斜 / 每步蛇形 |
| B4 | 5 + push | | | | | | | 恢复耗时 / 是否单向累积 |

交付物：上表 + 一句话判定（A / B / C）+ 一段录像。

---

## 8. 决策门（修正版）

| 结果 | 下一步 |
|---|---|
| 漂移比 ≤ 0.08 且 `c(vx,yaw)` < 0.3 | 已达标，**不改**；把漂移比加进训练日志防回归 |
| 判定 A（`c` > 0.5，安静段 `\|vx\|` < 0.02） | 补 heading 观测（`heading_sincos`），**必须重训**；改奖励治不好 |
| 判定 B（`c` < 0.3，安静段 `\|vx\|` ≥ 0.02 且同号 > 0.7） | 先把 `lateral_drift` 从 `(x−x0)²` 改成一次项（现在每回合只 −0.003 分，等于没有），或加 `lin_vel_x_zero`；可续训 1000~1500 iter |
| 判定 C（B4 恢复 > 2 s，或 Δx 同号 > 75%） | 先做扰动课程 + 恢复奖励，最后再补观测 |
| `track_ratio` < 0.65 | 策略本身还没走稳，**漂移不是首要矛盾**，先修推进/步态 |

> v1 的一个误判要记住：判定 B 时"加 `lin_vel_x_zero`"不是新增项——`lateral_drift` 已经在跑，
> 问题是二次项在 0.2 m 处只有 −0.00016/步。改之前先确认这一项的实际量级，别再加一条重复惩罚。

---

## 9. 坑（v1 的 + 新增的）

1. **别用训练日志的 `Mean reward` 判断好坏**：`forward_progress_y` 与 `yaw_drift` 都出现过自洽地虚高。
2. **α 必须对齐**：`ELF3_ALPHA` 是"固定"，续训要用 `ELF3_ALPHA_START`，别混。
3. **录像机位**：`diagnose_policy.py --record` 是跟拍，机器人永远在画面中心，看不出位移。
   判"整条斜"用 `ELF3_VIEWER_ORIGIN=world` 的 play，或直接看漂移比。
4. **16 个 env 的列不是均匀的**：`terrain_types`（flat/easy/mid/hard）创建时随机分配、终身不变。
   脚本收尾已按 type 分组打印，组间差异大时**别只看总均值**。
5. **`--terrain-level` 会禁用课程**（`_disable_curriculum`），这是刻意的，别去掉。
6. **play 里每次 reset 都会 `randomize_terrain`**：摔倒重开后地形换了，跨回合的 dx 不可比。
   摔得多的那档，漂移比可信度也低——先看摔倒次数。
7. **`--dump-scan` 不要和 `--trace` 同一次跑**：它内部会多走一步，trace 第 0 步会错位。
8. **本轮不改 `train/mjlab_tasks/crab/*`**：改了奖励，基线就不可比了。

---

## 附：本版改了哪些代码（审阅用）

| 文件 | 改动 |
|---|---|
| `scripts/diagnose_policy.py` | 新增 `--freeze-command`：三重冻结（ranges 钉常数 / `resampling_time_range` 置大 / 每步覆写指令张量）<br>新增 `--train-env`：P0 不通过时的退路<br>`--trace` 增记 `dy_ep` / `yaw_rel` / `ep_t` / `push`<br>收尾漂移比改成同基准，并按 `terrain_types` 分组 |
| `scripts/analyze_trace.py` | 新增。判定 A/B/C；量纲修正；主判别量换 `corr(vx, yaw_rel)`；冷启动按回合屏蔽；推挤恢复分析 |
