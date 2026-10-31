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
| **B 类硬判据** | `mean \|v_lon\|`：机身系纵向速度（沿机身 +X），同号率 | ≥0.02 m/s 且同号 ≥0.7 → B |
| A 类判据 | `corr(vx, yaw_rel)` | > 0.5 且 `\|v_lon\|` < 0.02 → A |
| `v_lat` | 机身系侧向速度 | 应 ≈ `cmd_y` |
| `track_ratio` | `mean(vy) / cmd_y` | ≥ 0.70 |
| 摔倒 | episode 计数（play 不超时，所以只有摔） | = 0 |
| `sat` | 残差饱和比例 | ≤ 40% |

---

## 2. 前置（5 分钟，GPU 服务器仓库根目录）

变量一次赋好，后面全部复用。**`ALPHA` 是硬性要求**，理由见 §2.1。

```bash
cd ~/elf3
export MUJOCO_GL=egl
TASK=Elf3-Crab-Residual-Play

# 最新 checkpoint（按修改时间倒序取第一个）
CKPT=$(ls -t logs/rsl_rl/elf3_crab/*/model_*.pt | head -1)
[ -n "$CKPT" ] || { echo "[ERR] 没找到 checkpoint"; return 1 2>/dev/null || exit 1; }

# 该 run 训练时的参考步态幅度 α（脚本把数字打到 stdout，说明走 stderr）
ALPHA=$( .venv/bin/python scripts/read_alpha.py --ckpt "$CKPT" )

# 任务是否注册（输出为空 = venv 里那份任务包是旧的，先跑 bash train/install_task.sh）
.venv/bin/python -c "import mjlab.tasks; from mjlab.tasks.registry import list_tasks; print([t for t in list_tasks() if 'Crab' in t])"

printf 'TASK=%s\nCKPT=%s\nALPHA=%s\n' "$TASK" "$CKPT" "$ALPHA"
```

### 2.1 为什么 α 必须显式给（这是最贵的静默失败）

`env_cfg_residual._alpha_fixed` 的取值规则：

| 情形 | α |
|---|---|
| `play=True`（默认） | **1.0** |
| `play=False`（即 `--train-env`）且没给 `ELF3_ALPHA` | **走课程，从 `alpha_start=0` 开始 ramp** |

课程总长 1500 iter × 24 步，而动作项每次都是新建实例、`_step_i` 从 0 起。所以

```
--train-env 跑 1000 步  ->  α = 1000 / 36000 = 0.028
```

α=0.028 时参考步态 ≈ **站姿**，残差（限幅 ±0.55 rad）要独自憋出整个步态。实测表现：
**60% 维度顶死限幅、每 2.7 s 摔一次、20 s 自转 1.7 圈、跟踪比 2.6~3.5**。
看起来像"策略很烂"，其实所有数字都和策略水平无关。

所以：**所有命令都带 `--alpha "$ALPHA"`**，脚本会在建完环境后立刻打印 `ref_alpha`，
小于 0.9 会打一整块 `!!!!` 警告。

> `ALPHA` 是**每个 shell 会话独立**的。新开窗口必须重跑本节，或命令里带兜底：
> `[ -n "$ALPHA" ] || ALPHA=$(.venv/bin/python scripts/read_alpha.py)`。

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
[ -n "$ALPHA" ] || ALPHA=$(.venv/bin/python scripts/read_alpha.py --ckpt "$CKPT")
printf 'TASK=%s\nCKPT=%s\nALPHA=%s\n' "$TASK" "$CKPT" "$ALPHA"

for lvl in 3 5 7; do
  echo "=== terrain-level $lvl ==="
  .venv/bin/python scripts/diagnose_policy.py \
    --task "$TASK" --checkpoint "$CKPT" \
    --num-envs 32 --steps 400 --terrain-level $lvl --freeze-command 0.17 \
    --alpha "$ALPHA" \
    > /tmp/elf3_drift/pre_p0_$lvl.log 2>&1
  echo "exit=$?"
  grep -E "rise 列|terrain_levels|指令 cmd|ref_alpha" /tmp/elf3_drift/pre_p0_$lvl.log \
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
- 某一列三档几乎相同 → `--terrain-level` 在本版本上是空操作，见 3.1。

### 3.1 实测结论（2026-10-31，32 env）：play 下 P0 **不通过**，改用 `--train-env`

三档实测（`--num-envs 32 --steps 400`，每格是该列 `rise` 均值，单位 m）：

| level | 列0 | 列1 | 列2 | 列3 | `rise=0.00` 占比 |
|---|---|---|---|---|---|
| 3 | 0.085 | 0.046 | 0.075 | 0.073 | 25%（8/32）|
| 5 | 0.056 | 0.091 | 0.063 | 0.089 | 19%（6/32）|
| 7 | 0.053 | 0.037 | 0.021 | 0.056 | 38%（12/32）|

三条否决理由：

1. **四列没有一列随 level 单调上升**，level 7 反而最平（列2：0.075→0.063→0.021）。
2. **"列0"出现 0.19 / 0.22**，而配置里 flat 列的上限只有 2 cm。这在
   `terrain_generator.curriculum=True` 的"列=类型"映射下不可能出现 —— 说明 play 里
   `curriculum=False`（`env_cfg.py:421`）已经把列含义打乱了：每格地形类型独立抽样，
   列均值是随机混合，跨 level 不可比。
3. **`rise=0.00` 占比随 level 上升**，level 7 有 12/32 个 env 根本没检测到上升边
   （摔/重置太多，一直在碗底）。

还有个更根本的问题：play 每次 reset 都触发 `randomize_terrain`（`env_cfg.py:417`），
**地形块本身会变**。就算 level 生效，跨回合的 dx 也不可比。

**所以 §4/§5 一律加 `--train-env`**：训练配置里 `curriculum=True`（行=难度、列=类型，
与 `terrain_cfg.py` 注释一致），且没有 `randomize_terrain`。
代价是 episode 固定 20 s —— 脚本已记录 `timeout`，`analyze_trace.py` 会把超时从"摔倒"里
剔除，不会把正常跑完当成摔。

> **别用全 env 的 `rise_est.mean()` 判断**：`terrain_types` 是创建时随机分配、终身不变的，
> 而各列的高度范围差一个量级（flat 0~2 cm、hard 8~22 cm）。32 个 env 里多抽到几个 hard，
> 均值就能差出 ±0.04 —— 比 level 3→7 本该有的梯度还大。8 个 env 时这个噪声直接盖过信号，
> 会得出"level 无效"的错误结论。**必须按列比。**

---

## 4. 基线 B0~B3（一条命令跑完，每个约 20 s 仿真）

```bash
mkdir -p /tmp/elf3_drift
CMD=0.17                    # 训练指令区间 (0.14, 0.20) 的中点
[ -n "$ALPHA" ] || ALPHA=$(.venv/bin/python scripts/read_alpha.py --ckpt "$CKPT")

for spec in "b1 3" "b0 5" "b2 7"; do
  name=${spec%% *}; lvl=${spec##* }
  echo "########## $name  (terrain-level $lvl, alpha $ALPHA)"
  .venv/bin/python scripts/diagnose_policy.py \
    --task "$TASK" --checkpoint "$CKPT" \
    --num-envs 16 --steps 1000 --terrain-level $lvl --train-env \
    --freeze-command $CMD --alpha "$ALPHA" \
    --trace /tmp/elf3_drift/trace_$name.npz \
    2>&1 | tee /tmp/elf3_drift/$name.log | tail -32
done
```

> `--train-env` 不是可选项：play 下 P0 已判不通过（§3.1）。
> **`--alpha` 更不是可选项**：不给的话 α 会从 0 开始 ramp，跑完只到 0.028（§2.1）。

**先看这一行**，它决定这批数据能不能用：

```
指令 cmd       : x +0.000 / y +0.170 / wz +0.000  ✅ 已冻结（Δx 不含指令成分）
```

出现 `❌ 没冻住` 就停下先修这条，否则所有 Δx 都是指令驱动的。

B3 只录一段看形态（跟拍机位，**看不出位移**，只看"整条斜"还是"每步蛇形"）：

```bash
.venv/bin/python scripts/diagnose_policy.py \
  --task "$TASK" --checkpoint "$CKPT" --num-envs 1 --steps 1000 \
  --terrain-level 5 --freeze-command $CMD --train-env --alpha "$ALPHA" \
  --record /tmp/elf3_drift/b3.mp4 --record-elev -35 --record-dist 3.5
```

---

## 5. B4 扰动恢复（区分"走歪"和"被打歪"）

```bash
[ -n "$ALPHA" ] || ALPHA=$(.venv/bin/python scripts/read_alpha.py --ckpt "$CKPT")
.venv/bin/python scripts/diagnose_policy.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num-envs 16 --steps 1000 --terrain-level 5 --train-env \
  --freeze-command $CMD --alpha "$ALPHA" --push \
  --trace /tmp/elf3_drift/trace_b4.npz \
  2>&1 | tee /tmp/elf3_drift/b4.log | tail -32
```

`--push` 把扰动补回来（3~6 s 一次、x/y ±0.2 m/s、yaw ±0.1 rad/s，与 `_retune_safety` 参数一致），
并**记录每次推挤发生在哪一步**——没有这个时间戳，"恢复时间"只能靠猜。

> `--train-env` 下 `push_robot` 本来就在配置里（`_retune_safety` 已调成同样的 3~6 s / ±0.2），
> 不加 `--push` 时脚本会主动 pop 掉；加了 `--push` 则换成带记录的包装版本。二者不冲突。

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
| `机身系 v_lon` | 沿机身 +X（前后）的速度，**与朝向无关** | \|v_lon\| < 0.02；≥0.02 且同号 ≥0.7 → B |
| `机身系 v_lat` | 沿机身 +Y（侧向）的速度 | 应 ≈ `cmd_y` |
| `c(vx, yaw_rel)` | 横向速度与朝向的相关 | >0.5 且 \|v_lon\| 小 → A |
| `安静段 vx` | 朝向没歪时的横向速度（辅助，样本可能很少） | \|vx\| < 0.02 |
| `c(Δdx, Δyaw)` | 参考量（v1 用过，差分会消掉共模） | 别看它下结论 |
| `c(速度方向,朝向)` | ≈1 说明策略在 heading 系里追指令 | 佐证 A |
| `[push] 耗时 / Δx 同号占比` | 恢复能力 | >2 s 或同号 >75% → 没有横向纠偏 |

> 为什么用 `v_lon` 而不是"安静段 vx"：策略持续偏航时，`|yaw_rel|<3°` 的样本很快就没有了
> （实测 16 个 env 里只有 3~5 个还剩样本，且都集中在开局 1 s 内）。`v_lon` 是机身系分解，
> 机器人转着走也能量，是 B 类唯一不受偏航污染的判据。

---

## 7. 基线表（执行后填写）

| run | terrain | 漂移比 (m/m) | 安静段 vx | c(vx,yaw) | track_ratio | 摔/超 | sat | 结论 |
|---|---|---|---|---|---|---|---|---|
| B0 | 5 | | | | | | | |
| B1 | 3 | | | | | | | |
| B2 | 7 | | | | | | | |
| B3 | 5（录像） | — | — | — | — | | — | 整条斜 / 每步蛇形 |
| B4 | 5 + push | | | | | | | 恢复耗时 / 是否单向累积 |

> 全部用 `--train-env`（§3.1）。`摔/超` 里超时是正常跑完，不是失败。

交付物：上表 + 一句话判定（A / B / C）+ 一段录像。

---

## 8. 决策门（修正版）

| 结果 | 下一步 |
|---|---|
| 漂移比 ≤ 0.08 且 `\|v_lon\|` < 0.02 且 `c` < 0.3 | 已达标，**不改**；把漂移比加进训练日志防回归 |
| 判定 B（`\|v_lon\|` ≥ 0.02 且同号 ≥ 0.7） | 先把 `lateral_drift` 从 `(x−x0)²` 改成一次项（现在每回合只 −0.003 分，等于没有），或加 `lin_vel_x_zero`；可续训 1000~1500 iter |
| 判定 A（`c` > 0.5 且 `\|v_lon\|` < 0.02） | 补 heading 观测（`heading_sincos`），**必须重训**；改奖励治不好 |
| 判定 C（B4 恢复 > 2 s，或 Δx 同号 > 75%） | 先做扰动课程 + 恢复奖励，最后再补观测 |
| `track_ratio` < 0.65 或 > 1.4 | 策略还没走稳（走不动 / 超速冲），**漂移不是首要矛盾**，先修推进/步态 |

> B 排在 A 前面：`v_lon` 是直接测量、与偏航无关；`c(vx,yaw)` 在策略持续转圈时会被周期性
> yaw 稀释，容易漏判（实测 16 个 env 只有 4~6 个超过 0.5）。两个都成立时按 B 先治——
> 步态有纵向偏置时，光补 heading 观测等于给一个歪着走的策略加导航。

> v1 的一个误判要记住：判定 B 时"加 `lin_vel_x_zero`"不是新增项——`lateral_drift` 已经在跑，
> 问题是二次项在 0.2 m 处只有 −0.00016/步。改之前先确认这一项的实际量级，别再加一条重复惩罚。

---

## 9. 坑（v1 的 + 新增的）

1. **别用训练日志的 `Mean reward` 判断好坏**：`forward_progress_y` 与 `yaw_drift` 都出现过自洽地虚高。
2. **α 必须钉死，而且必须传给脚本**：`--train-env` 下不给 `--alpha`，α 会从 0 ramp，
   1000 步只到 **0.028** —— 参考步态退化成站姿，实测「60% 维度顶死限幅、每 2.7 s 摔一次、
   20 s 自转 1.7 圈」，看着像策略烂，其实是参数错。脚本现在会在建完环境后立刻打印
   `ref_alpha`，< 0.9 会打 `!!!!` 警告。`ELF3_ALPHA` 是"固定"，续训要用 `ELF3_ALPHA_START`，别混。
3. **录像机位**：`diagnose_policy.py --record` 是跟拍，机器人永远在画面中心，看不出位移。
   判"整条斜"用 `ELF3_VIEWER_ORIGIN=world` 的 play，或直接看漂移比。
4. **16 个 env 的列不是均匀的**：`terrain_types`（flat/easy/mid/hard）创建时随机分配、终身不变，
   各列高度范围差一个量级。**别拿全 env 的 `rise_est.mean()` 当难度判据** —— 它主要由
   "这次抽到几个 hard" 决定，抖动 ±0.04 就盖过 level 3→7 的梯度。必须按列比。
5. **`--terrain-level` 会禁用课程**（`_disable_curriculum`），这是刻意的，别去掉。
6. **play 里每次 reset 都会 `randomize_terrain`**：地形块本身会变，跨回合的 dx 不可比。
   这也是 §4 改用 `--train-env` 的原因之一。
7. **`--dump-scan` 不要和 `--trace` 同一次跑**：它内部会多走一步，trace 第 0 步会错位。
8. **本轮不改 `train/mjlab_tasks/crab/*`**：改了奖励，基线就不可比了。
9. **`--train-env` 下 episode 固定 20 s**，跑满会 `timeout` 而不是摔。脚本已记录 `timeout` 标记，
   `analyze_trace.py` 的"摔倒/超时"是分开数的；自己另写脚本时别再拿 `dones` 当摔倒。

---

## 10. 判定结果（2026-10-31，α 修正后重测）

三轮基线（16 env × 1000 步，`--train-env --alpha 1.000`）实测：

| 指标 | b1 (L3) | b0 (L5) | b2 (L7) | 判据 |
|---|---|---|---|---|
| 终止 | 全超时，0 摔 | 同 | 同 | — |
| 跟踪比 | 1.02 | 1.17 | 1.51 | 0.65~1.4 |
| 漂移比 | 0.380 | 0.222 | 0.266 | ≤ 0.08 |
| **`v_lon` 偏置** | **+0.125** | **+0.098** | **+0.109** | \|·\| < 0.02 |
| `v_lat` | 0.152 | 0.174 | 0.230 | ≈ cmd 0.17 |
| `c(vx, yaw)` | −0.38 | −0.20 | −0.40 | >0.5 才算 A |

**判定：B（步态/残差不对称）。** 依据：机身系纵向偏置三档一致（+0.10~0.13 m/s，
是侧向指令的 60%），**与地形难度无关** —— 这是步态自带的，不是台阶造成的；
而 A 的证据（`c`）三档都不到 0.5。

> 逐帧 `|v_lon|` 有 0.18，那是参考步态本身的"冲-停-退"摆动（v_y 峰值 1.618、26% 帧倒退），
> **不能拿来判**。判据用 EMA(1.8 s) 之后的带符号均值。

### 10.1 已做的奖励改动

`train/mjlab_tasks/crab/mdp_extra.py` 新增 `lateral_bias`，`env_cfg_residual` 挂上去：

```python
r["lateral_bias"] = RewardTermCfg(
    func=X.lateral_bias, weight=-80.0, params={"tau": 1.8, "deadzone": 0.03},
)
```

标定（回合 ≈ 600 步，正奖励约 50~75 分）：

| 偏置 | 每步 | 每回合 | 占比 |
|---|---|---|---|
| 0.03 | 0 | 0 | 死区内，不罚 |
| 0.10 | −0.39 | −4.7 | ~7%，形成梯度不主导 |
| 0.20 | −2.31 | −27.7 | ~40%，明确压制 |

为什么不用现有 `lateral_drift`：`(x−x0)²` 在 0.2 m 处每步只 −0.00016 分（回合 −0.003），
量级等于没罚；而且它罚"已经漂出去的位移"，梯度在起点最弱。

### 10.2 验证方法

续训 1000~1500 iter（`ELF3_ALPHA_START=<当前α>`），再跑一遍 §4 的同一套基线，看两个数：

- `v_lon` 偏置应从 +0.10 降到 <0.03；
- 漂移比从 0.22~0.38 降到 ≤0.08。

若偏置降了但漂移比没降 → 剩下的来自偏航（`yaw 累计 219~231°`，判据 15°），转去治 A。

### 10.3 顺带发现：一部分 env 根本爬不上第一级台阶

b0 的 env3~7（`terrain_types=1`）Δy 全部卡在 **0.81~0.89 m**，`v_lon≈0`、`末|dx|` 只有 0.03~0.19。
`easy_stairs` 的 `platform_width=1.2` → 碗底半宽 **0.80 m**，正好是第一级台阶边缘。
b1 同样（0.85/0.81/0.89/0.84/0.84）。这些 env 站了 19 s 没上去。

这是**比漂移更基础的问题**：漂移比 0.22~0.38 主要是"能走"的那些 env 贡献的，
而这一列 env 的指标是"没动"（漂移比小反而是假象）。修 B 之前或之后，都要单独确认
抬脚高度是否够（参考 `Metrics/peak_height_mean` 与 `feet_swing_height` 的标定）。

---

## 附：本版改了哪些代码（审阅用）

| 文件 | 改动 |
|---|---|
| `scripts/diagnose_policy.py` | 新增 `--freeze-command`：三重冻结（ranges 钉常数 / `resampling_time_range` 置大 / 每步覆写指令张量）<br>新增 `--train-env`：play 的地形语义不可用时的主路径<br>新增 `--alpha`：写 `ELF3_ALPHA`，并在建完环境后立刻校验 `ref_alpha`（<0.9 打 `!!!!` 警告）<br>`--trace` 增记 `dy_ep` / `yaw_rel` / `ep_t` / `push` / `timeout`<br>收尾漂移比改同基准；`rise/tread` 按 `terrain_types` 分列打印 |
| `scripts/read_alpha.py` | 新增。从 run 的 tensorboard 读 `Metrics/ref_alpha` 末值，stdout 只给数字 |
| `scripts/analyze_trace.py` | 新增。判定 A/B/C；量纲修正；主判别量换 `corr(vx, yaw_rel)`；冷启动按回合屏蔽；摔倒/超时分开数；推挤恢复分析；**加机身系 `v_lon` 偏置作 B 类硬判据** |
| `train/mjlab_tasks/crab/mdp_extra.py` | 新增 `lateral_bias`：EMA(1.8 s) 后的机身系纵向偏置，死区 0.03 后平方 |
| `train/mjlab_tasks/crab/env_cfg_residual.py` | 挂上 `r["lateral_bias"]`，weight −80（标定见 §10.1） |
