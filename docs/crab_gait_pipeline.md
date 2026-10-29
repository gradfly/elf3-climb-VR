# 螃蟹步横移上楼 —— 从随机台阶到网页回放

> 本文说明网页里那段「脚本步态 · 螃蟹步横移上楼」动画到底由哪些文件构成、由谁生成、
> 每一步做了什么。看完应当能独立复现整条链路，也能在数据对不上时定位到具体环节。
>
> **一句话**：网页播放的不是实时仿真，而是一段**离线解算好的关节轨迹**。
> `gen_crab_stairs.py` 随机造一套楼梯 → `crab_climb.py` 用四相位准静态控制器解出
> 每 20 ms 一帧的 `qpos` → `bake_crab.py` 烘焙成网页资源 → 浏览器按时间插值回放。
> 全程**没有策略网络**，也**不跑物理步进**（只有 `mj_kinematics` 正运动学）。
>
> **与其它文档的关系**：`docs/crab_rl_training.md` 讲的是把这条链路升级成闭环策略的
> 训练设计（脚本步态只作为 `q_ref` 参考）；本文只讲现有的**开环脚本步态**怎么产出
> 网页动画。网页端整体说明见 `web/README.md`。

---

## 1. 全链路一览

```
                        ┌──────────────────────────────────────┐
 ① 造地形               │ gen_crab_stairs.py  (--seed 42)      │
    （随机台阶）         │ 级高 0.12~0.25 m / 进深 0.20~0.40 m  │
                        └───────────────┬──────────────────────┘
                                        │ crab_stairs_obstacles.xml（geom）
                                        │ crab_stairs.json（真值表：tilt/off）
                                        │ crab_stairs.xml（include 入口）
                                        │ elf3_crab.xml（换地形 include）
                        ┌───────────────▼──────────────────────┐
 ② 解轨迹               │ crab_climb.py + crab_ik.py           │
    （四相位准静态 IK）   │ LOAD → SWING → LAND → SETTLE         │
                        │ 闭式解种子 + DLS 精修，只调 mj_kinematics│
                        └───────────────┬──────────────────────┘
                                        │ crab_climb_traj.npz
                                        │   time[] / qpos[] / dt=0.02
                        ┌───────────────▼──────────────────────┐
 ③ 烘焙                 │ web/tools/bake_crab.py               │
                        │ 抽 crab_* geom → 场景片段             │
                        │ nq 38→36 按关节名重排                 │
                        │ float32-le → .bin + .json + manifest │
                        └───────────────┬──────────────────────┘
                                        │ motion_crab.bin / .json
                                        │ scene_crab.xml
                        ┌───────────────▼──────────────────────┐
 ④ 回放                 │ web/js/motion.js  (MotionPlayer)     │
                        │ 线性插值 + 四元数 slerp               │
                        │ → data.qpos → mj_kinematics → three  │
                        └──────────────────────────────────────┘
```

一条命令跑完 ①②③：

```bash
python web/tools/bake_crab.py            # 约 20 s
```

单独跑某一段、或只重烘焙已有轨迹：

```bash
python web/tools/bake_crab.py --skip-gen             # 复用已有地形，只重解轨迹 + 烘焙
python web/tools/bake_crab.py --skip-gen --skip-plan # 只重烘焙（改过打包逻辑时用）
python web/tools/bake_crab.py --traj some.npz        # 用指定的轨迹文件烘焙
```

---

## 2. 产物清单

| 文件 | 大小 | 内容 | 生成者 |
|---|---|---|---|
| `sim/.../terrains/universal_test/crab_stairs_obstacles.xml` | 2 KB | 12 个 `crab_*` box geom + 起终点 site | `gen_crab_stairs.py` |
| `sim/.../terrains/universal_test/crab_stairs.json` | 4 KB | 真值表：13 个落脚面的 `y0/y1/top/rise/depth/off/tilt` | `gen_crab_stairs.py` |
| `sim/.../terrains/universal_test/crab_stairs.xml` | 243 B | include 入口（assets + base + 台阶） | `gen_crab_stairs.py --write-entry` |
| `sim/.../mujoco_simulation/elf3_crab.xml` | — | 复制 `elf3.xml` 只把地形 include 换掉 | `gen_crab_stairs.py --write-scene` |
| `sim/.../mujoco_simulation/crab_climb_traj.npz` | ~200 KB | `time[]` / `qpos[nframes,38]` / `dt` | `crab_climb.py` |
| `web/assets/scene_crab.xml` | 2 KB | 台阶几何片段，被 `scene_bx.xml` include | `bake_crab.py` |
| `web/assets/motion_crab.bin` | 185,040 B | `float32-le` 行优先 `(1285, 36)` | `bake_crab.py` |
| `web/assets/motion_crab.json` | 1.7 KB | 元数据 + 统计（HUD 用） | `bake_crab.py` |
| `web/assets/manifest.json` | 3 KB | 资源清单（含 `scene_crab.xml`） | `bake_crab.py` |

> 尺寸关系可以直接对账：`1285 × 36 × 4 = 185,040`。对不上就是帧数或 `nq` 变了。

---

## 3. 第一步：随机台阶地形（`gen_crab_stairs.py`）

### 3.1 坐标系约定

机器人**朝 +X 站立**，左腿在 +Y 侧，**楼梯沿 +Y 逐级升高**。也就是说上楼方向是
机器人的**侧向**，所以是"螃蟹步横移"而不是正面爬楼。

每级台阶是一个从 `z=0` 到 `z=top` 的**实心 box**（不是薄板），相邻级在 y 上首尾相接。
MuJoCo 的 `size` 是半尺寸，所以第 `i` 级：

```
pos  = (0, (y0+y1)/2, top/2)
size = (width/2, (y1-y0)/2, top/2)
```

出生点在 `y=0`、`z=1.0339`（ISAAC 默认位形下脚底贴地的实测值），楼梯从 `y=1.5` 开始。

### 3.2 采样规则

| 参数 | 默认 | 说明 |
|---|---|---|
| `--seed` | 42 | 换 seed = 换一套楼梯，控制器自动适配 |
| `--n-up` | 6 | 上行级数 |
| `--n-down` | 3 | 下行级数**下限**，实际会被自动调整（见下） |
| `--rise-min/max` | 0.12 / 0.25 | 级高范围，超出侧倾限位会直接报错退出 |
| `--tread-min/max` | 0.20 / 0.40 | 踏面进深范围 |
| `--width` | 1.2 | 楼梯净宽（沿 X） |
| `--landing-depth` | 0.30 | 中间平台进深（实际在 `[d, d+0.08]` 内随机） |

**下行级数会被自动改掉**。因为下楼总降高必须**恰好等于**上楼总升高，否则末端会悬在
半空（最后一级踩空掉 0.68 m）。级数由总高反推：

```
need = ceil(总升高 / rise_max)          # 1.2758 / 0.25 → 6
```

默认 `n_down=3` 时，单级要 0.425 m 远超上限，于是打印 `[调整] 下楼级数 3 -> 6`
并生成 6 级。所以 seed 42 的实际形状是 **6 级上 + 1 平台 + 6 级下 = 13 个落脚面**。

### 3.3 真值表字段

`crab_stairs.json` 的每条 `treads[i]`：

| 字段 | 含义 |
|---|---|
| `name` / `section` | `crab_up_0` / `crab_landing` / `crab_dn_2`…，段属 `up`/`landing`/`down` |
| `y0` `y1` `center` `depth` | 踏面在 Y 上的范围与进深 |
| `top` | 该踏面**距地面的绝对高度**（不是相对上一级） |
| `rise` | 相对上一级的高差，下行段为负 |
| `tilt` | 该级 **SPLIT 相位需要的 `waist_x` 骨盆侧倾**（rad，正 = +Y 侧髋抬高） |
| `off` | 双脚同踏面（GATHER）时相对踏面中心的 Y 偏移 |
| `hip_h_window` | 该级允许的髋高窗口（闭式解的可行区间） |

控制器**直接读这张表**，不重新规划几何。改 seed 后 `tilt` / `off` 自动跟着变。

### 3.4 为什么 13 个落脚面只有 12 个 geom

最后一级下台阶的 `top` 降到 0，正好是地面本身。零厚度的 box 会让 MuJoCo 报
`size 2 must be positive`，所以 `emit()` 里 `top > 1e-4` 才写 geom，但**仍然记进真值表**
—— 控制器需要知道"这里还有一步下到地面"。网页端 `scene_crab.xml` 因此是 12 个 geom，
而 HUD 显示"台阶 13 级"，两者都对。

---

## 4. 第二步：四相位准静态轨迹（`crab_climb.py` + `crab_ik.py`）

### 4.1 落脚面序列与 GATHER / SPLIT

落脚面按顺序排成：

```
[起步地面] up_0 ... up_5 [中间平台] dn_0 ... dn_5 [终点地面]
```

每个踏面给 L / R 各分配**一个固定落点**（中心 ∓ `off`），于是每次转换**只有一只脚在动**：

| 转换 | L 落点 | R 落点 | 谁动 |
|---|---|---|---|
| `GATHER(P_i)` | `c_i + off_i` | `c_i - off_i` | — |
| `SPLIT(P_i→P_i+1)` | `c_{i+1} + off_{i+1}` | `c_i - off_i` | **L** |
| `GATHER(P_i+1)` | `c_{i+1} + off_{i+1}` | `c_{i+1} - off_{i+1}` | **R** |

`GATHER→SPLIT` 只动 L，`SPLIT→GATHER` 只动 R。这个"一次一条腿"的纪律是准静态的
前提，也是后面误差能压到亚毫米的原因。

### 4.2 每一步内部的四相位

| 相位 | 做什么 |
|---|---|
| **LOAD** | 重心平移到**支撑脚正上方**，摆动脚不动 |
| **SWING** | 摆动脚抬到 `max(z0,z1) + clearance`（默认 5.5 cm），横移越过踏面边界 |
| **LAND** | 摆动脚落到目标踏面（脚掌严格水平贴面） |
| **SETTLE** | 重心回到双脚中点，骨盆侧倾回到目标值 |

髋高必须用**世界系**插值。用"相对某只脚"会被摆腿时那只脚的
`级高 + 抬腿余量 ≈ 0.3 m` 带飞 —— 这是踩过的坑。

### 4.3 两个硬约束

**① 级高瓶颈 → 用 `waist_x` 骨盆侧倾解决。**
脚掌 24 cm(X) × 8 cm(Y)，髋间距 27.2 cm。平脚 + 踝限位 `[-50, +45]°` 把级高卡在
22.7 cm。侧倾让上坡侧髋抬高、下坡侧髋降低，两条腿的 ρ 约束同时放松：
**25 cm 级高只需 10.4°（限位 15°）**。同样效果若靠前后错位需要双脚叉开 62 cm。

**② 侧移上限。**
脚掌保持水平时 `ankle_x = -hip_x`，而 `ankle_x` 限位 ±20°，于是单脚相对髋的侧移

```
|ΔY| ≤ ρ · sin(20°) ≈ 0.21 m      （ρ 为腿长方向可达距离）
```

所以不做前后错位、改用"双脚对称放在踏面中心 ±off"就够 —— 窄踏面（20 cm）也不需要叉开。

**注意关节角不是简单相加**：`hip_y` 会改变 `hip_x` 的轴方向。所以闭式解只当**种子**，
每一步都再跑 DLS 数值 IK 收敛。

### 4.4 求解与输出

```
闭式 IK 种子  →  DLS（阻尼最小二乘）数值精修  →  qpos
                 每步只调 mj_kinematics（不跑物理、不积分）
```

- 时间步：`dt = 0.02`（50 Hz），`step_time = 0.9 s`（每步），`hold = 0.5 s`（起步站定）
- 输出：`np.savez(out, time=..., qpos=..., dt=0.02)`
- 体检口径（`--check-only` 可只体检不写轨迹）：

| 指标 | seed 42 实测 |
|---|---|
| 帧数 / 时长 | 1285 帧 / 25.7 s |
| 位置误差 | **0.97 mm** |
| 姿态误差 | 脚掌严格平贴，全程不超关节限位 |
| 支撑面内帧占比 | **仅 57%** —— 43% 的帧质心在支撑面外，最长连续 0.24 s |

最后一行是这套方法的**本质局限**，不是 bug：站姿间距（20~40 cm）约等于踏面进深，
而脚掌沿行进方向只有 8 cm 宽，重心完全移到支撑脚时后腿的侧移需求
`ρ ≈ 0.71 m > 腿长 0.64 m` —— 够不到。所以抬脚与重心转移**必须同时进行**：
纯准静态站不住，真跑 `mj_step` 会摔。这正是 `docs/crab_rl_training.md` 要解决的问题，
也是网页回放**只做运动学**的原因（放出来好看，不代表物理上站得住）。

---

## 5. 第三步：烘焙（`web/tools/bake_crab.py`）

四件事，缺一不可。

### 5.1 抽台阶几何 → `web/assets/scene_crab.xml`

网页端的 MuJoCo 是 WASM，模型**启动时一次编译完成，不能中途换地形**。原来
`scene_bx.xml` 里只有沿 +X 排布的紧凑障碍，螃蟹步的楼梯沿 +Y 上升，是另一套几何。
所以把 `crab_stairs_obstacles.xml` 里名字以 `crab_` 开头的 `<geom>` 整行抽出来，
写成独立的 include 片段：

```xml
<mujocoinclude>
  <geom name="crab_up_0" type="box" pos="0 1.6774 0.0885" size="0.6000 0.1774 0.0885" .../>
  ...
</mujocoinclude>
```

只搬 geom、**不搬 site**（`mjscene.js` 只渲染 geom，site 搬过去也看不见）。
`scene_bx.xml` 的 `<worldbody>` 末尾 `<include file="scene_crab.xml"/>` 把它接进来。

### 5.2 qpos 重排：`nq 38 → 36`

| 模型 | nq | 差异 |
|---|---|---|
| 仿真 `elf3_crab.xml` | 38 | 自由基座 7 + 29 策略关节 + `head_z` / `head_y` |
| 网页 `web/assets/elf3.xml` | 36 | 没有两个头部被动关节 |

`bake_climb.build_qpos_remap()` 按**关节名**逐个对齐（不是按索引截断），顺序变了
也不会录出"关节乱掉"的回放。对齐后断言 `ix.shape[0] == web_model.nq`，
再 `frames = qpos[:, ix]`。

### 5.3 写 `motion_crab.bin` + `motion_crab.json`

```python
arr = np.ascontiguousarray(frames, dtype="<f4")
(arr_path).write_bytes(arr.tobytes())
```

- 格式：`float32-le, row-major, (nframes, nq)`
- 元数据里除了 `fps/nq/nframes/duration`，还写：
  - `version = "<时间戳>-<帧数>"` —— 网页拿它当缓存版本号，重录后 URL 就变
  - `jointNames[]` —— 调试用
  - `kind: "crab"` —— 网页靠它走螃蟹步分支（切地形、切机位、换说明文字）
  - `stats{}` —— 台阶级数 / 级高 / 进深 / 最高 / 横移距离 / 躯干 z 区间，HUD 直接读

### 5.4 刷 `manifest.json`

网页启动时要把**所有被 include 到的 XML** 灌进 MuJoCo 的虚拟文件系统。
漏了 `scene_crab.xml` 就会在编译模型时报 `include not found`。
`bake_crab.py` 内置 `refresh_manifest()`（与 `tools/make-manifest.mjs` 逻辑一致），
不依赖 node。

---

## 6. 网页端回放

### 6.1 加载与插值（`web/js/motion.js`）

`MotionPlayer.load()` 先取 `motion_crab.json`，再用 `?v=<version>` 取 `.bin`，
校验 `frames.length === nframes * nq` 后构造播放器。每帧：

```js
x  = time * fps                 // 浮点帧号
i0 = floor(x); a = x - i0; i1 = i0 + 1
dst[k] = f[i0][k] + (f[i1][k] - f[i0][k]) * a     // 位置线性插值
_slerpQuat(dst, i0, i1, a)                        // 四元数单独处理
```

### 6.2 为什么必须 slerp

基座姿态是四元数。**直接对四元数做线性插值会让旋转矩阵出现缩放**，而
`mj_forward` / `mj_kinematics` 不会帮你归一化 —— 模型会"缩小"或"拉伸"。
所以 `qpos[3..6]` 走最短路径 slerp 再归一化，其余通道走线性插值。

### 6.3 `mj_kinematics` 而非 `mj_forward`

回放只需要几何位姿，不需要接触与约束。`mj_kinematics` 省掉碰撞检测，比
`mj_forward` 便宜很多（代码里做了 feature detect，wasm 构建没导出时回退 `mj_forward`）。

### 6.4 两套地形共用一份模型

模型里同时有紧凑障碍（x≈4.8~13.2）和螃蟹步台阶（y≈1.5~6.0），空间上不重叠。
`mjscene.js` 在建模时按 geom 名把世界几何分成两组：

```js
if (gname.startsWith('crab_')) this.crabGeomIds.push(gi);
else                           this.envGeomIds.push(gi);
```

回放哪段就只显示哪段（`main.js` 的 `setTerrain({stairs, crab})`）：

| 回放 | 楼梯/障碍 | 螃蟹步台阶 |
|---|---|---|
| 打拳（平地录） | 关（可手动勾回） | 关 |
| 爬楼梯策略 | **开** | 关 |
| 螃蟹步 | 关 | **开** |

### 6.5 机位

台阶剖面在 **YZ 平面**上，相机必须站在机器人**侧方**（MuJoCo +X 侧，three 里是 -Z）
才看得见"台阶"；站在沿台阶轴的方向只会看到一堵斜面墙。

```js
crab: { pos: [-1.35, 2.50, -6.00], target: [-1.35, 1.00, 0] },
```

选这段回放时自动 `applyView('crab')`。

---

## 7. 复现与调参

### 7.1 常用命令

```bash
# 全链（生成 + 规划 + 烘焙），约 20 s
python web/tools/bake_crab.py

# 换地形
python web/tools/bake_crab.py --seed 7 --rise-max 0.18

# 走慢一点（更容易站住）
python web/tools/bake_crab.py --step-time 1.2

# 只重解轨迹（地形不变）
python web/tools/bake_crab.py --skip-gen

# 只重烘焙（改过打包逻辑 / 改了场景抽取规则）
python web/tools/bake_crab.py --skip-gen --skip-plan
```

### 7.2 `bake_crab.py` 参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `--seed` | 42 | 地形随机种子 |
| `--n-up` / `--n-down` | 6 / 3 | 上/下行级数（下行会被自动调整） |
| `--rise-min/max` | 0.12 / 0.25 | 级高范围 |
| `--tread-min/max` | 0.20 / 0.40 | 踏面进深范围 |
| `--landing-depth` | 0.30 | 中间平台进深 |
| `--no-landing` | — | 不要中间平台 |
| `--step-time` | 0.9 | 每步时长（s），走不稳就调大 |
| `--traj` | `.../crab_climb_traj.npz` | 指定轨迹文件 |

### 7.3 网页端验证

```bash
start.bat                                     # 或 node server.mjs
# 浏览器打开 http://localhost:8080/?motion=crab
node tools/verify.mjs http://localhost:8080/  # 端到端自检
```

---

## 8. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| 编译模型报 `include not found: scene_crab.xml` | `manifest.json` 没刷新 / 没灌进 VFS | 重跑 `bake_crab.py`，或 `node tools/make-manifest.mjs` |
| 机器人踩空、穿过台阶 | `scene_crab.xml` 和 `motion_crab.*` 不是同一次生成的 | 两者必须成对更新，跑完整链路（不要只 `--skip-plan`） |
| 关节乱掉、手臂反折 | `qpos` 没按关节名重排（错误地按索引截断） | 确认走的是 `build_qpos_remap()` |
| 模型看起来被压扁 / 拉长 | 四元数做了线性插值没归一化 | 确认 `MotionPlayer._slerpQuat()` 生效 |
| 台阶像一堵斜面墙 | 机位不对（站在沿台阶轴方向） | 点「螃蟹步」预设，或用 `?motion=crab` 自动切 |
| 画面里同时出现两套地形 | 只勾了一个开关 | 正常现象也可关掉；两套地形空间不重叠 |
| 动画播起来"很飘"、像滑步 | 这是准静态开环轨迹，本来就没有物理反馈 | 正常。要闭环见 `docs/crab_rl_training.md` |

---

## 9. 局限与下一步

这套链路的价值是**把地形生成、步态规划、网页烘焙串成一条可复现的产线**：
换 seed 就换一套楼梯，20 秒后网页里就能看。它的局限也很明确：

1. **开环**。轨迹离线解好，碰到扰动、地面摩擦变化、台阶比规划的高一点，都不会修正。
2. **准静态不成立**。43% 的帧质心在支撑面外，真跑 `mj_step` 会摔。
3. **不泛化**。级高超 `--rise-max`（侧倾 15° 覆盖范围）直接报错，不是"学着爬过去"。

下一步是把 `crab_climb.py` 的解作为参考动作 `q_ref`，训残差策略（只输出 `Δq`），
让机器人真正闭环地踩上去 —— 设计见 `docs/crab_rl_training.md`。
训练资产的 `scene_crab.xml` 与网页、仿真同源，烘焙链路不用改。
