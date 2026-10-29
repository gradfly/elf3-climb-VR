# GPU 云服务器部署与换机手册

> 面向「换一台 GPU 云主机继续跑螃蟹步 RL 训练」这一场景，把依赖要求、机器选型、
> 部署步骤、验收标准、常见报错和换机检查清单集中在一处。
> 训练任务本身的设计见 `docs/crab_rl_training.md`，训练代码用法见 `train/README.md`。

---

## 0. 先记住一句话

这套环境**只有 PyTorch 一条依赖链**，唯一的版本冲突点是
「驱动的 CUDA 版本」与「torch wheel 的 CUDA 版本」谁高谁低。
其余框架（TensorFlow / PaddlePaddle / TensorRT / tritonserver）**一个都不需要**。

---

## 1. 依赖清单

### 1.1 必须装

| 组件 | 版本要求 | 说明 |
|---|---|---|
| NVIDIA 驱动 | **≥ 550（CUDA 12.8）**，最低 12.4 | 决定能装哪个 cuXXX 的 torch |
| Ubuntu | 22.04 LTS | 24.04 也可，但 EGL 相关包名略有差异 |
| Python | **3.11**（严格） | 3.10 缺 wheel，3.13 对 mujoco-warp 支持不全 |
| uv | 最新 | 建 venv 与装包，比 pip 快很多 |
| PyTorch | 2.x + 对应 cuXXX 构建 | 由 `bootstrap_gpu.sh` 按驱动自动选 |
| mujoco / mujoco-warp | 跟随 mjlab 解析 | 不要手钉版本 |
| mjlab | **GitHub main 分支** | PyPI 版没有 `mjlab.tasks.registry`，`install_task.sh` 会直接失败 |
| rsl_rl / onnx / onnxruntime | 由 mjlab 与 `export_onnx.py` 带入 | 导出与回放用 |
| 系统库 | `libegl1 libgl1 libglib2.0-0 libgomp1` | 缺了 `MUJOCO_GL=egl` 离屏渲染直接崩 |

### 1.2 绝对不要装

| 组件 | 结论 | 原因 |
|---|---|---|
| TensorFlow | 不装 | 仓库零引用；它会拽自己的 cuDNN 版本，和 mujoco-warp 抢 |
| PaddlePaddle | 不装 | 仓库零引用；对 numpy 版本极敏感，容易把 mujoco 连带降级 |
| tritonserver | 不装 | 推理链路是 onnxruntime（`crab_replay.py`），不需要推理服务端 |
| TensorRT | 不装（可选） | 只有真机低延迟部署才值得做，训练/回放阶段没有任何收益 |

> 易混点：pip 里的 `triton` 不是 `tritonserver`。前者是 torch 的间接依赖
> （`torch.compile` 用），装 torch 时自动带上，不用管也不用单独指定版本。

---

## 2. 版本矩阵

### 2.1 唯一的硬规则

```
torch.version.cuda  ≤  驱动支持的 CUDA 版本
```

反了就是 `torch.cuda.is_available() == False`，而且报错信息不会告诉你原因。
历史上踩过一次：服务器驱动只到 CUDA 12.4，但依赖把 torch 顶到了 cu130，
训练脚本启动即失败。

### 2.2 选型对照

| 驱动 CUDA | 装哪个 torch | bootstrap 行为 |
|---|---|---|
| ≥ 13.0 | cu130 | 自动 |
| 12.8 – 12.9 | cu128（最省心） | 自动 |
| 12.6 – 12.7 | cu126 | 自动 |
| 12.4 – 12.5 | cu124 | 自动 |
| < 12.4 | 先升级驱动 | 不建议硬撑 |

`bootstrap_gpu.sh` 会读 `nvidia-smi` 里的 `CUDA Version:` 自动落到上表某一档；
也可以强制指定：`TORCH_CU=cu126 bash scripts/bootstrap_gpu.sh`。

### 2.3 安装顺序（这条比版本号更重要）

```
系统库 → uv + Python 3.11 → mujoco / mujoco-warp / mjlab → torch（最后，--force-reinstall）
```

**torch 必须最后装并强制重装**。如果先装 torch，随后装 mjlab / mujoco-warp 时，
它们的依赖声明会把 torch 升级成更高 CUDA 的构建，`cuda_available` 就废了。
`bootstrap_gpu.sh` 已经按这个顺序写好，不要自己拆开逐条装。

---

## 3. 机器选型

| GPU | 显存 | 建议 `num_envs` | 备注 |
|---|---|---|---|
| RTX 4090 | 24 G | 4096 | 性价比首选；4096 env 约占 20 G |
| RTX 3090 | 24 G | 3072 | 算力略低，先按 3072 |
| L40S | 48 G | 8192 | 显存充裕，适合一次跑满课程 |
| A100 80G | 80 G | 12288–16384 | 大 batch 更稳，单价高 |

其他配置：

| 项 | 建议 | 原因 |
|---|---|---|
| CPU | **≥ 16 核** | 地形生成与 env reset 在主进程，常是真实瓶颈 |
| 内存 | ≥ 64 G | 4096 env 的地形缓存在内存里 |
| 系统盘 | ≥ 100 G | 依赖 + checkpoint |
| 数据盘 | 有则更好 | `logs/` 软链过去，回收实例不丢进度 |
| 计费 | 按量优先 | 竞价实例被回收会丢课程进度（见 §6.3） |

---

## 4. 部署步骤

### 4.1 验驱动

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

`driver_version` 对应的 CUDA 版本低于 12.4 就别往下走了，先换镜像或升驱动。

### 4.2 系统依赖

```bash
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  libegl1 libgl1 libglib2.0-0 libgomp1 libglu1-mesa-dev \
  ca-certificates curl git build-essential
```

`bootstrap_gpu.sh` 里也会装一遍，这里先装是为了后续 `git clone` 不被缺包卡住。

### 4.3 取代码

```bash
git clone https://github.com/gradfly/elf3-climb-VR.git ~/elf3
cd ~/elf3
```

服务器到 `github.com:443` 若被拦（国内机房常见），用镜像前缀或本地打包上传：

```bash
# 方式 A：镜像
git clone https://ghproxy.net/https://github.com/gradfly/elf3-climb-VR.git ~/elf3
# 方式 B：本地打包（在本机仓库根）
tar czf elf3.tar.gz --exclude=.git --exclude=.workbuddy --exclude=node_modules .
scp elf3.tar.gz user@<IP>:~/ && ssh user@<IP> 'mkdir -p ~/elf3 && tar xzf ~/elf3.tar.gz -C ~/elf3'
```

### 4.4 一键建环境

```bash
cd ~/elf3
bash scripts/bootstrap_gpu.sh
source .venv/bin/activate
```

脚本依次做六件事：系统库 → 装 uv → Python 3.11 venv → mujoco/mujoco-warp/mjlab
→ torch（按驱动选 cuXXX，`--force-reinstall`）→ 校验 EGL 离屏渲染与 `torch.cuda`。
末尾会把 `MUJOCO_GL=egl` 写进 `.venv/bin/activate.d/`，之后激活即生效。

**期望输出**（最后两行，缺一不可）：

```
EGL 离屏渲染 OK，帧尺寸 (64, 64, 3)
torch 2.x.x+cuXXX | torch.version.cuda 12.x
GPU: NVIDIA GeForce RTX 4090
```

### 4.5 生成资产 + 注册任务

```bash
python train/assets/make_robot_xml.py   # 期望：nq=38 nv=37 nu=0，plane geom = 0
bash train/install_task.sh              # 期望：已注册 4 个 Elf3-Crab-* 任务
```

`install_task.sh` 会把 `train/mjlab_tasks/crab/*.py` 拷进 mjlab 的 `tasks/crab/`
（mjlab 只自动导入自己 `tasks/` 下的子包）。**每次改完任务代码都要重跑一次。**

### 4.6 生成残差参考表（残差任务必需）

参考表是派生产物，不入库，新机器必须重算一次（纯 CPU，约 90 s）：

```bash
python train/assets/build_ref_tables.py
# 产出 train/assets/ref_tables/crab_ref_v1.npz（48 组 rise × tread，约 0.2 MB）
```

自检会验证 IK 误差、帧间二阶差、`dy_cycle ≈ tread`、限位余量，任一不过会直接报错。
不生成就跑 `Elf3-Crab-Residual`，会在 reset 时报找不到参考表。

---

## 5. 首次验收：三层冒烟

**不要跳过任何一层**，尤其不要一上来就 4096 env——出错时你分不清是环境问题还是规模问题。

```bash
# 第 1 层：能建环境、能跑 1–2 s 不报错（摔倒是正常的，PD 无平衡能力）
play Elf3-Crab-Residual-Play --agent zero

# 第 2 层：随机动作，验 MDP 各模块（观测/奖励/终止）不炸
play Elf3-Crab-Residual-Play --agent random

# 第 3 层：小规模训练 5 轮，验 reward 曲线有值、显存够
train Elf3-Crab-Residual --env.scene.num-envs 256 --agent.max-iterations 5
```

第 3 层通过后，先看三条曲线再放大：

- `Episode_Reward/ref_tracking` 有值且在动（不是恒 0）
- `Episode_Reward/forward_progress_y` ≥ 0
- `Curriculum/terrain_levels/mean` 从 0 开始

---

## 6. 训练运行规范

### 6.1 正式训练

```bash
tmux new -s crab
source .venv/bin/activate
WANDB_MODE=disabled train Elf3-Crab-Residual \
  --env.scene.num-envs 4096 --agent.max-iterations 3000 \
  --log-root logs/rsl_rl
```

- 先用 2048 跑 50 轮确认显存余量，再上 4096。
- `WANDB_MODE=disabled` 在没有 wandb 账号时必加，否则启动即卡在登录。

### 6.2 轮数的真实语义（最容易误判的一条）

| 你以为 | 实际 |
|---|---|
| `--agent.max-iterations 3000` 是总轮数 | 是**本次运行再跑多少轮** |
| 从 800 轮续训到 3000 | 会再跑 3000 轮，不是跑到 3000 |
| checkpoint 编号是全局累计 | 是 `model_<本次轮数>.pt` |

### 6.3 续训会重置课程

速度课程和地形等级在 resume 时回到起点（速度 0.05–0.12、地形等级 0）。
所以**尽量一次跑够**，用 tmux 挂着，不要切成多次续训——每次都要重新爬课程，
等于白跑几百轮。

### 6.4 checkpoint 与产物

- 每 50 iter 落盘一次，在 `logs/rsl_rl/elf3_crab/<时间戳>/`。
- 竞价实例：把 `logs/` 软链到数据盘，并挂一个定时回传（见 §8）。

---

## 7. 报错速查

| 报错 | 原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available() == False` | torch 的 CUDA 版本高于驱动 | 按 §2.2 重装匹配的 cuXXX，`--force-reinstall` |
| `ModuleNotFoundError: mjlab.tasks.registry` | 装的是 PyPI 版 mjlab | `uv pip install --force-reinstall "mjlab @ git+https://github.com/..."` 装 main |
| `ImportError: libEGL.so.1` | 缺系统库 | `apt install libegl1 libgl1` |
| `注册失败：没有找到 Crab 任务` | 没跑 `install_task.sh`，或改完代码没重跑 | `bash train/install_task.sh` |
| 找不到 `crab_ref_v1.npz` | 参考表没生成 | `python train/assets/build_ref_tables.py` |
| `KeyError: 'robot/imu_ang_vel'` | 该 mjlab 版本不自动建 IMU 传感器 | 改用 `mdp.base_ang_vel` / `mdp.base_lin_vel` |
| `invalid scale` / `actuator_names` 不匹配 | `ACTION_SCALE_MAP` 的键与执行器名不一致 | 把 `actuator_names=(".*",)` 改成具体关节名元组 |
| `solver iterations limit reached` 刷屏 | mujoco-warp 用 float32，`tolerance=1e-8` 永远不成立 | 已修为 `iterations=20`、`tolerance=1e-6`；旧代码自行改 |

---

## 8. 换机检查清单

按顺序打勾，一条不落：

1. [ ] `nvidia-smi` 的 CUDA 版本 ≥ 12.4，且记下具体值
2. [ ] `bash scripts/bootstrap_gpu.sh` 末尾两行校验全绿（EGL + CUDA）
3. [ ] `make_robot_xml.py` 输出 `nq=38 nv=37 nu=0`
4. [ ] `install_task.sh` 输出 4 个任务
5. [ ] `build_ref_tables.py` 跑通（残差任务）
6. [ ] 三层冒烟全过
7. [ ] **`uv pip freeze > requirements.lock.txt` 并提交**——否则下次换机版本又会漂
8. [ ] `logs/` 指向数据盘；`sync_back.sh` 定时回传配好

第 7 条是这份手册存在的根本原因：项目里没有 lock 文件，环境全靠脚本现场解析，
同样的脚本在不同时间跑会解析出不同的 torch / mjlab 版本。**跑通一次就锁一次。**

回传产物（在**本地**执行，不在服务器）：

```bash
SERVER=user@<IP> bash scripts/sync_back.sh
# 只拉 *.onnx / *.pt / *.ckpt / *.json / *.bin / tfevents，不碰大体积中间数据
```

导出可部署策略：

```bash
python train/export_onnx.py --task Elf3-Crab-Residual \
  --checkpoint logs/rsl_rl/elf3_crab/<run>/model_3000.pt \
  --out train/checkpoints/elf3_crab_residual.onnx
```

---

## 9. 时长与成本参考

| 项 | 参考值 | 来源 |
|---|---|---|
| 4090 上单轮迭代 | 1–5 s（含前向 + PPO 更新） | `docs/crab_rl_training.md` §3.3，经验值 |
| 2000 轮 | 约 1–3 h | 同上 |
| 完整课程到可用策略 | 1–2 天 | 同上 |
| 环境搭建（含下载） | 10–20 min | 视镜像与网络 |
| 参考表重算 | 约 90 s（CPU） | `build_ref_tables.py` |

实际吞吐以训练日志里的 `fps` 为准，上表只用于估预算。

---

## 10. 与其他文档的关系

| 文档 | 内容 |
|---|---|
| `docs/crab_rl_training.md` | 任务建模、奖励、课程、PPO 超参、验收指标 |
| `train/README.md` | 任务安装、冒烟、训练、续训、导出、残差任务细节 |
| `README.md` §3 | 脚本步态（无 RL）的一键跑法与体检基线 |
| 本文 | 机器怎么选、环境怎么装、换机怎么验 |
