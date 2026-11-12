# elf3 · MuJoCo 网页端仿真 + VR 沉浸式观看

在浏览器里跑**真正的 MuJoCo 物理引擎**（官方 WebAssembly 构建），把 `elf3.xml` +  
`scene_bx.xml`（地形）渲染成三维场景，并支持用 **VR 头显沉浸式观看**。  
默认状态是**静止站立**：机器人保持 XML 里的初始位形，不跑物理。  
也可以在网页 / VR 里回放**策略跑出来的运动仿真**：`elf3_play.py` 那条
MuJoCo + `policy.onnx` 闭环、跟踪 `fight.npz` 打拳动作的画面，以及
`climb_stairs.py` 那条**深度视觉闭环爬楼梯**的画面（地形为 `universal_test`
紧凑障碍），以及**蹲下 / 起立的 29 关节参考动作**（纯运动学回放，无需训练 / GPU）
—— 见 §3、§7.8。

---

## 1. 启动本地仿真界面

> **本地 MuJoCo 仿真界面 = 在浏览器里跑 MuJoCo 官方 WebAssembly 引擎。**
> 不需要安装 Python / MuJoCo / vuer，只用 Node.js 起一个静态服务器即可。
> MuJoCo WASM、three.js、模型与网格都已随工程放在 `vendor/` 和 `assets/` 里，**离线可用**。

### 1.1 前提：Node.js 18+

仅用于起静态服务器，**不安装任何 npm 依赖**。

- 本机已装 WorkBuddy 时，启动脚本会自动找到内置 Node，无需另装；
- 否则到 [nodejs.org](https://nodejs.org/) 下载 LTS 版；
- 自查版本：`node -v`。

### 1.2 方式一：一键启动（推荐）

| 系统 | 操作 |
| --- | --- |
| Windows | 进入 `web/` 目录，**双击 `start.bat`** |
| macOS / Linux | 进入 `web/` 目录，执行 `./start.sh`（首次需 `chmod +x start.sh`） |

脚本会自动完成：定位 Node → 启动服务器（HTTP + HTTPS 双端口）→ 打开默认浏览器。
**关闭那个命令行窗口即停止服务。**

### 1.2.1 一键看「运动仿真」（推荐给 VR 观看）

| 系统 | 操作 |
| --- | --- |
| Windows | 进入 `web/` 目录，**双击 `start_vr_motion.bat`** |
| macOS / Linux | 进入 `web/` 目录，执行 `./start_vr_motion.sh` |

它和 `start.bat` / `start.sh` **并存、互不覆盖**，区别是**直接进入运动仿真回放**：
自动检查回放数据（缺了就本机跑一遍 MuJoCo 仿真录出来）→ 起服务器 → 打开的地址带
`?motion=fight_policy`，所以一进页面就是机器人在打拳，VR 头显的地址也会一并打印。

```bash
./start_vr_motion.sh --rebake          # 重新跑一遍本机仿真，刷新回放数据
./start_vr_motion.sh --https-port 9443 # 其它参数原样转给 server.mjs
```

### 1.3 方式二：命令行启动

```bash
cd "E:/mujoco/robot test/web"   # 关键：必须先 cd 到 web/ 目录
node server.mjs --open          # 起服务器并自动打开浏览器
```

去掉 `--open` 就只起服务器，自己手动访问终端打印的地址：

```bash
node server.mjs
```

### 1.4 命令行参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--http-port <n>` | `8080` | HTTP 端口，桌面浏览器用 |
| `--https-port <n>` | `8443` | HTTPS 端口，VR 头显用 |
| `--no-https` | 关闭 | 只开 HTTP，跳过证书生成（纯桌面预览时最省事） |
| `--open` | 关闭 | 启动后自动用默认浏览器打开 |
| `--open-path <path>` | `/` | `--open` 时打开的路径，如 `"/?motion=fight_policy"` 直接进运动仿真回放 |
| `--log` | 关闭 | 在终端打印每条请求（排查加载问题时用） |

```bash
node server.mjs --http-port 9000 --https-port 9443   # 换端口
node server.mjs --no-https                          # 只开 HTTP
```

> 端口被占用时会自动向后顺延（+1），**以终端实际打印的地址为准**。

### 1.5 启动成功的标志

终端会打印本机与局域网访问地址：

```
  本机桌面浏览器：
     http://localhost:8080/
     https://localhost:8443/

  VR 头显（同一局域网内在头显浏览器打开）：
     https://192.168.22.19:8443/
```

浏览器打开 `http://localhost:8080/`，等加载进度条走完、出现机器人的三维场景，即启动成功。

### 1.6 停止服务

- 命令行方式：在运行服务器的终端按 **`Ctrl+C`**；
- 双击 `start.bat` 方式：直接关掉那个黑色命令窗口。

### 1.7 启动失败排查

| 现象 | 原因 / 处理 |
| --- | --- |
| 提示 `[ERROR] 未找到 Node.js` | 装 Node.js 18+ 后重试；或手动指定内置 Node：<br>`"%USERPROFILE%\.workbuddy\binaries\node\versions\22.22.2-6\node.exe" server.mjs --open` |
| 启动报 `SyntaxError: Unexpected reserved word`（指向 `server.mjs` 的 `await listen(...)`） | Node < 14.8 不支持**顶层 await**。两种解法：<br>① 装 Node 18+；<br>② 用零依赖的 Python 后备服务：`python web/tools/serve.py --port 8080`（MIME / 禁缓存与 `server.mjs` 一致，只少了 HTTPS） |
| 端口被占用 | 会自动 +1 顺延，也可用 `--http-port` / `--https-port` 指定其它端口 |
| 页面一直转圈或报错 | 必须通过 `server.mjs` 访问（`http://…`）；**别直接双击 `index.html`**，`file://` 会被 CORS / WASM 限制拦住 |
| 打开是空白页 | 按 `F12` 看 Console 报错，多半是没在 `web/` 目录下启动导致资源 404 |

### 1.8 可选：MuJoCo 原生查看器（Python）

项目根目录的 `view_mujoco.py` 用 MuJoCo 自带的 OpenGL 窗口打开同一个模型，
不经过 three.js，适合核对模型的**原始装配状态**。它依赖本地 Python 装了 `mujoco`：

```bash
pip install mujoco numpy
python view_mujoco.py          # 静止保持初始装配姿态（核对角度用）
python view_mujoco.py --sim    # 真实物理：跑 mj_step（无控制器会倒地）
python view_mujoco.py --pin    # 钉住躯干，只让关节自由摆动
python view_mujoco.py --drop   # 落地对齐：消除脚底悬空
```

> 若本地 Python 没装 `mujoco`，用 1.2 / 1.3 的网页端即可，功能完全不受影响。

---

## 2. 桌面端预览

用 Chrome / Edge 打开 `http://localhost:8080/` 即可。  
（`localhost` 属于安全上下文，所以 WebXR 在 HTTP 下也可用。）

- **左键拖动** = 旋转 · **滚轮** = 缩放 · **右键拖动** = 平移
- 右侧面板：**场景**（单人 / 矩阵 8×12，默认矩阵）、显示开关（机器人 / 楼梯 / 地面 / 阴影 / 碰撞体）、
  9 个视角预设（含**螃蟹步**侧视、**蹲起 3/4** 与**矩阵全景**）、物理开关与复位按钮
- **矩阵场景**（默认）= 把机器人复制成 **8 行 × 12 列 = 96 份**（间距 1.08 / 1.17 m），
  整体占地约 13 m × 8 m。
  同一段动画同步播放。
  实现用 `InstancedMesh`：几何 / 材质只有一份，绘制批次仍是 30 出头（不是 32×96），
  每帧只改实例矩阵的平移列（实测 0.15 ms/帧）。地形与阴影不参与复制（只在单人里有）。
  代价是三角形总量 ×96：约 **1980 万面/帧**（双目 3950 万），独立显卡流畅。
  **进 VR 也保持矩阵**（不再降载，见 4.3），一体机上会明显掉帧 —— 想流畅就手动点「单人」。
- 底部并排两个 XR 入口按钮（在不支持 XR 的设备上会显示为灰色不可用）：

  | 按钮 | 模式 | 效果 |
  | --- | --- | --- |
  | **进入 VR** | `immersive-vr` | 沉浸式：虚拟世界完全遮住现实 |
  | **半透明 VR** | `immersive-ar` | 透视（Passthrough）：背景透明，**能看到眼镜外的真实环境** |

---

## 3. 运动仿真回放（打拳 / 爬楼梯）

页面默认只摆位、不跑物理。右侧面板第一栏「运动仿真」可以把**策略跑出来的运动仿真**
放到网页 / VR 里看 —— 有两条链路，回放数据都是本机先跑一遍录下来的：

| 回放数据 | 内容 | 时长 |
| --- | --- | --- |
| 策略仿真 · 打拳 | `elf3_play.py` 链路：本机跑 **MuJoCo + policy.onnx** 闭环仿真录制的 qpos（含物理接触、关节跟踪） | 146.8 s |
| 参考动作 · 打拳 | 直接取 `fight.npz` 里重定向后的运动（纯运动学，无物理） | 244.9 s |
| 策略仿真 · 爬楼梯 | `climb_stairs.py` 链路：本机跑**深度视觉闭环**（MuJoCo 渲染深度图 → `normal_depth.onnx` → PD 力矩 → `mj_step`），在 `universal_test` 紧凑地形上从低楼梯爬过终点线 | 26.4 s |
| 参考动作 · 蹲下起立循环 | `logs/export/squat_{down,up}.npz` 烘焙的 29 关节极限蹲：下蹲 → 蹲姿保持 1 s → 起立 → 站姿保持 1 s，首尾同姿态无缝循环 | 6.5 s |
| 参考动作 · 螃蟹步横移上楼 | `crab_climb.py` 四相位准静态控制器解出的关节轨迹（闭式 IK + DLS，无策略网络），沿 +Y 侧向横移上台阶 | 25.7 s |
| 参考动作 · 走正步（前进） | `launch/goose_step.py` 逐帧解析两连杆 IK 生成的正步行进（腿绷直、脚不滑） | 6.8 s |
| 参考动作 · 正步 + 敬礼（连贯） | `scripts/build_parade_sequence.py` 把正步（341 帧，走 3.15 m 收步立定）与敬礼（225 帧）拼成一条序列：中间 50 帧 smootherstep 收臂过渡（过渡段**只插值关节、基座沿用正步终点**，否则会倒着滑回原点），敬礼段整段平移 +3.15 m，末尾 25 帧立正定格。播完冻结 | 12.8 s |
| 参考动作 · 贴地爬行 | `scripts/build_prone_crawl_sequence.py` 把动作 1（站立→蹲下→**双手撑地**→双脚后伸→趴姿，250 帧）与动作 2（俯卧爬行 4 循环，399 帧）拼成一条 38 维序列，再按关节名烘焙成 36 维；接缝 qpos 差 1.3e-4 rad，播完冻结 | 13.0 s |

> 「参考动作」是**统一叫法**：面板里只区分「参考动作」（本机烘焙好的轨迹：打拳参考动作 /
> 蹲起 / 螃蟹步 / 正步）与「策略仿真」（`policy.onnx`、`normal_depth.onnx` 闭环录制）。
> 单段的「下蹲」「起立」、「原地正步」「敬礼」四段数据仍在 `assets/` 里，
> 但**已从下拉菜单移除**（`?motion=squat_down` / `goose_inplace` / `salute` 仍可直达）。
> 正步与敬礼的「连贯版」已经加进下拉（`?motion=parade_seq`），就是下面那张表的最后一行。

浏览器里的 MuJoCo WASM 没有策略网络（没有 onnxruntime），所以网页端**不现算策略**：
先在本地把仿真跑一遍，把每个控制步（50 Hz）的 `qpos` 录成 float32 二进制
（`assets/motion_*.bin` + 同名 `.json`），网页端按时间插值回放（位置线性插值、
四元数最短路径 slerp），再用 `mj_kinematics` 刷新几何位姿 —— 50 Hz 数据 + 插值，
VR 里动作平滑不抖。

**蹲下 / 起立这几段是「参考动作」，不是策略仿真**：数据直接来自
`scripts/export_squat_npz.py` 导出的 `squat_down.npz` / `squat_up.npz`（29 关节
极限蹲动作），`bake_squat.py` 只做 38→36 维关节对账和 bin 编码，**不跑 `mj_step`、
不需要 ONNX、不需要 GPU**。所以回放里机器人永远保持设计好的姿态，不会像
本地物理那样摔倒 —— 它展示的是**动作本身**，不是 RL 策略的控制效果。
想看策略闭环真实跑出来的下蹲画面，需要在服务器把 MuJoCo + `squat_down.onnx`
闭环的 qpos 录成同格式 npz，再指给 `bake_squat.py --src-dir`（键名一致即可）。

重新录制（换了模型 / 策略 / 动作数据之后）：

```bash
# —— 打拳两段（elf3_play.py 链路）——
python web/tools/bake_motion.py                       # 整段录，约 20~40 s
python web/tools/bake_motion.py --steps 1500          # 只录前 30 s
python web/tools/bake_motion.py --only ref            # 只录参考动作
python web/tools/bake_motion.py --stop-on tracking    # 改用 mjlab 严格跟踪判据截断

# —— 爬楼梯（climb_stairs.py 深度视觉链路）——
python web/tools/bake_climb.py                        # 默认 34 s，楼梯前 1.2 m 出生
python web/tools/bake_climb.py --duration 20          # 只录前 20 s
python web/tools/bake_climb.py --cmd-vx 0.4           # 走慢一点
python web/tools/bake_climb.py --spawn-x 0            # 从世界原点出发（先走 4.6 m 平地）

# —— 螃蟹步横移上楼（crab_climb.py 脚本烘焙的参考动作，无策略网络）——
python web/tools/bake_crab.py                         # 全链：生成随机台阶 → 规划轨迹 → 烘焙
python web/tools/bake_crab.py --seed 7 --rise-max 0.18        # 换一套台阶重录
python web/tools/bake_crab.py --step-time 1.2                 # 走慢一点（更容易站住）
python web/tools/bake_crab.py --skip-gen --skip-plan          # 只重烘焙已有轨迹

# —— 蹲下 / 起立（29 关节参考动作，纯运动学，无需 GPU）——
python scripts/export_squat_npz.py                 # 先有 squat_down/up.npz（若还没有）
python web/tools/bake_squat.py                    # 生成 down / up / cycle 三段回放
python web/tools/bake_squat.py --hold-frames 75   # 循环版蹲姿 / 站姿各停 1.5 s
python web/tools/bake_squat.py --no-cycle         # 只要两个单段

# —— 贴地爬行（动作 1 + 动作 2 合并序列，纯运动学）——
python scripts/build_prone_crawl_sequence.py      # 先拼成 prone_crawl_sequence_traj.npz
python web/tools/bake_crawl.py                     # 再烘焙成 assets/motion_crawl.bin/.json
node web/tools/make-manifest.mjs                   # 改完 assets 顺手刷新清单
# python web/tools/bake_crawl.py --preset fast --name crawl_fast   # 快档（单帧跳变大，默认不下发）

# —— 正步 + 敬礼（连贯序列，纯运动学）——
python sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch/goose_step.py
python sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch/salute.py
python scripts/build_parade_sequence.py           # 拼成 parade_sequence_traj.npz（641 帧）
python web/tools/bake_actions.py                  # 同时重烘 goose_walk / goose_inplace / salute / parade_seq
python scripts/check_parade_actions.py            # 验收（含拆段复检 + 接缝/位移专项）
# python scripts/build_parade_sequence.py --salute-cycles 2 --blend-frames 75
```

> `bake_crab.py` 除了写 `motion_crab.*`，还会把台阶几何同步成
> `assets/scene_crab.xml` 并刷新 `manifest.json` —— 网页端是"一次编译"的模型，
> 地形必须和回放数据一起就位。

面板上的控件：

| 控件 | 说明 |
| --- | --- |
| 回放 | 关闭（静止站立）/ 策略仿真 · 打拳 / 参考动作 · 打拳 / 策略仿真 · 爬楼梯 / **参考动作 · 螃蟹步横移上楼** / 策略仿真 · 螃蟹步横移上楼 / **参考动作 · 蹲下起立循环** / **参考动作 · 走正步（前进）** / **参考动作 · 正步 + 敬礼（连贯）** |
| 倍速 | 0.25× / 0.5× / 1× / 1.5× |
| 视角跟随机器人 | 机器人会满场走动（世界坐标约 ±5 m），打开后镜头自动跟住它 |
| 从头播放 / 暂停 | 控制回放进度（「复位姿态」在回放中也等价于从头播放）。**单段蹲起**播完会冻结在末帧、按钮变「继续」，此时点它等于从头播放 |

几点说明：

- **蹲起三段**是原地动作：选它们时会**自动隐藏两套台阶**、切到能同时看清膝角与
  躯干前倾的 **3/4 机位**。单段「下蹲 / 起立」播完停在末帧（不会突然跳回站姿），
  循环版「蹲下起立」首尾帧完全相同，所以循环处没有跳变。
- **打拳两段**回放是**在平地上**录的，所以选它们时会自动把「楼梯 / 障碍」关掉避免
  视觉穿模；想看地形可以自己再勾回来。
- **爬楼梯那段**回放本身就**踩在这块地形上**：选它时不会隐藏「楼梯 / 障碍」，而且
  如果之前被关掉了还会自动打开（录制起点在低楼梯前约 1.2 m，一路爬过终点线）。
- **螃蟹步横移**用的是**另一套地形**（楼梯沿 +Y 上升，不是沿 +X）。选它时会自动
  勾上「螃蟹步台阶」、同时取消「楼梯 / 障碍」，并切到「螃蟹步」机位 —— 台阶剖面在
  YZ 平面上，只有站在机器人侧方（MuJoCo +X 侧）才看得见台阶而不是一堵斜面墙。
  两套地形在同一个模型里互不重叠（一套在 x≈4.8~13.2，一套在 y≈1.5~6.0），
  所以也可以手动两个都勾上看全景。
- 爬楼梯回放的机器人**从 x≈3.5 m 出发**，不在世界原点；开着「视角跟随机器人」
  镜头会自动跟过去（默认开着）。想看全景可以点「全景」预设。
- **正步 + 敬礼（连贯）**也是满场走的：正步段会前进 3.15 m 才收步，敬礼在原地完成。
  请保持「视角跟随机器人」勾着（默认勾选），否则走完就出画了；想要"一步到位"的
  构图，可以切「单人」+ 手动拉远相机。
- 开启回放会**自动关掉「运行 MuJoCo 物理」**：两者互斥（一个是回放录好的 qpos，
  一个是本地现算 `mj_step`）。
- 打拳策略在 t≈148 s 会失稳摔倒，所以录制在**最后一次姿态健康**的时刻截断
  （倾角 < 30°、根高 > 0.80 m），回放里不会看到倒地画面；爬楼梯那段则是一路
  跑到越过终点线（x > 13.4 m）自动收尾，全程未摔倒。
- VR 里的「视角跟随」是**舒适窗口式**的：机器人在 1.7 ~ 4.8 m 内且大致在身前就
  完全不动，跑出这个范围才慢慢把它拉回正前方 —— 不会和你自己的摇杆移动打架。
  A / X「把机器人摆到正前方」也按它的**实际位置**摆位（回放时它不在世界原点）。
  **右摇杆手动转向（或点 6 个视角按钮）会自动取消「视角跟随机器人」的勾选**，
  免得你刚转到侧面就被慢慢拉回正面；想恢复跟随再勾回去即可。角速度可在控制台
  现场改 `__elf.vrNav.turnSpeed`（默认 π/2 rad/s = 90°/s），死区改 `turnDeadzone`。
- URL 参数：`?motion=fight_policy&speed=0.5&follow=0`
  （一键脚本用的就是这条；爬楼梯用 `?motion=climb_policy`，螃蟹步用 `?motion=crab`，
  **蹲下起立循环**用 `?motion=squat_cycle`，单段用 `?motion=squat_down` / `squat_up`）。
  场景默认是矩阵，想看单个机器人加 `&scene=single`（例：`?scene=single&motion=crab`，
  否则矩阵模式下不切「螃蟹步」机位、也不显示台阶）。

---

## 4. VR 头显沉浸式观看

1. **电脑和头显连同一个局域网**（同一个 Wi-Fi / 路由器）。
2. 在头显浏览器里打开终端打印的 **HTTPS** 局域网地址，例如  
   `https://192.168.22.19:8443/`。
   - ⚠️ 必须用 **https**（WebXR 的安全上下文要求），http 地址在头显里不会出现 ENTER VR。
3. 首次打开会提示「证书不受信任」→ 选 **高级 → 继续访问**。  
   （想彻底免提示：把 `certs/cert.pem` 传到头显里安装为受信任证书。）
4. 点页面下方的 **进入 VR**（或 **半透明 VR**，见 4.1）。

进入 VR 后：

| 操作    | 作用                    |
| ----- | --------------------- |
| 左摇杆   | 前后左右移动（1.6 m/s）       |
| 右摇杆左右 | 连续转向：推到底 90°/s，死区 0.15，半推可慢转 |
| A / X | 把机器人摆到你正前方 2.6 m 并正对你 |
| B / Y | 世界复位（机器人回到世界原点）       |
| 扳机 / 侧键 | 点击头显里的设置面板（见 4.2）      |
| 摇杆按下  | 收起 / 展开设置面板            |
| 标题栏「50% / 100%」 | 整个头显面板在 50% 半透明与不透明之间一键切换 |
| 标题栏「归位」 | 在「跟随头部（头锁置顶）」与「固定在世界」之间切换 |

技术细节：参考空间用 `local-floor`，头显里**你的脚踩的地面**就是 MuJoCo 的  
`z = 0` 地面。机器人身高约 1.43 m，站在离你 2.6 m 处，原地 1:1 比例观看，  
可以走近绕一圈看细节。

### 4.1 半透明 VR（透视 / Passthrough）

「半透明 VR」按钮就在「进入 VR」旁边，点它进入的是 **WebXR 的 `immersive-ar`
会话**：虚拟场景的背景不再渲染，没被机器人 / 地面遮住的地方直接透出头显摄像头
拍到的真实房间 —— 机器人像站在你自己的客厅里打拳，你还是能看到沙发、桌子和走动
的家人。

| | 进入 VR | 半透明 VR |
| --- | --- | --- |
| 会话模式 | `immersive-vr` | `immersive-ar` |
| 混合模式 | `opaque` | `alpha-blend`（设备上报） |
| 背景 | 天空盒贴图（不透明） | 透明（`scene.background = null` + 清屏 alpha = 0） |
| 地面网格 | 不透明 | 淡化到 0.3（真实地面能透出来，又保留位置参照） |
| 手柄漫游 / A·X·B·Y | ✅ | ✅（完全一样） |

实现要点（详见 §7.5）：渲染器必须带 alpha 通道；进入透视时摘掉天空盒与雾、
清屏 alpha 置 0；退出会话时**全部还原**，不影响普通 VR / 桌面观感。

> ⚠️ 透视需要头显有**彩色摄像头**：Quest 3 / Quest Pro / Quest 2（灰度透视）都可以；
> Quest 1 没有，按钮会显示「透视不可用」。桌面浏览器不支持 `immersive-ar`，
> 所以电脑上两个按钮都是灰色 —— 这是设备能力限制，不是页面坏了。
> 首次进入时系统可能询问「允许使用摄像头 / 透视」，选允许即可。

### 4.2 头显里的设置面板（`js/xrpanel.js`）

WebXR 会话里浏览器**只合成 WebGL 画布，DOM 不会进到虚拟画面**（Quest 浏览器也不支持
`dom-overlay`），所以桌面右侧那块 `#panel` 在头显里是看不到的。为了在头显里也能改开关，
页面另外画了一块面板：2D canvas 画出精简后的控件 → `CanvasTexture` 贴到一块
**0.80 × 0.23 m** 的宽扁平面上（约 3.5:1）→ 头锁摆在视线上方 0.5 m、水平距离 1.5 m，
平面朝眼睛下倾约 10.8°，手柄射线打到平面上再换算成画布像素。
整个贴图面板默认 **50% 半透明**（文字、边框、按钮一起透），能看见后面的机器人；
材质 `transparent:true + depthWrite:false`，由标题栏的透明度 chip 在 0.5 / 1.0 之间切换。

| 操作 | 作用 |
| --- | --- |
| 扳机 / 侧键 | 点击面板上的单选、下拉、复选框、按钮 |
| 左 / 右摇杆按下 | 收起 / 展开面板（走远了够不到小签时的退路） |
| 标题栏「50% / 100%」 | 一键切换整个面板的透明度 |
| 标题栏「归位」 | 切换「头锁跟随 / 固定在世界」；切回跟随时立即摆回当前视野上方 |
| 标题栏「退出 VR」 | 直接结束 XR 会话，不用翻系统菜单 |
| 标题栏「收起」 | 收成一枚小签，点它重新展开 |

**头显面板只保留最小控件集**，按设计稿固定两栏：

| 左栏「场景」 | 右栏「视角」 |
| --- | --- |
| 单人 / 矩阵 8×12（单选） | 正面 / 背面 / 侧面 / 蹲起 3/4 / 矩阵全景 / 全景（6 键） |
| 回放（下拉，11 段） | 视角跟随机器人（复选） |
| 从头播放 / 继续（按钮） | |

显示组 6 个复选、物理组、倍速下拉、头部特写 / 俯视 / 螃蟹步三键、底部状态条、
标题提示文字都不画。这份清单是 `xrpanel.js` 顶部的 `LAYOUT` 白名单，**桌面
`web/index.html` 的 `#panel` DOM 一个控件都没删**，两边互不干扰。

控件内容仍是从 `#panel` 的 DOM **实时投影**的：label、下拉选项、选中态、按钮文案都读 DOM，
所以文案不会两份不同步；点击等价于在 DOM 上 `dispatchEvent(new Event('change'))` / `.click()`，
业务逻辑一份都不用复制。反方向靠 3 Hz 采样 DOM 签名，桌面端改完头显立刻跟着变。
「回放」下拉有 11 项，宽扁面板放不下：打开时画布临时撑到 782 px 并按顶边锚定，
关闭后缩回 290 px，命中坐标按 `canvas.height` 换算，所以射线拾取不用改。

悬停射线按 30 Hz 跑、重绘按 70 ms 合并：一次重绘 = 全画布 1024×~290 的 2D 重画 +
纹理上传 + mipmap 生成，而 `update()` 是从 XR 的帧回调里同步调的 —— 手柄扫过
面板时悬停项每秒能变十几次，不合并就会把主线程拖成长任务（头显里表现为「手柄都迟钝」）。
头锁每帧只改 group 变换、不触发重绘，所以空闲时不会反复上传纹理。

**说明段落（`p.note`）在头显里不画** —— 又长又静态，头显里既占地方又看不清。
进 VR / 半透明 VR 时二维面板也同时进精简模式（`#panel.compact` → `.note{display:none}`），
标题与控件标签照旧，退出会话还原。

> 无头环境也能验：非 XR 时面板用桌面相机摆位，`__elf.xrPanel.tap('check:opt-follow')`
> 直接点控件，`__elf.xrPanel.exportPNG()` 把画好的面板导成 PNG 看中文有没有变豆腐块
> （见 `tools/verify.mjs` 的 7.6 节，22 项：半透明材质与切换、头锁置顶与法向、
> 归位切换、覆盖层撑高、最小控件集白名单、视角与回放按钮派发、收起小签）。

### 4.3 VR 画质：进会话自动降载（`main.js` 的 `enterXRQuality`）

一体机（PICO / Quest）进 VR 后默认按「能跑满刷新率」的档位渲染，退出会话**还原阴影**：

| 项 | 桌面 / 退出 VR | 进 VR 后 | 为什么 |
| --- | --- | --- | --- |
| 场景 | 默认矩阵 8×12（96 份） | **保持你的选择**（不降载） | 用户要求：进 VR 仍是矩阵；想流畅就在面板里点「单人」 |
| 阴影 | 按「阴影」开关 | **关** | 一次深度 pass + PCFSoft 9 tap × 全屏 × 2 眼 |
| 帧缓冲缩放 | — | **1.0×（默认不缩）** | 0.8× 在 PICO 投影层上会全黑，见下方警告 |
| 抗锯齿 | 开 | **关**（一体机默认） | 投影层 `samples: 4` 是最大一笔带宽 |
| 注视点渲染 | — | **0.5（不拉满）** | 见下方警告：拉满会「单眼丢机器人」 |

**场景在 VR 里可自由切换**：矩阵 8×12 = 1980 万三角/单目、3950 万/双目，一体机上会明显
掉帧（这是保留矩阵的代价），但不再拦截。退出 VR 后**保留**你在头显里选的场景，只有阴影
会还原。URL 逃生口：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `?scene=single` | matrix | 启动就用单人（一体机想直接流畅可以用这个） |
| `?xrss=0.8` | 1.0 | XR 帧缓冲缩放，clamp 到 0.5~1.0；越低越糊越省。**先在真机确认能出图再开** |
| `?aa=1` | 一体机关、桌面开 | 强制开抗锯齿（`?aa=0` 强制关） |
| `?ffr=0` | 0.5 | 注视点渲染等级 0~1。**能省片元，但拉满会出单眼问题**（见下） |
| `?interp=linear` | cr | 回放插值：默认 Catmull-Rom，退成旧的逐帧线性（见 4.4） |
| `?cull=0` | on | 关掉矩阵模式的逐实例视锥剔除（排查边缘闪烁用） |
| `?noprefetch=1` | off | 不预取回放数据（见 4.4） |
| `?mat=standard` | lambert | 切回 `MeshStandardMaterial`（PBR + 环境贴图，金属感；默认是省片元的 Lambert，见 4.5） |

> `antialias` / `framebufferScaleFactor` 只能在**启动时**定：前者固化在 WebGL context
> attributes 里，后者必须早于 `renderer.xr.setSession` 才有意义。
> `ffr` 例外，会话里也能现场调：`__elf.xrQuality.foveation(0)` 立刻生效，用来 A/B 排查。

> ⚠️ **帧缓冲缩放不要默认开 0.8×**。three 在有 `XRWebGLBinding` 的设备上走投影层分支，
> 用 `scaleFactor` 创建 `XRProjectionLayer` 再按 `textureWidth/Height` 建 render target
> （`vendor/three.module.js:27427-27453`）。PICO 4 Ultra 上 0.8 会拿到无效尺寸 / 合成器
> 不接受该纹理，**整帧被丢弃 → 进入 VR 全黑**（连机器人和地面都没有）。
> 已实测：`?xrss=1` 恢复，`0.8` 全黑。所以默认值是 1.0；`xr.js` 也只在 `<1` 时才调
> `setFramebufferScaleFactor`。真要省性能，先确认设备能正常出图，再用 `?xrss=0.8` 试。

> ⚠️ **注视点不要默认拉满 1.0**。它按着色率（shading rate）分块实现：边缘每 N×N 像素
> 才真正着色一次，细长 / 面积小 / 远的几何能不能被采样到取决于**亚像素对齐** —— 左右眼
> 是两个独立采样网格，于是可能出现「左眼有、右眼整只机器人没了」。地面连成一片所以看不
> 出来，表现为「只有机器人丢」。PICO 4 上 1.0 能稳定复现，默认值因此是 0.5。

> 真机上跑 `__elf.perf()` 看实际数字。单人：`tris ≈ 41 万`（双目）；矩阵 96 份：
> `tris ≈ 3960 万`，剔除后约 `2000 万`；`calls` 60~100。

### 4.4 流畅度：预加载 + 插值 + 剔除

这一批改动都**不改画面观感**，只在「同样的画面」上少做无用功。

**① 回放数据预加载（`motion.js`）**

原来切一段回放要等 `motion_*.json` + `.bin` 两个网络往返（Wi-Fi 上 0.3~2 s），
VR 里就是纯冻帧。现在：

- `MotionPlayer.load` 带进程内缓存（`_cache`）与 in-flight 去重（`_pending`），
  同名数据只下载一次，切回去是同步操作；
- 页面就绪后由 `schedulePrefetch()` 在**空闲时**并发 6 路预取全部 14 段（3.5 MB），
  默认片 `fight_policy` 排第一、1.76 MB 的 `fight_ref` 压轴；
- 服务端对 `motion_*.bin` 发 `immutable`（URL 自带 `?v=<version>`，重烘后 URL 会变），
  其余文件仍是 `no-cache` 并新增 `ETag` / 304 —— **`.json` 绝不能 immutable**，
  它的 URL 不带版本号，缓存住会永远读到旧 version。

**② 插值曲线（`motion.js: writeQpos`）**

线性插值的速度在每 20 ms 边界上是分段常数，而 72/90 Hz 与 50 Hz 不整除，折点会不断
落在不同的显示帧上 → 节拍性「顿一下」。改成均匀 Catmull-Rom（C1 连续）后同一份数据
明显更顺，**不需要重新烘焙**。

两个护栏：`a == 0` 时逐位命中关键帧（端点断言依赖它）；每通道结果钳在相邻两帧之间
（循环接缝处有几段数据首尾差 2~10 m，不钳制会把「瞬移」摊成 5~6 帧滑行）。实测只有
0.4%~6% 的采样点被钳到。`?interp=linear` 可回退。

**③ 矩阵实例逐实例剔除（`mjscene.js: cullGrid`）**

`InstancedMesh` 的包围球罩住整片网格，three 只能「整体画 / 整体不画」，所以 96 份以前
是**全画**。VR 里按头显双目视锥逐份判包围球，把不在视野里的剔掉：站在机器人前 2.6 m
的典型站姿下**约 96 份只剩 46 份**，几何负载直接砍半（`__elf.perf().gridDrawn` 可查）。
桌面/无头不剔除（`count` 恒为 96），`?cull=0` 可关。

**④ 其他（都不改观感）**

| 改动 | 位置 | 说明 |
| --- | --- | --- |
| 共享材质 | `mjscene.js` | 32 个视觉 geom 同色，材质从 32 份减到 **1 份**（key 按量化后的 rgba，`0.75294` 与 `0.752941` 必须算同一个） |
| 帧时间从 XR timestamp 取 | `main.js: frame(time)` | 不再用 `clock.getDelta()`（它把本帧自身耗时也算进 dt，掉帧时正反馈放大抖动）；`perf().ft` 给 p50/p95/max/std —— **p95/p50 ≈ 1 说明稳定被 GPU 卡住，> 2 说明有长任务** |
| 删掉重复的全树矩阵更新 | `vrnav.js` | 5 处 `stage.updateMatrixWorld(true)` 删除：`renderer.render` 每帧本来就会做 |
| 静态 geom 只写一次 | `mjscene.js` | 地板和地形盒位姿恒定，不再每帧重写 |
| 预热着色器 | `enterXRQuality` | 关阴影会让 program 失效，趁会话刚建立时先 `renderer.compile`，别让重编译落到第一帧 |
| HUD 不在 XR 里写 | `main.js` | 沉浸态 DOM 不会合成，写 `textContent` 只是标脏样式/布局 |
| 面板各向异性 8 → 2 | `xrpanel.js` | 面板是 1.57:1 缩小采样，8× 没有可见收益却要每片元 8 次 |

> **算术上限**：矩阵 96 份 × 20.6 万 = 1980 万三角/单目，剔除后约 1000 万/眼，而 XR2 在
> 72 Hz 的几何预算约 140~210 万/帧 —— 仍超 5 倍。所以矩阵模式在 VR 里注定上不到 72 Hz，
> 这一批只能把它从「幻灯片」救到「20~30 fps + 重投影可用」。要真流畅必须动份数或几何
> （见 `.workbuddy/plans` 里的下一轮清单）。

### 4.5 机器人材质：默认省片元的 MeshLambert（`?mat=standard` 切回 PBR）

上面 4.3/4.4 都不改观感；这一条是**故意的有损**，用来换帧率。现在**默认就是 Lambert**
（PBR 的 GGX + 环境贴图片元太贵，一体机跑不动矩阵模式），想看金属质感再显式切回
`MeshStandardMaterial`。

| | `lambert`（默认） | `standard`（`?mat=standard`） |
| --- | --- | --- |
| 材质 | `MeshLambertMaterial`（逐顶点漫反射） | `MeshStandardMaterial` + PMREM 环境贴图 |
| 参数 | 只有 `color` | `metalness 0.72 / roughness 0.34 / envMapIntensity 1.15` |
| 片元成本 | 约 **1/2~1/3** | GGX 求值 + 2 次 cubemapUV 采样 + 一整套 BRDF |
| 观感 | 哑光塑料、整体偏亮、暗部变平 | 金属高光、暗部有层次、头部/肩部有环境反射 |

只换材质引用（几何、矩阵、实例化都不动），矩阵模式下 32 个 `InstancedMesh` 会跟着换。
启动：`?mat=standard`；运行期也能切：`__elf.setRobotMaterial('standard')` / `('lambert')`，
`__elf.perf().robotMaterial` 可查。

> 实测观感差异**比预期小**：机器人本来是近白色（rgba 0.753）+ 两盏很亮的平行光 + 半球光，
> PBR 的金属感在这种浅色表面上本来就不强。主要损失是**暗部对比度**和头部那块镜面高光的形状。
> 对比图见 `logs/mat_compare/side_by_side.png`（左 PBR / 右 Lambert，上矩阵下近景）。
>
> 怀疑「单眼丢物体」时先跑 `__elf.xrDiag()`（会话内、退出前都行）：它把**左右眼各自的视口 +
> 机器人包围盒在那一只眼里的 NDC 范围**打出来。两眼不一致 → 视口 / 投影问题；
> 两眼几乎一致而画面里确实还在丢 → 几何没丢，丢的是**片元**（降采样 / FFR），
> 把注视点调成 0 复测即可确认（`?ffr=0` 或 `__elf.xrQuality.foveation(0)`）。

---

## 5. 界面功能

| 控件           | 说明                                          |
| ------------ | ------------------------------------------- |
| 场景           | **单人**（1 个机器人）/ **矩阵**（8 行 × 12 列 = 96 份同步播放，默认）。矩阵用实例化渲染，绘制批次不变；地形与阴影不复制。VR 里同样可用 |
| 机器人          | 显示 / 隐藏 32 个视觉网格（group 2，含两只手的 mesh）；矩阵模式下控制的是 96 份实例 |
| 楼梯 / 障碍      | 显示 / 隐藏 `scene_bx.xml` 里的地形（`universal_test` 紧凑障碍：10 cm 低楼梯 / 15 cm 高楼梯 / 壕沟堤岸） |
| 地面网格         | 显示 / 隐藏地板                                   |
| 阴影           | 开关实时阴影                                      |
| 碰撞体（调试图层）    | 显示 group 3 的 36 个碰撞几何（绿色半透明 capsule）        |
| 视角预设         | 正面 / 背面 / 侧面 / 头部特写 / 俯视 / 螃蟹步 / **蹲起 3/4** / **矩阵全景** / 全景 |
| 运行 MuJoCo 物理 | 默认**关闭**；打开后真实跑 `mj_step`，机器人会因重力倒地（模型无控制器） |
| 复位姿态         | 回到落地对齐后的静止站立姿态                              |
| 复位世界位置       | 把世界平移 / 旋转复位                                |

---

## 6. 目录结构

```
web/
├── index.html            页面骨架（importmap + 三个面板 + 加载层）
├── css/app.css           全部样式
├── js/
│   ├── main.js           启动流程、渲染循环、UI、WebXR 会话、调试句柄 __elf
│   ├── mjscene.js        MuJoCo（Z-up）→ three.js（Y-up）场景同步
│   ├── motion.js         运动仿真回放：读 motion_*.bin，位置插值 + 四元数 slerp
│   ├── vrnav.js          WebXR 漫游：定位、摇杆移动、连续转向、跟随机器人
│   ├── xr.js             XR 入口按钮：进入 VR / 半透明 VR（透视），会话与背景合成
│   └── xrpanel.js        头显内设置面板：canvas 画控件 → CanvasTexture 贴图 + 手柄射线命中
├── vendor/
│   ├── mujoco.js/.wasm   MuJoCo 官方 WebAssembly 构建（simulate 版，含 mjv*）
│   └── three.module.js   three.js r170
├── jsm/                  three.js addons（OrbitControls / RoomEnvironment）
├── assets/
│   ├── elf3.xml          机器人模型（第 463 行 <include file="scene_bx.xml"/>）
│   ├── scene_bx.xml      地形：universal_test 紧凑障碍（地面 + 低楼梯 / 高楼梯 / 壕沟堤岸）
│   ├── scene_crab.xml    地形：螃蟹步横移台阶（沿 +Y 上升，bake_crab.py 生成）
│   ├── meshes/*.STL      34 个网格
│   ├── motion_fight_policy.bin/.json   策略仿真回放数据（bake_motion.py 生成）
│   ├── motion_fight_ref.bin/.json      参考动作回放数据
│   ├── motion_climb_policy.bin/.json   爬楼梯回放数据（bake_climb.py 生成）
│   ├── motion_crab.bin/.json           螃蟹步横移回放数据（bake_crab.py 生成）
│   ├── motion_squat_{down,up,cycle}.bin/.json   蹲下 / 起立 / 无缝循环回放数据
│   │                                   （scripts/export_squat_npz.py → bake_squat.py）
│   └── manifest.json     资源清单（加载进度用；不含回放数据）
├── certs/                自签证书（首次启动自动生成）
├── server.mjs            本地服务器：HTTP + HTTPS 双监听
├── start.bat / start.sh               一键启动（静止站立查看）
├── start_vr_motion.bat / .sh          一键启动 + 自动进入运动仿真回放
└── tools/                构建 / 调试工具（见 §8）
```

---

## 7. 实现要点（踩过的坑）

### 7.1 网格顶点必须用 MuJoCo 编译后的数据

MuJoCo 编译 STL 时会把顶点**重心归零**，偏移量记在 `model.mesh_pos` 里  
（例如大腿 `l_hip_z_link` 的 `mesh_pos.z = -0.2229`）。

> **错误做法**：直接拿 STL 原始顶点 × `geom_xmat` + `geom_xpos`。  
> 漏掉 `mesh_pos` 后，腿部会整体错位 22 cm，看起来像"腿断了"。

**正确做法**：直接用 `model.mesh_vert` / `model.mesh_face` / `model.mesh_normal`  
（已经做过重心归零，和 MuJoCo 内部渲染逐顶点一致），再用 `data.geom_xpos` /  
`data.geom_xmat` 定位。这样也不需要在前端解析 STL。

### 7.2 坐标映射

MuJoCo 是 Z-up，three.js / WebXR 是 Y-up：

```
(x, y, z)_mujoco  →  three.js 里套两层 Group：
  flip.rotation.x = -π/2      Z-up → Y-up
  yaw .rotation.y = +π/2      机器人正面（MuJoCo +x）朝向 three 的 -Z
```

### 7.3 姿态与落地对齐

- 静止站立 = `mj_forward` 后把 `data.geom_xpos/geom_xmat` 一次性写进 three.js 对象，  
  默认**不跑物理**（`renderer.shadowMap.autoUpdate = false`，省开销）。
- 模型初始 free joint 高度（`qpos0[2] = 1.1`）会让脚底离地约 3.7 cm，  
  所以加载时按视觉网格最低点自动对齐一次，让脚底正好落在 `z = 0`。  
  复位按钮用的也是这个对齐后的姿态。

### 7.4 VR 里的漫游

相机固定在世界原点（由 three.js 用头显姿态驱动），**平移 / 旋转的是装着整个  
MuJoCo 世界的 `stage` 组**，等价于世界坐标系漫游，不会和 XR 的头显位姿打架。

### 7.5 半透明 VR（透视）是怎么实现的

「看到眼镜外面」= WebXR 的 passthrough 合成，关键链条有四环，缺一不可：

1. **WebGL 上下文带 alpha 通道**：`new THREE.WebGLRenderer({ alpha: true })`。
   three 给 XR 会话建 `XRWebGLLayer` 时写死 `alpha: true`，但层缓冲的 alpha
   能不能用，取决于上下文本身 —— 没有 alpha 通道就永远透不出去。
2. **会话模式决定混合模式**：`environmentBlendMode` 是只读的，由 UA 按
   `immersive-vr`（→ `opaque`）/ `immersive-ar`（→ `alpha-blend`）决定。
   所以「半透明」必须走 `immersive-ar`，普通 VR 会话做不到。
3. **别让虚拟背景填满画面**：three r170 的 `WebGLBackground` 在 alpha-blend 下
   会把清屏 alpha 置 0，但 `scene.background` 若是贴图，会先画一个全屏四边形、
   alpha 写成 1，现实就被盖住了 → 进入透视时把 `scene.background` 与
   `scene.fog` 置 null（雾会把远处物体拉向雾色，透视下不合适）。
4. **地面别挡住真实地面**：不透明的虚拟地板会把真实地板整个盖掉，
   所以透视时把地面材质调成半透明（`MjScene.setFloorOpacity(0.3)`，
   同时关 `depthWrite`），既保留位置参照又能看到真实地面。

这些全部封装在 `js/xr.js` 的 `XRControls` 里（会话生命周期、参考空间探测、
按钮状态、透视开关 `setSeeThrough()`），`main.js` 只在 `onEnter / onExit /
onSeeThrough` 三个回调里接 UI。`setSeeThrough()` 幂等且不依赖真头显，
`tools/verify.mjs` 直接调它做回归。

---

### 7.6 爬楼梯回放（`motion_climb_policy`）的三个关键点

1. **关节顺序对账** —— 物理仿真跑的是官方 `data/mujoco_simulation/elf3.xml`
   （nq=38：自由基座 7 + 29 个策略关节 + `head_z_joint` / `head_y_joint` 两个头部
   关节）；网页端用的是老版 `assets/elf3.xml`（nq=36，没有那两个头部关节）。
   两者前 36 个 qpos 的**关节顺序与 `jnt_qposadr` 完全一致**，所以
   `bake_climb.py` 是按关节名逐一对齐后写出的（当前就是原样截取 `qpos[:36]`）——
   哪一侧以后调了关节顺序也能自动重排，不会录出"关节乱掉"的回放。
2. **地形拍平** —— `assets/scene_bx.xml` 现在是 `terrains/universal_test/` 下
   `assets.xml`（材质）+ `base.xml`（光照 / 地面 / 出生点）+ `compact_obstacles.xml`
   （低楼梯 / 高楼梯 / 壕沟）三个文件的拍平版，几何尺寸、位置、摩擦逐条照抄。
   之所以拍平而不是改 `elf3.xml` 的 include：网页端是 `from_xml_string(xml, vfs)`
   编译的，VFS 里没有"文件所在目录"的概念，嵌套 include 的相对路径不可靠。
3. **世界坐标必须一致** —— 回放写的是**绝对 qpos**，所以网页里的地形要与录制时
   逐条相同（`pos` / `size` 完全同值），`bake_climb.py` 也是直接拿官方那份
   "机器人 + 地形"完整场景跑的。

两个小细节：

- 地面 geom 仍叫 `floor`（官方叫 `ut_infinite_ground`）：`elf3.xml` 的 `<contact>`
  里有 14 条足底-地面 pair 写死了 `geom2="floor"`，改名会让模型编译失败。
- 地形里的起点 / 终点 / 壕沟断口标记在官方 XML 里是 `<site>`，而
  `js/mjscene.js` 只渲染 geom —— 所以它们在网页里不显示（MuJoCo 原生查看器 /
  Python 渲染里照常可见）。

---

### 7.7 螃蟹步横移回放（`motion_crab`）

这一段和前两段的性质不同：**没有策略网络**。轨迹由
`sim/.../launch/crab_climb.py` 的四相位准静态控制器解出来 ——
闭式 IK 给种子、DLS 数值精修到 0.00 mm，每 20 ms 一帧，网页端照常插值回放。

三个要点：

1. **多了一套地形** —— 螃蟹步的楼梯沿 **+Y** 上升（机器人朝 +X，侧身横移），
   和沿 +X 排布的紧凑障碍是两回事。网页端的 MuJoCo 模型在启动时一次编译完成，
   不能中途换地形，所以台阶被抽成独立的 `assets/scene_crab.xml`，由
   `scene_bx.xml` 的 `<worldbody>` 末尾 `<include>` 进来。
   两套地形在空间上不重叠，靠 geom 名前缀 `crab_` 分成两组显隐
   （`js/mjscene.js` 的 `crabGeomIds` / `showCrab`）。
2. **台阶是生成的，不是手写的** —— `gen_crab_stairs.py` 按 seed 随机生成
   级高 12~25 cm / 进深 20~40 cm 的台阶，同时给出每级的 `tilt`（骨盆侧倾）和
   `off`（双脚落点偏移）真值表。改 seed 就换一套楼梯，控制器自动适配。
   所以 `bake_crab.py` 每次都会重写 `scene_crab.xml` 并刷新 `manifest.json`
   —— 地形和回放数据必须成对更新，否则机器人会踩空。
   注意 `web/tools/make-manifest.mjs` 也已同步加上这个文件；手工跑它同样安全。
3. **机位** —— 台阶剖面在 YZ 平面上，所以虚拟相机必须站在机器人**侧方**
   （MuJoCo +X 侧，three 里是 -Z 方向）才看得见"台阶"；站在沿台阶轴的方向
   只会看到一堵斜面墙。`VIEWS.crab` 就是按这个算的，选这段回放时自动切过去。

一个已知的物理事实（不是 bug）：这个步态**本质上是动态的**。站姿间距约等于
踏面进深（20~40 cm），而脚掌沿行进方向只有 8 cm 宽，重心完全移到支撑脚时后腿
的侧移需求 ρ ≈ 0.71 m > 腿长 0.64 m —— 够不到。所以抬脚与重心转移必须同时进行：
体检里约 43% 的帧质心落在支撑面外，最长连续 0.24 s。纯运动学回放看不出来，
真跑 `mj_step` 会摔 —— 这正是后续 RL 要解决的问题。

> 完整链路（随机台阶 → 四相位准静态 IK → 烘焙 → 浏览器插值回放）、数据格式、
> 产物清单与排错对照表：见 `docs/crab_gait_pipeline.md`。

---

### 7.8 蹲下 / 起立参考动作回放（`motion_squat_*`）

数据从 `scripts/export_squat_npz.py` 来：`squat_down.npz`（站姿 → 极限蹲，124 帧）
与 `squat_up.npz`（极限蹲 → 站姿，102 帧），都是 50 Hz、`qpos [T,38]`。
`web/tools/bake_squat.py` 干四件事：

1. **38 → 36 维按关节名重映射**。npz 的布局是
   `0:7 自由基座 | 7:36 29 关节 | 36:38 head_z/head_y`，网页模型 `elf3.xml`
   没有那两个被动头关节。脚本按 `jnt_qposadr` + npz 的 `names` 生成取列索引，
   当前恰好就是原样取前 36 列；将来任一侧改了顺序也不会「关节乱掉」。
   同时断言 `q == qpos[:,7:36]`、`z == qpos[:,2]`、`dt == 0.02`、单帧跳变 < 0.30 rad。
2. **逐帧落地自检（最容易踩的坑）**。按 `MjScene.robotMinZ()` 的**同一套公式**
   算视觉网格最低点：`mesh_vert` 顶点乘 `geom_xmat` + `geom_xpos`。
   **不要再叠加 `model.mesh_pos`** —— MuJoCo 编译 STL 时已把质心偏移折进
   `geom_xpos`，重复加会让算出来的脚底低约 7.6 cm，进而误判「轨迹穿地」并
   给整段回放加一个错误的 z 偏移。当前这批 226 帧实测最低点全为 0.00 cm，
   所以 `zOffset = 0`；超出 ±1.2 cm 会直接报错，可用 `--z-offset` 手动补偿。
3. **拼无缝循环**。`down[-1] == up[0]` 且 `up[-1] == down[0]`（脚本断言两者
   `max|Δ| < 1e-9`），所以 `cycle = down + 蹲姿保持 + up[1:] + 站姿保持`，
   保持段各 50 帧（1 s），共 323 帧 / 6.46 s，首尾帧完全相同 → 循环处不跳。
4. **写出 + 回归**。复用 `bake_climb.save()` 的 `motion_*.bin/.json` 格式，
   落盘后再核对 `nq=36`、帧数、`loop` 字段和 bin 字节数。

`meta.loop` 是前端行为开关：单段 `down` / `up` 写 `false`，`MotionPlayer`
播完**冻结在末帧**（`progress` 停在 1，暂停按钮自动变「继续」），与真机部署的
`freeze_at_end` 语义一致；循环版不设该字段，默认 `true`。这一段也修掉了
`MotionPlayer.seek()` 在 `time == duration` 时用取模回到 0 的老问题（原来只在
`loop=true` 的回放里用，看不出来）。

页面侧（`main.js`）给 `kind === 'squat'` 加了三件事：隐藏两套台阶、切 3/4 机位、
显示「参考动作」说明。`js/motion.js` 的 `describe()` 把四类轨迹统一显示成
「参考动作」（只有 onnx 闭环录制才叫「策略仿真」）。
`tools/verify.mjs` 新增自检（下拉项 / 循环首尾帧 / 基座高度变化 / 地形与机位 /
HUD / 冻结末帧 / 蹲姿高度 / 暂停按钮 / 从头播放）。

> 完整部署语义（时长、结束状态、ONNX、npz 键）：见 `docs/squat_rl_training.md` §6~§9。

---

## 8. 工具（`tools/`）

| 脚本                  | 用途                                                   |
| ------------------- | ---------------------------------------------------- |
| `verify.mjs`        | **端到端自检**：加载 / 模型 / UI / 场景（单人 · 矩阵 96 份）/ 开关 / 视角 / 物理 / 复位 / 回放（打拳 + 爬楼梯 + 螃蟹步 + 蹲起 + 贴地爬行）/ XR（进 VR 保持矩阵 + 可自由切换 + 降载 + 帧时间 + 预取 + 插值不越界 + 实例剔除）/ 半透明透视 / 头显面板（最小控件集 + 点击派发 + 50% 透明切换 + 头锁置顶 + 覆盖层撑高 + 收起小签 + 右摇杆连续转向配置 + 材质切换），125 项 |
| `shoot.mjs`         | 用 CDP 等页面就绪后截图（可注入 JS 改相机、改可见性）                      |
| `dump-geoms.mjs`    | 列出每个视觉网格的世界 AABB，用于和 MuJoCo 对账                       |
| `make-manifest.mjs` | 重新生成 `assets/manifest.json`                          |
| `bake_motion.py`    | 本机跑 MuJoCo + `policy.onnx`，把每步 qpos 录成 `assets/motion_*.bin` |
| `bake_climb.py`     | 本机跑 MuJoCo + `normal_depth.onnx` 深度视觉闭环（复用 climb stairs 工作区的 `climb_stairs.py`），把爬楼梯的每步 qpos 录成 `assets/motion_climb_policy.bin` |
| `bake_crab.py`      | 跑 `gen_crab_stairs.py` + `crab_climb.py`（脚本烘焙的参考动作，无策略网络），录成 `assets/motion_crab.bin`，并把台阶几何同步到 `assets/scene_crab.xml` |
| `bake_squat.py`     | 把 `logs/export/squat_{down,up}.npz` 的 38 维 qpos 按关节名对账成网页 36 维，逐帧做落地自检，烘焙出 `motion_squat_{down,up,cycle}.bin/.json`（纯运动学，不跑策略） |
| `bake_crawl.py`     | 把 `scripts/build_prone_crawl_sequence.py` 合并的贴地爬行序列（38 维）按关节名重映射成 36 维，四元数归一化 + 站姿段落地自检，烘焙出 `motion_crawl.bin/.json`（趴姿接触点是手/躯干，不套用脚底全程贴地判据） |
| `fetch-three.mjs`   | 下载 three.js 与 addons                                 |
| `selfsign.mjs`      | 纯 JS 生成自签证书（无 openssl 依赖，SAN 含本机所有内网 IP）             |
| `make-cert.mjs`     | 证书生成的封装                                              |
| `ref-render.py`     | 用 MuJoCo 官方 Python 渲染器出参考图（需另装 `mujoco`）             |

自检：

```bash
node tools/verify.mjs http://localhost:8080/
node tools/verify.mjs https://192.168.22.19:8443/   # 头显走的这条路
```

---

## 9. 常见问题

**Q：双击 `start.bat` 没反应 / 窗口一闪而过 / `http://localhost:8080` 打不开？**  
A：`start.bat` 和 `start_vr_motion.bat` 是 **cmd 批处理**，而 `cmd.exe` 是**按字节**逐行
解析它们的，用的是系统 OEM 代码页（中文 Windows = CP936/GBK）。所以这两个文件必须是
**纯 ASCII + CRLF 行尾**：一旦写进 UTF-8 中文（一个汉字 3 字节，按 GBK 两字节分组会
**吞掉后面的换行**）或改成裸 LF，解析立刻错位，**其后所有命令都被当成乱码命令丢弃** ——
表现就是双击秒关、端口根本没起来。  
`chcp 65001` **救不了这个问题**：它只改控制台输出的代码页，不改变 bat 文件自身被解析的
方式。  
（2026-10-13 已修复这两个文件。）如果你要改它们，请保持纯英文注释和提示，并用 Python
写文件：`open(p, "wb").write(text.replace("\n", "\r\n").encode("ascii"))` ——
别用记事本「另存为 UTF-8」。

**Q：点「半透明 VR」没反应 / 按钮是灰的？**  
A：透视走的是 WebXR `immersive-ar`，需要**带彩色摄像头的头显**（Quest 2/3/Pro
的浏览器都支持）。桌面 Chrome / 手机浏览器不支持，所以按钮显示「透视不可用」；
Quest 1 没有彩色摄像头也不支持。另外首次进入时系统会请求「摄像头 / 透视」权限，
拒绝了就进不去，可在头显的站点设置里重新允许。

**Q：头显里打开页面没有 ENTER VR？**  
A：① 必须用 https 地址；② 确认头显和电脑在同一局域网、能 ping 通；③  
用头显自带浏览器（Quest 用 Meta Quest Browser），不要用某些内置的简易浏览器。

**Q：提示证书不安全 / NET::ERR_CERT_AUTHORITY_INVALID？**  
A：自签证书的正常现象，选「高级 → 继续访问」。想免提示就把  
`certs/cert.pem` 装进头显的受信任证书列表。

**Q：换了网络环境（IP 变了）后头显打不开？**  
A：删除 `certs/` 目录后重启服务器，会按当前 IP 重新生成证书。

**Q：一体机（PICO 4 / Quest）进 VR 后卡顿、连手柄都迟钝？**  
A：现在进 VR 会**关阴影 + 关 MSAA**（帧缓冲默认不缩，见 §4.3），但**场景保持你的选择**——
如果进去时是矩阵 8×12，那就是 1980 万三角/单目、3950 万/双目，一体机 GPU 差一个数量级，
必然掉帧。想流畅：在头显面板里点「单人」（41 万/双目），或启动时加 `?scene=single`。如果还卡：
① `?xrss=0.8` 再把帧缓冲降一档（画面会软一点；**若进入后全黑就退回 1.0**）；
② 确认走的是 `https` 且没在下载别的（Wi-Fi 抢带宽会表现为掉帧）；
③ 真机连 `chrome://inspect` 远程调试，跑 `__elf.perf()` —— 单人下 `tris` 应在 40 万上下、
`calls` 60~100、`shadows.enabled === false`；矩阵下 `tris` 约 2600 万是**正常**的。

**Q：进入 VR 后全黑，连机器人和地面都没有？**  
A：先看地址栏有没有 `?xrss=0.8` —— 投影层缩放会在 PICO 4 Ultra 上让合成器丢掉整帧，
去掉参数或显式加 `?xrss=1` 再进一次（默认已经是 1.0）。如果还黑，再试 `?ffr=0`；
仍黑就用 `chrome://inspect` 看控制台里「· XR 帧缓冲 W×H」是不是 `0×0`
（`xr.js` 现在会在尺寸为 0 时打一条显式告警）。

**Q：切一段回放要等一两秒（VR 里是纯冻帧）？**  
A：页面就绪后会自动在空闲时预取全部 14 段（3.5 MB），之后切段是内存命中
（实测 1.4 ms）。如果关掉了预取（`?noprefetch=1`）或者预取还没跑完就切，仍会走网络。
另外服务端对 `motion_*.bin` 发了 `immutable`（URL 自带 `?v=`），`motion_*.json` 走
`ETag`/304，所以第二次刷新页面不会重下模型和回放数据。

**Q：动作看起来一顿一顿的（不是掉帧）？**  
A：50 Hz 源数据 + 原来的线性插值在每 20 ms 边界上速度突变，而 72/90 Hz 与 50 Hz 不整除，
折点会不断落在不同的显示帧上。现在默认用 Catmull-Rom（C1 连续），同一份数据明显更顺。
想对比：`?interp=linear` 回退旧曲线。**不要**靠调 `?speed=1.5` 看，那个在 72 Hz 下本来
就欠采样（每显示帧推进 20.8 ms > 20 ms 源帧间隔）。

**Q：加载很慢？**  
A：首次要下载约 10 MB 的 MuJoCo WASM + 约 12 MB 的 STL 网格 + 3.5 MB 回放数据，
之后浏览器会缓存（`ETag`/304 + `.bin` 的 `immutable`）。  
加载页有进度条和实时日志。

**Q：面板里选「策略仿真 · 打拳」提示"加载失败"？**  
A：缺 `assets/motion_fight_policy.bin`。跑一次 `python web/tools/bake_motion.py`
生成即可（或直接用 `start_vr_motion.bat` / `.sh`，它会自动补上）。

**Q：面板里选「策略仿真 · 爬楼梯」提示"加载失败"？**  
A：缺 `assets/motion_climb_policy.bin`。跑一次 `python web/tools/bake_climb.py`
生成即可。注意它是**深度视觉闭环**，需要 vendor 里的 `climb_stairs.py` +
`normal_depth.onnx` + `universal_test` 地形；工作区路径默认由脚本自身位置推导，
目录搬过家也不用改，特殊情况才用 `--ws <路径>` 指定。

**Q：选了「参考动作 · 蹲下起立循环」（或 `?motion=squat_down`）提示"加载失败"？**  
A：缺 `assets/motion_squat_*.bin/.json`。依次跑
`python scripts/export_squat_npz.py` 和 `python web/tools/bake_squat.py` 即可，
全程只依赖本机 numpy + mujoco，不需要 GPU 和训练。若报「脚底没有贴在 z=0 上」，
说明轨迹与 `web/assets/elf3.xml` 不配套，按提示用 `--z-offset` 补偿。

**Q：为什么蹲起回放里的机器人不会倒，也不像训练回放那样有抖动？**  
A：这几段是**纯运动学参考动作**，网页端只做 `mj_kinematics` + 插值摆位，
不跑 `mj_step`、没有重力和接触，也不加载 ONNX 策略。要看 RL 策略实际控制的
效果，得在服务器把闭环仿真的 qpos 录下来再烘焙；详见 §7.8 与
`docs/squat_rl_training.md` §9。

**Q：面板里选「参考动作 · 螃蟹步横移上楼」提示"加载失败"？**  
A：缺 `assets/motion_crab.bin`（或 `assets/scene_crab.xml`）。跑一次
`python web/tools/bake_crab.py` 生成即可，它会顺带写好台阶几何和 `manifest.json`。
换一套台阶：`python web/tools/bake_crab.py --seed 7`。

**Q：选「参考动作 · 螃蟹步横移上楼」后看不到台阶？**  
A：地形由「显示 → **螃蟹步台阶**」这个复选框控制，选回放时会自动勾上。
如果手动取消了，机器人会像在踏空 —— 再勾回来即可。注意它和「楼梯 / 障碍」是
两套独立地形，选不同回放时脚本会自动切换，也可以两个都勾上看全景。

**Q：选了「策略仿真 · 爬楼梯」，但机器人脚下没有地形？**  
A：正常情况下选这一项会自动把「楼梯 / 障碍」勾上。如果你手动取消勾选，地形就会
藏起来而机器人仍在爬（看起来像在踏空）—— 再勾回来即可。

**Q：页面显示"启动失败"？**  
A：加载层会直接打印错误信息。常见原因是资源没就位——确认是在 `web/` 目录下、  
通过 `server.mjs` 访问（直接双击 `index.html` 用 `file://` 打开会因为  
CORS 与 WASM 限制而失败）。
