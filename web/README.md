# elf3 · MuJoCo 网页端仿真 + VR 沉浸式观看

在浏览器里跑**真正的 MuJoCo 物理引擎**（官方 WebAssembly 构建），把 `elf3.xml` +  
`scene_bx.xml`（地形）渲染成三维场景，并支持用 **VR 头显沉浸式观看**。  
默认状态是**静止站立**：机器人保持 XML 里的初始位形，不跑物理。  
也可以在网页 / VR 里回放**策略跑出来的运动仿真**：`elf3_play.py` 那条
MuJoCo + `policy.onnx` 闭环、跟踪 `fight.npz` 打拳动作的画面，以及
`climb_stairs.py` 那条**深度视觉闭环爬楼梯**的画面（地形为 `universal_test`
紧凑障碍）—— 见 §3。

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
- 右侧面板：显示开关（机器人 / 楼梯 / 地面 / 阴影 / 碰撞体）、6 个视角预设、  
  物理开关与复位按钮
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

浏览器里的 MuJoCo WASM 没有策略网络（没有 onnxruntime），所以网页端**不现算策略**：
先在本地把仿真跑一遍，把每个控制步（50 Hz）的 `qpos` 录成 float32 二进制
（`assets/motion_*.bin` + 同名 `.json`），网页端按时间插值回放（位置线性插值、
四元数最短路径 slerp），再用 `mj_kinematics` 刷新几何位姿 —— 50 Hz 数据 + 插值，
VR 里动作平滑不抖。

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

# —— 螃蟹步横移上楼（crab_climb.py 脚本步态，无策略网络）——
python web/tools/bake_crab.py                         # 全链：生成随机台阶 → 规划轨迹 → 烘焙
python web/tools/bake_crab.py --seed 7 --rise-max 0.18        # 换一套台阶重录
python web/tools/bake_crab.py --step-time 1.2                 # 走慢一点（更容易站住）
python web/tools/bake_crab.py --skip-gen --skip-plan          # 只重烘焙已有轨迹
```

> `bake_crab.py` 除了写 `motion_crab.*`，还会把台阶几何同步成
> `assets/scene_crab.xml` 并刷新 `manifest.json` —— 网页端是"一次编译"的模型，
> 地形必须和回放数据一起就位。

面板上的控件：

| 控件 | 说明 |
| --- | --- |
| 回放 | 关闭（静止站立）/ 策略仿真 · 打拳 / 参考动作 · 打拳 / 策略仿真 · 爬楼梯 / **脚本步态 · 螃蟹步横移上楼** |
| 倍速 | 0.25× / 0.5× / 1× / 1.5× |
| 视角跟随机器人 | 机器人会满场走动（世界坐标约 ±5 m），打开后镜头自动跟住它 |
| 从头播放 / 暂停 | 控制回放进度（「复位姿态」在回放中也等价于从头播放） |

几点说明：

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
- 开启回放会**自动关掉「运行 MuJoCo 物理」**：两者互斥（一个是回放录好的 qpos，
  一个是本地现算 `mj_step`）。
- 打拳策略在 t≈148 s 会失稳摔倒，所以录制在**最后一次姿态健康**的时刻截断
  （倾角 < 30°、根高 > 0.80 m），回放里不会看到倒地画面；爬楼梯那段则是一路
  跑到越过终点线（x > 13.4 m）自动收尾，全程未摔倒。
- VR 里的「视角跟随」是**舒适窗口式**的：机器人在 1.7 ~ 4.8 m 内且大致在身前就
  完全不动，跑出这个范围才慢慢把它拉回正前方 —— 不会和你自己的摇杆移动打架。
  A / X「把机器人摆到正前方」也按它的**实际位置**摆位（回放时它不在世界原点）。
- URL 参数：`?motion=fight_policy&speed=0.5&follow=0`
  （一键脚本用的就是这条；爬楼梯用 `?motion=climb_policy`，螃蟹步用 `?motion=crab`）。

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
| 右摇杆左右 | 30° 转向                |
| A / X | 把机器人摆到你正前方 2.6 m 并正对你 |
| B / Y | 世界复位（机器人回到世界原点）       |

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

---

## 5. 界面功能

| 控件           | 说明                                          |
| ------------ | ------------------------------------------- |
| 机器人          | 显示 / 隐藏 30 个视觉网格（group 2）                   |
| 楼梯 / 障碍      | 显示 / 隐藏 `scene_bx.xml` 里的地形（`universal_test` 紧凑障碍：10 cm 低楼梯 / 15 cm 高楼梯 / 壕沟堤岸） |
| 地面网格         | 显示 / 隐藏地板                                   |
| 阴影           | 开关实时阴影                                      |
| 碰撞体（调试图层）    | 显示 group 3 的 36 个碰撞几何（绿色半透明 capsule）        |
| 视角预设         | 正面 / 背面 / 侧面 / 头部特写 / 俯视 / 全景               |
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
│   ├── vrnav.js          WebXR 漫游：定位、摇杆移动、快速转向、跟随机器人
│   └── xr.js             XR 入口按钮：进入 VR / 半透明 VR（透视），会话与背景合成
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

## 8. 工具（`tools/`）

| 脚本                  | 用途                                                   |
| ------------------- | ---------------------------------------------------- |
| `verify.mjs`        | **端到端自检**：加载 / 模型 / UI / 开关 / 视角 / 物理 / 复位 / 回放（打拳 + 爬楼梯）/ XR / 半透明透视，51 项 |
| `shoot.mjs`         | 用 CDP 等页面就绪后截图（可注入 JS 改相机、改可见性）                      |
| `dump-geoms.mjs`    | 列出每个视觉网格的世界 AABB，用于和 MuJoCo 对账                       |
| `make-manifest.mjs` | 重新生成 `assets/manifest.json`                          |
| `bake_motion.py`    | 本机跑 MuJoCo + `policy.onnx`，把每步 qpos 录成 `assets/motion_*.bin` |
| `bake_climb.py`     | 本机跑 MuJoCo + `normal_depth.onnx` 深度视觉闭环（复用 climb stairs 工作区的 `climb_stairs.py`），把爬楼梯的每步 qpos 录成 `assets/motion_climb_policy.bin` |
| `bake_crab.py`      | 跑 `gen_crab_stairs.py` + `crab_climb.py`（脚本步态，无策略网络），录成 `assets/motion_crab.bin`，并把台阶几何同步到 `assets/scene_crab.xml` |
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

**Q：加载很慢？**  
A：首次要下载约 10 MB 的 MuJoCo WASM + 约 12 MB 的 STL 网格，之后浏览器会缓存。  
加载页有进度条和实时日志。

**Q：面板里选「策略仿真 · 打拳」提示"加载失败"？**  
A：缺 `assets/motion_fight_policy.bin`。跑一次 `python web/tools/bake_motion.py`
生成即可（或直接用 `start_vr_motion.bat` / `.sh`，它会自动补上）。

**Q：面板里选「策略仿真 · 爬楼梯」提示"加载失败"？**  
A：缺 `assets/motion_climb_policy.bin`。跑一次 `python web/tools/bake_climb.py`
生成即可。注意它是**深度视觉闭环**，需要 vendor 里的 `climb_stairs.py` +
`normal_depth.onnx` + `universal_test` 地形；工作区路径默认由脚本自身位置推导，
目录搬过家也不用改，特殊情况才用 `--ws <路径>` 指定。

**Q：面板里选「脚本步态 · 螃蟹步横移上楼」提示"加载失败"？**  
A：缺 `assets/motion_crab.bin`（或 `assets/scene_crab.xml`）。跑一次
`python web/tools/bake_crab.py` 生成即可，它会顺带写好台阶几何和 `manifest.json`。
换一套台阶：`python web/tools/bake_crab.py --seed 7`。

**Q：选「脚本步态 · 螃蟹步横移上楼」后看不到台阶？**  
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
