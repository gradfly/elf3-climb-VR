# 深度行走 Mod

`com.bxi.normal_depth` 只负责深度策略及其专属预处理，不再打开相机 SDK，也不再
发布相机话题。RealSense 和 Orbbec Gemini 335 由独立的 `bxi_depth_camera` ROS 2
包管理：

```bash
ros2 launch bxi_depth_camera cameras.launch.py
```

相机包会发现并打开所有受支持的深度相机。真机与仿真使用相同的逻辑相机名和
话题结构，仅沿用 launch 已有的环境前缀：

```text
/simulation/head_depth_camera/depth/image_rect_raw
/simulation/head_depth_camera/depth/camera_info

/hardware/head_depth_camera/depth/image_rect_raw
/hardware/head_depth_camera/depth/camera_info
```

MuJoCo 与真机均采用相同的 ROS 流级目录。状态只根据 `/topic_prefix` 替换
`simulation` 或 `hardware`，不需要为两种环境单独配置话题。

策略使用 MuJoCo XML 中的逻辑名称 `head_depth_camera`。单相机真机会由相机管理器
自动回退到该名称；多相机部署通过 `cameras.head_depth_camera.serial_no` 把这个逻辑
位置映射到实际设备序列号。

## 选择策略相机

在 `mod.yaml` 的 `states.normal_depth.params` 中配置逻辑相机名：

```yaml
camera_name: head_depth_camera
topic: ""
camera_info_topic: ""
```

状态根据控制节点已有的 `/topic_prefix` 自动订阅仿真或真机话题，不包含序列号和
环境判断。多相机的位置映射由每台机器的部署配置明确指定，不根据枚举顺序猜测。
如果使用 rosbag 或其他驱动，也可以清空逻辑名称并同时显式配置：

```yaml
camera_name: ""
topic: /other_camera/depth/image_rect_raw
camera_info_topic: /other_camera/depth/camera_info
```

两种方式都未配置时 Mod 仍能加载，但深度状态不可进入。

## 预处理边界

相机包发布经过设备 SDK 空域、时域和孔洞滤波的完整深度图，不应用策略距离范围或
ROI。`NormalDepthState.depth_image_callback()` 缓存 `CameraInfo`，并在 ROS 2 回调中
依次完成：

1. 将 `16UC1` 或 `32FC1` 转换为米；
2. 根据真实内参和策略目标 FOV 计算中心 ROI；
3. 最近邻缩放到策略输入前尺寸；
4. 对有效深度应用距离限幅并保持无效值 `0`；
5. 保留原有旋转方向，然后交给策略的归一化和历史缓冲。

`origin_camera` 默认生成旋转前 `36x48`、FOV `45.2° x 58.0616969°`、距离
`[0.2, 3.0] m`；`depth_walk` 默认生成旋转前 `64x36`、FOV
`89.24° x 58.06°`、距离 `[0.2, 2.5] m`。这些值都可在状态参数中覆盖。

`origin_camera` 同时支持循环 ONNX 策略的 `h_in/h_out`。隐状态在进入状态时
清零，只有真实推理帧会推进，Transition 预览不会修改已提交的隐状态。

## RKNN INT8 校准数据

策略代码不包含采集线程或监控分支。使用通用采集工具包装控制器：

```bash
python3 tools/benchmark/collect_calibration.py \
  --output /tmp/bxi_rknn_calibration \
  --every 5 \
  --max-samples 500 \
  --skip-first 10 \
  -- ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py
```

完整校验和转换命令见 `tools/benchmark/README.md`。
