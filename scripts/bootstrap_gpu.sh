#!/usr/bin/env bash
# =============================================================================
# bootstrap_gpu.sh — 云服务器训练环境一键搭建
# =============================================================================
# 在已 `git clone` 本项目的云服务器（Ubuntu 22.04 + NVIDIA GPU）上运行，
# 自动装好：uv -> Python 3.11 虚拟环境 -> PyTorch(CUDA) -> MuJoCo 生态(mjlab /
# mujoco-warp) -> 校验 GPU 与无头 EGL 离屏渲染。
#
# 用法：
#   bash scripts/bootstrap_gpu.sh
#
# 可选环境变量（都有合理默认值）：
#   REPO_DIR    项目根目录           默认：脚本所在仓库根
#   PY_VER      Python 版本          默认：3.11   （mjlab/warp 对 3.13 支持不全）
#   TORCH_CU    PyTorch CUDA 索引    默认：按驱动自动（cu124/cu126/cu128/cu130）
#   PIP_INDEX   普通包镜像           默认：阿里云（国内快）
#
# 设计要点（来自迁移踩坑记录）：
#   - 必须用 3.11 而非系统自带的 3.10，且避开 3.13
#   - 无头渲染依赖 libegl1/libgl1，否则 mujoco 离屏渲染直接崩
#   - torch 必须最后装并 --force-reinstall：mjlab/mujoco-warp 的依赖会把它升级成
#     与驱动不匹配的 CUDA 版本（曾出现驱动 12.4 却装上 cu130，cuda_available=False）
#   - 把 MUJOCO_GL=egl 写进 venv 的 activate.d，crab_replay --record 自动走离屏
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PY_VER="${PY_VER:-3.11}"
TORCH_CU="${TORCH_CU:-}"          # 留空 = 按驱动自动选择（cu124/cu126/cu128/cu130）
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
VENV="$REPO_DIR/.venv"

echo "==> 项目目录 : $REPO_DIR"
echo "==> 目标 Python: $PY_VER | torch: ${TORCH_CU:-自动} | 镜像: $PIP_INDEX"
cd "$REPO_DIR"

echo "==> [1/6] 安装系统依赖（无头渲染 + 编译工具）"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  libegl1 libgl1 libglib2.0-0 libgomp1 libglu1-mesa-dev \
  ca-certificates curl git build-essential

echo "==> [2/6] 安装 uv（若缺失）"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || { echo "uv 安装失败，请手动安装后重试"; exit 1; }

echo "==> [3/6] 准备 Python $PY_VER 并建虚拟环境"
uv python install "$PY_VER" || true
uv venv --python "$PY_VER" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> [4/6] 探测驱动支持的 CUDA 版本，选择匹配的 torch 构建"
if [ -z "$TORCH_CU" ]; then
  DRV_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
  if [ -z "$DRV_CUDA" ]; then
    echo "    [!] 未检测到 nvidia-smi / 驱动；默认 cu124，若失败请先修 GPU 驱动"
    TORCH_CU="cu124"
  else
    MAJ=${DRV_CUDA%%.*}; MIN=${DRV_CUDA#*.}
    if   [ "$MAJ" -ge 13 ]; then TORCH_CU="cu130"
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then TORCH_CU="cu128"
    elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then TORCH_CU="cu126"
    else TORCH_CU="cu124"; fi
    echo "    驱动 CUDA $DRV_CUDA  ->  选用 torch-$TORCH_CU"
  fi
else
  echo "    使用指定的 TORCH_CU=$TORCH_CU"
fi

echo "==> [5/6] 先装 MuJoCo 生态（mujoco / mujoco-warp / mjlab）"
uv pip install --index-url "$PIP_INDEX" \
  mujoco mujoco-warp mjlab numpy pyyaml
# 若 mjlab 在镜像缺失，回退官方 PyPI
python -c "import mjlab" 2>/dev/null || uv pip install mjlab
echo "    再装 torch（$TORCH_CU）—— 必须放最后并强制重装，"
echo "    否则 mjlab/mujoco-warp 的依赖会把 torch 升级成与驱动不匹配的 CUDA 版本"
uv pip install --index-url "https://download.pytorch.org/whl/$TORCH_CU" --force-reinstall torch

echo "==> [6/6] 校验 GPU 与无头 EGL 离屏渲染"
cat > /tmp/_egl_test.py <<'PY'
import os
os.environ["MUJOCO_GL"] = "egl"
import mujoco
from mujoco import Renderer

# 先验 EGL（与 CUDA 无关，缺 libegl 会在这里失败）
xml = """<mujoco><worldbody>
<light pos="0 0 3"/>
<body><joint type="free"/><geom type="box" size=".1 .1 .1"/></body>
<camera name="cam" pos="0 0 2" xyaxes="0 1 0 0 0 1"/>
</worldbody></mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = Renderer(m, 64, 64)
mujoco.mj_step(m, d)
r.update_scene(d)
print("EGL 离屏渲染 OK，帧尺寸", r.render().shape)

# 再验 CUDA（torch 构建与驱动不匹配时这里失败）
import torch
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda)
assert torch.cuda.is_available(), (
    "CUDA 不可用：当前 torch 是 %s；若驱动较旧，请装匹配的 cuXXX 版本"
    "（例如驱动 CUDA 12.4 时用 cu124），或升级 NVIDIA 驱动后重装。"
    % torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
PY
"$VENV/bin/python" /tmp/_egl_test.py
rm -f /tmp/_egl_test.py

# 让后续 crab_replay --record 默认走 EGL（无 DISPLAY 也能录）
mkdir -p "$VENV/bin/activate.d"
printf 'export MUJOCO_GL=egl\n' > "$VENV/bin/activate.d/mujoco.sh"

echo
echo "==> 完成。后续步骤："
echo "    source $VENV/bin/activate"
echo "    冒烟： python -c 'import torch, mujoco, mjlab; print(\"ok\")'"
echo "    复现 baseline： ./run_crab.sh --dry-run          # 看生效配置"
echo "                  ./run_crab.sh --stages gen,plan   # 生成地形 + 轨迹"
echo "                  ./run_crab.sh --record            # 验证 EGL 能离屏录（无窗口）"
echo "    回传产物： SERVER=user@<IP> bash scripts/sync_back.sh"
