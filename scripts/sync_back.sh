#!/usr/bin/env bash
# =============================================================================
# sync_back.sh — 把云服务器的训练产物拉回本地
# =============================================================================
# 在本地（Windows Git Bash / macOS / Linux）运行，通过 rsync 把服务器上训练
# 产出的 checkpoint、日志、以及烘焙后的网页回放数据同步回本地仓库。
#
# 用法：
#   SERVER=root@1.2.3.4 bash scripts/sync_back.sh
#
# 环境变量（都有默认）：
#   SERVER      必填，服务器登录串，例如 root@1.2.3.4
#   REMOTE_DIR  服务器项目根          默认：~/elf3
#   LOCAL_DIR   本地项目根            默认：脚本所在仓库根
#
# 只拉轻量产物（*.onnx/*.pt/*.ckpt/*.json/*.bin/events），不碰大体积中间数据。
# 需要本地装 rsync（Windows 用 Git Bash 自带；macOS/Linux 自带）。
# =============================================================================
set -euo pipefail

if [ -z "${SERVER:-}" ]; then
  echo "请先设置 SERVER，例如："
  echo "    SERVER=root@1.2.3.4 bash $0"
  exit 1
fi

REMOTE_DIR="${REMOTE_DIR:-~/elf3}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "==> 从 $SERVER:$REMOTE_DIR 拉回训练产物 -> $LOCAL_DIR"
mkdir -p "$LOCAL_DIR/logs"

# 1) 训练 checkpoint / 日志（只取轻量文件，排除大体积中间数据）
rsync -am --progress \
  --include='*/' \
  --include='*.onnx' --include='*.pt' --include='*.ckpt' \
  --include='*.json' --include='*.bin' \
  --include='events.out.tfevents.*' \
  --exclude='*' \
  "$SERVER:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"

# 2) 训练后烘焙的网页回放数据（没有则跳过）
rsync -a --ignore-missing-args --progress \
  "$SERVER:$REMOTE_DIR/web/assets/motion_crab.bin" \
  "$SERVER:$REMOTE_DIR/web/assets/motion_crab.json" \
  "$LOCAL_DIR/web/assets/" 2>/dev/null || echo "（无 motion_crab 产物，跳过）"

echo "==> 完成。本地 logs/ 与 web/assets/ 已更新"
echo "    本地查看回放：启动 web 后在下拉选择「脚本步态 · 螃蟹步横移上楼」"
