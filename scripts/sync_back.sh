#!/usr/bin/env bash
# =============================================================================
# sync_back.sh — 把云服务器的训练产物拉回本地
# =============================================================================
# 在【本地】运行（Windows Git Bash / macOS / Linux），不是服务器上。
# 优先用 rsync；若本地没有 rsync（Windows Git Bash 常见），自动改用 ssh+tar。
#
# 用法（在仓库根目录，或任意目录用绝对路径调用）：
#   SERVER=vipuser@1.2.3.4 bash scripts/sync_back.sh
#
# 环境变量（都有默认）：
#   SERVER      必填，服务器登录串，例如 vipuser@1.2.3.4
#   REMOTE_DIR  服务器项目根          默认：~/elf3
#   LOCAL_DIR   本地项目根            默认：脚本所在仓库根
#
# 只拉轻量产物（*.onnx/*.pt/*.ckpt/*.json/*.bin/events），不碰大体积中间数据。
# =============================================================================
set -euo pipefail

if [ -z "${SERVER:-}" ]; then
  echo "请先设置 SERVER，例如："
  echo "    SERVER=vipuser@1.2.3.4 bash $0"
  exit 1
fi

REMOTE_DIR="${REMOTE_DIR:-~/elf3}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PATTERNS="-name '*.onnx' -o -name '*.pt' -o -name '*.ckpt' -o -name '*.json' -o -name '*.bin' -o -name 'events.out.tfevents.*'"

echo "==> 从 $SERVER:$REMOTE_DIR 拉回训练产物 -> $LOCAL_DIR"
mkdir -p "$LOCAL_DIR/logs" "$LOCAL_DIR/web/assets"

if command -v rsync >/dev/null 2>&1; then
  echo "==> 使用 rsync"
  rsync -am --progress \
    --include='*/' \
    --include='*.onnx' --include='*.pt' --include='*.ckpt' \
    --include='*.json' --include='*.bin' \
    --include='events.out.tfevents.*' \
    --exclude='*' \
    "$SERVER:$REMOTE_DIR/logs/" "$LOCAL_DIR/logs/"

  rsync -a --ignore-missing-args --progress \
    "$SERVER:$REMOTE_DIR/web/assets/motion_crab.bin" \
    "$SERVER:$REMOTE_DIR/web/assets/motion_crab.json" \
    "$LOCAL_DIR/web/assets/" 2>/dev/null || echo "（无 motion_crab 产物，跳过）"
else
  echo "==> 未找到 rsync，改用 ssh+tar"
  # 远端只把指定类型的文件打包，本地解包到仓库根（保留 logs/ 前缀）
  ssh "$SERVER" "cd $REMOTE_DIR && mkdir -p logs && find logs -type f \\( $PATTERNS \\) -print0 2>/dev/null | tar --null -T - -czf - 2>/dev/null" \
    | tar xzf - -C "$LOCAL_DIR"

  # 网页回放数据（没有则跳过）
  scp -q "$SERVER:$REMOTE_DIR/web/assets/motion_crab.bin" "$LOCAL_DIR/web/assets/" 2>/dev/null || true
  scp -q "$SERVER:$REMOTE_DIR/web/assets/motion_crab.json" "$LOCAL_DIR/web/assets/" 2>/dev/null || true
fi

echo "==> 完成。本地 logs/ 与 web/assets/ 已更新"
echo "    本地查看回放：启动 web 后在下拉选择「脚本步态 · 螃蟹步横移上楼」"
