#!/usr/bin/env bash
# ============================================================
#  elf3 MuJoCo Web Viewer -- macOS / Linux 启动脚本
# ============================================================
set -e
cd "$(dirname "$0")"

if command -v node >/dev/null 2>&1; then
  NODE=node
elif [ -x "$HOME/.workbuddy/binaries/node/versions/22.22.2-6/node.exe" ]; then
  NODE="$HOME/.workbuddy/binaries/node/versions/22.22.2-6/node.exe"
elif [ -x "$HOME/.workbuddy/binaries/node/versions/22.22.2-6/bin/node" ]; then
  NODE="$HOME/.workbuddy/binaries/node/versions/22.22.2-6/bin/node"
else
  echo "[ERROR] 未找到 Node.js，请先安装 Node.js 18+" >&2
  exit 1
fi

echo "使用 Node: $NODE"
exec "$NODE" server.mjs --open "$@"
