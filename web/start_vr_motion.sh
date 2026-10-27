#!/usr/bin/env bash
# ============================================================
#  elf3 运动仿真 · 一键在网页端 / VR 头显里观看（macOS / Linux）
#
#  和 start.sh 的区别：本脚本直接进入「运动仿真回放」模式
#  （策略跟踪 fight.npz 的打拳动作），并自动准备好回放数据。
#  start.bat / start.sh 保持原样，只做「静止站立」查看。
#
#  用法：./start_vr_motion.sh  [--rebake] [--http-port N] [--https-port N]
#    --rebake        重新跑一遍本机 MuJoCo 仿真，刷新回放数据
# ============================================================
set -e
cd "$(dirname "$0")"

REBAKE=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --rebake) REBAKE=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

# ---- Node ----
if command -v node >/dev/null 2>&1; then
  NODE=node
elif [ -x "$HOME/.workbuddy/binaries/node/versions/22.22.2-6/node.exe" ]; then
  NODE="$HOME/.workbuddy/binaries/node/versions/22.22.2-6/node.exe"
elif [ -x "$HOME/.workbuddy/binaries/node/versions/22.22.2-6/bin/node" ]; then
  NODE="$HOME/.workbuddy/binaries/node/versions/22.22.2-6/bin/node"
else
  echo "[错误] 未找到 Node.js 18+，请先安装：https://nodejs.org/" >&2
  exit 1
fi

# ---- Python（只用来录制回放数据，缺了也能看静止站立）----
PY=""
for cand in \
  "$HOME/.workbuddy/binaries/python/envs/default/bin/python" \
  "$HOME/.workbuddy/binaries/python/envs/default/Scripts/python.exe"; do
  if [ -x "$cand" ]; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python; fi
fi

MOTION="assets/motion_fight_policy.bin"
if [ "$REBAKE" = 1 ] || [ ! -f "$MOTION" ]; then
  echo
  echo " [1/3] 本地录制运动仿真数据（MuJoCo + policy.onnx 闭环）…"
  echo
  if [ -n "$PY" ]; then
    "$PY" tools/bake_motion.py || echo " [警告] 录制失败，沿用已有回放数据。"
  else
    echo " [警告] 缺少 $MOTION 且未找到 Python，页面里「运动仿真」会提示加载失败。"
    echo "        补数据：python web/tools/bake_motion.py"
  fi
fi

echo
echo " [2/3] 启动本地服务器（HTTP + HTTPS 双端口）…"
echo " [3/3] 浏览器会自动打开「运动仿真」页面。"
echo "       VR 头显：连同一个 Wi-Fi，在头显浏览器里打开下面打印的"
echo "       https 局域网地址（带 ?motion=fight_policy 的那一条）。"
echo
echo " 使用 Node: $NODE"
exec "$NODE" server.mjs --open --open-path "/?motion=fight_policy" "${ARGS[@]}"
