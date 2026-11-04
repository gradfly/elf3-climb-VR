#!/usr/bin/env bash
# =============================================================================
# run_squat_train.sh — 蹲下/起立跟踪策略：冒烟 + 训练（在服务器上跑，VNC 里直接执行）
# =============================================================================
# 用法（仓库根目录）：
#   bash scripts/run_squat_train.sh smoke      # 只冒烟：zero 动作 + 5 轮小规模训练
#   bash scripts/run_squat_train.sh all        # 冒烟通过后顺序训 down、up（默认）
#   bash scripts/run_squat_train.sh down       # 只训下蹲
#   bash scripts/run_squat_train.sh up         # 只训起立
#   bash scripts/run_squat_train.sh play down  # 回放：自动找 elf3_squat_down 下最新的 model_*.pt
#   bash scripts/run_squat_train.sh play up    #   指定某个 ckpt：CKPT=logs/rsl_rl/elf3_squat_down/<run>/model_1000.pt ...
#
# 可调环境变量：
#   NUM_ENVS=1024     并行环境数
#   ITERS=6000        训练轮数（--agent.max-iterations，是"本次再跑多少轮"）
#   HOLD=5.0          参考播完后还必须站住几秒（训练段；0 = 旧行为，播完即结束）
#   ARM_GUARD=1       手臂-腿接触检测开关（本版 mjlab 不支持自碰撞传感器时设 0）
#   SMOKE_ITERS=5     冒烟轮数
#   VENV=.venv        虚拟环境目录（相对仓库根）
#   CKPT=...          play 模式下显式指定 checkpoint（不给就自动找最新）
#
# 日志：~/train_squat_{down,up}.log，曲线/checkpoint 在
#   logs/rsl_rl/elf3_squat_{down,up}/<时间戳>/（两边实验名分开，别混）
# =============================================================================
set -euo pipefail

MODE="${1:-all}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-6000}"
SMOKE_ITERS="${SMOKE_ITERS:-5}"
VENV="${VENV:-.venv}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BIN_TRAIN="$REPO/$VENV/bin/train"
BIN_PLAY="$REPO/$VENV/bin/play"
if [ ! -x "$BIN_TRAIN" ]; then
  echo "[ERR] 找不到 $BIN_TRAIN"
  echo "      先激活/安装环境：bash scripts/deploy_gpu.sh"
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
# 这两个在任务包 import 时就生效（配置是 import 时 build 的），必须先 export
export ELF3_SQUAT_HOLD="${HOLD:-5.0}"
export ELF3_SQUAT_ARM_GUARD="${ARM_GUARD:-1}"

step() { echo; echo "==> $*"; }

check_tasks() {
  step "自检任务注册"
  "$REPO/$VENV/bin/python" - <<'PY'
import sys
from mjlab.tasks.registry import list_tasks
import mjlab.tasks.crab  # noqa: F401
want = ["Elf3-Squat-Down", "Elf3-Squat-Down-Play", "Elf3-Squat-Up", "Elf3-Squat-Up-Play"]
have = [t for t in list_tasks() if "Squat" in t]
print("已注册:", have)
missing = [w for w in want if w not in have]
if missing:
    print(f"[ERR] 缺任务 {missing}")
    print("      跑 bash train/install_task.sh；仍失败看上面的 [squat][ERR] 提示")
    sys.exit(1)
print("蹲起任务齐全 ✔")
PY
}

smoke() {
  check_tasks
  step "冒烟 ① zero 动作（残差=0，纯参考开环；播完会一直保持蹲姿，Ctrl-C 退出）"
  timeout 180 "$BIN_PLAY" Elf3-Squat-Down-Play --agent zero || {
    echo "[warn] zero 冒烟退出码非 0（交互式 viewer 被 timeout 杀掉是正常的）"
  }
  for t in down up; do
    tag="$(python -c "print('$t'.capitalize())")"
    step "冒烟 ② $t：${SMOKE_ITERS} 轮 × 256 env"
    "$BIN_TRAIN" "Elf3-Squat-$tag" \
      --env.scene.num-envs 256 --agent.max-iterations "$SMOKE_ITERS" \
      2>&1 | tee "$HOME/train_squat_${t}_smoke.log" | tail -25
  done
  echo
  echo "[OK] 冒烟跑完。真实训练：bash scripts/run_squat_train.sh all"
  echo "     先确认上面有这两行（没有就是配置没生效，别开训）："
  echo "       [squat] down: 参考 2.46 s + 保持 5.0 s = 7.96 s/回合"
  echo "       [squat] 已加手臂-腿接触传感器 arm_leg_contact"
}

train_one() {
  local which="$1" tag
  tag="$(python -c "print('$which'.capitalize())")"
  step "训练 Elf3-Squat-$tag：${NUM_ENVS} env × ${ITERS} iter"
  "$BIN_TRAIN" "Elf3-Squat-$tag" \
    --env.scene.num-envs "$NUM_ENVS" --agent.max-iterations "$ITERS" \
    2>&1 | tee "$HOME/train_squat_${which}.log" | tail -40
}

# 回放：默认自动找该实验目录下修改时间最新的 model_*.pt
latest_ckpt() {
  local exp="logs/rsl_rl/elf3_squat_$1"
  local ck=""
  if [ -n "${CKPT:-}" ]; then
    echo "$CKPT"; return 0
  fi
  # run 目录按时间戳命名，同目录里 model_<iter>.pt 按字母序 == 按迭代数序
  ck="$(ls -1dt "$exp"/*/ 2>/dev/null | head -1 || true)"   # 最新 run 目录
  if [ -z "$ck" ]; then
    ck="$(ls -1dt "$exp"*/ 2>/dev/null | head -1 || true)"  # 没有 run 子目录的兜底
  fi
  if [ -z "$ck" ]; then
    echo "[ERR] 没找到任何 checkpoint（$exp/*/model_*.pt）" >&2
    echo "      先训练：bash scripts/run_squat_train.sh $1" >&2
    return 1
  fi
  local pt
  pt="$(ls -1 "$ck"model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
  if [ -z "$pt" ]; then
    # 最新 run 里还没有 model（刚开训没到 save_interval），往前找其它 run
    pt="$(find logs/rsl_rl/elf3_squat_$1 -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d' ' -f2- || true)"
  fi
  if [ -z "$pt" ]; then
    echo "[ERR] 最新 run 目录 $ck 里没有 model_*.pt，全目录也没找到" >&2
    return 1
  fi
  echo "$pt"
}

play_one() {
  local which="$1" tag pt
  tag="$(python -c "print('$which'.capitalize())")"
  pt="$(latest_ckpt "$which")" || exit 1
  step "回放 Elf3-Squat-$tag-Play，checkpoint = $pt"
  echo "    （想换别的：CKPT=<路径> bash scripts/run_squat_train.sh play $which）"
  "$BIN_PLAY" "Elf3-Squat-$tag-Play" --checkpoint-file "$pt"
}

case "$MODE" in
  smoke) smoke ;;
  down)  check_tasks; train_one down ;;
  up)    check_tasks; train_one up ;;
  play)
    check_tasks
    case "${2:-}" in
      down) play_one down ;;
      up)   play_one up ;;
      *)    echo "用法：play down | play up（想同时看两个就开两个终端）"; exit 2 ;;
    esac
    ;;
  all)   check_tasks; train_one down; train_one up ;;
  *) echo "未知参数：$MODE（可用 smoke / down / up / play / all）"; exit 2 ;;
esac

echo
echo "==> 完成。看指标："
echo "    grep -o 'Metrics/traj_err_rms_leg: [0-9.]*' ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/sym_err_rad: [0-9.]*'     ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/base_xy: [0-9.]*'         ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/base_yaw_deg: [0-9.]*'    ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/arm_leg_touch: [0-9.]*'   ~/train_squat_down.log | tail -5"
echo "    tensorboard --logdir logs/rsl_rl/elf3_squat_down --port 6006"
