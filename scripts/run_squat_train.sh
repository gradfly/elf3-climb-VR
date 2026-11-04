#!/usr/bin/env bash
# =============================================================================
# run_squat_train.sh — 蹲下/起立跟踪策略：冒烟 + 训练（在服务器上跑，VNC 里直接执行）
# =============================================================================
# 用法（仓库根目录）：
#   bash scripts/run_squat_train.sh smoke      # 只冒烟：zero 动作 + 5 轮小规模训练
#   bash scripts/run_squat_train.sh all        # 冒烟通过后顺序训 down、up（默认）
#   bash scripts/run_squat_train.sh down       # 只训下蹲
#   bash scripts/run_squat_train.sh up         # 只训起立
#
# 可调环境变量：
#   NUM_ENVS=1024     并行环境数
#   ITERS=4000        训练轮数（--agent.max-iterations，是"本次再跑多少轮"）
#   SMOKE_ITERS=5     冒烟轮数
#   VENV=.venv        虚拟环境目录（相对仓库根）
#
# 日志：~/train_squat_{down,up}.log，曲线/checkpoint 在 logs/rsl_rl/elf3_squat/<时间戳>/
# =============================================================================
set -euo pipefail

MODE="${1:-all}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-4000}"
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
  step "冒烟 ① zero 动作（残差=0，纯参考开环，看 2.5 s 内会不会自己塌）"
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
}

train_one() {
  local which="$1" tag
  tag="$(python -c "print('$which'.capitalize())")"
  step "训练 Elf3-Squat-$tag：${NUM_ENVS} env × ${ITERS} iter"
  "$BIN_TRAIN" "Elf3-Squat-$tag" \
    --env.scene.num-envs "$NUM_ENVS" --agent.max-iterations "$ITERS" \
    2>&1 | tee "$HOME/train_squat_${which}.log" | tail -40
}

case "$MODE" in
  smoke) smoke ;;
  down)  check_tasks; train_one down ;;
  up)    check_tasks; train_one up ;;
  all)   check_tasks; train_one down; train_one up ;;
  *) echo "未知参数：$MODE（可用 smoke / down / up / all）"; exit 2 ;;
esac

echo
echo "==> 完成。看指标："
echo "    grep -o 'Metrics/traj_progress: [0-9.]*' ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/traj_err_rms_leg: [0-9.]*' ~/train_squat_down.log | tail -5"
echo "    tensorboard --logdir logs/rsl_rl/elf3_squat --port 6006"
