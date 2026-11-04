#!/usr/bin/env bash
# =============================================================================
# run_squat_train.sh — 蹲下/起立跟踪 + 蹲姿保持：冒烟 + 训练（服务器上跑，VNC 直接执行）
# =============================================================================
# 用法（仓库根目录）：
#   bash scripts/run_squat_train.sh smoke        # 只冒烟：zero 动作 + 每个任务 5 轮小规模训练
#   bash scripts/run_squat_train.sh all          # 顺序训 down、up、hold_squat、hold_stand
#   bash scripts/run_squat_train.sh down         # 只训下蹲（站姿 -> 极限蹲，2.46 s）
#   bash scripts/run_squat_train.sh up           # 只训起立（极限蹲 -> 站姿，2.02 s）
#   bash scripts/run_squat_train.sh hold         # 只训两个「保持」任务（先 squat 后 stand）
#   bash scripts/run_squat_train.sh hold-squat   # 只训极限蹲保持（全程蹲着不动，20 s 回合）
#   bash scripts/run_squat_train.sh hold-stand   # 只训站姿保持
#   bash scripts/run_squat_train.sh play down        # 回放：自动找 logs/rsl_rl/elf3_squat_down 下最新 pt
#   bash scripts/run_squat_train.sh play hold-squat  #   同上，保持任务
#   CKPT=logs/rsl_rl/.../model_1000.pt bash scripts/run_squat_train.sh play hold-squat
#
# 四个任务、四份实验目录（别混，回放/续训都按目录找）：
#   down        Elf3-Squat-Down     -> logs/rsl_rl/elf3_squat_down
#   up          Elf3-Squat-Up       -> logs/rsl_rl/elf3_squat_up
#   hold-squat  Elf3-Squat-Hold     -> logs/rsl_rl/elf3_squat_hold_squat
#   hold-stand  Elf3-Stand-Hold     -> logs/rsl_rl/elf3_squat_hold_stand
#
# 可调环境变量：
#   NUM_ENVS=1024        并行环境数
#   ITERS=6000           训练轮数（--agent.max-iterations，是"本次再跑多少轮"）
#   HOLD=5.0             蹲起任务里参考播完后还必须站住几秒（0 = 旧行为，播完即结束）
#   HOLD_EPISODE=20.0    「保持」任务的回合长度（秒），整个回合都是保持段
#   ARM_GUARD=1          手臂-腿接触检测开关（本版 mjlab 不支持自碰撞传感器时设 0）
#   SMOKE_ITERS=5        冒烟轮数
#   VENV=.venv           虚拟环境目录（相对仓库根）
#   CKPT=...             play 模式下显式指定 checkpoint（不给就自动找最新）
#
# 日志：~/train_squat_<mode>.log（如 ~/train_squat_hold_squat.log）
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
# 这几个在任务包 import 时就生效（配置是 import 时 build 的），必须先 export
export ELF3_SQUAT_HOLD="${HOLD:-5.0}"
export ELF3_SQUAT_ARM_GUARD="${ARM_GUARD:-1}"
export ELF3_HOLD_EPISODE="${HOLD_EPISODE:-20.0}"

step() { echo; echo "==> $*"; }

# mode -> mjlab 任务 ID。四个任务共用一套训练流程，只有 ID 不同。
task_of() {
  case "$1" in
    down)       echo "Elf3-Squat-Down" ;;
    up)         echo "Elf3-Squat-Up" ;;
    hold_squat) echo "Elf3-Squat-Hold" ;;
    hold_stand) echo "Elf3-Stand-Hold" ;;
    *) echo "[ERR] 未知任务 $1" >&2; return 1 ;;
  esac
}

check_tasks() {
  step "自检任务注册"
  "$REPO/$VENV/bin/python" - <<'PY'
import sys
from mjlab.tasks.registry import list_tasks
import mjlab.tasks.crab  # noqa: F401
want = [
    "Elf3-Squat-Down", "Elf3-Squat-Down-Play",
    "Elf3-Squat-Up", "Elf3-Squat-Up-Play",
    "Elf3-Squat-Hold", "Elf3-Squat-Hold-Play",
    "Elf3-Stand-Hold", "Elf3-Stand-Hold-Play",
]
have = [t for t in list_tasks() if "Squat" in t or "Stand" in t]
print("已注册:", have)
missing = [w for w in want if w not in have]
if missing:
    print(f"[ERR] 缺任务 {missing}")
    print("      跑 bash train/install_task.sh；仍失败看上面的 [squat]/[hold][ERR] 提示")
    sys.exit(1)
print("蹲起 + 保持任务齐全 ✔")
PY
}

smoke() {
  check_tasks
  step "冒烟 ① zero 动作（残差=0，纯参考开环；播完会一直保持蹲姿，Ctrl-C 退出）"
  timeout 180 "$BIN_PLAY" Elf3-Squat-Down-Play --agent zero || {
    echo "[warn] zero 冒烟退出码非 0（交互式 viewer 被 timeout 杀掉是正常的）"
  }
  step "冒烟 ② 保持任务 zero 动作（冻在极限蹲，20 s 一回合，Ctrl-C 退出）"
  timeout 180 "$BIN_PLAY" Elf3-Squat-Hold-Play --agent zero || {
    echo "[warn] zero 冒烟退出码非 0（交互式 viewer 被 timeout 杀掉是正常的）"
  }
  for t in down up hold_squat hold_stand; do
    task="$(task_of "$t")"
    step "冒烟 ③ $t（$task）：${SMOKE_ITERS} 轮 × 256 env"
    "$BIN_TRAIN" "$task" \
      --env.scene.num-envs 256 --agent.max-iterations "$SMOKE_ITERS" \
      2>&1 | tee "$HOME/train_squat_${t}_smoke.log" | tail -25
  done
  echo
  echo "[OK] 冒烟跑完。真实训练：bash scripts/run_squat_train.sh all"
  echo "     先确认上面有这几行（没有就是配置没生效，别开训）："
  echo "       [squat] down: 参考 2.46 s + 保持 5.0 s = 7.96 s/回合"
  echo "       [squat] 已加手臂-腿接触传感器 arm_leg_contact"
  echo "       [hold] squat（极限蹲保持）：20.0 s/回合，参考冻结在 down 第 123/123 帧"
  echo "       [traj] down: ... 冻结在第 123 帧（不推进，参考=静态位形，z=0.523 m）"
}

train_one() {
  local which="$1" task
  task="$(task_of "$which")" || exit 1
  step "训练 $task（$which）：${NUM_ENVS} env × ${ITERS} iter"
  "$BIN_TRAIN" "$task" \
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
  local which="$1" task pt
  task="$(task_of "$which")" || exit 1
  pt="$(latest_ckpt "$which")" || exit 1
  step "回放 $task-Play，checkpoint = $pt"
  echo "    （想换别的：CKPT=<路径> bash scripts/run_squat_train.sh play ${which//_/-}）"
  "$BIN_PLAY" "$task-Play" --checkpoint-file "$pt"
}

case "$MODE" in
  smoke) smoke ;;
  down)  check_tasks; train_one down ;;
  up)    check_tasks; train_one up ;;
  hold)  check_tasks; train_one hold_squat; train_one hold_stand ;;
  hold-squat) check_tasks; train_one hold_squat ;;
  hold-stand) check_tasks; train_one hold_stand ;;
  play)
    check_tasks
    case "${2:-}" in
      down|up|hold-squat|hold-stand)
        play_one "${2//-/_}" ;;
      *) echo "用法：play down | play up | play hold-squat | play hold-stand"; exit 2 ;;
    esac
    ;;
  all)
    check_tasks
    train_one down
    train_one up
    train_one hold_squat
    train_one hold_stand
    ;;
  *) echo "未知参数：$MODE（可用 smoke / down / up / hold / hold-squat / hold-stand / play / all）"
     exit 2 ;;
esac

echo
echo "==> 完成。看指标（把 down 换成 up / hold_squat / hold_stand 即可）："
echo "    grep -o 'Metrics/traj_err_rms_leg: [0-9.]*' ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/sym_err_rad: [0-9.]*'     ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/base_xy: [0-9.]*'         ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/base_yaw_deg: [0-9.]*'    ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/arm_leg_touch: [0-9.]*'   ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/joint_speed: [0-9.]*'     ~/train_squat_hold_squat.log | tail -5"
echo "    tensorboard --logdir logs/rsl_rl/elf3_squat_down --port 6006"
