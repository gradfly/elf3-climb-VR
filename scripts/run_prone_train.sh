#!/usr/bin/env bash
# =============================================================================
# run_prone_train.sh — 趴姿三段动作的 RL 残差策略训练（服务器上跑，VNC 直接执行）
# =============================================================================
# 三个任务共用「参考轨迹 + 残差」结构（参考来自 scripts/build_prone_parts.py 的 npz）：
#
#   stand2crawl   Elf3-Prone-Stand2Crawl   站立 -> 蹲 -> 撑地 -> 趴姿   2.6~6.0 s
#   crawl         Elf3-Prone-Crawl         趴姿向前爬行                 2.5~5.5 s
#   crawl2stand   Elf3-Prone-Crawl2Stand   趴姿 -> 撑地 -> 蹲 -> 站立   2.6~6.0 s
#
# 用法（仓库根目录）：
#   bash scripts/run_prone_train.sh smoke          # 冒烟：zero 策略 + 每个任务 5 轮
#   bash scripts/run_prone_train.sh all            # 顺序训三段
#   bash scripts/run_prone_train.sh crawl          # 只训爬行
#   bash scripts/run_prone_train.sh stand2crawl    # 只训「站立->趴」
#   bash scripts/run_prone_train.sh crawl2stand    # 只训「趴->站立」
#   bash scripts/run_prone_train.sh play crawl     # 回放最新 checkpoint
#   CKPT=logs/rsl_rl/elf3_prone_crawl/2026-.../model_3000.pt \
#     bash scripts/run_prone_train.sh play crawl
#
# 实验目录（回放/续训都按目录找，别混）：
#   logs/rsl_rl/elf3_prone_stand2crawl
#   logs/rsl_rl/elf3_prone_crawl
#   logs/rsl_rl/elf3_prone_crawl2stand
#
# 可调环境变量：
#   NUM_ENVS=1024        并行环境数
#   ITERS=6000           本次训练的轮数（--agent.max-iterations）
#   PRESET=fast          fast（默认）或 normal —— 决定用哪份参考表和回合长度
#   PUSH=1               回合中是否推机器人（0 = 关掉，先验证能不能学会时用）
#   SMOKE_ITERS=5        冒烟轮数
#   VENV=.venv           虚拟环境目录（相对仓库根）
#   CKPT=...             play 模式显式指定 checkpoint（不给就自动找最新）
#   CR_FIX=0             设 1 把 \r 进度条转成换行（旧版 rsl_rl/tqdm）
#
# 日志：~/train_prone_<which>.log
#
# 首次使用前必须先烘焙参考表（缺表时任务注册会打印 [prone][ERR]）：
#   python scripts/build_prone_parts.py --preset fast      # 生成 fast 动作 npz
#   python train/assets/build_prone_ref.py --preset fast   # 烘焙 fast 参考表（默认档）
#   想改用 normal 档：去掉上面两条命令的 --preset fast 重新执行，并 PRESET=normal 启动
# =============================================================================
set -euo pipefail

MODE="${1:-all}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-6000}"
SMOKE_ITERS="${SMOKE_ITERS:-5}"
VENV="${VENV:-.venv}"
CR_FIX="${CR_FIX:-0}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BIN_TRAIN="$REPO/$VENV/bin/train"
BIN_PLAY="$REPO/$VENV/bin/play"
BIN_PY="$REPO/$VENV/bin/python"
if [ ! -x "$BIN_TRAIN" ]; then
  echo "[ERR] 找不到 $BIN_TRAIN"
  echo "      先激活/安装环境：bash scripts/deploy_gpu.sh"
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
# 任务 cfg 在 import 时 build，这几个必须在这之前 export
export ELF3_PRONE_PRESET="${PRESET:-fast}"
export ELF3_PRONE_PUSH="${PUSH:-1}"

step() { echo; echo "==> $*"; }

# stdout 同时打到终端 + tee 写日志。不要在管道末端接 tail：tail 要等 EOF，
# 训练结束前一行都不吐，看起来就是「指标没有实时输出」。
run_logged() {
  local log="$1"; shift
  echo "    实时日志：$log"
  echo "    另开终端看逐轮指标：tail -F \"$log\" | grep --line-buffered -E 'Learning iteration|Mean reward|Metrics/'"
  if [ "$CR_FIX" = "1" ]; then
    "$@" 2>&1 | sed -u 's/\r/\n/g' | tee "$log"
  else
    "$@" 2>&1 | tee "$log"
  fi
}

# mode -> mjlab 任务 ID
task_of() {
  case "$1" in
    stand2crawl) echo "Elf3-Prone-Stand2Crawl" ;;
    crawl)       echo "Elf3-Prone-Crawl" ;;
    crawl2stand) echo "Elf3-Prone-Crawl2Stand" ;;
    *) echo "[ERR] 未知任务 $1" >&2; return 1 ;;
  esac
}

check_ref() {
  local table="train/assets/ref_tables/prone_ref_v1"
  if [ "$ELF3_PRONE_PRESET" = "fast" ]; then table="${table}_fast"; fi
  if [ ! -f "${table}.npz" ]; then
    echo "[ERR] 缺参考表 ${table}.npz"
    echo "      补救："
    if [ "$ELF3_PRONE_PRESET" = "fast" ]; then
      echo "        python scripts/build_prone_parts.py --preset fast"
      echo "        python train/assets/build_prone_ref.py --preset fast"
    else
      echo "        python scripts/build_prone_parts.py"
      echo "        python train/assets/build_prone_ref.py"
    fi
    echo "      若想先用 normal 档：PRESET=normal bash scripts/run_prone_train.sh <mode>"
    exit 1
  fi
  echo "    参考表：${table}.npz（PRESET=$ELF3_PRONE_PRESET）"
}

check_tasks() {
  step "自检任务注册（PRESET=$ELF3_PRONE_PRESET，PUSH=$ELF3_PRONE_PUSH）"
  check_ref
  "$BIN_PY" - <<'PY'
import sys
from mjlab.tasks.registry import list_tasks
import mjlab.tasks.crab  # noqa: F401
want = [
    "Elf3-Prone-Stand2Crawl", "Elf3-Prone-Stand2Crawl-Play",
    "Elf3-Prone-Crawl", "Elf3-Prone-Crawl-Play",
    "Elf3-Prone-Crawl2Stand", "Elf3-Prone-Crawl2Stand-Play",
]
have = [t for t in list_tasks() if "Prone" in t]
print("已注册:", have)
missing = [w for w in want if w not in have]
if missing:
    print(f"[ERR] 缺任务 {missing}")
    print("      往上翻 [prone][ERR] 的具体原因；参考表缺失是最常见的")
    sys.exit(1)
print("趴姿三段任务齐全 ✔")
PY
}

smoke() {
  check_tasks
  for t in stand2crawl crawl crawl2stand; do
    local task; task="$(task_of "$t")" || exit 1
    step "冒烟 ① $t：zero 策略（残差=0，纯参考开环；Ctrl-C 退出）"
    timeout 180 "$BIN_PLAY" "$task-Play" --agent zero || {
      echo "[warn] zero 冒烟退出码非 0（交互式 viewer 被 timeout 杀掉是正常的）"
    }
    step "冒烟 ② $t（$task）：${SMOKE_ITERS} 轮 × 256 env"
    run_logged "$HOME/train_prone_${t}_smoke.log" \
      "$BIN_TRAIN" "$task" \
        --env.scene.num-envs 256 --agent.max-iterations "$SMOKE_ITERS"
  done
  echo
  echo "[OK] 冒烟跑完。真实训练：bash scripts/run_prone_train.sh all"
  echo "     先确认上面能看到这几行（没有就是配置没生效，别开训）："
  echo "       [prone] crawl: 出生 (x=+0.140, y=+0.000, z=0.100) m"
  echo "       [prone] crawl: 参考 2.66 s（含冻结尾） -> 回合 3.16 s（fast 档）"
  echo "       [prone] stand2crawl: 出生 (x=+0.000, z=1.034) m，参考 5.98 s"
  echo "       [prone] crawl2stand: 出生 (x=+0.618, z=0.100) m，参考 5.96 s"
}

train_one() {
  local which="$1" task
  task="$(task_of "$which")" || exit 1
  step "训练 $task（$which）：${NUM_ENVS} env × ${ITERS} iter"
  run_logged "$HOME/train_prone_${which}.log" \
    "$BIN_TRAIN" "$task" \
      --env.scene.num-envs "$NUM_ENVS" --agent.max-iterations "$ITERS"
}

# 回放：自动找该实验目录下最新的 model_*.pt
latest_ckpt() {
  local exp="logs/rsl_rl/elf3_prone_$1"
  if [ -n "${CKPT:-}" ]; then echo "$CKPT"; return 0; fi
  local run pt
  run="$(ls -1dt "$exp"/*/ 2>/dev/null | head -1 || true)"
  if [ -n "$run" ]; then
    pt="$(ls -1 "$run"model_*.pt 2>/dev/null | sort -V | tail -1 || true)"
    [ -n "$pt" ] && { echo "$pt"; return 0; }
  fi
  pt="$(find "$exp" -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2- || true)"
  if [ -z "$pt" ]; then
    echo "[ERR] 没找到 checkpoint（$exp/*/model_*.pt）" >&2
    echo "      先训练：bash scripts/run_prone_train.sh $1" >&2
    return 1
  fi
  echo "$pt"
}

play_one() {
  local which="$1" task pt
  task="$(task_of "$which")" || exit 1
  pt="$(latest_ckpt "$which")" || exit 1
  step "回放 $task-Play，checkpoint = $pt"
  echo "    （换别的：CKPT=<路径> bash scripts/run_prone_train.sh play $which）"
  "$BIN_PLAY" "$task-Play" --checkpoint-file "$pt"
}

case "$MODE" in
  smoke) smoke ;;
  stand2crawl) check_tasks; train_one stand2crawl ;;
  crawl)       check_tasks; train_one crawl ;;
  crawl2stand) check_tasks; train_one crawl2stand ;;
  all)
    check_tasks
    train_one stand2crawl
    train_one crawl
    train_one crawl2stand
    ;;
  play)
    check_tasks
    case "${2:-}" in
      stand2crawl|crawl|crawl2stand) play_one "$2" ;;
      *) echo "用法：play stand2crawl | play crawl | play crawl2stand"; exit 2 ;;
    esac
    ;;
  *)
    echo "未知参数：$MODE"
    echo "可用：smoke / stand2crawl / crawl / crawl2stand / all / play <which>"
    exit 2
    ;;
esac

echo
echo "==> 完成。"
echo "    回放：bash scripts/run_prone_train.sh play crawl"
echo "    指标（把 crawl 换成 stand2crawl / crawl2stand）："
echo "      grep -o 'Metrics/prone_err_rms_leg: [0-9.]*'  ~/train_prone_crawl.log | tail -5"
echo "      grep -o 'Metrics/prone_err_rms_all: [0-9.]*'  ~/train_prone_crawl.log | tail -5"
echo "      grep -o 'Metrics/prone_xy_err: [0-9.]*'       ~/train_prone_crawl.log | tail -5"
echo "      grep -o 'Metrics/prone_z: [0-9.]*'            ~/train_prone_crawl.log | tail -5"
echo "      grep -o 'Metrics/prone_progress: [0-9.]*'     ~/train_prone_crawl.log | tail -5"
echo "      tensorboard --logdir logs/rsl_rl/elf3_prone_crawl --port 6006"
