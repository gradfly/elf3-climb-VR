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
#   bash scripts/run_squat_train.sh video down       # 回放导出 mp4（自动找最新 pt、无需窗口）
#   CKPT=logs/rsl_rl/.../model_1000.pt bash scripts/run_squat_train.sh play hold-squat
#   CKPT=logs/rsl_rl/.../model_1000.pt bash scripts/run_squat_train.sh video hold-squat
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
#   CKPT=...             play / video 模式下显式指定 checkpoint（不给就自动找最新）
#   VIDEO_STEPS=1000     mp4 录多少步（控制 50 Hz，1000 步 = 20 s；hold 默认 6000 = 120 s）
#   VIDEO_W=1280         mp4 宽（框架默认只有 320，必须放大）
#   VIDEO_H=720          mp4 高
#   VIDEO_TIMEOUT=900    等 mp4 写完的最长秒数
#   VIDEO_BOOL=0         设 1 把 --video 写成 "--video True"（新版 tyro 要求显式布尔值）
#   CR_FIX=0             设 1 把 \r 原地覆盖式进度条转成换行（旧版 rsl_rl / tqdm 用 \r 打印时）
#
# 日志：~/train_squat_<mode>.log（如 ~/train_squat_hold_squat.log）
#
# ⚠️ 实时输出：训练 stdout 直接打到终端，同时 tee 同步写日志。
#    不要在 tee 后面再接 `tail -N` —— tail 要读到 EOF 才能确定「最后 N 行」，
#    训练不结束它就一行都不吐，看起来就是「每轮指标没有实时输出」。
#    （2026-11-09 修：旧版 smoke/train_one 末尾接了 tail -25 / tail -40。）
# =============================================================================
set -euo pipefail

MODE="${1:-all}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-6000}"
SMOKE_ITERS="${SMOKE_ITERS:-5}"
VENV="${VENV:-.venv}"
CR_FIX="${CR_FIX:-0}"
VIDEO_W="${VIDEO_W:-1280}"
VIDEO_H="${VIDEO_H:-720}"
VIDEO_TIMEOUT="${VIDEO_TIMEOUT:-900}"
VIDEO_BOOL="${VIDEO_BOOL:-0}"

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

# 训练进程的 stdout/stderr 同时打到终端 + tee 写日志，两边都是实时的。
#
# ⚠️ 管道末端绝不能接 `tail -N`：tail 必须等 stdin EOF 才知道最后 N 行是哪几行，
#    所以在训练结束前它一行都不吐。旧版两处调用都接了 tail（smoke -25 / train -40），
#    表现就是「训练中屏幕不动、结束时突然冒出几十行」。
#    只想要末尾摘要的话，训练结束后自己去 tail 日志文件，别让它挡在管道里。
#
# CR_FIX=1：个别旧版 rsl_rl / tqdm 用 `\r` 原地覆盖打印进度，tee 写进日志会变成
#    一整行（没有换行符），grep / tail -f 都抓不到逐轮数据。这时把 \r 转成 \n。
#    sed -u 是无缓冲的，不会再引入一层延迟；CRLF 输出会多出空行，不影响内容。
run_logged() {
  local log="$1"; shift
  echo "    实时日志：$log"
  echo "    另开终端看逐轮指标：tail -F \"$log\" | grep --line-buffered -E 'Learning iteration|Mean reward|Mean episode length|Metrics/'"
  if [ "$CR_FIX" = "1" ]; then
    "$@" 2>&1 | sed -u 's/\r/\n/g' | tee "$log"
  else
    "$@" 2>&1 | tee "$log"
  fi
}

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
    run_logged "$HOME/train_squat_${t}_smoke.log" \
      "$BIN_TRAIN" "$task" \
        --env.scene.num-envs 256 --agent.max-iterations "$SMOKE_ITERS"
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
  run_logged "$HOME/train_squat_${which}.log" \
    "$BIN_TRAIN" "$task" \
      --env.scene.num-envs "$NUM_ENVS" --agent.max-iterations "$ITERS"
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

# 回放导出 mp4：走 mjlab play 自带的 VideoRecorder（不用自己一帧帧渲染）。
#
# 几个必须处理的坑：
#   * 必须带 --checkpoint-file：agent=zero/random 时 play.py 会直接禁用录像；
#   * --num-envs 1：离屏相机跟着 asset body，多 env 时容易录到一片空地；
#   * --video-length 是「步数」不是秒，控制 50 Hz，步数 / 50 = 秒；
#   * 框架默认只录 200 步（4 s）、320x240，两个都得显式放大；
#   * 产物固定叫 rl-video-step-0.mp4，重录会覆盖，所以先挪走旧的、再复制成带任务名的文件；
#   * 录满 video_length 后 viewer 不会自己退出，所以后台跑 + 轮询文件，写完就收掉进程。
video_one() {
  local which="$1" task pt outdir tmp dest steps def_steps pid sz last stable n run_tag
  task="$(task_of "$which")" || exit 1
  pt="$(latest_ckpt "$which")" || exit 1
  if [[ "$which" == hold* ]]; then def_steps=6000; else def_steps=1000; fi
  steps="${VIDEO_STEPS:-$def_steps}"

  outdir="$(dirname "$pt")/videos/play"
  tmp="$outdir/rl-video-step-0.mp4"
  mkdir -p "$outdir"
  [ -f "$tmp" ] && mv -f "$tmp" "$outdir/rl-video-step-0.prev.mp4" || true

  local video_args
  if [ "$VIDEO_BOOL" = "1" ]; then video_args=(--video True); else video_args=(--video); fi

  step "导出 mp4：$task-Play，ckpt=$pt"
  echo "    ${steps} 步 = $((steps / 50)) s @50 fps，${VIDEO_W}x${VIDEO_H}"
  "$BIN_PLAY" "$task-Play" --checkpoint-file "$pt" --num-envs 1 \
    "${video_args[@]}" --video-length "$steps" \
    --video-width "$VIDEO_W" --video-height "$VIDEO_H" &
  pid=$!

  # 等 mp4 写完：文件出现且大小连续 3 次不变，或 play 自己退出。
  last=-1; stable=0; n=0
  while kill -0 "$pid" 2>/dev/null; do
    if [ -s "$tmp" ]; then
      sz="$(stat -c %s "$tmp" 2>/dev/null || echo 0)"
      if [ "$sz" = "$last" ]; then stable=$((stable + 1)); else stable=0; fi
      last="$sz"
      [ "$stable" -ge 3 ] && { echo "    mp4 写完：$tmp ($sz B)"; break; }
    fi
    sleep 1
    n=$((n + 1))
    if [ "$n" -ge "$VIDEO_TIMEOUT" ]; then echo "[warn] 等了 ${n}s 还没写完，先收尾"; break; fi
  done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true

  if [ ! -s "$tmp" ]; then
    echo "[ERR] 没生成 $tmp。常见原因："
    echo "      --video 写法不对（试 VIDEO_BOOL=1）、viewer 起不来（VNC 里要有 DISPLAY）、"
    echo "      或显存被同时跑的训练占满。上面 play 的输出里有具体报错。"
    exit 1
  fi
  run_tag="$(basename "$(dirname "$pt")" | tr -cd 'A-Za-z0-9._-')"
  dest="$HOME/squat_${which}_${run_tag}_$(basename "$pt" .pt).mp4"
  cp -f "$tmp" "$dest"
  echo "[OK] mp4 已导出：$dest"
  echo "     框架原始产物：$tmp（固定文件名，下次录会被挪走）"
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
  video)
    check_tasks
    case "${2:-}" in
      down|up|hold-squat|hold-stand)
        video_one "${2//-/_}" ;;
      *) echo "用法：video down | video up | video hold-squat | video hold-stand"; exit 2 ;;
    esac
    ;;
  all)
    check_tasks
    train_one down
    train_one up
    train_one hold_squat
    train_one hold_stand
    ;;
  *) echo "未知参数：$MODE（可用 smoke / down / up / hold / hold-squat / hold-stand / play / video / all）"
     exit 2 ;;
esac

echo
echo "==> 完成。回放导出 mp4：bash scripts/run_squat_train.sh video down"
echo "    （保持任务默认录 120 s：VIDEO_STEPS=6000 bash scripts/run_squat_train.sh video hold-squat）"
echo
echo "    完整日志：~/train_squat_<mode>.log（训练过程中一直实时写入）"
echo "    训练中另开终端实时看逐轮指标（grep 必须带 --line-buffered，否则又会被缓冲）："
echo "    tail -F ~/train_squat_down.log | grep --line-buffered -E 'Learning iteration|Mean reward|Mean episode length|Metrics/'"
echo
echo "    训完看指标（把 down 换成 up / hold_squat / hold_stand 即可）："
echo "    grep -o 'Metrics/traj_err_rms_leg: [0-9.]*' ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/sym_err_rad: [0-9.]*'     ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/base_xy: [0-9.]*'         ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/base_yaw_deg: [0-9.]*'    ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/arm_leg_touch: [0-9.]*'   ~/train_squat_down.log | tail -5"
echo "    grep -o 'Metrics/joint_speed: [0-9.]*'     ~/train_squat_hold_squat.log | tail -5"
echo "    tensorboard --logdir logs/rsl_rl/elf3_squat_down --port 6006"
