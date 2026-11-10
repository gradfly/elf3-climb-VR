#!/usr/bin/env bash
# ============================================================
#  双手双脚贴地爬行（prone crawl）- 一键：生成 -> 验收 -> 回放
#
#  用法：
#      ./run_crawl.sh                 # 默认拆成两个动作：蹲下->双手撑地->双脚后伸->俯卧 + 向前爬行
#      ./run_crawl.sh --merge         # 不拆，出一整段（含 5.0 s 过渡 + 4 循环）
#      ./run_crawl.sh --no-replay     # 只生成 + 验收（服务器/无显示）
#      ./run_crawl.sh --phys          # 额外跑一遍物理 PD 回放
#      ./run_crawl.sh --record        # 离屏出 PNG 序列到 logs/prone_crawl/frames*
#      ./run_crawl.sh --speed 0.10 --cycles 6   # 参数透传给生成器
#      ./run_crawl.sh --preset fast   # 0.5 m/s 快档（缩周期 + 收站距）
#
#  为什么要拆：完整动作里"过渡段末帧"与"循环段首帧"姿态差 55°，接缝那一帧的
#  二阶差分 514 rad/s²，是全程角加速度峰值，怎么修都压不住（两段分别求解，
#  接缝处 IK 的解就是不一样）。拆成两个独立动作后接缝消失：
#  fast 档下动作 1 峰值 289.4、动作 2 峰值 164.9 rad/s²（判据 400）。
#  动作 1 现为四阶段（站立→下蹲→双手撑地→双脚向后伸→趴姿），2026-11-09 起
#  **手撑地(0.44)必须早于脚后伸(0.44~0.74)**：v8 是脚提前后撤 + 边前倾边伸臂 +
#  腿长时间静止，观感不协调。v9 用基座前移（x_frac 峰值 2.1）把髋拉回蹲姿脚
#  上方，才可能在「脚还没后伸」时就把手撑到地上，详见 docs/prone_crawl.md §3.4。
#  normal / fast × 动作 1 / 2 四套轨迹均通过数值验收（normal 动作 2 质心 98.0%）。
#  末段对齐默认 48 帧（--blend-frames 可改）：32 帧时右上臂会扫进地面 38.7 mm。
#  注意：验收只查穿地、不查离地；改「脚后伸」段时必须同时看
#  python scripts/render_prone_preview.py 出的 a1_sheet_p*.png 或 GIF，
#  以及 python scripts/check_prone_entry.py 的「顺序核对」。
#  下蹲段腿部电机抖动另有一条判据（常规验收查不出高频小幅反向噪声）：
#  python scripts/check_prone_shake.py [--preset fast|normal]。生成器末尾的
#  smooth_leg_joints()（leg_shake_sigma=2.0）已处理，详见 docs/prone_crawl.md §3.4 末。
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="$ROOT/sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch"
DATA="$ROOT/sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/data/mujoco_simulation"

PY="${PYTHON_EXE:-${PY:-}}"
if [[ -z "$PY" ]]; then
    CAND="$HOME/.workbuddy/binaries/python/versions/3.13.12/python.exe"
    if [[ -x "$CAND" ]]; then PY="$CAND"; else PY="python"; fi
fi

NO_REPLAY=0
DO_PHYS=0
DO_RECORD=0
SPLIT=1
PRESET="normal"
GEN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-replay) NO_REPLAY=1; shift ;;
        --phys)      DO_PHYS=1;   shift ;;
        --record)    DO_RECORD=1; shift ;;
        --split)     SPLIT=1; shift ;;
        --merge)     SPLIT=0; shift ;;
        --preset)    PRESET="$2"; GEN_ARGS+=("$1" "$2"); shift 2 ;;
        -h|--help)   sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           GEN_ARGS+=("$1"); shift ;;
    esac
done

# 预设决定轨迹文件名（generator 里 preset_out() 的命名），验收/回放要跟着走
case "$PRESET" in
    fast) SUFFIX="_fast" ;;
    *)    SUFFIX="" ;;
esac
TRAJ="$DATA/prone_crawl${SUFFIX}_traj.npz"
TRAJ_A="$DATA/prone_standup${SUFFIX}_traj.npz"
TRAJ_B="$DATA/prone_gait${SUFFIX}_traj.npz"

export PYTHONUNBUFFERED=1
echo "Using python: $PY"
echo

echo "== 1/4 生成场景与轨迹 =="
if [[ $SPLIT -eq 1 ]]; then
    "$PY" "$LAUNCH/prone_crawl.py" --write-scene --split "${GEN_ARGS[@]+"${GEN_ARGS[@]}"}"
else
    "$PY" "$LAUNCH/prone_crawl.py" --write-scene "${GEN_ARGS[@]+"${GEN_ARGS[@]}"}"
fi

if [[ $SPLIT -eq 1 ]]; then
    echo
    echo "== 2/4 数值验收（两个动作分别验）=="
    "$PY" "$ROOT/scripts/check_prone_crawl.py" --kind standup \
        --traj "$TRAJ_A" \
        --report "$ROOT/logs/prone_crawl/report_standup.txt" \
        --csv "$ROOT/logs/prone_crawl/contact_standup.csv" || CHECK_RC=$?
    "$PY" "$ROOT/scripts/check_prone_crawl.py" --kind gait \
        --traj "$TRAJ_B" \
        --report "$ROOT/logs/prone_crawl/report_gait.txt" \
        --csv "$ROOT/logs/prone_crawl/contact_gait.csv" || CHECK_RC=$?

    if [[ $DO_RECORD -eq 1 ]]; then
        echo
        echo "== 3/4 离屏渲染 PNG 序列 =="
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ_A" --mode kin --no-gui \
            --record "$ROOT/logs/prone_crawl/frames_standup" --every 4
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ_B" --mode kin --no-gui \
            --record "$ROOT/logs/prone_crawl/frames_gait" --every 4
    fi

    if [[ $NO_REPLAY -eq 0 ]]; then
        echo
        echo "== 4/4 回放：先播动作 1（站立->俯卧），再循环动作 2（向前爬行）=="
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ_A" --mode kin || true
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ_B" --mode kin --loop || true
    fi

    if [[ $DO_PHYS -eq 1 ]]; then
        echo
        echo "== 附加：动作 2 的物理 PD 回放（开环，只看诊断数字）=="
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ_B" --mode phys --no-gui || true
    fi
else
    echo
    echo "== 2/4 数值验收 =="
    # 验收脚本失败时不要直接退出：先让用户看见报告，再返回非零
    "$PY" "$ROOT/scripts/check_prone_crawl.py" --traj "$TRAJ" || CHECK_RC=$?

    if [[ $DO_RECORD -eq 1 ]]; then
        echo
        echo "== 3/4 离屏渲染 PNG 序列 =="
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ" --mode kin --no-gui --record "$ROOT/logs/prone_crawl/frames" --every 2
    fi

    if [[ $NO_REPLAY -eq 0 ]]; then
        echo
        echo "== 4/4 运动学回放（关掉窗口或按 R 重播）=="
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ" --mode kin --loop || true
    fi

    if [[ $DO_PHYS -eq 1 ]]; then
        echo
        echo "== 附加：物理 PD 回放（开环，预期爬不动，只看诊断数字）=="
        "$PY" "$LAUNCH/prone_replay.py" --traj "$TRAJ" --mode phys --no-gui || true
    fi
fi

if [[ "${CHECK_RC:-0}" -ne 0 ]]; then
    echo
    echo "[警告] 数值验收未全项通过，详见 logs/prone_crawl/report_*.txt" >&2
    exit "$CHECK_RC"
fi
echo
echo "[OK] 全部完成"
