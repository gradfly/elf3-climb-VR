#!/usr/bin/env bash
# =============================================================================
# deploy_gpu.sh — 新云服务器「一条命令」部署螃蟹步爬楼梯仿真 + RL 训练
# =============================================================================
# 前置：已经 clone 好仓库，且在仓库根目录（脚本自己会 cd 到根）。
#
# >>> USAGE
# 部署 / 冒烟：
#   bash scripts/deploy_gpu.sh                  # 体检 → 装环境 → 资产 → 注册 → 冒烟
#   bash scripts/deploy_gpu.sh --no-smoke       # 跳过冒烟（赶时间，之后再手跑）
#   bash scripts/deploy_gpu.sh --force          # 删掉 .venv 重装（换 GPU / torch 装错时用）
#   bash scripts/deploy_gpu.sh --num-envs 2048 --iters 3000
#                                               # 只改最后打印的建议训练命令
#
# 它按顺序做 6 件事：
#   [0] 体检：GPU / 驱动 CUDA / 显存 / 磁盘 / CPU，不合规直接拦下
#   [1] 环境：scripts/bootstrap_gpu.sh（系统库 → uv → py3.11 → mujoco 生态 → torch 最后装）
#   [2] 校验：mjlab.tasks.registry 可导入、torch.cuda 可用
#   [3] 资产：train/assets/make_robot_xml.py（nq=38 nv=37 nu=0，geom group 0 为空）
#   [4] 注册：train/install_task.sh（4 个 Elf3-Crab-* 任务）
#   [5] 冒烟：zero / random / 256 env × 5 iter
#
# 幂等：重复执行会跳过已就绪的 .venv；不会覆盖 logs/ 里的历史 run。
# <<< USAGE
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
VENV="$REPO_DIR/.venv"
PY="$VENV/bin/python"
LOG_DIR="$REPO_DIR/logs"
SMOKE_LOG="$LOG_DIR/deploy_smoke.log"
SMOKE_ITERS=5

FORCE=0
SMOKE=1
NUM_ENVS=""
ITERS=3000

step() { echo; echo "==> $*"; }
warn() { echo "    [!] $*"; }
die()  { echo "[ERR] $*" >&2; exit 1; }

usage() {
  sed -n '/^# >>> USAGE/,/^# <<< USAGE/p' "$0" \
    | sed '1d;$d' \
    | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --force)    FORCE=1 ;;
    --no-smoke) SMOKE=0 ;;
    --num-envs) NUM_ENVS="${2:?--num-envs 需要一个数字}"; shift ;;
    --iters)    ITERS="${2:?--iters 需要一个数字}"; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "未知参数：$1"; echo; usage; exit 2 ;;
  esac
  shift
done

# -----------------------------------------------------------------------------
step "[0/6] 体检"
# -----------------------------------------------------------------------------
[ "$(uname -s)" = "Linux" ] || die "只支持 Linux，当前是 $(uname -s)"

command -v nvidia-smi >/dev/null 2>&1 \
  || die "没有 nvidia-smi —— 先装 NVIDIA 驱动（CUDA >= 12.4）再回来跑"

GPU_INFO="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)"
[ -n "$GPU_INFO" ] || die "nvidia-smi 跑不通（驱动没装好？），先修驱动再回来"
echo "    GPU: $GPU_INFO"

DRV_CUDA="$(nvidia-smi 2>/dev/null \
  | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
if [ -z "$DRV_CUDA" ]; then
  warn "读不到 nvidia-smi 里的 CUDA Version，bootstrap 会退回 cu124"
else
  echo "    驱动 CUDA: $DRV_CUDA"
  MAJ="${DRV_CUDA%%.*}"
  MIN="${DRV_CUDA#*.}"
  if [ "$MAJ" -lt 12 ] || { [ "$MAJ" -eq 12 ] && [ "$MIN" -lt 4 ]; }; then
    die "驱动只到 CUDA $DRV_CUDA（< 12.4）—— 先升级驱动，硬装 torch 只会得到 cuda_available=False"
  fi
fi

VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
  | head -1 | tr -d ' ' || true)"
VRAM_MIB="${VRAM_MIB:-0}"
case "$VRAM_MIB" in ''|*[!0-9]*) VRAM_MIB=0 ;; esac
suggest_envs() {
  if   [ "$VRAM_MIB" -ge 70000 ]; then echo 12288
  elif [ "$VRAM_MIB" -ge 40000 ]; then echo 8192
  elif [ "$VRAM_MIB" -ge 22000 ]; then echo 4096
  elif [ "$VRAM_MIB" -ge 14000 ]; then echo 2048
  else echo 1024
  fi
}
[ -n "$NUM_ENVS" ] || NUM_ENVS="$(suggest_envs)"
echo "    显存: $((VRAM_MIB / 1024)) GB  ->  建议 num_envs=$NUM_ENVS"

FREE_GB="$(df -Pk "$REPO_DIR" 2>/dev/null | awk 'NR==2 {printf "%d", $4 / 1048576}' || true)"
case "$FREE_GB" in ''|*[!0-9]*) FREE_GB=0 ;; esac
echo "    磁盘可用: ${FREE_GB} GB"
[ "$FREE_GB" -ge 40 ] || warn "可用磁盘 < 40 GB：依赖 + checkpoint 可能不够，建议挂数据盘"

CORES="$(nproc 2>/dev/null || echo 0)"
case "$CORES" in ''|*[!0-9]*) CORES=0 ;; esac
echo "    CPU 核数: $CORES"
[ "$CORES" -ge 16 ] || warn "核数偏少：地形生成与 env reset 在主进程，可能成为瓶颈（建议 >= 16）"

# -----------------------------------------------------------------------------
step "[1/6] Python 环境（venv: $VENV）"
# -----------------------------------------------------------------------------
if [ "$FORCE" -eq 0 ] && [ -x "$PY" ] \
   && "$PY" -c "import mjlab.tasks.registry, torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "    已有可用 .venv（mjlab + torch.cuda 都 OK），跳过 bootstrap；要重装加 --force"
else
  if [ "$FORCE" -eq 1 ]; then
    rm -rf "$VENV"
    echo "    已删除旧 .venv"
  fi
  bash scripts/bootstrap_gpu.sh
fi
[ -x "$PY" ] || die "bootstrap 结束后没有 $PY —— 看上面 bootstrap 输出定位失败步骤"

# -----------------------------------------------------------------------------
step "[2/6] 校验 mjlab / torch"
# -----------------------------------------------------------------------------
"$PY" - <<'PY'
import mujoco, torch, mjlab
import mjlab.tasks.registry  # noqa: F401  PyPI 版没有这个模块

print(f"    mujoco : {mujoco.__version__}")
print(f"    mjlab  : {mjlab.__file__}")
print(f"    torch  : {torch.__version__}  (built for cu{torch.version.cuda})")
assert torch.cuda.is_available(), (
    "torch.cuda.is_available() = False：装的 cuXXX 高于驱动支持，"
    "按 docs/gpu_server_deploy.md §2.2 重装匹配版本")
print(f"    device : {torch.cuda.get_device_name(0)}")
PY

# -----------------------------------------------------------------------------
step "[3/6] 生成纯机器人 XML"
# -----------------------------------------------------------------------------
"$PY" train/assets/make_robot_xml.py
"$PY" - <<'PY'
from collections import Counter

import mujoco

p = ("sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3"
     "/data/mujoco_simulation/elf3_robot_only.xml")
m = mujoco.MjModel.from_xml_path(p)
g = Counter(int(m.geom_group[i]) for i in range(m.ngeom))
assert (m.nq, m.nv, m.nu) == (38, 37, 0), \
    f"期望 nq=38 nv=37 nu=0，实际 {m.nq}/{m.nv}/{m.nu}"
assert g.get(0, 0) == 0, (
    f"group 0 里还有 {g[0]} 个 geom：terrain_scan 只打 group 0，"
    "height_scan 会打到机器人自己的腰腿，rise_est 会被顶到上限")
print(f"    nq=38 nv=37 nu=0 ✔ | geom group {dict(sorted(g.items()))}")
PY

# -----------------------------------------------------------------------------
step "[4/6] 注册 crab 任务"
# -----------------------------------------------------------------------------
bash train/install_task.sh

# -----------------------------------------------------------------------------
step "[5/6] 参考残差表"
# -----------------------------------------------------------------------------
REF="$REPO_DIR/train/assets/ref_tables/crab_ref_v1.npz"
if [ -f "$REF" ]; then
  echo "    已随仓库提供：$REF（$(du -h "$REF" | cut -f1)）"
else
  warn "缺 $REF，现场重建（纯 CPU，约 90 s）"
  "$PY" train/assets/build_ref_tables.py
fi

# -----------------------------------------------------------------------------
step "[6/6] 冒烟"
# -----------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
if [ "$SMOKE" -eq 0 ]; then
  echo "    --no-smoke，跳过。建议之后手动跑："
  echo "      MUJOCO_GL=egl .venv/bin/play Elf3-Crab-Residual-Play --agent random"
else
  export MUJOCO_GL=egl WANDB_MODE=disabled PYTHONUNBUFFERED=1

  # 第 1/2 层：能建环境、跑起来不报错就算过。play 会一直跑到窗口关闭，
  # 无头机器上靠 timeout 收尾，退出码 124 属于预期，不算失败。
  run_play() {
    local agent="$1" rc=0
    printf '    play --agent %-6s ... ' "$agent"
    timeout 60 "$VENV/bin/play" Elf3-Crab-Residual-Play --agent "$agent" \
      >>"$SMOKE_LOG" 2>&1 || rc=$?
    case "$rc" in
      0)        echo "OK" ;;
      124)      echo "OK（60 s 到点主动结束）" ;;
      *)        echo "失败(rc=$rc)，详见 $SMOKE_LOG" ; return 1 ;;
    esac
  }
  # 这两层只是「能建环境、能跑」的粗筛：无头机器没有窗口时 play 自己会退，
  # 失败先确认日志里有没有 Traceback，别急着判环境坏。
  run_play zero   || warn "zero 冒烟没过：先 grep -n Traceback $SMOKE_LOG，别急着开大规模"
  run_play random || warn "random 冒烟没过：同上"

  # 第 3 层：小规模真训练，这层必须过。
  echo "    train 256 env × $SMOKE_ITERS iter（首次会编译 warp kernel，1–3 min）"
  if ! "$VENV/bin/train" Elf3-Crab-Residual \
        --env.scene.num-envs 256 --agent.max-iterations "$SMOKE_ITERS" \
        --log-root "$LOG_DIR/rsl_rl" \
        2>&1 | tee -a "$SMOKE_LOG"; then
    die "小规模训练失败 —— 对照 docs/gpu_server_deploy.md §7 报错速查，日志在 $SMOKE_LOG"
  fi

  RUN_DIR="$(ls -1dt "$LOG_DIR"/rsl_rl/elf3_crab/*/ 2>/dev/null | head -1 || true)"
  if [ -n "$RUN_DIR" ]; then
    echo "    本次 run: $RUN_DIR"
    ls -1 "$RUN_DIR" | sed 's/^/      /'
    echo "    看曲线：tensorboard --logdir $LOG_DIR/rsl_rl"
  else
    warn "没找到 logs/rsl_rl/elf3_crab/<run> —— 检查 --log-root 与写盘权限"
  fi
fi

# -----------------------------------------------------------------------------
step "完成"
# -----------------------------------------------------------------------------
echo "    cd $REPO_DIR && source .venv/bin/activate"
echo
echo "    正式训练（VNC 里前台跑，断线重连进程还在；改用 SSH 才需要 tmux）："
echo "      WANDB_MODE=disabled PYTHONUNBUFFERED=1 train Elf3-Crab-Residual \\"
echo "        --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS \\"
echo "        --log-root logs/rsl_rl 2>&1 | tee ~/train.log"
echo
echo "    续训（注意：resume 会把速度/地形课程重置到起点）："
echo "      WANDB_MODE=disabled PYTHONUNBUFFERED=1 train Elf3-Crab-Residual \\"
echo "        --agent.resume True --env.scene.num-envs $NUM_ENVS --agent.max-iterations $ITERS"
echo "      # 本版 CLI 没有 --checkpoint-file（旧文档写错了，会 Unrecognized options）。"
echo "      # resume=True 自己去 logs/rsl_rl/elf3_crab/ 找最新 run 的最新 checkpoint。"
echo "      # 要从指定文件起跑：train ... --help | grep -iE 'resume|load|checkpoint' 查真实字段名"
echo
echo "    回放 / 导出（在本地机器执行回传）："
echo "      python train/export_onnx.py --task Elf3-Crab-Residual \\"
echo "        --checkpoint logs/rsl_rl/elf3_crab/<run>/model_$ITERS.pt \\"
echo "        --out train/checkpoints/elf3_crab_residual.onnx"
echo "      # 本地： SERVER=user@<IP> bash scripts/sync_back.sh"
echo
echo "    logs/ 想放数据盘：ln -sfn /data/logs $REPO_DIR/logs（先备份已有 logs）"
