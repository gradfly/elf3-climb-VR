#!/usr/bin/env bash
# 把 crab 任务包挂进 mjlab（mjlab 只会自动导入 mjlab/tasks/ 下的子包）
#
# 用法：
#   bash train/install_task.sh          # 安装/刷新（默认软链接）
#   bash train/install_task.sh --copy   # 强制用复制方式（软链接不被允许时）
#   bash train/install_task.sh --check  # 只自检，不动文件
#
# 为什么要软链接
# --------------
# 之前是 `cp` 到 site-packages。结果 `git pull` 只更新仓库，venv 里那份副本
# 不会跟着变 —— 拉到了新的 env_cfg_residual.py，注册表里却还是旧任务，
# 跑 play 直接 KeyError: 'Elf3-Crab-Residual-Play'。
# 软链接后仓库即唯一真源，pull 完立刻生效，不需要再 install。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/mjlab_tasks/crab"
MODE="link"
case "${1:-}" in
  --copy)  MODE="copy" ;;
  --check) MODE="check" ;;
  "")      ;;
  *) echo "未知参数：$1（可用 --copy / --check）"; exit 2 ;;
esac

# 默认用仓库自带的 venv，避免装到系统 python 里
if [ -z "${PYTHON_EXE:-}" ]; then
  if [ -x "$HERE/../.venv/bin/python" ]; then
    PY="$HERE/../.venv/bin/python"
  elif [ -x "$HERE/../.venv/Scripts/python.exe" ]; then
    PY="$HERE/../.venv/Scripts/python.exe"
  else
    PY="python"
  fi
else
  PY="$PYTHON_EXE"
fi

MJLAB_DIR="$("$PY" -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))")"
DEST="$MJLAB_DIR/tasks/crab"

echo "==> python : $PY"
echo "==> mjlab  : $MJLAB_DIR"
echo "==> 源目录 : $SRC"
echo "==> 目标   : $DEST  (mode=$MODE)"

if [ "$MODE" != "check" ]; then
  mkdir -p "$MJLAB_DIR/tasks"
  # 先清掉旧的实体目录 / 旧链接，换成指向仓库的软链接
  if [ -L "$DEST" ]; then
    rm -f "$DEST"
  elif [ -d "$DEST" ]; then
    BAK="$DEST.bak.$(date +%Y%m%d%H%M%S)"
    mv "$DEST" "$BAK"
    echo "==> 旧副本已备份到 $BAK，确认无误后可删"
  fi

  linked=0
  if [ "$MODE" = "link" ]; then
    if ln -sfn "$SRC" "$DEST" 2>/dev/null && [ -d "$DEST" ]; then
      linked=1
      echo "==> 已软链接：$DEST -> $SRC"
    else
      echo "==> 软链接失败（文件系统不支持？），回退到复制"
    fi
  fi

  if [ "$linked" -eq 0 ]; then
    mkdir -p "$DEST"
    cp -f "$SRC"/*.py "$DEST/"
    echo "==> 已复制 $(ls -1 "$SRC"/*.py | wc -l) 个 .py 到 $DEST"
  fi

  # 清掉可能过期的字节码，避免旧 __pycache__ 遮住改动
  # -L：DEST 通常是软链接，不跟进去就清不到仓库里的 __pycache__
  find -L "$DEST" -maxdepth 2 -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
fi

echo "==> 自检注册"
"$PY" - "$DEST" <<'PY'
import sys
from mjlab.tasks.registry import list_tasks
import mjlab.tasks.crab  # noqa: F401  触发注册

want = [
    "Elf3-Crab-Stairs",
    "Elf3-Crab-Stairs-Play",
    "Elf3-Crab-Residual",
    "Elf3-Crab-Residual-Play",
]
have = [t for t in list_tasks() if "Crab" in t]
print("已注册:", have)
missing = [w for w in want if w not in have]
if missing:
    print(f"[ERR] 缺任务: {missing}")
    print("      说明装进去的包里没有这些任务 —— 检查 train/mjlab_tasks/crab/__init__.py")
    sys.exit(1)
print("四个任务齐全 ✔")
PY

echo "==> 完成。示例："
echo "    MUJOCO_GL=egl .venv/bin/play Elf3-Crab-Residual-Play \\"
echo "        --checkpoint-file logs/rsl_rl/elf3_crab/<run>/model_XXXX.pt"
