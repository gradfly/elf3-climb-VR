#!/usr/bin/env bash
# 把 crab 任务包安装进 mjlab（mjlab 只会自动导入 mjlab/tasks/ 下的子包）
# 用法： bash train/install_task.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON_EXE:-python}"

MJLAB_DIR="$("$PY" -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))")"
DEST="$MJLAB_DIR/tasks/crab"

echo "==> mjlab: $MJLAB_DIR"
echo "==> 安装任务包 -> $DEST"

mkdir -p "$DEST"
cp -f "$HERE/mjlab_tasks/crab/"*.py "$DEST/"

echo "==> 自检注册"
"$PY" - <<'PY'
from mjlab.tasks.registry import list_tasks
import mjlab.tasks.crab  # noqa: F401  (触发注册)
ts = [t for t in list_tasks() if "Crab" in t]
print("已注册:", ts)
assert ts, "注册失败：没有找到 Crab 任务"
PY

echo "==> 完成。训练："
echo "    MUJOCO_GL=egl uv run train Elf3-Crab-Stairs --env.scene.num-envs 4096"
