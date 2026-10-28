#!/usr/bin/env bash
# ============================================================
#  Crab-walk stair climbing - one-click launcher (Git Bash / Linux / macOS)
#  Usage:
#      ./run_crab.sh                      # full pipeline + replay window
#      ./run_crab.sh --preset easy
#      ./run_crab.sh --seed 7 --step-time 1.2
#      ./run_crab.sh --show-config
#      ./run_crab.sh --stages gen,plan    # no window
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$ROOT/sim/vendor/bxi_controller_ros2/src/bxi_example_py_elf3/launch/run_crab.py"

# Python: $PYTHON_EXE > $PY > workbuddy bundled > python
PY="${PYTHON_EXE:-${PY:-}}"
if [[ -z "$PY" ]]; then
    CAND="$HOME/.workbuddy/binaries/python/versions/3.13.12/python.exe"
    if [[ -x "$CAND" ]]; then PY="$CAND"; else PY="python"; fi
fi

if [[ ! -f "$RUNNER" ]]; then
    echo "[ERROR] runner not found: $RUNNER" >&2
    exit 1
fi

echo "Using python: $PY"
echo

"$PY" "$RUNNER" "$@"
