#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SITE="$SCRIPT_DIR/.venv/lib/python3.11/site-packages"

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$SITE/nvidia/cusparselt/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Pick a single GPU with the most free memory BEFORE the process starts CUDA.
# This must happen here (not in Python) so that torch/llama-cpp only ever see one
# card and never split the model across devices. Requires >= 8192 MiB free,
# otherwise CUDA_VISIBLE_DEVICES is left empty and the code falls back to CPU.
MIN_FREE_MIB=8192
if command -v nvidia-smi >/dev/null 2>&1; then
    BEST_LINE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | sort -t',' -k2 -nr | head -1)
    BEST_IDX=$(echo "$BEST_LINE" | cut -d',' -f1 | tr -d ' ')
    BEST_FREE=$(echo "$BEST_LINE" | cut -d',' -f2 | tr -d ' ')
    if [ -n "$BEST_FREE" ] && [ "$BEST_FREE" -ge "$MIN_FREE_MIB" ] 2>/dev/null; then
        export CUDA_VISIBLE_DEVICES="$BEST_IDX"
        echo "[run.sh] Selected GPU $BEST_IDX ($BEST_FREE MiB free)."
    else
        export CUDA_VISIBLE_DEVICES=""
        echo "[run.sh] No GPU with >= ${MIN_FREE_MIB} MiB free (max=${BEST_FREE:-none}); using CPU."
    fi
else
    export CUDA_VISIBLE_DEVICES=""
    echo "[run.sh] nvidia-smi not found; using CPU."
fi

# Sanity check: warn loudly if the torch install can't run on this GPU's arch
# (e.g. if some `uv sync` reverted torch to a build without sm_120 kernels).
# This does not abort the run — the Python code will fall back to CPU — but it
# makes the root cause obvious in the log instead of a cryptic CUDA crash.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    "$SCRIPT_DIR/.venv/bin/python" - <<'PY' || echo "[run.sh] WARNING: torch may not support this GPU arch; reranking could fall back to CPU."
import sys
try:
    import torch
    if not torch.cuda.is_available():
        sys.exit(1)
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    sys.exit(0 if cap in torch.cuda.get_arch_list() else 1)
except Exception:
    sys.exit(1)
PY
fi

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/main.py" "$@"
