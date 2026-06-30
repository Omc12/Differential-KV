#!/bin/zsh
# Clean, sequential, authoritative measurement batch for the DiffKV paper.
# Run with nothing else competing for CPU (the NumPy SVD during prefill is
# CPU-bound; concurrent processes inflate prefill time ~2x).
set -e
cd "$(dirname "$0")/../.."
PY=diffkv_venv/bin/python3
echo "=== [1/3] decode-mode ablation 4k-32k (compressed + exact) ==="
$PY paper/scripts/measure_active.py --ctx 4096 8192 16384 32768 \
    --modes compressed exact --gen 128 \
    --out paper/generated/active_modes_sweep_v2.json
echo "=== [2/3] 64k context reach (compressed + exact) ==="
$PY paper/scripts/measure_active.py --ctx 65536 \
    --modes compressed exact --gen 128 \
    --out paper/generated/active_modes_sweep_64k.json
echo "=== [3/3] residual-budget trade-off at 16k (R in 0,8,16,32,64) ==="
$PY paper/scripts/measure_residual_sweep.py --ctx 16384 \
    --residuals 0 8 16 32 64 --gen 96 \
    --out paper/generated/residual_sweep.json
echo "=== ALL MEASUREMENTS DONE ==="