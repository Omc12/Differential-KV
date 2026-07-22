#!/usr/bin/env bash
# ============================================================================
# run_paper_evals.sh — Re-run the original paper benchmarks + RULER/LongBench
# ============================================================================
#
# Phase A1: Main NIAH sweep (active vs dense, 4k/8k/16k/32k/64k)
#           Re-verifies consistency_report.md §6 headline numbers.
#
# Phase C1: LongBench (NarrativeQA, Qasper, HotpotQA, GovReport)
#           20 samples each, compare dense vs DiffKV.
#
# Phase C2: RULER (NIAH-single, NIAH-multi-keys, NIAH-multi-values,
#           NIAH-multi-queries, variable-tracking, CWE, FWE, QA)
#           at 4k/8k/16k.
#
# Usage:
#   ./benchmarks/run_paper_evals.sh           # run all three
#   ./benchmarks/run_paper_evals.sh A1        # only re-run paper sweep
#   ./benchmarks/run_paper_evals.sh C1 C2     # only LongBench + RULER
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO/diffkv_venv/bin/python3"
BENCH="$REPO/benchmarks"
ACTIVE="$REPO/ACTIVE_RUNTIME"
RESULTS="$BENCH/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RESULTS/paper_run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}" | tee -a "$LOG_DIR/master.log"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL: $*${NC}" | tee -a "$LOG_DIR/master.log"; }

run_phase() {
    local phase="$1"; local label="$2"; shift 2
    log "==========  $phase: $label  =========="
    local t0=$SECONDS
    if "$@" 2>&1 | tee "$LOG_DIR/${phase}.log"; then
        local elapsed=$(( SECONDS - t0 ))
        log "DONE $phase in ${elapsed}s"
        echo "$phase: OK (${elapsed}s)" >> "$LOG_DIR/summary.txt"
    else
        local code=$?
        fail "$phase exited $code"
        echo "$phase: FAIL (exit=$code)" >> "$LOG_DIR/summary.txt"
        return 1
    fi
}

if [[ ! -x "$VENV_PY" ]]; then
    fail "diffkv_venv not found. Run: cd $REPO && make setup && pip install mlx mlx-lm datasets"
    exit 1
fi
"$VENV_PY" -c "import mlx.core" || { fail "MLX not installed. Run: pip install mlx mlx-lm"; exit 1; }

PHASES=("$@")
should_run() {
    local p="$1"
    [[ ${#PHASES[@]} -eq 0 ]] && return 0
    for ph in "${PHASES[@]}"; do [[ "$ph" == "$p" ]] && return 0; done
    return 1
}

# Paper-config env for all runs
export DIFFKV_COMPRESSED_DECODE=1
export DIFFKV_MAX_RESIDUAL=128
export DIFFKV_SPARSE_PREFILL=1
export DIFFKV_DECODE_CACHE=1
export DIFFKV_SPARSE_BIAS=auto
export DIFFKV_SEED=1234
export TOKENIZERS_PARALLELISM=false

# ============================================================================
# A1 — Re-run original paper NIAH sweep: active vs dense, 4k–64k
#      Matches consistency_report.md §6 config exactly.
#      Output: results/results_<timestamp>.json + results/summary.md
# ============================================================================
if should_run "A1"; then
    run_phase "A1" "Main NIAH sweep — active vs dense (4k/8k/16k/32k/64k)" \
        "$VENV_PY" "$BENCH/run_bench.py" \
            --engines active dense \
            --contexts 4096 8192 16384 32768 65536 \
            --gen 128 \
            --timeout 2400 \
            --ram-cap-gb 8.2
fi

# ============================================================================
# C1 — LongBench: NarrativeQA, Qasper, HotpotQA, GovReport
#      20 examples each, dense vs DiffKV side-by-side (--compare).
#      Output: results/longbench_compare_<timestamp>.json
# ============================================================================
if should_run "C1"; then
    LONGBENCH_OUT="$RESULTS/longbench_compare_${TIMESTAMP}.json"
    run_phase "C1" "LongBench — NarrativeQA / Qasper / HotpotQA / GovReport (dense vs DiffKV)" \
        "$VENV_PY" "$ACTIVE/run_longbench.py" \
            --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
            --preset mid \
            --serving-mode long-context \
            --rank 32 \
            --num-samples 20 \
            --max-input-tokens 8000 \
            --compare \
            --output "$LONGBENCH_OUT" \
            --datasets narrativeqa,qasper,hotpotqa,govreport
fi

# ============================================================================
# C2 — RULER: Standard long-context benchmark suite
#      Tasks: NIAH-single, NIAH-multi-keys, NIAH-multi-values,
#             NIAH-multi-queries, variable-tracking, CWE, FWE, QA
#      Contexts: 4k, 8k, 16k. 10 examples per task.
#      Output: results/ruler_results_<timestamp>.json
# ============================================================================
if should_run "C2"; then
    RULER_OUT="$RESULTS/ruler_results_${TIMESTAMP}.json"
    run_phase "C2" "RULER — standard long-context benchmark (4k/8k/16k)" \
        "$VENV_PY" "$BENCH/run_ruler_mlx.py" \
            --output "$RULER_OUT" \
            --contexts 4096 8192 16384 \
            --num-samples 10
fi

# ============================================================================
log "==========  All done  =========="
echo ""
echo "=== Phase Summary ==="
cat "$LOG_DIR/summary.txt" 2>/dev/null
echo ""
echo "=== Outputs ==="
ls -lh "$RESULTS/"*.json 2>/dev/null | tail -10
