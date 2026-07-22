#!/usr/bin/env bash
# ============================================================================
# run_all_evals.sh — Full DiffKV Evaluation Suite (MLX Active Runtime)
# ============================================================================
# Runs ALL benchmarks for the paper in a fixed sequence:
#
#  Phase A — Paper baselines (re-verify existing paper numbers)
#    A1: Main NIAH sweep — active vs dense @ 4k/8k/16k/32k/64k
#
#  Phase B — Extended paper evals (already in paper sections E7–E12)
#    B1: Multi-needle NIAH (4 needles @ 4k/16k/32k)
#    B2: Multi-hop NIAH (chain recall @ 4k/16k/32k)
#    B3: Perplexity eval (dense vs DiffKV @ 4k/8k/16k)
#    B4: Llama-3.2-3B cross-arch NIAH (@ 4k/8k/16k, depths 0.1/0.5/0.9)
#    B5: Residual signal ablation (owner/edge capture @ 8k)
#    B6: Lego prefill peak memory (@ 16k/32k/48k)
#
#  Phase C — New benchmarks (for grant applications / broader credibility)
#    C1: LongBench (NarrativeQA, Qasper, HotpotQA, GovReport — compare mode)
#
# Usage:
#   ./benchmarks/run_all_evals.sh           # run everything
#   ./benchmarks/run_all_evals.sh A1        # run only phase A1
#   ./benchmarks/run_all_evals.sh B1 B2 B3  # run specific phases
#
# Results land in:
#   benchmarks/results/           (JSON + per-run logs)
#   ACTIVE_RUNTIME/               (LongBench JSON)
# ============================================================================

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO/diffkv_venv/bin/python3"
BENCH="$REPO/benchmarks"
ACTIVE="$REPO/ACTIVE_RUNTIME"
RESULTS="$BENCH/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$RESULTS/run_all_${TIMESTAMP}"

mkdir -p "$LOG_DIR"

# ── Helpers ─────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] $*${NC}" | tee -a "$LOG_DIR/master.log"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN: $*${NC}" | tee -a "$LOG_DIR/master.log"; }
fail() { echo -e "${RED}[$(date '+%H:%M:%S')] FAIL: $*${NC}" | tee -a "$LOG_DIR/master.log"; }

run_phase() {
    local phase="$1"; local label="$2"; shift 2
    log "=========================================="
    log "Starting $phase: $label"
    log "=========================================="
    local t0=$SECONDS
    if "$@" 2>&1 | tee "$LOG_DIR/${phase}.log"; then
        local elapsed=$(( SECONDS - t0 ))
        log "DONE $phase in ${elapsed}s"
        echo "$phase: OK (${elapsed}s)" >> "$LOG_DIR/summary.txt"
    else
        local exit_code=$?
        fail "$phase exited with code $exit_code"
        echo "$phase: FAIL (exit=$exit_code)" >> "$LOG_DIR/summary.txt"
        return 1
    fi
}

# ── Check venv ────────────────────────────────────────────────────────────────
if [[ ! -x "$VENV_PY" ]]; then
    fail "diffkv_venv not found. Run: cd $REPO && make setup"
    exit 1
fi
log "Using Python: $VENV_PY"
"$VENV_PY" -c "import mlx.core; print('MLX OK')" 2>/dev/null || { fail "MLX not installed in venv. Run: make setup"; exit 1; }

# ── Phase selector ────────────────────────────────────────────────────────────
PHASES=("$@")
should_run() {
    local p="$1"
    [[ ${#PHASES[@]} -eq 0 ]] && return 0
    for ph in "${PHASES[@]}"; do [[ "$ph" == "$p" ]] && return 0; done
    return 1
}

# ── Env defaults for all runs ─────────────────────────────────────────────────
export DIFFKV_COMPRESSED_DECODE=1
export DIFFKV_MAX_RESIDUAL=128
export DIFFKV_SPARSE_PREFILL=1
export DIFFKV_DECODE_CACHE=1
export DIFFKV_SPARSE_BIAS=auto
export TOKENIZERS_PARALLELISM=false
export DIFFKV_SEED=1234

# ============================================================================
# PHASE A1 — Main NIAH sweep (active vs dense, 4k–64k)
# Re-verifies paper Table 3 / consistency_report.md Section 6
# Uses run_bench.py with --engines active dense, contexts 4k–64k
# ============================================================================
if should_run "A1"; then
    run_phase "A1" "Main NIAH sweep — active vs dense (4k–64k)" \
        "$VENV_PY" "$BENCH/run_bench.py" \
            --engines active dense \
            --contexts 4096 8192 16384 32768 65536 \
            --gen 128 \
            --timeout 2400 \
            --ram-cap-gb 7.8
fi

# ============================================================================
# PHASE B1 — Multi-needle NIAH (4 needles @ 4k/16k/32k)
# Paper section E7 — re-runs with current runtime
# ============================================================================
if should_run "B1"; then
    run_phase "B1" "Multi-needle NIAH (4 needles, 4k/16k/32k)" \
        "$VENV_PY" "$BENCH/run_multi_needle_mlx.py"
fi

# ============================================================================
# PHASE B2 — Multi-hop NIAH (chain recall @ 4k/16k/32k)
# Paper section E8
# ============================================================================
if should_run "B2"; then
    run_phase "B2" "Multi-hop NIAH (relational chain recall)" \
        "$VENV_PY" "$BENCH/run_multihop_mlx.py"
fi

# ============================================================================
# PHASE B3 — Perplexity (dense vs DiffKV @ 4k/8k/16k)
# Paper section E9
# ============================================================================
if should_run "B3"; then
    run_phase "B3" "Perplexity eval (dense vs DiffKV)" \
        "$VENV_PY" "$BENCH/run_ppl_mlx.py"
fi

# ============================================================================
# PHASE B4 — Llama-3.2-3B cross-arch NIAH
# Paper section E10
# ============================================================================
if should_run "B4"; then
    run_phase "B4" "Llama-3.2-3B cross-arch NIAH (@ 4k/8k/16k, depths 0.1/0.5/0.9)" \
        "$VENV_PY" "$BENCH/run_llama3b_mlx.py"
fi

# ============================================================================
# PHASE B5 — Residual signal ablation
# Paper section E12 (signal ablation table)
# ============================================================================
if should_run "B5"; then
    run_phase "B5" "Residual signal ablation (owner/edge capture @ 8k)" \
        "$VENV_PY" "$BENCH/run_signal_ablation_mlx.py"
fi

# ============================================================================
# PHASE B6 — Lego streaming prefill peak memory
# Paper section E12 (lego table)
# ============================================================================
if should_run "B6"; then
    run_phase "B6" "Lego streaming prefill peak memory (16k/32k/48k)" \
        "$VENV_PY" "$BENCH/run_lego_mem_mlx.py"
fi

# ============================================================================
# PHASE B7 — Real decode latency breakdown at 16k
# Uses mode-difference approach (not hardcoded estimates)
# Measures: full_sparse, no_cache, dense_exact, topk1
# ============================================================================
if should_run "B7"; then
    run_phase "B7" "Real decode latency breakdown at 16k (mode-difference method)" \
        "$VENV_PY" "$BENCH/run_latency_breakdown_mlx.py" --ctx 16000
fi

# ============================================================================
# PHASE C1 — LongBench (NarrativeQA, Qasper, HotpotQA, GovReport)
# NEW: not yet in paper — grant-critical for broader credibility
# Uses run_longbench.py in ACTIVE_RUNTIME with --compare (dense vs DiffKV)
# ============================================================================
if should_run "C1"; then
    LONGBENCH_OUT="$RESULTS/longbench_compare_${TIMESTAMP}.json"
    run_phase "C1" "LongBench (NarrativeQA, Qasper, HotpotQA, GovReport — compare mode)" \
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
# Final summary report
# ============================================================================
log "=========================================="
log "All requested phases complete."
log "Results directory: $LOG_DIR"
log "=========================================="
echo ""
echo "=== Phase Summary ==="
cat "$LOG_DIR/summary.txt" 2>/dev/null || echo "(no summary written)"
echo ""
echo "=== JSON Outputs ==="
ls -lh "$RESULTS/"*.json 2>/dev/null | tail -20
echo ""
echo "=== Quick Numbers ==="
for f in \
    "$RESULTS/test1_multi_needle.json" \
    "$RESULTS/test2_multihop.json" \
    "$RESULTS/test3_perplexity.json" \
    "$RESULTS/test4_llama3b_niah.json" \
    "$RESULTS/test5_signal_ablation.json" \
    "$RESULTS/test6_latency_breakdown.json" \
    "$RESULTS/test7_lego_prefill_mem.json"; do
    if [[ -f "$f" ]]; then
        echo "  $(basename $f): $(wc -c < $f) bytes"
    fi
done
