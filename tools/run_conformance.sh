#!/usr/bin/env bash
# run_conformance.sh — one-command decode conformance guardrail.
#
# WHY THIS EXISTS: the golden vectors (tools/conformance_vectors.bin) are a
# gitignored, generated artifact whose shape depends on the decode config (rank,
# MAX_RESIDUAL, head dims). If that config changes in gen_decode_vectors.py /
# conformance_test.cpp but the vectors are NOT regenerated, the native conformance
# test fails on STALE vectors — a silent guardrail break (this is what happened when
# the rank default moved 16→32 in 8058506; discrepancy jumped to 0.40). This script
# regenerates the vectors from the CURRENT config, then runs the native CPU
# conformance check, so the two can never drift. Run it after ANY change to the
# sparse-decode math or config, and treat a non-zero exit as a red guardrail.
#
# Usage: tools/run_conformance.sh   (from anywhere; resolves the repo root itself)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python}"
if [ -x "$REPO_ROOT/dkv_venv/bin/python" ]; then
    PY="$REPO_ROOT/dkv_venv/bin/python"
fi

BIN="$REPO_ROOT/dkv_native/build/conformance_test"
if [ ! -x "$BIN" ]; then
    echo "[run_conformance] conformance_test not built — building (-j4)..."
    cmake --build "$REPO_ROOT/dkv_native/build" --target conformance_test -j4
fi

echo "[run_conformance] 1/2 regenerating golden vectors from current config..."
"$PY" tools/gen_decode_vectors.py

echo "[run_conformance] 2/2 running native CPU conformance..."
echo "======================================================================"
"$BIN"
status=$?
echo "======================================================================"
if [ $status -eq 0 ]; then
    echo "[run_conformance] RESULT: PASS"
else
    echo "[run_conformance] RESULT: FAIL (exit $status) — native decode diverges from the golden reference"
fi
exit $status
