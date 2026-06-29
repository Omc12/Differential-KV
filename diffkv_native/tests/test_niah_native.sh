#!/bin/bash
set -e

# Run NIAH test for C++ native runtime
# We test 4k, 8k, 16k, 32k contexts.

BINARY="../build/diffkv_native"
MODEL="../qwen2.5-1.5b-instruct-q4_k_m.gguf"
CONTEXTS=(4000 8000 16000 32000)
DEPTHS=(0.1 0.5 0.9)
NEEDLE="The secret passcode is OMEGA-7741-DELTA."
QUESTION="What is the secret passcode? Repeat it exactly."

export DIFFKV_ENGAGE_THRESHOLD=1024  # ensure sparse at 4k+
export DIFFKV_NATIVE_ATTN=1          # enable native attention engine
export DIFFKV_CPU_EXACT_ATTN=1       # use CPU path: per-token RoPE + SVD residual correction
export DIFFKV_NO_FUSED_OP=1          # disable fused Metal op (required for CPU_EXACT_ATTN branch)
export DIFFKV_TEMPERATURE=0
export DIFFKV_DISABLE_VSL=1          # disable VSL anti-hallucination mask for NIAH
export DIFFKV_ENABLE_FACTUAL=0        # disable factual store building for NIAH

# Activate virtualenv for the make_niah_prompt.py script
source ../../diffkv_venv/bin/activate

pass=0; fail=0
for ctx in "${CONTEXTS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        echo "============================================================"
        echo "Testing Native NIAH: ctx=$ctx depth=$depth"
        echo "============================================================"
        prompt=$(python3 make_niah_prompt.py "$ctx" "$depth" "$NEEDLE" "$QUESTION")
        # Run C++ binary
        output=$("$BINARY" "$MODEL" "$prompt" 2>/dev/null)
        
        # Check output
        if echo "$output" | grep -qi "OMEGA-7741-DELTA"; then
            echo "Result: PASS ✓"
            ((pass++))
        else
            echo "Result: FAIL ✗"
            echo "Output was: $output"
            ((fail++))
        fi
        echo ""
    done
done

echo "============================================================"
echo "Native NIAH Results: $pass PASS / $fail FAIL"
echo "============================================================"
[ "$fail" -eq 0 ] && exit 0 || exit 1
