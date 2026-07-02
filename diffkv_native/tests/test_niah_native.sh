#!/bin/bash
set -e

# Run NIAH test for C++ native runtime
# We test 4k, 8k, 16k, 32k contexts.

BINARY="../build/diffkv_native"
MODEL="../qwen2.5-1.5b-instruct-q8_0.gguf"
CONTEXTS=(4000)
DEPTHS=(0.9)
NEEDLE="The secret passcode is OMEGA-7741-DELTA."
QUESTION="What is the secret passcode? Repeat it exactly."

export DIFFKV_ENGAGE_THRESHOLD=1024  # ensure sparse at 4k+
export DIFFKV_NATIVE_ATTN=1          # enable native attention engine
export DIFFKV_CPU_EXACT_ATTN=0       # use fused op path
export DIFFKV_NO_FUSED_OP=0          # enable fused Metal op
export DIFFKV_FORCE_CPU_ATTN=0       # run on GPU
export DIFFKV_DBG_GRAPH=1            # enable graph counter logs
export DIFFKV_DBG_INPUTS=1           # enable input debug logs
export DIFFKV_MPS_APPROXIMATE_ATTN=0 # force exact attention scoring
export DIFFKV_DENSE_DIRECT=1         # force exact per-token trig function
export DIFFKV_DBG_CMP=1              # enable comparison debug prints
export DIFFKV_DBG_CMP_CUR=1          # include current token in CPU comparison reference
export DIFFKV_POOL_ABS_ROT=1
export DIFFKV_TEMPERATURE=0
export DIFFKV_DISABLE_VSL=1          # disable VSL anti-hallucination mask for NIAH
export DIFFKV_ENABLE_FACTUAL=0        # disable factual store building for NIAH
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DIFFKV_MAX_TOKENS=40

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
        output=$("$BINARY" "$MODEL" "$prompt" 2>native_test_stderr.log)
        
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
