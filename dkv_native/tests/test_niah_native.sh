#!/bin/bash
set -e

# Run NIAH test for C++ native runtime.
# Sweep 4k/8k/16k × depth 0.5/0.9. The filler intentionally contains a digit
# token ("2010s") — it competes with the needle digits for residual slots,
# which is the real accuracy frontier. Do NOT sanitize the filler to make
# this pass; a single-cell or digit-free run hides regressions.
# (32k is excluded: native OOMs there on the 8GB dev machine.)

BINARY="${DKV_TEST_BINARY:-../build/dkv_native}"
# Overridable so this runs on a box that does not carry this exact file.
# It hardcoded one .gguf path, so the suite was simply unrunnable anywhere
# that file was missing -- which is how a Mac session ended up driving the
# binary by hand instead of running the certified script.
MODEL="${DKV_TEST_MODEL:-../qwen2.5-1.5b-instruct-q8_0.gguf}"
if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    echo "set DKV_TEST_MODEL=/path/to/model.gguf" >&2
    exit 2
fi
CONTEXTS=(4000 8000 16000)
DEPTHS=(0.5 0.9)
NEEDLE="The secret passcode is OMEGA-7741-DELTA."
QUESTION="What is the secret passcode? Repeat it exactly."

export DKV_ENGAGE_THRESHOLD=1024  # ensure sparse at 4k+
export DKV_NATIVE_ATTN=0          # enable native attention engine (default path)
# export DKV_CPU_EXACT_ATTN=0       # use fused op path
# export DKV_NO_FUSED_OP=0          # enable fused Metal op
export DKV_FORCE_CPU_ATTN=0       # run on GPU
export DKV_DBG_GRAPH=1            # enable graph counter logs
export DKV_DBG_INPUTS=1           # enable input debug logs
export DKV_MPS_APPROXIMATE_ATTN=1 # use approximate attention scoring
export DKV_DENSE_DIRECT=1         # force exact per-token trig function
export DKV_DBG_CMP=1              # enable comparison debug prints
export DKV_DBG_CMP_CUR=1          # include current token in CPU comparison reference
export DKV_POOL_ABS_ROT=1
export DKV_TEMPERATURE=0
export DKV_DISABLE_VSL=1          # disable VSL anti-hallucination mask for NIAH
export DKV_ENABLE_FACTUAL=0        # disable factual store building for NIAH
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DKV_MAX_TOKENS=40

# Activate virtualenv for the make_niah_prompt.py script
source ../../dkv_venv/bin/activate

pass=0; fail=0
for ctx in "${CONTEXTS[@]}"; do
    for depth in "${DEPTHS[@]}"; do
        echo "============================================================"
        echo "Testing Native NIAH: ctx=$ctx depth=$depth"
        echo "============================================================"
        prompt=$(python3 make_niah_prompt.py "$ctx" "$depth" "$NEEDLE" "$QUESTION")
        # Run C++ binary
        output=$("$BINARY" "$MODEL" "$prompt" 2>native_test_stderr.log) || true
        
        # Check output
        if echo "$output" | grep -qi "OMEGA-7741-DELTA"; then
            echo "Result: PASS ✓"
            pass=$((pass+1))
        else
            echo "Result: FAIL ✗"
            echo "Output was: $output"
            fail=$((fail+1))
        fi
        echo ""
    done
done

echo "============================================================"
echo "Native NIAH Results: $pass PASS / $fail FAIL"
echo "============================================================"
[ "$fail" -eq 0 ] && exit 0 || exit 1
