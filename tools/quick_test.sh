#!/bin/bash

# Quick performance verification - just run one test

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Quick Performance Test - Optimized Native Attention             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
export DKV_NATIVE_ATTN=1
export DKV_VERBOSE=1
export DKV_MAX_CTX_TK=8192
export DKV_PRESET=mid
export DKV_MAX_TOKENS=50

# Find model
MODEL=""
if [ -f "dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf"
elif [ -f "dkv_native/qwen2.5-1.5b-instruct-q8_0.gguf" ]; then
    MODEL="qwen2.5-1.5b-instruct-q8_0.gguf"
elif [ -f "dkv_native/qwen2.5-0.5b-instruct.gguf" ]; then
    MODEL="qwen2.5-0.5b-instruct.gguf"
else
    echo "❌ No model found. Place a GGUF model in dkv_native/"
    exit 1
fi

echo "Model: $MODEL"
echo "Config: Native Attn=ON, Max Tokens=50"
echo ""
echo "Paste your prompt and press Ctrl+D (or type a prompt):"
echo "───────────────────────────────────────────────────────────────────"

cd dkv_native
../dkv_venv/bin/python3 serving/cli.py \
    --model "$MODEL" \
    --binary-path build/dkv_native \
    --preset mid \
    --max-tokens 50 \
    2>&1 | tee -a ../quick_test.log

echo ""
echo "───────────────────────────────────────────────────────────────────"
echo ""

# Extract TPS
TPS=$(grep -oE "[0-9]+\.[0-9]+ tok/s" ../quick_test.log | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")

echo "Result: $TPS tokens/second"
echo ""

if [ "$TPS" != "N/A" ]; then
    TPS_INT=$(echo "$TPS" | cut -d. -f1)
    if [ "$TPS_INT" -ge 30 ]; then
        echo "✅ EXCELLENT! Performance is in target range (40+ TPS)"
    elif [ "$TPS_INT" -ge 10 ]; then
        echo "✓ GOOD! Significant improvement, but below target"
    elif [ "$TPS_INT" -ge 2 ]; then
        echo "⚠ MODERATE - Better than before (0.3-0.4), needs investigation"
    else
        echo "❌ SLOW - Check logs for issues"
    fi
fi

# Check for optimization message
if grep -q "Persistent buffers initialized" ../quick_test.log; then
    echo "✓ Persistent buffer optimization is active"
else
    echo "⚠ Could not confirm optimization (check quick_test.log)"
fi

echo ""
echo "Full log: quick_test.log"
