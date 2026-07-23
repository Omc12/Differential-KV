#!/bin/bash

# Test script for performance fix verification
# This will run the optimized code and measure TPS improvement

set -e

echo "═══════════════════════════════════════════════════════════════════"
echo "  DKV Native - Performance Fix Verification"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if binary exists
if [ ! -f "dkv_native/build/dkv_native" ]; then
    echo -e "${RED}❌ Binary not found. Building...${NC}"
    cd dkv_native
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j$(sysctl -n hw.ncpu)
    cd ..
    echo -e "${GREEN}✅ Build complete${NC}"
fi

# Check for model
MODEL=""
if [ -f "dkv_native/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf"
elif [ -f "dkv_native/qwen2.5-1.5b-instruct-q8_0.gguf" ]; then
    MODEL="qwen2.5-1.5b-instruct-q8_0.gguf"
elif [ -f "dkv_native/qwen2.5-0.5b-instruct.gguf" ]; then
    MODEL="qwen2.5-0.5b-instruct.gguf"
else
    echo -e "${RED}❌ No model found${NC}"
    echo "Please place a GGUF model in dkv_native/"
    exit 1
fi

echo -e "${GREEN}✓ Using model: $MODEL${NC}"
echo ""

# Test configuration
export DKV_MAX_CTX_TK=8192  # Smaller for quick test
export DKV_PRESET=mid
export DKV_PREFILL_CHUNK_SIZE=512
export DKV_MICRO_BLOCK_SIZE=256
export DKV_RANK=16
export DKV_GPU_BUDGET_GB=1.5
export DKV_MPS_APPROXIMATE_ATTN=1
export DKV_MAX_TOKENS=128
export DKV_VERBOSE=1  # Enable verbose logging

echo "Configuration:"
echo "  Context: 8192 tokens"
echo "  Preset: mid"
echo "  Chunk size: 512"
echo "  Micro block: 256"
echo "  Max generation: 128 tokens"
echo ""

# Create test prompt
if [ -f "dkv_native/scratch_longprompt.txt" ]; then
    TEST_PROMPT=$(cat dkv_native/scratch_longprompt.txt | tr '\n' ' ')
    echo "✓ Loaded long prompt from scratch_longprompt.txt ($(wc -c < dkv_native/scratch_longprompt.txt) bytes)"
else
    TEST_PROMPT="Write a detailed summary of the major events in World War II, including the key battles, turning points, and outcomes."
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  Test 1: With Native Attention (OPTIMIZED)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
export DKV_NATIVE_ATTN=1

echo "Test prompt: $TEST_PROMPT"
echo ""
echo "Running... (this may take a moment for first-time initialization)"
echo ""

cd dkv_native
echo "$TEST_PROMPT" | ../dkv_venv/bin/python3 serving/cli.py \
    --model "$MODEL" \
    --binary-path build/dkv_native \
    --preset mid \
    --max-tokens 128 \
    2>&1 | tee ../test_native_on.log || true
cd ..

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Test 2: Without Native Attention (for comparison)"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
export DKV_NATIVE_ATTN=0

cd dkv_native
echo "$TEST_PROMPT" | ../dkv_venv/bin/python3 serving/cli.py \
    --model "$MODEL" \
    --binary-path build/dkv_native \
    --preset mid \
    --max-tokens 128 \
    2>&1 | tee ../test_native_off.log || true
cd ..

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Results Analysis"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Extract TPS from logs
echo "Extracting performance metrics..."
echo ""

TPS_ON=$(grep -oE "[0-9]+\.[0-9]+ tok/s" test_native_on.log 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
TPS_OFF=$(grep -oE "[0-9]+\.[0-9]+ tok/s" test_native_off.log 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")

TTFT_ON=$(grep -oE "TTFT: [0-9]+\.[0-9]+ms" test_native_on.log 2>/dev/null | head -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
TTFT_OFF=$(grep -oE "TTFT: [0-9]+\.[0-9]+ms" test_native_off.log 2>/dev/null | head -1 | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")

echo "┌─────────────────────────────────────────┬──────────┬───────────┐"
echo "│ Metric                                  │ Native=1 │ Native=0  │"
echo "├─────────────────────────────────────────┼──────────┼───────────┤"
echo "│ Tokens Per Second (TPS)                 │ $TPS_ON    │ $TPS_OFF     │"
echo "│ Time To First Token (TTFT ms)           │ $TTFT_ON   │ $TTFT_OFF    │"
echo "└─────────────────────────────────────────┴──────────┴───────────┘"
echo ""

# Check for persistent buffer initialization message
if grep -q "Persistent buffers initialized" test_native_on.log; then
    echo -e "${GREEN}✅ Persistent buffer optimization is ACTIVE${NC}"
    echo "   (Found initialization message in logs)"
else
    echo -e "${YELLOW}⚠️  Could not confirm persistent buffer optimization${NC}"
    echo "   (Check test_native_on.log for details)"
fi

echo ""

# Performance expectation check
if [ "$TPS_ON" != "N/A" ]; then
    TPS_INT=$(echo "$TPS_ON" | cut -d. -f1)
    if [ "$TPS_INT" -ge 30 ]; then
        echo -e "${GREEN}✅ EXCELLENT: Native attention achieving $TPS_ON TPS (target: 40+ TPS)${NC}"
        echo "   Performance is in the expected range!"
    elif [ "$TPS_INT" -ge 10 ]; then
        echo -e "${YELLOW}⚠️  GOOD: Native attention achieving $TPS_ON TPS${NC}"
        echo "   Better than before, but below target of 40+ TPS"
        echo "   This may be due to small test size. Try with longer context."
    elif [ "$TPS_INT" -ge 2 ]; then
        echo -e "${YELLOW}⚠️  MODERATE: Native attention achieving $TPS_ON TPS${NC}"
        echo "   Improvement from 0.3-0.4 TPS, but not meeting full potential"
        echo "   Check if persistent buffers were actually used (see logs)"
    else
        echo -e "${RED}❌ SLOW: Native attention only achieving $TPS_ON TPS${NC}"
        echo "   Expected 40+ TPS. Something may be wrong."
        echo "   Check test_native_on.log for errors"
    fi
fi

echo ""
echo "Full logs saved to:"
echo "  - test_native_on.log  (with optimization)"
echo "  - test_native_off.log (without optimization)"
echo ""

# Check for errors
if grep -qi "error\|failed\|warning" test_native_on.log; then
    echo -e "${YELLOW}⚠️  Warnings/errors detected in test_native_on.log${NC}"
    echo "Relevant messages:"
    grep -i "error\|failed\|warning" test_native_on.log | head -10
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo "  Test Complete"
echo "═══════════════════════════════════════════════════════════════════"
