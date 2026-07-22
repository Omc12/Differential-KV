#!/bin/bash
set -e

# Generate prompts
../../dkv_venv/bin/python3 make_niah_prompt.py 8000 0.5 "The secret passcode is OMEGA-7741-DELTA." "What is the secret passcode? Repeat it exactly." > temp_prompt_8k.txt
../../dkv_venv/bin/python3 make_niah_prompt.py 16000 0.5 "The secret passcode is OMEGA-7741-DELTA." "What is the secret passcode? Repeat it exactly." > temp_prompt_16k.txt

export DKV_MAX_TOKENS=10

# Profile f16 8k
echo "[*] Profiling F16 8K..."
export DKV_KV_QUANT=f16
../../dkv_venv/bin/python3 ../monitor_memory_native.py --log mem_f16_8k.json --cmd "../build/dkv_native ../qwen2.5-1.5b-instruct-q8_0.gguf $(cat temp_prompt_8k.txt)"

# Profile q8_0 8k
echo "[*] Profiling Q8_0 8K..."
export DKV_KV_QUANT=q8_0
../../dkv_venv/bin/python3 ../monitor_memory_native.py --log mem_q8_8k.json --cmd "../build/dkv_native ../qwen2.5-1.5b-instruct-q8_0.gguf $(cat temp_prompt_8k.txt)"

# Profile f16 16k
echo "[*] Profiling F16 16K..."
export DKV_KV_QUANT=f16
../../dkv_venv/bin/python3 ../monitor_memory_native.py --log mem_f16_16k.json --cmd "../build/dkv_native ../qwen2.5-1.5b-instruct-q8_0.gguf $(cat temp_prompt_16k.txt)"

# Profile q8_0 16k
echo "[*] Profiling Q8_0 16K..."
export DKV_KV_QUANT=q8_0
../../dkv_venv/bin/python3 ../monitor_memory_native.py --log mem_q8_16k.json --cmd "../build/dkv_native ../qwen2.5-1.5b-instruct-q8_0.gguf $(cat temp_prompt_16k.txt)"

# Print summary
echo "=========================================="
echo "Memory Profile Summary (Peak RSS)"
echo "=========================================="
echo "F16 8K:  $(python3 -c "import json; print(json.load(open('mem_f16_8k.json'))['peak_rss_mb'])") MB"
echo "Q8 8K:  $(python3 -c "import json; print(json.load(open('mem_q8_8k.json'))['peak_rss_mb'])") MB"
echo "F16 16K: $(python3 -c "import json; print(json.load(open('mem_f16_16k.json'))['peak_rss_mb'])") MB"
echo "Q8 16K: $(python3 -c "import json; print(json.load(open('mem_q8_16k.json'))['peak_rss_mb'])") MB"
echo "=========================================="

# Cleanup
rm -f temp_prompt_8k.txt temp_prompt_16k.txt
