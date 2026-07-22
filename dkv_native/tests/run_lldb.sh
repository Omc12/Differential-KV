#!/bin/bash
source ../../dkv_venv/bin/activate
prompt=$(python3 make_niah_prompt.py 4000 0.5 "The secret passcode is OMEGA-7741-DELTA." "What is the secret passcode? Repeat it exactly.")
export DKV_ENGAGE_THRESHOLD=1024
export DKV_NATIVE_ATTN=1
export DKV_TEMPERATURE=0
export DKV_DISABLE_VSL=1
export DKV_ENABLE_FACTUAL=0

# Create lldb batch file to run and print backtrace on crash
echo "run" > lldb_cmds.txt
echo "bt" >> lldb_cmds.txt
echo "quit" >> lldb_cmds.txt

echo "$prompt" | lldb -batch -s lldb_cmds.txt -- ../build/dkv_native ../qwen2.5-1.5b-instruct-q4_k_m.gguf -
rm lldb_cmds.txt
