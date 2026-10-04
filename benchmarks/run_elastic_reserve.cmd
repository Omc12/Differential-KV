@echo off
rem D2 v3 at 65k spilled with 2.0 GB elastic headroom (12.6 GB reserved); try 3.5.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_KEY_QUANT=pc4
set DKV_RESID_ATTN=1
set DKV_STREAMING_COMPRESS=1
set DKV_STREAM_ELASTIC=1
set DKV_STREAM_ELASTIC_RESERVE_GB=3.5
"%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 --out paper\results\diag\TEST_granite_hybrid_elastic3_65k_res3.5.jsonl
echo ELASTIC RESERVE COMPLETE
