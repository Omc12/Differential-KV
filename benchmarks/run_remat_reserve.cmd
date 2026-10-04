@echo off
rem Hybrid streaming spills at 65k on granite (12.17 GB reserved; shipped DKV 9.36):
rem the memory-sized remat cache fills headroom up to its gate, and the gate's fixed
rem 1.5 GB step headroom is smaller than the 65k decode transient. Try 2.5 / 3.5 GB.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_KEY_QUANT=pc4
set DKV_RESID_ATTN=1
set DKV_STREAMING_COMPRESS=1
for %%R in (2.5 3.5) do (
  set DKV_REMAT_RESERVE_GB=%%R
  "%PY%" benchmarks\context_ladder.py --model ibm-granite/granite-4.2-8b --arms dkv --gen 128 --contexts 65536 --out paper\results\diag\TEST_granite_hybrid_stream_65k_res%%R.jsonl
)
echo RESERVE COMPLETE
