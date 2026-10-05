@echo off
rem V3 PART C -- the hybrid store as a DKV mode (DKV_STORE=hybrid: per-channel
rem 4-bit keys, low-rank values, residuals ranked by error x attention), in
rem exact AND streaming mode, mirroring the low-rank DKV rows item for item.
rem Nothing of it exists yet. Resumable. Est. ~14-16 h (two nights): run C1
rem first, C2 the night after if needed -- each half stands alone.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set DKV_STORE=hybrid
set DKV_PREFILL_SDPA=1
set DKV_STREAM_ELASTIC=1

if /i "%1"=="c2" goto c2

rem ===== C1. hybrid, EXACT mode (compare: granite-4.2-8b_mid_nf4, *_dkv_mid_len12000,
rem      Qwen3.5-4B_dkv_mid_max32768 / max65536) =====
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 4096 8192 16384 20480 24576 --timeout 7200 --out %R%\ladder\V3_granite_hybrid_exact.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 8192 16384 32768 65536 98304 131072 --timeout 7200 --out %R%\ladder\V3_Qwen3.5-4B_hybrid_exact.jsonl
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm dkv --preset mid --max-length 12000 --num-samples 20 --out %R%\longbench\V3_granite-4.2-8b_hybrid_mid_len12000.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --max-length 32768 --out %R%\ruler\V3_Qwen3.5-4B_hybrid_mid_max32768.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 65536 --max-length 65536 --out %R%\ruler\V3_Qwen3.5-4B_hybrid_mid_max65536.jsonl
echo V3 PART C1 COMPLETE
if /i "%1"=="c1" goto end

:c2
rem ===== C2. hybrid, STREAMING mode (compare: part A's V3_* rows) =====
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dkv --gen 128 --contexts 24576 32768 49152 65536 98304 131072 --timeout 7200 --out %R%\ladder\V3_granite_hybrid_auto.jsonl
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 128 --contexts 131072 163840 196608 229376 262144 --timeout 7200 --out %R%\ladder\V3_Qwen3.5-4B_hybrid_auto.jsonl
endlocal
setlocal
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\run_longbench_cuda.py --model %GR% --arm dkv --preset mid --max-length 12000 --num-samples 20 --out %R%\longbench\V3_granite-4.2-8b_hybrid_mid_stream_len12000.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 32768 --max-length 32768 --per-task 10 --out %R%\ruler\V3_Qwen3.5-4B_hybrid_mid_forcedstream_32k.jsonl
endlocal
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=16384
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid --min-length 24576 --max-length 32768 --per-task 10 --out %R%\ruler\V3_BEYOND_granite_hybrid_mid_auto_24k-32k.jsonl
set DKV_STREAM_AUTO_TOKENS=98304
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 131072 --max-length 131072 --per-task 5 --out %R%\ruler\V3_BEYOND_Qwen3.5-4B_hybrid_mid_auto_131k.jsonl
endlocal
echo V3 PART C2 COMPLETE

:end
echo V3 PART C COMPLETE
