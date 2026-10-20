@echo off
rem V4 PART A -- RULER on Qwen3.5-4B within dense's limit (4k-32k), on the
rem FIXED harness (RULER's answer prefix attached; before, vt scored ~0 for every
rem arm). Every arm re-run so the whole table is one protocol. DKV runs as
rem shipped: auto mode, N = the measured exact ceiling (131,072), so every
rem length here is exact prefill.
rem Plus, at 32k: tiled KIVI-4 and KIVI-2, DKV attending every block (routing
rem ablation), and DKV forced to stream (the cost of streaming).
rem Resumable. Est. ~25 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set SL={\"n_sink\": 4, \"recency_window\": 2044}
set H2={\"budget\": 2048, \"recency_window\": 512, \"prefill_chunk\": 256}
set SN={\"budget\": 2016, \"window\": 32}

"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense --max-length 32768 --baseline-params "{\"defrag\": true}" --out %R%\V4_Qwen3.5-4B_dense_max32768.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --max-length 32768 --out %R%\V4_Qwen3.5-4B_dkv_mid_max32768.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset high --max-length 32768 --out %R%\V4_Qwen3.5-4B_dkv_high_max32768.jsonl
endlocal
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm snapkv --max-length 32768 --baseline-params "%SN%" --out %R%\V4_Qwen3.5-4B_snapkv_max32768.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked --max-length 32768 --baseline-params "%SL%" --out %R%\V4_Qwen3.5-4B_sl_chunked_max32768.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm h2o_chunked --max-length 32768 --baseline-params "%H2%" --out %R%\V4_Qwen3.5-4B_h2o_chunked_max32768.jsonl

rem -- 32k extras
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_tiled --min-length 32768 --max-length 32768 --out %R%\V4_Qwen3.5-4B_kivi4_tiled_32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi2_tiled --min-length 32768 --max-length 32768 --out %R%\V4_Qwen3.5-4B_kivi2_tiled_32k.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
set DKV_TOPK_BLOCKS=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 32768 --max-length 32768 --out %R%\V4_Qwen3.5-4B_dkv_mid_attendall_32k.jsonl
endlocal
setlocal
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 32768 --max-length 32768 --out %R%\V4_Qwen3.5-4B_dkv_mid_forcedstream_32k.jsonl
endlocal

echo V4 PART A COMPLETE
