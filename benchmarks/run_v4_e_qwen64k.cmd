@echo off
rem V4 PART E -- RULER on Qwen3.5-4B at 65,536 on the fixed harness (10 per
rem task): the longest length the allocator-fixed dense cache serves, so the
rem last point where every compressor has an exact reference.
rem Resumable. Est. ~11 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set L=--min-length 65536 --max-length 65536 --per-task 10
set SL={\"n_sink\": 4, \"recency_window\": 2044}
set H2={\"budget\": 2048, \"recency_window\": 512, \"prefill_chunk\": 256}

"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense %L% --baseline-params "{\"defrag\": true}" --out %R%\V4_Qwen3.5-4B_dense_64k.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %L% --out %R%\V4_Qwen3.5-4B_dkv_mid_64k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset high %L% --out %R%\V4_Qwen3.5-4B_dkv_high_64k.jsonl
endlocal
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_tiled %L% --out %R%\V4_Qwen3.5-4B_kivi4_tiled_64k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked %L% --baseline-params "%SL%" --out %R%\V4_Qwen3.5-4B_sl_chunked_64k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm h2o_chunked %L% --baseline-params "%H2%" --out %R%\V4_Qwen3.5-4B_h2o_chunked_64k.jsonl

echo V4 PART E COMPLETE
