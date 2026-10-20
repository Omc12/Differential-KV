@echo off
rem V4 PART C -- RULER on granite-4.2-8b on the fixed harness: dense within its
rem limit (4k-16k, 20 per task), then 24k and 32k (10 per task) past every dense
rem cache that stays on the GPU. Same arms as part B. Equal-byte eviction budgets
rem match DKV/mid's measured bytes per length (0.916 GB at 24k -> 5,632 tokens;
rem 1.138 GB at 32k -> 6,912 tokens; within 1%%).
rem Resumable. Est. ~27 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set GR=ibm-granite/granite-4.2-8b
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set B=--min-length 24576 --max-length 32768 --per-task 10
set SL={\"n_sink\": 4, \"recency_window\": 2044}
set H2={\"budget\": 2048, \"recency_window\": 512, \"prefill_chunk\": 256}

"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dense --max-length 16384 --baseline-params "{\"defrag\": true}" --out %R%\V4_granite-4.2-8b_dense_max16384.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=24576
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dkv --preset mid %B% --out %R%\V4_granite-4.2-8b_dkv_mid_24k-32k.jsonl
endlocal
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi4_tiled %B% --out %R%\V4_granite-4.2-8b_kivi4_tiled_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm streamingllm_chunked %B% --baseline-params "%SL%" --out %R%\V4_granite-4.2-8b_sl_chunked_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm h2o_chunked %B% --baseline-params "%H2%" --out %R%\V4_granite-4.2-8b_h2o_chunked_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dense %B% --baseline-params "{\"offload\": true}" --out %R%\V4_granite-4.2-8b_dense_offload_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi2_tiled %B% --out %R%\V4_granite-4.2-8b_kivi2_tiled_24k-32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm streamingllm_chunked --min-length 24576 --max-length 24576 --per-task 10 --baseline-params "{\"n_sink\": 4, \"recency_window\": 5628}" --out %R%\V4_granite-4.2-8b_sl_chunked_eqbytes_24k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm streamingllm_chunked --min-length 32768 --max-length 32768 --per-task 10 --baseline-params "{\"n_sink\": 4, \"recency_window\": 6908}" --out %R%\V4_granite-4.2-8b_sl_chunked_eqbytes_32k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm h2o_chunked --min-length 24576 --max-length 24576 --per-task 10 --baseline-params "{\"budget\": 5632, \"recency_window\": 1408, \"prefill_chunk\": 256}" --out %R%\V4_granite-4.2-8b_h2o_chunked_eqbytes_24k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm h2o_chunked --min-length 32768 --max-length 32768 --per-task 10 --baseline-params "{\"budget\": 6912, \"recency_window\": 1728, \"prefill_chunk\": 256}" --out %R%\V4_granite-4.2-8b_h2o_chunked_eqbytes_32k.jsonl

echo V4 PART C COMPLETE
