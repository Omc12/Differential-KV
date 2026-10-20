@echo off
rem V4 PART B -- RULER on Qwen3.5-4B at 131,072, past every dense cache that
rem stays on the GPU, on the fixed harness. 10 items per task (was 5).
rem   reference:  exact KV offloaded to host memory (what a user can already do)
rem   compressors that keep every token: DKV as shipped, tiled KIVI-4, tiled KIVI-2
rem   eviction at the old 2,048-token budget AND at DKV's own bytes (23,808
rem   tokens = DKV/mid's measured 0.776 GB at this length, within 1%%)
rem   routing ablation: DKV attending every block (5 per task)
rem Resumable. Est. ~30 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\ruler
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set L=--min-length 131072 --max-length 131072
set SL={\"n_sink\": 4, \"recency_window\": 2044}
set H2={\"budget\": 2048, \"recency_window\": 512, \"prefill_chunk\": 256}
set SLQ={\"n_sink\": 4, \"recency_window\": 23804}
set H2Q={\"budget\": 23808, \"recency_window\": 5952, \"prefill_chunk\": 256}

setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %L% --per-task 10 --out %R%\V4_Qwen3.5-4B_dkv_mid_131k.jsonl
endlocal
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi4_tiled %L% --per-task 10 --out %R%\V4_Qwen3.5-4B_kivi4_tiled_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked %L% --per-task 10 --baseline-params "%SL%" --out %R%\V4_Qwen3.5-4B_sl_chunked_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm h2o_chunked %L% --per-task 10 --baseline-params "%H2%" --out %R%\V4_Qwen3.5-4B_h2o_chunked_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm streamingllm_chunked %L% --per-task 10 --baseline-params "%SLQ%" --out %R%\V4_Qwen3.5-4B_sl_chunked_eqbytes_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense %L% --per-task 10 --baseline-params "{\"offload\": true}" --out %R%\V4_Qwen3.5-4B_dense_offload_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi2_tiled %L% --per-task 10 --out %R%\V4_Qwen3.5-4B_kivi2_tiled_131k.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm h2o_chunked %L% --per-task 10 --baseline-params "%H2Q%" --out %R%\V4_Qwen3.5-4B_h2o_chunked_eqbytes_131k.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
set DKV_TOPK_BLOCKS=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid %L% --per-task 5 --out %R%\V4_Qwen3.5-4B_dkv_mid_attendall_131k.jsonl
endlocal

echo V4 PART B COMPLETE
