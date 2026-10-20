@echo off
rem V4 SMOKE -- every new arm/switch on 2 items at 4k (attend-all at 32k, where
rem routing actually selects), before any long run depends on it. Writes to
rem paper\results\smoke\ (not a paper store).
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set S=paper\results\smoke
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set DKV_TRITON_STRICT=1
set RULER_ANSWER_PREFIX=1
set T=--max-length 4096 --per-task 1 --tasks niah_single_1 vt

"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm kivi2_tiled %T% --out %S%\V4Z_qwen_kivi2_tiled.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm kivi2_tiled %T% --out %S%\V4Z_granite_kivi2_tiled.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dense %T% --baseline-params "{\"offload\": true}" --out %S%\V4Z_qwen_offload.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %GR% --arm dense %T% --baseline-params "{\"offload\": true}" --out %S%\V4Z_granite_offload.jsonl
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm snapkv %T% --baseline-params "{\"budget\": 2016, \"window\": 32, \"defrag\": true}" --out %S%\V4Z_qwen_snapkv_defrag.jsonl
setlocal
set DKV_STREAMING_COMPRESS=auto
set DKV_STREAM_AUTO_TOKENS=131072
set DKV_TOPK_BLOCKS=0
"%PY%" benchmarks\run_ruler_cuda.py --model %QW% --arm dkv --preset mid --min-length 32768 --max-length 32768 --per-task 1 --tasks niah_multikey_2 vt --out %S%\V4Z_qwen_dkv_attendall.jsonl
endlocal
echo V4 SMOKE COMPLETE
