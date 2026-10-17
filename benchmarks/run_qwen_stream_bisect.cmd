@echo off
rem Qwen3.5-4B streaming peak regression (131k auto: 18.58 GB vs 6.71 before the
rem 2026-10-04..11 defaults). Forced streaming at 65k, one new default off per arm.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set DKV_TRITON_STRICT=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set QW=Qwen/Qwen3.5-4B
set O=paper\results\diag\BISECT_qwen_stream65k
set DKV_STREAMING_COMPRESS=1
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 32 --contexts 65536 --timeout 3600 --out %O%_defaults.jsonl
setlocal
set DKV_STREAM_ELASTIC=0
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 32 --contexts 65536 --timeout 3600 --out %O%_noelastic.jsonl
endlocal
setlocal
set DKV_PREFILL_SDPA=0
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 32 --contexts 65536 --timeout 3600 --out %O%_nosdpa.jsonl
endlocal
setlocal
set DKV_EXACT_LEAN=0
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dkv --gen 32 --contexts 65536 --timeout 3600 --out %O%_nolean.jsonl
endlocal
echo BISECT COMPLETE
