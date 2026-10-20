@echo off
rem V4 PART F -- reach (ladder) and timing (systems-v2) for the arms V4 adds:
rem   tiled KIVI-2; SnapKV with the allocator fix dense gets (it was measured on
rem   the stock allocator, where it spills on fragmentation); exact KV offloaded
rem   to host memory (granite stops at 65k: 98k of KV is 16 GB of host RAM).
rem Resumable. Est. ~8 h.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results
set GR=ibm-granite/granite-4.2-8b
set QW=Qwen/Qwen3.5-4B
set NOP={}
set SND={\"budget\": 2016, \"window\": 32, \"defrag\": true}
set OFF={\"offload\": true}

"%PY%" benchmarks\context_ladder.py --model %GR% --arms kivi2_tiled --gen 128 --contexts 16384 32768 65536 98304 131072 --timeout 7200 --out %R%\ladder\V4_granite_kivi2_tiled.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms kivi2_tiled --gen 128 --contexts 65536 131072 196608 262144 --timeout 7200 --out %R%\ladder\V4_Qwen3.5-4B_kivi2_tiled.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms snapkv --gen 128 --contexts 16384 20480 24576 --baseline-params "%SND%" --timeout 7200 --out %R%\ladder\V4_granite_snapkv_defrag.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms snapkv --gen 128 --contexts 49152 65536 98304 122880 131072 --baseline-params "%SND%" --timeout 7200 --out %R%\ladder\V4_Qwen3.5-4B_snapkv_defrag.jsonl
"%PY%" benchmarks\context_ladder.py --model %GR% --arms dense --gen 128 --contexts 24576 32768 65536 --baseline-params "%OFF%" --timeout 7200 --out %R%\ladder\V4_granite_dense_offload.jsonl
"%PY%" benchmarks\context_ladder.py --model %QW% --arms dense --gen 128 --contexts 131072 196608 262144 --baseline-params "%OFF%" --timeout 7200 --out %R%\ladder\V4_Qwen3.5-4B_dense_offload.jsonl

set S=paper\results\systems
set G=64
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms kivi2_tiled --gen %G% --contexts 4096 16384 32768 65536 98304 --baseline-params "%NOP%" --out %S%\V4_granite_kivi2_tiled.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms kivi2_tiled --gen %G% --contexts 8192 32768 131072 --baseline-params "%NOP%" --out %S%\V4_Qwen3.5-4B_kivi2_tiled.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %GR% --arms dense --gen %G% --contexts 16384 32768 65536 --baseline-params "%OFF%" --out %S%\V4_granite_dense_offload.jsonl
"%PY%" benchmarks\bench_systems_cuda.py --model %QW% --arms dense --gen %G% --contexts 32768 131072 262144 --baseline-params "%OFF%" --out %S%\V4_Qwen3.5-4B_dense_offload.jsonl

echo V4 PART F COMPLETE
