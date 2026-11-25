@echo off
rem V13 cross-architecture fidelity panel -- does the method, unchanged and with
rem nothing set per model, keep each model's next-token distribution at 32k?
rem First decode step KL / top-1 vs plain HF (dense_true) and DKV (baseline),
rem nf4 on both sides, 5 depths, shipped mid preset. Models chosen for distinct
rem attention layouts (all already cached; no downloads):
rem   SmolLM3-3B        Llama-style GQA, NoPE on every 4th layer, hd 128
rem   Phi-4-mini        fused qkv_proj, partial rotary 0.75, hd 128
rem   Qwen2.5-7B        dense GQA with qkv bias, hd 128, 32k native
rem   Qwen3.5-2B        hybrid: linear attention + 1-in-4 full, hd 256
rem   gemma-4-e2b/e4b   sliding + global, shared KV layers, composite config
rem   gemma-4-12B       unified composite, sliding + global
rem   Ministral-3-8B    composite mistral3 wrapper, dense GQA, hd 128
rem   Qwen3.5-4B        re-anchor (V12 block-size run: baseline KL 0.00180)
rem   granite-4.2-8b    dense GQA, 40 full layers, flat config
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set R=paper\results\fidelity
set DKV_TRITON_STRICT=1
rem Models as arguments run only those (resume: v13b.log reran SmolLM3 after the
rem NoPE fix, then the six the paused run had not reached).
set MODELS=%*
if "%MODELS%"=="" set MODELS=HuggingFaceTB/SmolLM3-3B microsoft/Phi-4-mini-reasoning Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3.5-2B google/gemma-4-e2b-it google/gemma-4-e4b-it google/gemma-4-12B-it mistralai/Ministral-3-8B-Instruct-2512-BF16 Qwen/Qwen3.5-4B ibm-granite/granite-4.2-8b

for %%M in (%MODELS%) do (
  echo === V13 PANEL %%M ===
  "%PY%" colab\logit_fidelity.py --model %%M --quant nf4 --ctx 32768 --arms dense_true baseline
  echo === V13 PANEL %%M DONE ===
)
echo V13 PANEL COMPLETE
