@echo off
rem V13e -- Phi-4-mini: is the compressed-decode gap capacity or a defect?
rem V13d: decode off 0.047, baseline 0.679, all blocks 1.049 (reconstruction, not
rem routing). resid200 = more exact residual rows; rank128 = higher stored rank.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_OFFLINE=1
cd /d "%~dp0..\.."
set PY=C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe
set DKV_TRITON_STRICT=1
"%PY%" colab\logit_fidelity.py --model microsoft/Phi-4-mini-reasoning --quant nf4 --ctx 32768 --arms dense_true baseline resid200 rank128
echo V13E COMPLETE
