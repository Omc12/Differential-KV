@echo off
rem Start or RESUME the chunk-wise eviction campaign. After a power cut, just
rem double-click this again: finished steps are skipped and the step that was
rem running picks up from its last saved item.
rem   run_chunked_campaign.cmd --status    shows progress without running
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0.."
"C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe" benchmarks\campaign_chunked_eviction.py %*
pause
