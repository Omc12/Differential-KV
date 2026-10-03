@echo off
rem Waits for run_now.cmd to finish (it writes NOW COMPLETE to now.log), then
rem runs the overnight campaign. Detached from any shell, so it survives the
rem session that started it.
cd /d "%~dp0.."
:wait
findstr /c:"NOW COMPLETE" paper\results\campaign\now.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto wait
)
call benchmarks\run_overnight.cmd > paper\results\campaign\overnight.log 2> paper\results\campaign\overnight.log.err
