@echo off
rem V100 fork, step 15 - your real work, logged.  USES BOTH GPUs: this IS the server (port 8080, the browser page),
rem   started the same way as run-unsloth-ud-q4_k_xl.bat, with the engine's timing switches on.  Use it as usual.
rem   Everything goes to runs\workload-<date-time>\ (no prompt or answer text is written).
rem   Press Ctrl+C in this window when you are done: a summary and a zip to send back are written then.
rem   Extra server options after --, e.g.  15_workload.bat -- --api-key KEY
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_workload.py %*
pause
