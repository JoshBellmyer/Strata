@echo off
rem V100 fork, step 3 - starts the Strata server on both GPUs with the config step 1 wrote (http://127.0.0.1:8080).
rem (The same as the start script setup wrote, run-unsloth-ud-q4_k_xl.bat.)  Extra options go to the server.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m serve.server --engine strata --config strata-unsloth-ud-q4_k_xl.json %*
pause
