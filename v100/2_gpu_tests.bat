@echo off
rem V100 fork, step 2 - USES BOTH GPUs: stop your other GPU server first.
rem   Kernel parity tests on each card alone (V100 first), then the speed / quality benchmark of
rem   several engine settings (each loads the model: ~30-40 minutes in all).
rem   Logs: v100_gputest.log, v100_bench.log and v100_bench.json in the Strata folder.
rem   Options:  --parity-only   --bench-only   --variants baseline,kq256
setlocal
cd /d "%~dp0.."
echo This uses both GPUs for ~30-40 minutes. Make sure nothing else is running on them.
pause
".venv\Scripts\python.exe" tools\v100_gputest.py %*
echo.
echo Done. Send v100_gputest.log and v100_bench.log back for review.
pause
