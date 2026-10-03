@echo off
rem V100 fork, step 16 - the expert caches: your recorded routing replayed, then the adaptive swaps with a budget per card.
rem   Part 1 (no GPU): compiles the changed engine file (a few minutes; log v100_setup.log), then replays the routing
rem   of your workload runs (runs\workload-*\routing.bin) through a model of the expert caches (5-10 minutes, CPU
rem   only).  Log: v100_cachesim.log.
rem   Part 2 (USES BOTH GPUs, ~30-35 minutes, nothing compiled): six engine runs - the config, the swap budget per card
rem   at 64/32, 96/32, 160/32 and 96/16 (4070 Super/V100), the config again.  Log: v100_decodebench8.log (+ .json).
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\v100_cachesim.py
echo.
echo Part 1 is done. Part 2 uses both GPUs for about 30-35 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench8.log --variants baseline,stage64_32,stage96_32,stage160_32,stage96_16,baseline_end %*
echo.
echo Done. Send v100_cachesim.log and v100_decodebench8.log back for review.
pause
