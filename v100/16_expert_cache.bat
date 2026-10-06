@echo off
rem V100 fork, step 16 - the expert caches: your recorded routing replayed, then the adaptive swaps with a budget per card.
rem   Part 1 (no GPU): compiles the changed engine file (a few minutes; log v100_setup.log), then replays the routing
rem   of your workload runs (runs\workload-*\routing.bin) through a model of the expert caches (5-10 minutes, CPU
rem   only).  Log: v100_cachesim.log.
rem   (Its GPU part benched per-card swap budgets, --adapt-stage-swaps: no speed gain, dropped in the 0.1.40 merge.)
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
echo Done. Send v100_cachesim.log back for review.
pause
