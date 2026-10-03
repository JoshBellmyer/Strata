@echo off
rem V100 fork, step 8 - decode tuning, round 3.
rem   Part 1 (no GPU): compiles one changed engine file (the PLE read statistics in the log). Log: v100_setup.log.
rem   Part 2 (USES BOTH GPUs, ~45-50 minutes): ten engine runs - the config (now with --adapt-swaps 32) first and last,
rem   adaptive rounds every 2 windows, shorter drafts, both, three other split points between the cards, the RAM copy
rem   page-locked, and a bigger PLE row cache.
rem   Logs: v100_decodebench3.log (+ .json).
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo The engine is compiled. Part 2 uses both GPUs for about 45-50 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench3.log %*
echo.
echo Done. Send v100_decodebench3.log back for review.
pause
