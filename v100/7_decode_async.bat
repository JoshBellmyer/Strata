@echo off
rem V100 fork, step 7 - the adaptive expert swaps without decode stalls.
rem   Part 1 (no GPU): compiles the changed engine files (a few minutes). Log: v100_setup.log.
rem   Part 2 (USES BOTH GPUs, ~35-40 minutes, nothing else compiled): seven engine runs - the swaps as before, the new
rem   ones, no swaps, fewer swaps, more frequent swaps, shorter drafts, and the V100 alone.  Each writes 256 tokens for
rem   three short chats (twice) and 300 tokens after a ~60K-token document.
rem   Logs: v100_decodebench.log (+ .json); the engine's timing lines are copied into it.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo The engine is compiled. Part 2 uses both GPUs for about 35-40 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench2.log %*
echo.
echo Done. Send v100_decodebench2.log back for review.
pause
