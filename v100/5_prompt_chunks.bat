@echo off
rem V100 fork, step 5 - per-card prompt chunks (the V100 reads long prompts in bigger chunks than the 4070 Super).
rem   Part 1 (no GPU): compiles the changed engine files (a few minutes) and keeps the 256K-context config.
rem   Log: v100_setup.log.
rem   Part 2 (USES BOTH GPUs, ~25-35 minutes): three engine runs - 8192-token chunks on both cards (as before), the
rem   V100 at 16384, the V100 at 32768.  Each reads a ~16K and a ~120K-token prompt (with a code word to find) and
rem   answers a follow-up turn.  Logs: v100_prefillbench.log (+ .json) and strata-unsloth-ud-q4_k_xl.log.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo The engine is compiled. Part 2 uses both GPUs for about 25-35 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_prefillbench.py %*
echo.
echo Done. Send v100_prefillbench.log back for review.
pause
