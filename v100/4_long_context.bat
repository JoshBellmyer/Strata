@echo off
rem V100 fork, step 4 - long context.
rem   Part 1 (no GPU): rewrites the config for a 256K context (262,144 tokens, the model's full trained window; setup
rem   turns on KV streaming for it) - the engine is not compiled again.  Log: v100_setup.log.
rem   Part 2 (USES BOTH GPUs, ~45-60 minutes): starts the server on port 8091 and measures prompt reading and writing
rem   speed at 32K / 128K / 240K tokens, then a needle-in-a-haystack recall test at 240K (3 depths).
rem   Logs: v100_longbench.log (+ .json, + .server.txt) and the engine log strata-unsloth-ud-q4_k_xl.log.
rem   Another context: 4_long_context.bat 196608   (the first option is the context in tokens)
setlocal
cd /d "%~dp0.."
set CTX=%1
if "%CTX%"=="" set CTX=262144
".venv\Scripts\python.exe" tools\v100_setup.py --context %CTX%
if errorlevel 1 (
  echo Setup failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo The config is now set up for %CTX% tokens of context.
echo Part 2 uses both GPUs for about 45-60 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_longbench.py
echo.
echo Done. Send v100_longbench.log and strata-unsloth-ud-q4_k_xl.log back for review.
pause
