@echo off
rem V100 fork, step 17 - the fork merged onto upstream Strata 0.1.40: build it and check it is at least as fast.
rem   Part 1 (no GPU, ~20-40 minutes, mostly compiling): saves your config as strata-unsloth-ud-q4_k_xl.json.pre-merge,
rem   then runs setup for the merged code (the CUDA 12 engine now goes in engine-cuda12\, built in build-cuda12\),
rem   re-applies this PC's tuning and builds the GPU test programs.  Log: v100_setup.log.
rem   Part 2 (USES BOTH GPUs, ~35-45 minutes): decode - the new config (upstream's asynchronous adaptive tier on),
rem   the blocking tier, the config again; then prompt reading - a ~120K prompt with every card in 8192-token chunks
rem   and with the V100 in 32768-token chunks.  Logs: v100_decodebench9.log and v100_prefillbench2.log (+ .json).
rem   Before the merge: ~57-60 tok/s short chats, ~53-55 after 60K, ~814 tok/s on the 114K prompt.
setlocal
cd /d "%~dp0.."
if exist strata-unsloth-ud-q4_k_xl.json if not exist strata-unsloth-ud-q4_k_xl.json.pre-merge copy /y strata-unsloth-ud-q4_k_xl.json strata-unsloth-ud-q4_k_xl.json.pre-merge >nul
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo Part 1 is done. Part 2 uses both GPUs for about 35-45 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench9.log --variants baseline,async_off,baseline_end
".venv\Scripts\python.exe" tools\v100_prefillbench.py --log v100_prefillbench2.log --variants same8k,v100_32k
echo.
echo Done. Send v100_setup.log, v100_decodebench9.log and v100_prefillbench2.log back for review.
pause
