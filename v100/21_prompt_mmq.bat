@echo off
rem V100 fork, step 21 - faster prompt reading: llama.cpp's MMQ kernels for the experts (instead of dequantizing them to
rem   FP16 for cuBLAS, which was ~75% of the 4070 Super's prompt GPU time).
rem   Part 1 (no GPU, ~10-20 minutes): rebuilds the engine with the MMQ kernels and the test programs (log
rem   v100_setup.log). The config keeps them OFF (STRATA_MMQ_KQUANTS=0 in its env) until Part 2 has checked them.
rem   Part 2 (USES BOTH GPUs, ~30 minutes): the kernel tests on each card (v100_gputest3.log), then four engine runs,
rem   each reading a 16K and a ~120K prompt: MMQ off (as now), MMQ on both cards, MMQ on the 4070 Super only, and MMQ
rem   with the V100 reading bigger chunks (v100_prefillbench3.log + .json). Close other GPU programs first.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo Part 1 is done. Part 2 uses both GPUs for about 30 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_gputest.py --parity-only --log v100_gputest3.log
".venv\Scripts\python.exe" tools\v100_prefillbench.py --log v100_prefillbench3.log --variants mmq_off,mmq,mmq_cuda0,mmq_v100_37k %*
echo.
echo Done. Send v100_gputest3.log and v100_prefillbench3.log (and v100_setup.log if anything failed) back for review.
pause
