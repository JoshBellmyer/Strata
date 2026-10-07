@echo off
rem V100 fork, step 22 - prompt reading, round 2 (after the MMQ kernels).
rem   Part 1 (no GPU, a few minutes): rebuilds the engine with three new switches (log v100_setup.log). The config
rem   is not changed by them.
rem   Part 2 (USES BOTH GPUs, about an hour): ten engine runs, each reading a 16K and a ~120K prompt - the config as it
rem   is (first and again near the end), bigger V100 chunks, a smaller first hand-off to the V100, copy threads that
rem   sleep instead of spinning (also with MMQ on both cards), more copy threads, and part (16 / 28 GiB) or all of
rem   the RAM copy page-locked so the 4070 Super can copy it by DMA. The last one may fail to start (Windows can
rem   refuse that much page-locked memory); that is expected and only ends that run.
rem   Log: v100_prefillbench4.log (+ .json). Close other GPU programs first.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo Part 1 is done. Part 2 uses both GPUs for about an hour. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_prefillbench.py --log v100_prefillbench4.log --variants cfg,v100_big,first1,blocking,blocking_mmq2,stager8,pin16,pin28,cfg,pin_all %*
echo.
echo Done. Send v100_prefillbench4.log (and v100_setup.log if anything failed) back for review.
pause
