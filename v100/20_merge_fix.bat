@echo off
rem V100 fork, step 20 - the merged engine with the host thread hard-pinned again (as before the merge) and no PCIe
rem   share for the V100's x4 link.  Part 1 (no GPU, a few minutes): rebuilds the engine (log v100_setup.log).
rem   Part 2 (USES BOTH GPUs, ~40 minutes): six engine runs - the pre-merge engine, the rebuilt one, then the rebuilt one
rem   with each fix undone in turn (0.1.40's host placement, 0.1.40's V100 PCIe share, the async copy thread on its old
rem   spot), the pre-merge engine again.  Log: v100_decodebench12.log (+ .json).  Close other programs first.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
echo.
echo Part 1 is done. Part 2 uses both GPUs for about 40 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench12.log --variants pre_merge,baseline,host_cpuset,stage_pcie_up,job_spare,pre_merge_end %*
echo.
echo Done. Send v100_decodebench12.log (and v100_setup.log if anything failed) back for review.
pause
