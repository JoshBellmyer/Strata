@echo off
rem V100 fork, step 19 - which part of the merged engine makes it decode slower.  USES BOTH GPUs, ~50 minutes,
rem   nothing is compiled.  Eight engine runs: the pre-merge engine, the merged one as it is, then the merged one with
rem   one group of its newer decode paths turned back off per run (PCIe share 0, decode batching off, shared-expert
rem   stream off, the newer verify-window paths off, the host thread on the last core), the pre-merge engine again.
rem   Log: v100_decodebench11.log (+ .json).  Close other programs first.
setlocal
cd /d "%~dp0.."
if not exist strata-unsloth-ud-q4_k_xl.json.pre-merge (
  echo strata-unsloth-ud-q4_k_xl.json.pre-merge is missing - it is written by step 17.
  pause
  exit /b 1
)
echo This uses both GPUs for about 50 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench11.log --variants pre_merge,baseline,pcie0,decbatch0,shstream0,gpu_legacy,hostlast,pre_merge_end %*
echo.
echo Done. Send v100_decodebench11.log back for review.
pause
