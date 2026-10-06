@echo off
rem V100 fork, step 18 - is the merged engine slower, or was the PC slower that afternoon?  USES BOTH GPUs, ~30 minutes,
rem   nothing is compiled.  Four engine runs, alternating: the engine from before the merge (engine\strata.exe, with the
rem   config saved as strata-unsloth-ud-q4_k_xl.json.pre-merge), the merged one, the old one again, the merged one again.
rem   Log: v100_decodebench10.log (+ .json).  Close other programs first (browser tabs, games, anything busy).
setlocal
cd /d "%~dp0.."
if not exist strata-unsloth-ud-q4_k_xl.json.pre-merge (
  echo strata-unsloth-ud-q4_k_xl.json.pre-merge is missing - it is written by step 17.
  pause
  exit /b 1
)
echo This uses both GPUs for about 30 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench10.log --variants pre_merge,baseline,pre_merge_end,baseline_end %*
echo.
echo Done. Send v100_decodebench10.log back for review.
pause
