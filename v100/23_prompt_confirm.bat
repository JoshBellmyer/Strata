@echo off
rem V100 fork, step 23 - optional: confirms the new prompt settings together (bench 4 measured them one at a time).
rem   USES BOTH GPUs, about 25 minutes: the config before bench 4 and the config now, twice each, every run reading a
rem   16K and a ~120K prompt. No rebuild. Log: v100_prefillbench5.log. Close other GPU programs first.
setlocal
cd /d "%~dp0.."
echo This uses both GPUs for about 25 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_prefillbench.py --log v100_prefillbench5.log --variants prev,cfg,prev,cfg %*
echo.
echo Done. Send v100_prefillbench5.log back for review.
pause
