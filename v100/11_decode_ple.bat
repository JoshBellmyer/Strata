@echo off
rem V100 fork, step 11 - the PLE row reads (layer 1's table on the SSD) while decoding.
rem   USES BOTH GPUs, ~30 minutes, nothing is compiled.  Six engine runs: the config first and last, 48 and 16 threads
rem   issuing the reads, the reads submitted on the decode thread, and the SSD kept awake.
rem   Logs: v100_decodebench5.log (+ .json).
setlocal
cd /d "%~dp0.."
echo This uses both GPUs for about 30 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench5.log %*
echo.
echo Done. Send v100_decodebench5.log back for review.
pause
