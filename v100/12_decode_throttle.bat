@echo off
rem V100 fork, step 12 - whether Windows slows the engine's I/O threads while decoding.
rem   USES BOTH GPUs, ~20 minutes, nothing is compiled.  Four engine runs: the config, Windows power throttling
rem   (EcoQoS) off for the engine process, the engine at high priority, the config again.
rem   Logs: v100_decodebench6.log (+ .json).
setlocal
cd /d "%~dp0.."
echo This uses both GPUs for about 20 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench6.log --variants baseline,nothrottle,highprio,baseline_end %*
echo.
echo Done. Send v100_decodebench6.log back for review.
pause
