@echo off
rem V100 fork, step 6 - where the writing (decode) time goes.
rem   USES BOTH GPUs, ~35-45 minutes, nothing is compiled.  Eight engine runs: the config as it is, the same with the
rem   GPU stage stamps, the V100 alone, and five settings (CPU expert kernels, PCIe share, draft length).  Each writes
rem   256 tokens for three short chats (twice) and 300 tokens after a ~60K-token document.
rem   Logs: v100_decodebench.log (+ .json); the engine's timing lines are copied into it.
setlocal
cd /d "%~dp0.."
echo This uses both GPUs for about 35-45 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py %*
echo.
echo Done. Send v100_decodebench.log back for review.
pause
