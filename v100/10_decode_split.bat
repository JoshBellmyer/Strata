@echo off
rem V100 fork, step 10 - the split point between the cards and the draft length, measured more carefully.
rem   USES BOTH GPUs, ~45 minutes, nothing is compiled.  Eight engine runs: the config first and last, the 4070 Super
rem   with 20 / 22 / 24 layers, 20 layers with shorter drafts, shorter drafts alone, and the page-locked RAM copy with
rem   a smaller PCIe share.  Each writes 256 tokens for three short chats (twice) and 2 x 300 tokens after ~60K tokens.
rem   Logs: v100_decodebench4.log (+ .json).
setlocal
cd /d "%~dp0.."
echo This uses both GPUs for about 45 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench4.log %*
echo.
echo Done. Send v100_decodebench4.log back for review.
pause
