@echo off
rem V100 fork, step 13 - the PLE table (layer 1's n-gram table) in a file of its own.  NO GPU USED.
rem   Builds a small test program, measures the table's reads with and without a live mapping of the model shard
rem   (~2 minutes), and if the mapping slows them: copies the table (~29 GB, a few minutes) into its own file next
rem   to the model, checks it byte for byte, and points the config at it.  Log: v100_ple_split.log.
rem   Undo: 13_ple_split.bat --undo
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_ple_split.py %*
echo.
echo Done. Send v100_ple_split.log back for review.
pause
