@echo off
rem V100 fork, step 14 - the PLE table from its own file, measured.
rem   Part 1 (NO GPU, ~2 minutes): the table's reads timed again, now that the copy has settled on the drive.
rem     Log: v100_ple_split.log.
rem   Part 2 (USES BOTH GPUs, ~25 minutes): four engine runs - the config (table from its own file), the table read
rem     from the model's shard again, then both once more.  Log: v100_decodebench7.log.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_ple_split.py --retest
echo.
echo Part 2 uses both GPUs for about 25 minutes. Stop any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench7.log --variants baseline,ple_shard,baseline_end,ple_shard_end %*
echo.
echo Done. Send v100_ple_split.log and v100_decodebench7.log back for review.
pause
