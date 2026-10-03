@echo off
rem V100 fork, step 9 - how fast the model files' drive answers small random reads.  NO GPU USED: safe to run any
rem   time, even with the other server running (~1 minute).  Log: v100_disktest.log.
rem   Another drive to compare: 9_disk_test.bat --file D:\some\big\file.bin
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_disktest.py %*
echo.
echo Done. Send v100_disktest.log back for review.
pause
