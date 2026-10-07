@echo off
rem V100 fork, step 24 - where the writing (decode) time goes, and the RAM copy in large pages.
rem   Run v100\24a_grant_large_pages.bat first (then sign out and back in), or the large-page runs just use normal
rem   pages and say so in the log.
rem   Part 1 (no GPU, a few minutes): rebuilds the engine (log v100_setup.log), then measures how fast the CPU can
rem   read expert-sized blocks from RAM in normal and in large pages (log v100_membw.log, about a minute).
rem   Part 2 (USES BOTH GPUs, about 35 minutes): five engine runs, each writing after short chats and after a ~60K
rem   prompt - the config, the RAM copy in large pages, the GPU stage profile of each card, the config again, large
rem   pages again. Log: v100_decodebench13.log (+ .json). Close other GPU programs first.
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\v100_setup.py --context 262144
if errorlevel 1 (
  echo Setup or the build failed - send v100_setup.log back for review.
  pause
  exit /b 1
)
set MEMBW=build-cuda12\v100_membw.exe
if not exist "%MEMBW%" set MEMBW=build\v100_membw.exe
"%MEMBW%" > v100_membw.log 2>&1
type v100_membw.log
echo.
echo Part 1 is done. Part 2 uses both GPUs for about 35 minutes. Stop the server and any other GPU program first.
pause
".venv\Scripts\python.exe" tools\v100_decodebench.py --log v100_decodebench13.log --variants baseline,largepages,profile,baseline_end,largepages %*
echo.
echo Done. Send v100_membw.log and v100_decodebench13.log (and v100_setup.log if anything failed) back for review.
pause
