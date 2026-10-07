@echo off
rem V100 fork, step 24a - gives your Windows account the "Lock pages in memory" right, which the engine needs to keep
rem   its RAM copy of the experts in 2 MB large pages (tested by step 24). No GPU. It asks for administrator rights.
rem   Afterwards SIGN OUT of Windows and back in (or restart) before running step 24.
rem   To undo it later:  24a_grant_large_pages.bat --remove
setlocal
cd /d "%~dp0.."
net session >nul 2>&1
if errorlevel 1 (
  echo Asking for administrator rights...
  if "%~1"=="" (
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
  ) else (
    powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0' -ArgumentList '%*'"
  )
  exit /b
)
".venv\Scripts\python.exe" tools\v100_grant_lockpages.py %*
echo.
pause
