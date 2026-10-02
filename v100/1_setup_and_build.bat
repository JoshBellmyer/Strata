@echo off
rem V100 fork, step 1 - uses NO GPU (safe while your other server runs on both cards):
rem   finds the RTX 4070 Super and the Tesla V100 with nvidia-smi, runs setup for Unsloth's UD-Q4_K_XL
rem   (download / pack if needed, compile the engine for sm_70 + sm_89 with CUDA 12.9, write the config, no start),
rem   then compiles the GPU test programs.  Everything is logged to v100_setup.log in the Strata folder.
rem   Extra options are passed on, e.g.:  1_setup_and_build.bat --context 65536 --gguf-dir D:\models\ud-q4
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" call START-HERE.bat --check < nul
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing - run START-HERE.bat --check once, then this again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" tools\v100_setup.py %*
echo.
echo Done. Send v100_setup.log (in the Strata folder) back for review.
pause
