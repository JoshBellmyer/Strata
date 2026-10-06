@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\Josh\Documents\GitHub\Strata\build --target ple_reader_test -j 4
