@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

rem dartsam3 has torch+cu128, TensorRT 10.13 and DART installed.
set "PYEXE=C:\Users\aitor\anaconda3\envs\dartsam3\python.exe"
if not exist "%PYEXE%" (
  echo Could not find the dartsam3 environment at:
  echo   %PYEXE%
  echo Edit this file and point PYEXE at your python.exe.
  pause
  exit /b 1
)

"%PYEXE%" "%~dp0run_analysis.py"
if errorlevel 1 pause
