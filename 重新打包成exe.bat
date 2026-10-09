@echo off
set "PYEXE=python"
where python >nul 2>nul || set "PYEXE=py"
"%PYEXE%" "%~dp0app\build_exe.py"
