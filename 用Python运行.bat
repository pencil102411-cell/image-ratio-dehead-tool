@echo off
set "PYEXE=python"
where python >nul 2>nul || set "PYEXE=py"
"%PYEXE%" "%~dp0app\crop_tool.py" %*
if errorlevel 1 pause
