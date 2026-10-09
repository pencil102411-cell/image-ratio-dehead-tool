@echo off
set "PYEXE=python"
where python >nul 2>nul || set "PYEXE=py"
"%PYEXE%" -m pip install numpy scipy pillow "opencv-python-headless<5" tkinterdnd2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pause
