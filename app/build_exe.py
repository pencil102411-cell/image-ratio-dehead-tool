"""重新打包成 exe：python build_exe.py（或双击上一级的「重新打包成exe.bat」）。"""
import os
import shutil
import subprocess
import sys
import tempfile

APP = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(APP)                 # 源码文件夹（也是 GitHub 仓库）
TOOL_DIR = os.path.dirname(REPO)            # 再上一级：平时用 exe 的文件夹
EXE_NAME = "图片比例转换去头工具.exe"
MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"


def run(cmd):
    print(">", " ".join(cmd[:4]), "…")
    subprocess.check_call(cmd)


def main():
    print("第 1 步：确认打包工具已安装……")
    run([sys.executable, "-m", "pip", "install", "--quiet", "pyinstaller", "numpy", "scipy", "pillow",
         "opencv-python-headless<5", "tkinterdnd2", "-i", MIRROR])
    work = os.path.join(tempfile.gettempdir(), "croptool_build")
    print("第 2 步：打包中，大约 1 分钟……")
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
         "--name", "CropTool",
         "--icon", os.path.join(APP, "icon.ico"),
         "--splash", os.path.join(APP, "splash.png"),
         "--add-data", os.path.join(APP, "face_detection_yunet_2023mar.onnx") + ";.",
         "--add-data", os.path.join(APP, "icon.ico") + ";.",
         "--paths", APP,
         "--collect-all", "tkinterdnd2",   # 拖放组件自带的 tkdnd 动态库
         "--distpath", os.path.join(work, "dist"),
         "--workpath", os.path.join(work, "work"),
         "--specpath", work,
         "--exclude-module", "matplotlib", "--exclude-module", "pandas",
         "--exclude-module", "IPython", "--exclude-module", "pytest",
         os.path.join(APP, "crop_tool.py")])
    built = os.path.join(work, "dist", "CropTool.exe")
    os.makedirs(os.path.join(REPO, "dist"), exist_ok=True)
    targets = [os.path.join(REPO, "dist", EXE_NAME)]
    # 如果上一级已经放着旧版 exe（平时使用的位置），一起替换掉
    if os.path.exists(os.path.join(TOOL_DIR, EXE_NAME)):
        targets.append(os.path.join(TOOL_DIR, EXE_NAME))
    for t in targets:
        shutil.copy2(built, t)
        print("\n打包完成，新的 exe 已放到：", t)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n打包失败：", e, "\n请把上面的文字截图发给 Claude。")
    input("\n按回车键关闭…")
