@echo off
setlocal enabledelayedexpansion

echo ==============================
echo 双区域屏幕实时识别应用打包脚本
echo ==============================
echo.

:: 检查Python是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python。请先安装Python 3.8+并添加到PATH。
    pause
    exit /b 1
)

:: 检查是否已创建虚拟环境
if not exist "venv" (
    echo 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo 错误：创建虚拟环境失败。
        pause
        exit /b 1
    )
)

:: 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo 错误：激活虚拟环境失败。
    pause
    exit /b 1
)

:: 检查是否已安装依赖
echo 检查依赖包...
pip list | findstr /i "torch" >nul 2>&1
if errorlevel 1 (
    echo 安装依赖包...
    pip install torch torchvision pillow pyinstaller
    if errorlevel 1 (
        echo 错误：安装依赖包失败。
        pause
        exit /b 1
    )
)

:: 检查是否有模型文件，如果没有则创建示例模型
if not exist "best_model.pth" (
    echo 创建示例模型文件...
    python create_dummy_model.py
    if errorlevel 1 (
        echo 错误：创建示例模型失败。
        pause
        exit /b 1
    )
)

:: 执行打包
echo 开始打包应用...
pyinstaller screen_recognition.spec
if errorlevel 1 (
    echo 错误：打包失败。
    pause
    exit /b 1
)

:: 检查打包结果
if exist "dist\ScreenRecognition\ScreenRecognition.exe" (
    echo 打包成功！
    echo 可执行文件路径：dist\ScreenRecognition\ScreenRecognition.exe
    echo.
    echo 打包的文件包括：
    dir "dist\ScreenRecognition" /b
) else (
    echo 错误：打包后的可执行文件不存在。
    pause
    exit /b 1
)

:: 提示用户
echo.
echo ==============================
echo 打包完成！
echo ==============================
echo 您可以在 dist\ScreenRecognition 目录下找到打包好的应用。
echo 分发时，请将整个 ScreenRecognition 目录复制给用户。
echo 用户只需双击 ScreenRecognition.exe 即可运行应用。
echo.
echo 注意：应用可能需要管理员权限才能正常工作。
pause
endlocal
