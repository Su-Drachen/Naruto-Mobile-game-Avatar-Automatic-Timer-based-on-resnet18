# 双区域屏幕实时实时识别应用打包指南

## 项目概述

这是一个基于PyTorch和Tkinter的双区域屏幕实时识别应用，可以在没有Python环境的Windows电脑上运行。

## 打包方案

### 环境准备

1. **安装Python**
   - 下载并安装Python 3.8+（推荐3.9或3.10）
   - 确保勾选"Add Python to PATH"选项

2. **创建虚拟环境**
   ```bash
   # 创建虚拟环境
   python -m venv venv
   
   # 激活虚拟环境
   # Windows命令提示符
   venv\Scripts\activate
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

3. **安装依赖包**
   ```bash
   pip install torch torchvision pillow pillow pyinstaller
   ```

### 文件结构

```
screen_recognition/
├── screen_recognition.py      # 主程序文件
├── screen_recognition.spec   # PyInstaller配置文件
├── version_info.txt          # 版本信息文件
├── app_icon.ico              # 应用图标
├── best_model.pth            # 模型文件
├── packaging_guide.md        # 打包指南
└── create_dummy_model.py     # 创建示例模型的脚本
```

### 打包步骤

1. **创建示例模型（如果没有真实模型）**
   ```bash
   python create_dummy_model.py
   ```

2. **执行打包命令**
   ```bash
   pyinstaller screen_recognition.spec
   ```

3. **打包完成后**
   - 可执行文件和相关依赖会生成在`dist/ScreenRecognition/`目录下
   - 主程序文件：`ScreenRecognition.exe`

### 打包优化

1. **减小文件体积**
   - 使用UPX压缩（已在.spec文件中配置）
   - 只包含必要的依赖
   - 使用PyTorch的CPU版本

2. **性能优化**
   - 确保模型文件被正确打包
   - 检查所有动态链接库是否被正确包含

### 分发文件

打包完成后，`dist/ScreenRecognition/`目录下的所有文件都需要分发给用户。用户只需双击`ScreenRecognition.exe`即可运行应用。

### 注意事项

1. **管理员权限**
   - 应用可能需要管理员权限才能正确捕获屏幕
   - 在.spec文件中已设置`uac_admin=True`，运行时会请求管理员权限

2. **模型文件**
   - 确保`best_model.pth`文件在打包时被正确包含
   - 模型文件必须与可执行文件在同一目录下

3. **DPI问题**
   - 应用包含了DPI感知代码，确保在高DPI显示器上正常工作

4. **错误处理**
   - 如果运行时出现错误，可以尝试以管理员身份运行
   - 检查是否缺少必要的依赖文件

### 常见问题解决

1. **"找不到模型文件"错误**
   - 确保`best_model.pth`与可执行文件在同一目录下

2. **"无法捕获屏幕"错误**
   - 尝试以管理员身份运行应用
   - 检查是否有其他应用阻止屏幕捕获

3. **应用启动后无响应**
   - 检查是否有其他应用占用了大量系统资源
   - 尝试重新启动电脑后再运行应用

### 替代打包方案

如果PyInstaller出现问题，可以考虑以下替代方案：

1. **cx_Freeze**
   ```bash
   pip install cx_Freeze
   cxfreeze screen_recognition.py --target-dir dist --base-name Win32GUI
   ```

2. **conda-pack**
   如果使用conda环境：
   ```bash
   conda install conda-pack
   conda pack -n myenv -o screen_recognition_env.tar.gz
   ```

3. **Inno Setup**
   可以使用Inno Setup创建安装程序，将所有文件打包为一个安装包。

## 总结

通过以上步骤，您可以将双区域屏幕实时识别应用打包为可在没有Python环境的Windows电脑上运行的可执行文件。打包完成后，用户只需双击运行即可使用应用的所有功能。
