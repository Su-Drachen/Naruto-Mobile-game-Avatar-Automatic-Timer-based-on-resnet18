# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['screen_recognition.py'],
             pathex=['/home/user/vibecoding/workspace/screen_recognition'],
             binaries=[],
             datas=[('best_model.pth', '.'), ('app_icon.ico', '.')],
             hiddenimports=['torch._C', 'torch._C._fft', 'torch._C._linalg', 'torch._C._nn', 'torch._C._sparse'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='ScreenRecognition',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon='app_icon.ico',
          version='version_info.txt',
          uac_admin=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ScreenRecognition')
