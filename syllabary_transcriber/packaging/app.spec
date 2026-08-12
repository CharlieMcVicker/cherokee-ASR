# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.osx import BUNDLE
from PyInstaller.building.build_main import Analysis
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# SPECPATH is defined by PyInstaller when executing spec files
ui_dist = os.path.abspath(os.path.join(SPECPATH, '..', 'ui', 'dist'))

datas = [
    (ui_dist, os.path.join('syllabary_transcriber', 'ui', 'dist')),
]
binaries = []
hiddenimports = ['uvicorn', 'pywebview', 'engineio.async_drivers.asgi']

tv_datas, tv_binaries, tv_hiddenimports = collect_all('torchvision')
datas.extend(tv_datas)
binaries.extend(tv_binaries)
hiddenimports.extend(tv_hiddenimports)

ta_datas, ta_binaries, ta_hiddenimports = collect_all('torchaudio')
datas.extend(ta_datas)
binaries.extend(ta_binaries)
hiddenimports.extend(ta_hiddenimports)

a = Analysis(
    [os.path.join(SPECPATH, '..', '__main__.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cherokee Syllabary Transcriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cherokee Syllabary Transcriber',
)

app = BUNDLE(
    coll,
    name='Cherokee Syllabary Transcriber.app',
    icon=None,
    bundle_identifier='com.cherokee.syllabarytranscriber',
    info_plist={
        'NSMicrophoneUsageDescription': 'Cherokee Syllabary Transcriber needs access to your microphone to transcribe speech.',
    },
)

