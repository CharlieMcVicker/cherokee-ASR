# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis

block_cipher = None

ui_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ui', 'dist'))

datas = [
    (ui_dist, os.path.join('syllabary_transcriber', 'ui', 'dist')),
]

a = Analysis(
    ['../__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['uvicorn', 'pywebview'],
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

info_plist = {
    'NSMicrophoneUsageDescription': 'Cherokee Syllabary Transcriber needs access to your microphone to transcribe speech.',
}
