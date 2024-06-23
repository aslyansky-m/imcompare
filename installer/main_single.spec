# -*- mode: python ; coding: utf-8 -*-

import pkgutil

import rasterio

from glob import glob


# list all rasterio and fiona submodules, to include them in the package
additional_packages = list()
for package in pkgutil.iter_modules(rasterio.__path__, prefix="rasterio."):
    additional_packages.append(package.name)
	
dlls = glob("C:\\Users\\maxima\\anaconda3\\envs\\imcompare\\Lib\\site-packages\\rasterio.libs\\*.dll")
dlls = [(x,'.') for x in dlls]


resources = [('C:\\Users\\maxima\\Documents\\repos\\imcompare\\resources\\logo.jpg','resources\\logo.jpg'),
			("C:\\Users\\maxima\\anaconda3\\envs\\imcompare\\Lib\\site-packages\\matplotlib\\mpl-data", "matplotlib/mpl-data"),]

a = Analysis(
    ['C:\\Users\\maxima\\Documents\\repos\\imcompare\\main.py'],
    pathex=['C:\\Users\\maxima\\anaconda3\\envs\\imcompare\\Lib\\site-packages\\rasterio.libs'],
    binaries=dlls,
    datas=resources,
    hiddenimports=additional_packages,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=".",
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\maxima\\Documents\\repos\\imcompare\\resources\\logo.ico'],
)
