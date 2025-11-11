import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# Get the directory containing the spec file
spec_dir = Path(os.path.abspath(os.path.dirname(sys.argv[0])))

# Add packaging directory to path for imports
sys.path.insert(0, str(spec_dir))

from pyinstaller_hook_filter import filter_binaries
project_root = spec_dir.parent
vocalance_dir = project_root / 'vocalance'

# Collect all dependencies including shared libraries
llama_datas, llama_binaries, llama_hiddenimports = collect_all('llama_cpp')
vosk_datas, vosk_binaries, vosk_hiddenimports = collect_all('vosk')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')

a = Analysis(
    [str(project_root / 'vocalance.py')],
    pathex=[str(project_root)],
    binaries=llama_binaries + vosk_binaries + soundfile_binaries,
    datas=[
        (str(vocalance_dir / 'app' / 'assets'), 'vocalance/app/assets'),
    ] + llama_datas + vosk_datas + soundfile_datas,
    hiddenimports=[
        # Conditional/dynamic imports
        'tensorflow',
        'tensorflow.python.keras.api._v2.keras',
        'tensorflow.python.platform',
        'tensorflow.python._pywrap_tensorflow_internal',
        'customtkinter',
        'PIL.ImageTk',

        # C extension libraries - expanded for better coverage
        'llama_cpp',
        'llama_cpp.llama_cpp',
        'llama_cpp._ctypes_extensions',
        'vosk',
        'soundfile',
        'librosa',
    ] + llama_hiddenimports + vosk_hiddenimports + soundfile_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Filter proprietary binaries from the Analysis object (catches all packages, not just explicit ones)
a.binaries = filter_binaries(a.binaries)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vocalance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(vocalance_dir / 'app' / 'assets' / 'logo' / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vocalance',
)
