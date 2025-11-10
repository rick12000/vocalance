import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

# Get the directory containing the spec file
spec_dir = Path(os.path.abspath(os.path.dirname(sys.argv[0])))
project_root = spec_dir.parent
vocalance_dir = project_root / 'vocalance'

# Collect all dependencies including shared libraries
llama_datas, llama_binaries, llama_hiddenimports = collect_all('llama_cpp')
vosk_datas, vosk_binaries, vosk_hiddenimports = collect_all('vosk')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')
tensorflow_datas, tensorflow_binaries, tensorflow_hiddenimports = collect_all('tensorflow')

# Collect all DLLs from conda environment's Library/bin directory
# TensorFlow and other native dependencies require these runtime libraries
# Determine conda environment bin directory
if hasattr(sys, 'base_prefix'):
    conda_env_bin = Path(sys.base_prefix) / 'Library' / 'bin'
else:
    conda_env_bin = Path(sys.executable).parent.parent / 'Library' / 'bin'

extra_binaries = []
if conda_env_bin.exists():
    print(f"Collecting DLLs from conda environment: {conda_env_bin}")
    # Get all DLL files, excluding Windows API forwarding DLLs (api-ms-*.dll)
    dll_files = [f for f in conda_env_bin.glob('*.dll') if not f.name.startswith('api-ms-')]
    
    for dll_path in dll_files:
        extra_binaries.append((str(dll_path), '.'))
        print(f"  Adding: {dll_path.name}")
    
    print(f"Total conda environment DLLs bundled: {len(extra_binaries)}")
else:
    print(f"WARNING: Conda environment bin directory not found at {conda_env_bin}")
    print("TensorFlow and other native dependencies may not work correctly!")

a = Analysis(
    [str(project_root / 'vocalance.py')],
    pathex=[str(project_root)],
    binaries=llama_binaries + vosk_binaries + soundfile_binaries + tensorflow_binaries + extra_binaries,
    datas=[
        (str(vocalance_dir / 'app' / 'assets'), 'vocalance/app/assets'),
    ] + llama_datas + vosk_datas + soundfile_datas + tensorflow_datas,
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
    ] + llama_hiddenimports + vosk_hiddenimports + soundfile_hiddenimports + tensorflow_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

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
