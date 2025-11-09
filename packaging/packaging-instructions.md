# Vocalance Packaging

## Pyinstaller

From the project root directory, run:
```bash
pyinstaller packaging/vocalance.spec
```

The packaged application will be in `dist/vocalance/`. Run `vocalance.exe` to start.

### PyInstaller Bundling Logic

PyInstaller creates a self-contained executable by analyzing the application and bundling only the necessary components from the build environment.

#### Environment Consistency
- **Python Version**: Bundles the exact Python interpreter from the environment where PyInstaller runs
- **Package Versions**: Includes exact versions of all dependencies installed in the build environment (see `dist_requirements.txt`)

#### Dependency Collection (`vocalance.spec` lines 10-13)
Uses `collect_all()` to automatically detect and bundle dependencies for key libraries:
```python
llama_datas, llama_binaries, llama_hiddenimports = collect_all('llama_cpp')
vosk_datas, vosk_binaries, vosk_hiddenimports = collect_all('vosk')
soundfile_datas, soundfile_binaries, soundfile_hiddenimports = collect_all('soundfile')
```

#### Hidden Imports (`vocalance.spec` lines 22-35)
Explicitly declares modules that may not be automatically detected:
- Runtime/dynamic imports (tensorflow, customtkinter, PIL.ImageTk)
- C extension modules (llama_cpp, vosk, soundfile, librosa)

#### Inclusion Criteria
- **Included**: Only packages explicitly imported in `vocalance.py` or listed in `hiddenimports`
- **Excluded**: PyInstaller itself (build tool only), development dependencies, unused packages

#### Runtime Detection
Application detects bundled mode using PyInstaller runtime attributes (see `vocalance/app/config/app_config.py` lines 423-424):
```python
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Running from PyInstaller bundle
```

## Inno Setup
