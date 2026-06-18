# Headless Environment Testing

## Overview

Eight test modules require display, audio hardware, or OpenGL libraries. This document explains the skip mechanism and CI setup required to run all tests.

## Test Modules Requiring Display or Audio

| Module | Dependency | Library |
|--------|-----------|---------|
| `test_audio_capture_service.py` | `audio_capture_service` → `sounddevice` | Audio hardware |
| `test_automation_service.py` | `automation_service` → `pyautogui` | Display |
| `test_mark_service.py` | `mark_service` → `pyautogui` | Display |
| `test_text_input_service.py` | `text_input_service` → `pyautogui` | Display |
| `test_text_command_parse.py` | `text_command_parse` → `pyautogui` | Display |
| `test_centralized_command_parser.py` | `parser` → `text_command_parse` → `pyautogui` | Display |
| `test_dictation_coordinator_core.py` | `dictation_coordinator` → `text_input_service` → `pyautogui` | Display |
| `test_style_builder.py` | `qt_theme` → `PySide6.QtGui` | OpenGL |

## Skip Mechanism

### Implementation

A reusable guard function in `tests/conftest.py`:

```python
def skip_if_headless() -> None:
    """Skip the calling test module when display or audio hardware is unavailable."""
    try:
        import pyautogui  # noqa: F401
    except Exception:
        pytest.skip("requires display (pyautogui)", allow_module_level=True)
    try:
        import sounddevice  # noqa: F401
    except OSError:
        pytest.skip("requires audio hardware (sounddevice)", allow_module_level=True)
    try:
        from PySide6.QtGui import QFont  # noqa: F401
    except ImportError:
        pytest.skip("requires OpenGL libraries (PySide6.QtGui)", allow_module_level=True)
```

### Usage

Each of the 8 affected test modules includes these lines immediately after imports:

```python
from conftest import skip_if_headless
skip_if_headless()
```

Calling `pytest.skip(..., allow_module_level=True)` prevents collection errors and marks the entire module as skipped in the test report.

## CI/CD Configuration

### System Dependencies

The GitHub Actions runner must install:

```yaml
- name: Install system dependencies
  run: sudo apt-get install -y libportaudio2 xvfb libgl1 libegl1
```

- `libportaudio2`: Required by `sounddevice` to import successfully on Linux
- `xvfb`: Provides a virtual X11 display server for headless environments
- `libgl1`: Provides OpenGL libraries for PySide6/Qt on headless systems
- `libegl1`: Provides EGL libraries for PySide6/Qt OpenGL rendering

### Test Execution

Run pytest under `xvfb-run`:

```yaml
- name: Run unit tests
  run: xvfb-run --auto-servernum pytest --ignore=tests/integration -m "not slow and not integration and not memory and not stress"
```

- `xvfb-run --auto-servernum`: Launches a virtual display and assigns an unused display number automatically
- `--ignore=tests/integration`: Excludes integration tests (not unit test scope)
- `-m "not slow and not integration and not memory and not stress"`: Excludes slow, integration, memory, and stress tests

## Expected Behavior

### With Dependencies Installed and Virtual Display Available

All 551 unit tests pass. The 8 affected modules import `pyautogui`, `sounddevice`, and PySide6 successfully; the `skip_if_headless()` guard does not trigger.

### Without Dependencies or Virtual Display

The 8 affected modules skip at collection time with reason messages visible in the pytest report (marked with 's'). The remaining 543 tests pass normally. Total: 543 passed, 8 skipped.

## Local Development

To run tests locally with the same environment:

```bash
conda activate vocalance_env_dev
xvfb-run pytest tests/vocalance
```

Or if a display is available:

```bash
conda activate vocalance_env_dev
pytest tests/vocalance
```

If running on Windows or macOS with a display, `skip_if_headless()` will never trigger and all 551 tests execute.
