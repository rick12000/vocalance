# Directory Path Consistency

Vocalance writes to two root directories. Several files reference these paths
as literals and must be kept in sync whenever one of them changes.

## Owned Directories

| Name | Windows path | Contents |
|---|---|---|
| Install root | `%LOCALAPPDATA%\Programs\Vocalance` | App code, venv, bundled tools. Written by setup scripts only. |
| User data root | `%APPDATA%\Vocalance` | Settings, marks, click history, sound/LLM models, logs. Written by the running app. |

## Points That Must Stay Aligned

### User data root

The directory name `"Vocalance"` appears in four places and must be identical across all of them:

| File | Variable |
|---|---|
| `vocalance/app/config/app_config.py` | `APPDATA_DIR_NAME = "Vocalance"` |
| `scripts/bootstrapping/setup.ps1` | `$USER_DATA_DIR = Join-Path $env:APPDATA 'Vocalance'` |
| `scripts/bootstrapping/cleanup.ps1` | `$USER_DATA = Join-Path $env:APPDATA 'Vocalance'` |

`APPDATA_DIR_NAME` is also the value passed to `LoggingConfigModel(appdata_dir_name=...)` in the same file, so log files land under the same root automatically.

### Install root

| File | Variable |
|---|---|
| `scripts/bootstrapping/setup.ps1` | `$INSTALL_ROOT = Join-Path $env:LOCALAPPDATA 'Programs\Vocalance'` |
| `scripts/bootstrapping/cleanup.ps1` | `$INSTALL_ROOT = Join-Path $env:LOCALAPPDATA 'Programs\Vocalance'` |

The app does not read the install root at runtime; it is a setup script concern only.

### User data subdirectories

All subdirectory names under the user data root are defined as fields on `StorageConfig` in `app_config.py`. The scripts wipe the entire user data root rather than individual subdirectories, so adding or renaming a subdir does **not** require touching the scripts.

## Update Checklist

**Renaming the user data root:** update `APPDATA_DIR_NAME` in `app_config.py`, then update the matching literal in `setup.ps1` and `cleanup.ps1`.

**Renaming the install root:** update `$INSTALL_ROOT` in `setup.ps1` and `cleanup.ps1`.

## Verification

```powershell
# Confirm scripts and app_config agree on the user data dir name
Select-String -Path scripts\bootstrapping\setup.ps1, `
                    scripts\bootstrapping\cleanup.ps1, `
                    vocalance\app\config\app_config.py `
              -Pattern 'Vocalance'

# Confirm scripts agree on the install root
Select-String -Path scripts\bootstrapping\setup.ps1, `
                    scripts\bootstrapping\cleanup.ps1 `
              -Pattern 'INSTALL_ROOT'
```
