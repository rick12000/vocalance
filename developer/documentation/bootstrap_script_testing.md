# Bootstrap Script Testing

## Scripts

| Script | Path |
|--------|------|
| Install | `scripts/bootstrapping/setup.ps1` |
| Uninstall | `scripts/bootstrapping/cleanup.ps1` |

Production `setup.ps1` downloads the app release zip from GitHub (`rick12000/vocalance`, tag `v$VOCALANCE_VERSION`) and uv from Astral (pinned `$UV_VERSION`, SHA-256 verified). To test before a release, mock only the **app zip** so the branch under test is installed instead of the published artifact.

## Mock the app zip (temporary)

1. At the top of `setup.ps1`, after `$VOCALANCE_REPO`, set a local zip path:

   ```powershell
   $LOCAL_RELEASE_ZIP = 'C:\path\to\vocalance-v1.0.0.zip'
   ```

   Omit this variable in production (GitHub download only).

2. Replace the Vocalance download block with:

   ```powershell
   if ($LOCAL_RELEASE_ZIP) {
       if (-not (Test-Path -LiteralPath $LOCAL_RELEASE_ZIP)) {
           throw "Local release zip not found: $LOCAL_RELEASE_ZIP"
       }
       Write-Host "Using local release zip: $LOCAL_RELEASE_ZIP"
       Copy-Item -LiteralPath $LOCAL_RELEASE_ZIP -Destination $zipPath -Force
   } else {
       Write-Host "Downloading Vocalance v$VOCALANCE_VERSION..."
       Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
   }
   ```

   Do not commit `$LOCAL_RELEASE_ZIP` or this branch; revert before release.

## Build the local release zip

From the repository root (same contents as `.github/workflows/ci-cd.yml` release step):

```powershell
$zip = '.\.test-artifacts\vocalance-v1.0.0.zip'
New-Item -ItemType Directory -Force -Path (Split-Path $zip) | Out-Null
Compress-Archive -Path vocalance, vocalance.py, pyproject.toml, uv.lock, README.md, DISCLAIMER.md, NOTICES -DestinationPath $zip -Force
```

Use the version in `pyproject.toml` / `$VOCALANCE_VERSION` in the zip file name.

## Test flow

1. **Clean slate** — Run `cleanup.ps1` or ensure `%LOCALAPPDATA%\Programs\Vocalance\` is absent. Back up `%APPDATA%\Vocalance\` if needed; reinstall wipes it.
2. **Apply mock** — Set `$LOCAL_RELEASE_ZIP` and the conditional copy block above. Build the local zip (previous section).
3. **Install** — From repo root:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\bootstrapping\setup.ps1
   ```

   Answer the LLM prompt for your scenario. Confirm install tree and Start Menu shortcut.
4. **Manual GUI** — Launch from the Start Menu; confirm startup and behavior.
5. **Uninstall** — Run `cleanup.ps1`. Confirm `%LOCALAPPDATA%\Programs\Vocalance\`, `%APPDATA%\Vocalance\`, and `Vocalance.lnk` are gone.
6. **Roll back mock** — Remove `$LOCAL_RELEASE_ZIP` and the conditional; restore GitHub-only download. Remove `.test-artifacts\` (or other local zips).

Steps 3–6 passing means install/uninstall scripts are validated; with the mock rolled back, `setup.ps1` and `cleanup.ps1` are ready for production and the release bundle.

## uv pin changes

Bumping `$UV_VERSION` in `setup.ps1` requires new `$UV_ZIP_SHA256` entries from:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\security\compute_uv_binary_hash.ps1 -UvVersion <version>
```

Update `docs/security/supply_chain_integrity.rst` example if the documented version changes.
