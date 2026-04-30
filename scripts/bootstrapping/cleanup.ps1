#Requires -Version 5.1
<#
.SYNOPSIS
    Remove Vocalance user data and Start Menu shortcut.

.DESCRIPTION
    Removes:
    - %APPDATA%\vocalance_voice_assistant_data\ (user data, settings, models, aliases)
    - Start Menu shortcut (Vocalance.lnk)

    Does not remove the vocalance-prod folder or vocalance_env folder (delete those manually).

    Safe and robust: only targets exact known paths, uses -LiteralPath to prevent interpretation
    of special characters, and checks existence before deletion.
#>

$ErrorActionPreference = 'Stop'

# Define paths with exact names — no wildcards or variable expansion that could go wrong
$userDataRoot = Join-Path $env:APPDATA 'vocalance_voice_assistant_data'
$shortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Vocalance.lnk'

$removed = @()
$skipped = @()

# Remove user data directory
if (Test-Path -LiteralPath $userDataRoot) {
    try {
        Remove-Item -LiteralPath $userDataRoot -Recurse -Force -ErrorAction Stop
        $removed += "user data"
    } catch {
        Write-Host "[error] Failed to remove user data: $_" -ForegroundColor Red
        $skipped += "user data (permission denied or in use)"
    }
} else {
    $skipped += "user data (not found)"
}

# Remove shortcut file
if (Test-Path -LiteralPath $shortcutPath) {
    try {
        Remove-Item -LiteralPath $shortcutPath -Force -ErrorAction Stop
        $removed += "Start Menu shortcut"
    } catch {
        Write-Host "[error] Failed to remove shortcut: $_" -ForegroundColor Red
        $skipped += "Start Menu shortcut (permission denied or in use)"
    }
} else {
    $skipped += "Start Menu shortcut (not found)"
}

# Final summary
Write-Host ""
Write-Host "=== Cleanup Summary ===" -ForegroundColor Cyan
Write-Host ""

if ($removed.Count -gt 0) {
    Write-Host "Removed:" -ForegroundColor Green
    $removed | ForEach-Object { Write-Host "  - $_" }
}

if ($skipped.Count -gt 0) {
    Write-Host "Skipped:" -ForegroundColor Yellow
    $skipped | ForEach-Object { Write-Host "  - $_" }
}

Write-Host ""
