#Requires -Version 5.1
<#
.SYNOPSIS
    Uninstall Vocalance completely.

.DESCRIPTION
    Removes:
    - C:\Program Files\Vocalance\  (application files and virtual environment)
    - %APPDATA%\vocalance_voice_assistant_data\  (user data, settings, models, aliases)
    - Start Menu shortcut (Vocalance.lnk)

    Requires administrator privileges — will self-elevate via UAC if needed.
#>

$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

$INSTALL_ROOT = Join-Path $env:ProgramFiles 'Vocalance'
$USER_DATA    = Join-Path $env:APPDATA 'vocalance_voice_assistant_data'
$SHORTCUT     = Join-Path ([Environment]::GetFolderPath('Programs')) 'Vocalance.lnk'

$removed = @()
$skipped = @()

function Remove-IfExists {
    param([string] $Path, [string] $Label)
    if (Test-Path -LiteralPath $Path) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            $script:removed += $Label
        } catch {
            Write-Host "[error] Failed to remove ${Label}: $_" -ForegroundColor Red
            $script:skipped += "$Label (in use or permission denied)"
        }
    } else {
        $script:skipped += "$Label (not found)"
    }
}

Remove-IfExists -Path $INSTALL_ROOT -Label "application files ($INSTALL_ROOT)"
Remove-IfExists -Path $USER_DATA    -Label 'user data'
Remove-IfExists -Path $SHORTCUT     -Label 'Start Menu shortcut'

Write-Host ''
Write-Host '=== Uninstall Summary ===' -ForegroundColor Cyan
if ($removed.Count -gt 0) {
    Write-Host 'Removed:' -ForegroundColor Green
    $removed | ForEach-Object { Write-Host "  - $_" }
}
if ($skipped.Count -gt 0) {
    Write-Host 'Skipped:' -ForegroundColor Yellow
    $skipped | ForEach-Object { Write-Host "  - $_" }
}
Write-Host ''
