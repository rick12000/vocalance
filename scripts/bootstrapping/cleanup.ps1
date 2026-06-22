#Requires -Version 5.1
<#
.SYNOPSIS
    Uninstall Vocalance completely.

.DESCRIPTION
    Removes:
    - %LOCALAPPDATA%\Programs\Vocalance\  (application, virtual environment, bundled tools)
    - %APPDATA%\Vocalance\  (user data, settings, aliases)
    - Start Menu shortcut (Vocalance.lnk)

    No administrator privileges are required.
#>

$ErrorActionPreference = 'Stop'

$INSTALL_ROOT = Join-Path $env:LOCALAPPDATA 'Programs\Vocalance'
$USER_DATA    = Join-Path $env:APPDATA 'Vocalance'
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
