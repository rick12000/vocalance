#Requires -Version 5.1
<#
.SYNOPSIS
    Remove the Vocalance installation created by setup.ps1.

.DESCRIPTION
    This script removes Vocalance components. It works best if run from the same directory
    where you originally ran setup.ps1, but will still remove your user data and Start Menu
    shortcut from their fixed Windows locations if you're not sure.

    If you confirm you're running from the original installation directory, the script will
    also remove the repository clone and Python virtual environment.
#>

$ErrorActionPreference = 'Stop'

function Test-YesAnswer {
    param([string] $Raw)
    if ($null -eq $Raw) { return $false }
    $t = $Raw.Trim().ToLowerInvariant()
    return ($t -eq 'yes' -or $t -eq 'y')
}

function Test-NoAnswer {
    param([string] $Raw)
    if ($null -eq $Raw) { return $false }
    $t = $Raw.Trim().ToLowerInvariant()
    return ($t -eq 'no' -or $t -eq 'n')
}

function Write-Header {
    param([string] $Text)
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Remove-PathWithConfirm {
    param(
        [string] $Path,
        [string] $Label,
        [string] $Description
    )

    Write-Host ""
    Write-Host "  $Label" -ForegroundColor White
    Write-Host "  $Path" -ForegroundColor DarkGray
    Write-Host "  ($Description)" -ForegroundColor DarkGray

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "  [not found]" -ForegroundColor DarkGray
        return
    }

    $answer = Read-Host "  Remove? (yes/no)"
    if (Test-YesAnswer $answer) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            Write-Host "  [removed]" -ForegroundColor Green
        } catch {
            Write-Host "  [error] Could not remove: $_" -ForegroundColor Red
            Write-Host "  Close Vocalance if running, then try again or delete manually." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [skipped]" -ForegroundColor DarkGray
    }
}

# ──────────────────────────────────────────────────────────────────────────────
Write-Header "Vocalance Uninstaller"

Write-Host "Current directory: $(Get-Location)" -ForegroundColor Cyan
Write-Host ""

$locationChoice = Read-Host "Running from the same directory where you ran setup.ps1? (yes/no/not sure)"

$isOriginalLocation = $false
if (Test-YesAnswer $locationChoice) {
    $isOriginalLocation = $true
    Write-Host "  [confirmed] All steps will be performed." -ForegroundColor Green
} else {
    Write-Host "  [proceeding with safe steps only]" -ForegroundColor Yellow
}

Write-Host ""

# ──────────────────────────────────────────────────────────────────────────────
Write-Header "Step 1 of 4 — User Data"

$userDataRoot = Join-Path $env:APPDATA 'vocalance_voice_assistant_data'
Remove-PathWithConfirm `
    -Path $userDataRoot `
    -Label "User data directory" `
    -Description "marks, settings, custom commands, aliases, prompts, LLM models"

# ──────────────────────────────────────────────────────────────────────────────
Write-Header "Step 2 of 4 — Start Menu Shortcut"

$programs     = [Environment]::GetFolderPath('Programs')
$shortcutPath = Join-Path $programs 'Vocalance.lnk'

Remove-PathWithConfirm `
    -Path $shortcutPath `
    -Label "Start Menu shortcut" `
    -Description "Vocalance.lnk"

# ──────────────────────────────────────────────────────────────────────────────
if ($isOriginalLocation) {
    Write-Header "Step 3 of 4 — Python Virtual Environment"

    $cwd       = (Get-Location).Path
    $venvPath  = Join-Path $cwd 'vocalance_env'

    Remove-PathWithConfirm `
        -Path $venvPath `
        -Label "Virtual environment" `
        -Description "Python packages and dependencies"

    # ──────────────────────────────────────────────────────────────────────────

    Write-Header "Step 4 of 4 — Repository Clone"

    $cloneDir = Join-Path $cwd 'vocalance-prod'

    Remove-PathWithConfirm `
        -Path $cloneDir `
        -Label "Repository" `
        -Description "Source code and bundled assets"

} else {
    Write-Host ""
    Write-Host "Steps 3 & 4 skipped (location not confirmed)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "To remove the repository and virtual environment:" -ForegroundColor Yellow
    Write-Host "  1. Run this script again from the original installation directory" -ForegroundColor DarkGray
    Write-Host "  2. Or delete manually: vocalance-prod and vocalance_env folders" -ForegroundColor DarkGray
}

# ──────────────────────────────────────────────────────────────────────────────
Write-Header "Done"

Write-Host "Uninstall complete." -ForegroundColor Green
Write-Host ""
