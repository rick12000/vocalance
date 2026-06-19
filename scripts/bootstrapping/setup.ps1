#Requires -Version 5.1
<#
.SYNOPSIS
    Install Vocalance and create a Start Menu shortcut.

.NOTES
    UV (if missing): installed from Astral's official versioned release URL.

.PARAMETER SkipReinstallPrompt
    If Vocalance is already installed, skip the reinstall prompt and overwrite silently.
#>
param(
    [string] $Repo = 'rick12000/vocalance',
    [switch] $SkipReinstallPrompt
)

$ErrorActionPreference = 'Stop'

$VOCALANCE_VERSION = '0.0.1'
$UV_VERSION        = '0.11.22'

$INSTALL_ROOT = Join-Path $env:LOCALAPPDATA 'Programs\Vocalance'
$APP_DIR      = Join-Path $INSTALL_ROOT 'app'
$VENV_DIR     = Join-Path $INSTALL_ROOT 'env'
$PY_EXE       = Join-Path $VENV_DIR 'Scripts\python.exe'
$PYTHONW      = Join-Path $VENV_DIR 'Scripts\pythonw.exe'

function Test-YesAnswer {
    param([string] $Raw)
    if ($null -eq $Raw) { return $false }
    $t = $Raw.Trim().ToLowerInvariant()
    return ($t -eq 'yes' -or $t -eq 'y')
}

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts   = @()
    if ($machine) { $parts += $machine }
    if ($user)    { $parts += $user }
    if ($parts.Count -gt 0) { $env:Path = ($parts -join ';') }
}

Update-SessionPath
$env:Path = "$HOME\.local\bin;$env:Path"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    $answer = Read-Host 'UV is required but not found. Install it now? (yes/no)'
    if (-not (Test-YesAnswer $answer)) { exit 1 }
    powershell -ExecutionPolicy Bypass -c "irm https://releases.astral.sh/github/uv/releases/download/$UV_VERSION/uv-installer.ps1 | iex"
    Update-SessionPath
    $env:Path = "$HOME\.local\bin;$env:Path"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'UV is still not available. Open a new PowerShell window and run this script again.'
    exit 1
}

if (Test-Path -LiteralPath $INSTALL_ROOT) {
    if (-not $SkipReinstallPrompt) {
        $answer = Read-Host "Vocalance is already installed at $INSTALL_ROOT. Reinstall? (yes/no)"
        if (-not (Test-YesAnswer $answer)) { exit 1 }
    }
    Remove-Item -LiteralPath $INSTALL_ROOT -Recurse -Force
}

$zipUrl = "https://github.com/$Repo/releases/download/v$VOCALANCE_VERSION/vocalance-v$VOCALANCE_VERSION.zip"
$tmpZip = Join-Path $env:TEMP "vocalance-v$VOCALANCE_VERSION.zip"

Write-Host "Downloading Vocalance v$VOCALANCE_VERSION..."
Invoke-WebRequest -Uri $zipUrl -OutFile $tmpZip -UseBasicParsing
New-Item -ItemType Directory -Force -Path $APP_DIR | Out-Null
Expand-Archive -Path $tmpZip -DestinationPath $APP_DIR -Force
Remove-Item $tmpZip -Force

Write-Host "Creating virtual environment..."
uv venv --python 3.13.9 $VENV_DIR

$llmAnswer = Read-Host 'Enable LLM features? (requires ~2 GB and Microsoft C++ Build Tools) (yes/no)'

Write-Host 'Installing dependencies...'
$env:VIRTUAL_ENV            = $VENV_DIR
$env:UV_PROJECT_ENVIRONMENT = $VENV_DIR
Set-Location -LiteralPath $APP_DIR
if (Test-YesAnswer $llmAnswer) {
    uv sync --frozen --extra llm
} else {
    uv sync --frozen
}

$mainScript = Join-Path $APP_DIR 'vocalance.py'
$iconPath   = Join-Path $APP_DIR 'vocalance\app\assets\logo\icon.ico'

$shell        = New-Object -ComObject WScript.Shell
$programs     = [Environment]::GetFolderPath('Programs')
$shortcutPath = Join-Path $programs 'Vocalance.lnk'
$shortcut     = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath       = $PYTHONW
$shortcut.Arguments        = "`"$mainScript`""
$shortcut.WorkingDirectory = $APP_DIR
if (Test-Path -LiteralPath $iconPath) { $shortcut.IconLocation = "$iconPath,0" }
$shortcut.Save()

Write-Host "Setup complete. Launch Vocalance from the Start Menu."
