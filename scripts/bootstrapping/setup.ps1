#Requires -Version 5.1
<#
.SYNOPSIS
    Clone Vocalance (if needed), install UV/venv/deps, create Start Menu shortcut.

.NOTES
    Run from the directory that should contain the vocalance-prod folder (its parent). After setup, launch from Start Menu.

    Git must already be installed (PATH). Official Windows download: https://git-scm.com/download/win

    UV (if missing): optional install via Astral’s official script:
    https://docs.astral.sh/uv/installation/

.PARAMETER SkipVenvOverwritePrompt
    If an existing vocalance_env is present, do not prompt: keep it and only run uv sync (useful when re-running after updates).
#>
param(
    [string] $CloneUrl = 'https://github.com/rick12000/vocalance.git',
    [switch] $SkipVenvOverwritePrompt
)

$ErrorActionPreference = 'Stop'

function Test-YesAnswer {
    param([string] $Raw)
    if ($null -eq $Raw) { return $false }
    $t = $Raw.Trim().ToLowerInvariant()
    return ($t -eq 'yes' -or $t -eq 'y')
}

function Update-SessionPathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($machine) { $parts += $machine }
    if ($user) { $parts += $user }
    if ($parts.Count -gt 0) {
        $env:Path = ($parts -join ';')
    }
}

function Prepend-GitCmdIfPresent {
    $gitCmd = Join-Path $env:ProgramFiles 'Git\cmd'
    if (Test-Path -LiteralPath (Join-Path $gitCmd 'git.exe')) {
        $env:Path = "$gitCmd;$env:Path"
    }
}

Update-SessionPathFromRegistry
Prepend-GitCmdIfPresent

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git is required but was not found on PATH. Install the latest Git for Windows from https://git-scm.com/download/win then open a new PowerShell window and run this script again.'
}

Update-SessionPathFromRegistry
$env:Path = "$HOME\.local\bin;$env:Path"
Prepend-GitCmdIfPresent

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ''
    Write-Host 'UV was not found.'
    Write-Host 'The installer is Astral’s published script (same approach as https://docs.astral.sh/uv/installation/).'
    $answer = Read-Host 'Install UV now? Type yes or no'
    if (-not (Test-YesAnswer $answer)) {
        Write-Host 'Setup aborted (UV is required for dependencies and the virtual environment).'
        exit 1
    }
    Write-Host 'Installing UV...'
    powershell.exe -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    Update-SessionPathFromRegistry
    $env:Path = "$HOME\.local\bin;$env:Path"
    Prepend-GitCmdIfPresent
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'UV is still not available in this session. Close this window, open a new PowerShell, and run this script again.'
    exit 1
}

$cwd = (Get-Location).Path
$cloneDir = Join-Path $cwd 'vocalance-prod'

if (Test-Path -LiteralPath $cloneDir) {
    Write-Host ''
    Write-Host "A folder named vocalance-prod already exists at:"
    Write-Host "  $cloneDir"
    $answer = Read-Host 'Replace it with a fresh git clone? Type yes or no'
    if (-not (Test-YesAnswer $answer)) {
        Write-Host 'Setup aborted.'
        exit 1
    }
    Write-Host 'Removing existing folder...'
    Remove-Item -LiteralPath $cloneDir -Recurse -Force
    Write-Host "Cloning misc-optimizations branch into $cloneDir ..."
    git clone --branch misc-optimizations --single-branch $CloneUrl $cloneDir
} else {
    Write-Host "Cloning misc-optimizations branch into $cloneDir ..."
    git clone --branch misc-optimizations --single-branch $CloneUrl $cloneDir
}

$repoRoot = $cloneDir
Set-Location -LiteralPath $repoRoot

Update-SessionPathFromRegistry
$env:Path = "$HOME\.local\bin;$env:Path"
Prepend-GitCmdIfPresent

$parentDir = Split-Path -Parent $repoRoot
$venvPath = Join-Path $parentDir 'vocalance_env'
$pyExe = Join-Path $venvPath 'Scripts\python.exe'
$pythonw = Join-Path $venvPath 'Scripts\pythonw.exe'

$venvDirExists = Test-Path -LiteralPath $venvPath -PathType Container
$venvUsable = Test-Path -LiteralPath $pyExe

if ($venvDirExists -and $venvUsable) {
    if (-not $SkipVenvOverwritePrompt) {
        Write-Host ''
        Write-Host "A virtual environment named vocalance_env already exists at:"
        Write-Host "  $venvPath"
        $answer = Read-Host 'Overwrite it (delete and recreate)? Type yes or no'
        if (-not (Test-YesAnswer $answer)) {
            Write-Host 'Setup aborted.'
            exit 1
        }
        Write-Host 'Removing existing virtual environment...'
        Remove-Item -LiteralPath $venvPath -Recurse -Force
    }
} elseif ($venvDirExists -and -not $venvUsable) {
    if ($SkipVenvOverwritePrompt) {
        Write-Host "Removing incomplete vocalance_env at $venvPath ..."
        Remove-Item -LiteralPath $venvPath -Recurse -Force -ErrorAction Stop
    } else {
        Write-Host ''
        Write-Host "A folder named vocalance_env already exists at:"
        Write-Host "  $venvPath"
        Write-Host 'It does not look like a complete virtual environment.'
        $answer = Read-Host 'Remove it and create a fresh one? Type yes or no'
        if (-not (Test-YesAnswer $answer)) {
            Write-Host 'Setup aborted.'
            exit 1
        }
        Write-Host 'Removing folder...'
        Remove-Item -LiteralPath $venvPath -Recurse -Force -ErrorAction Stop
    }
}

if (-not (Test-Path -LiteralPath $pyExe)) {
    Write-Host "Creating virtual environment at $venvPath ..."
    uv venv --python 3.13.9 $venvPath
}

Write-Host 'Installing dependencies (uv sync)...'
uv sync --python $pyExe

$mainScript = Join-Path $repoRoot 'vocalance.py'
$iconPath = Join-Path $repoRoot 'vocalance\app\assets\logo\icon.ico'

$shell = New-Object -ComObject WScript.Shell
$programs = [Environment]::GetFolderPath('Programs')
$shortcutPath = Join-Path $programs 'Vocalance.lnk'
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "`"$mainScript`""
$shortcut.WorkingDirectory = $repoRoot
if (Test-Path -LiteralPath $iconPath) {
    $shortcut.IconLocation = "$iconPath,0"
}
$shortcut.Save()

Write-Host ''
Write-Host "Setup finished. Open Vocalance from the Start Menu shortcut (Vocalance)."
