#Requires -Version 5.1
<#
.SYNOPSIS
    Install Vocalance and create a Start Menu shortcut.

.NOTES
    Installs entirely within the current user's AppData\Local\Programs directory.
    No administrator privileges are required or requested.

    UV is bundled as a binary inside the install tree — no system-wide UV
    installation occurs and no downloaded script is ever executed.

.PARAMETER SkipReinstallPrompt
    If Vocalance is already installed, skip the reinstall prompt and overwrite silently.
#>
param(
    [switch] $SkipReinstallPrompt
)

$ErrorActionPreference = 'Stop'

$VOCALANCE_VERSION = '0.0.1'
$VOCALANCE_REPO    = 'rick12000/vocalance'

$UV_VERSION    = '0.11.22'
$UV_ZIP_SHA256 = @{
    'x86_64'  = 'b56939bac92d29996d351647f7c6f15b31cc69cf952d06d136de3e1e62eb64d1'  # scripts/security/compute_uv_binary_hash.ps1 -Arch x86_64
    'aarch64' = '30fa01e0fc7c78bdaf6f369ebac401f22f0f865d650f0732a26f1df3e2c6971e'  # scripts/security/compute_uv_binary_hash.ps1 -Arch aarch64
}

$INSTALL_ROOT = Join-Path $env:LOCALAPPDATA 'Programs\Vocalance'
$TOOLS_DIR    = Join-Path $INSTALL_ROOT 'tools'
$APP_DIR      = Join-Path $INSTALL_ROOT 'app'
$VENV_DIR     = Join-Path $INSTALL_ROOT 'env'
$UV_EXE       = Join-Path $TOOLS_DIR 'uv.exe'
$PYTHONW      = Join-Path $VENV_DIR 'Scripts\pythonw.exe'

function Test-YesAnswer {
    param([string] $Raw)
    if ($null -eq $Raw) { return $false }
    return ($Raw.Trim().ToLowerInvariant() -in @('yes', 'y'))
}

function Get-VerifiedDownload {
    <#
    .SYNOPSIS
        Download a file and verify its SHA-256 before returning control.
        Deletes the file and throws if the hash does not match.
    #>
    param(
        [string] $Uri,
        [string] $OutPath,
        [string] $ExpectedSha256
    )
    Invoke-WebRequest -Uri $Uri -OutFile $OutPath -UseBasicParsing
    $actual = (Get-FileHash -Path $OutPath -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $ExpectedSha256.ToLower()) {
        Remove-Item -LiteralPath $OutPath -Force -ErrorAction SilentlyContinue
        throw "Integrity check failed for '$(Split-Path $OutPath -Leaf)'.`n  Expected : $ExpectedSha256`n  Computed : $actual"
    }
}

$arch     = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'aarch64' } else { 'x86_64' }
$stageDir = Join-Path $env:TEMP "VocalanceSetup-$([System.IO.Path]::GetRandomFileName())"
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

try {
    if (-not (Test-Path -LiteralPath $UV_EXE)) {
        $uvZipName  = "uv-$arch-pc-windows-msvc.zip"
        $uvZipUrl   = "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$uvZipName"
        $uvZipPath  = Join-Path $stageDir $uvZipName
        $uvExpected = $UV_ZIP_SHA256[$arch]
        if (-not $uvExpected) { throw "No UV hash defined for architecture: $arch" }

        Write-Host "Downloading uv $UV_VERSION ($arch)..."
        Get-VerifiedDownload -Uri $uvZipUrl -OutPath $uvZipPath -ExpectedSha256 $uvExpected

        $uvExtractDir = Join-Path $stageDir 'uv-extract'
        Expand-Archive -Path $uvZipPath -DestinationPath $uvExtractDir -Force

        $uvBinary = Get-ChildItem -Path $uvExtractDir -Filter 'uv.exe' -Recurse |
                        Select-Object -First 1 -ExpandProperty FullName
        if (-not $uvBinary) { throw 'uv.exe not found in downloaded archive.' }

        New-Item -ItemType Directory -Force -Path $TOOLS_DIR | Out-Null
        Copy-Item -LiteralPath $uvBinary -Destination $UV_EXE -Force
    }

    if (Test-Path -LiteralPath $APP_DIR) {
        if (-not $SkipReinstallPrompt) {
            $answer = Read-Host "Vocalance is already installed at $INSTALL_ROOT. Reinstall? (yes/no)"
            if (-not (Test-YesAnswer $answer)) { exit 0 }
        }
        Remove-Item -LiteralPath $APP_DIR -Recurse -Force
        if (Test-Path -LiteralPath $VENV_DIR) {
            Remove-Item -LiteralPath $VENV_DIR -Recurse -Force
        }
    }

    $zipName = "vocalance-v$VOCALANCE_VERSION.zip"
    $zipUrl  = "https://github.com/$VOCALANCE_REPO/releases/download/v$VOCALANCE_VERSION/$zipName"
    $zipPath = Join-Path $stageDir $zipName

    Write-Host "Downloading Vocalance v$VOCALANCE_VERSION..."
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing

    New-Item -ItemType Directory -Force -Path $APP_DIR | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $APP_DIR -Force

    Write-Host "Creating virtual environment..."
    & $UV_EXE venv --python 3.13.9 $VENV_DIR

    $llmAnswer = Read-Host 'Enable LLM features? (requires ~2 GB and Microsoft C++ Build Tools) (yes/no)'

    Write-Host 'Installing dependencies...'
    $env:VIRTUAL_ENV            = $VENV_DIR
    $env:UV_PROJECT_ENVIRONMENT = $VENV_DIR
    if (Test-YesAnswer $llmAnswer) {
        & $UV_EXE sync --directory $APP_DIR --frozen --extra llm
    } else {
        & $UV_EXE sync --directory $APP_DIR --frozen
    }

    $mainScript   = Join-Path $APP_DIR 'vocalance.py'
    $iconPath     = Join-Path $APP_DIR 'vocalance\app\assets\logo\icon.ico'
    $shell        = New-Object -ComObject WScript.Shell
    $shortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Vocalance.lnk'
    $shortcut     = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath       = $PYTHONW
    $shortcut.Arguments        = "`"$mainScript`""
    $shortcut.WorkingDirectory = $APP_DIR
    if (Test-Path -LiteralPath $iconPath) { $shortcut.IconLocation = "$iconPath,0" }
    $shortcut.Save()

    Write-Host "Setup complete. Launch Vocalance from the Start Menu."

} finally {
    if (Test-Path -LiteralPath $stageDir) {
        Remove-Item -LiteralPath $stageDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
