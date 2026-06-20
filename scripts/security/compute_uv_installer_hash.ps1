#Requires -Version 5.1
<#
.SYNOPSIS
    Compute the SHA-256 hash of the UV installer script for a given UV version.

.DESCRIPTION
    Downloads uv-installer.ps1 for the specified UV version to a temp file,
    computes its SHA-256, prints it, and removes the temp file.

    Use this whenever you bump $UV_VERSION in setup.ps1. After running, paste
    the output hash into the $UV_INSTALLER_SHA256 variable in setup.ps1.

.PARAMETER UvVersion
    The UV release version to fetch (e.g. "0.11.22"). Defaults to the version
    currently hardcoded in setup.ps1.

.EXAMPLE
    .\compute_uv_installer_hash.ps1
    .\compute_uv_installer_hash.ps1 -UvVersion 0.11.23
#>
param(
    [string] $UvVersion = "0.11.22"
)

$ErrorActionPreference = 'Stop'

$installerUrl  = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-installer.ps1"
$tmpFile       = Join-Path $env:TEMP "uv-installer-$UvVersion-hash-check.ps1"

try {
    Write-Host "Downloading UV $UvVersion installer from:"
    Write-Host "  $installerUrl"
    Write-Host ""

    Invoke-WebRequest -Uri $installerUrl -OutFile $tmpFile -UseBasicParsing

    $hash = (Get-FileHash -Path $tmpFile -Algorithm SHA256).Hash.ToLower()

    Write-Host "SHA-256: $hash"
    Write-Host ""
    Write-Host "Paste into setup.ps1:"
    Write-Host "  `$UV_INSTALLER_SHA256 = '$hash'"
} finally {
    if (Test-Path -LiteralPath $tmpFile) {
        Remove-Item $tmpFile -Force
    }
}
