#Requires -Version 5.1
<#
.SYNOPSIS
    Compute the SHA-256 hash of the UV binary zip for a given version and architecture.

.DESCRIPTION
    Downloads uv-{arch}-pc-windows-msvc.zip for the specified UV version to a temp
    file, computes its SHA-256, prints it, and removes the temp file.

    Run this whenever you bump $UV_VERSION in setup.ps1 and paste the output hashes
    into the $UV_ZIP_SHA256 hashtable (one entry per architecture).

.PARAMETER UvVersion
    The UV release version to fetch (e.g. "0.11.22"). Defaults to the version
    currently hardcoded in setup.ps1.

.PARAMETER Arch
    Target architecture: "x86_64", "aarch64", or "all" (default) to hash both.

.EXAMPLE
    .\compute_uv_binary_hash.ps1
    .\compute_uv_binary_hash.ps1 -UvVersion 0.11.23
    .\compute_uv_binary_hash.ps1 -UvVersion 0.11.22 -Arch aarch64
#>
param(
    [string] $UvVersion = '0.11.22',
    [ValidateSet('x86_64', 'aarch64', 'all')]
    [string] $Arch = 'all'
)

$ErrorActionPreference = 'Stop'

$archs = if ($Arch -eq 'all') { @('x86_64', 'aarch64') } else { @($Arch) }
$results = [ordered]@{}

foreach ($a in $archs) {
    $zipName = "uv-$a-pc-windows-msvc.zip"
    $zipUrl  = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$zipName"
    $tmpFile = Join-Path $env:TEMP "uv-$UvVersion-$a-hash-check.zip"

    try {
        Write-Host "Downloading $zipName for uv $UvVersion from:"
        Write-Host "  $zipUrl"
        Write-Host ''

        Invoke-WebRequest -Uri $zipUrl -OutFile $tmpFile -UseBasicParsing

        $hash = (Get-FileHash -Path $tmpFile -Algorithm SHA256).Hash.ToLower()
        $results[$a] = $hash

        Write-Host "SHA-256 ($a): $hash"
        Write-Host ''
    } finally {
        if (Test-Path -LiteralPath $tmpFile) {
            Remove-Item -LiteralPath $tmpFile -Force
        }
    }
}

Write-Host "Paste into setup.ps1 `$UV_ZIP_SHA256:"
foreach ($a in $results.Keys) {
    Write-Host "    '$a' = '$($results[$a])'"
}
