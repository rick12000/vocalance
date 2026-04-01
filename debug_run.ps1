$repoRoot = "C:\Users\ricca\vocalance\vocalance-prod"
$venvPath = "C:\Users\ricca\vocalance\vocalance_env"
$pyExe = Join-Path $venvPath 'Scripts\python.exe'
$mainScript = Join-Path $repoRoot 'vocalance.py'

if (-not (Test-Path $pyExe)) {
    Write-Host "Python not found at $pyExe"
    exit
}
if (-not (Test-Path $mainScript)) {
    Write-Host "Script not found at $mainScript"
    exit
}

Set-Location $repoRoot
& $pyExe $mainScript
