$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "==============================================="
Write-Host " TradeAI Precision Live Signal Engine"
Write-Host "==============================================="
Write-Host ""

python -m TradeAI.precision_live_engine

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Engine exited with code $LASTEXITCODE"
    Read-Host "Press Enter to close"
}
