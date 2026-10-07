$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host ""
Write-Host "==============================================="
Write-Host " TradeAI Local Read-Only API"
Write-Host " http://127.0.0.1:8765"
Write-Host "==============================================="
Write-Host ""

python -m TradeAI.api.tradeai_api

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "API exited with code $LASTEXITCODE"
    Read-Host "Press Enter to close"
}
