$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "HEALTH"
curl.exe -s http://127.0.0.1:8765/health
Write-Host ""
Write-Host ""

Write-Host "STATUS"
curl.exe -s http://127.0.0.1:8765/status
Write-Host ""
Write-Host ""

Write-Host "POSITIONS"
curl.exe -s http://127.0.0.1:8765/positions
Write-Host ""
