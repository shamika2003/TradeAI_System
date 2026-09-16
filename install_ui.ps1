$ErrorActionPreference = "Stop"
Write-Host "Installing TradeAI desktop UI dependencies..." -ForegroundColor Cyan
python -m pip install --upgrade pip
python -m pip install -r requirements-ui.txt
Write-Host "Done. Run: python TradeAI\dashboard_app.py" -ForegroundColor Green
