TradeAI Local API v3 - Precision Demo Engine

This version matches the engine currently running with:
    python -m TradeAI.precision_demo_engine

It reads:
    artifacts/reports/precision_live_state.json
    artifacts/reports/precision_demo_execution_state.json

It uses precision-demo magic:
    26061055

Important:
The precision demo engine does NOT publish the newer RuntimeControlPlane heartbeat.
Therefore v3 reports ACTIVE_TELEMETRY / STALE_TELEMETRY rather than incorrectly
calling the engine OFFLINE. Telemetry normally updates when a new closed M5 bar
is processed.

Upgrade:
1. Stop ONLY the API server with Ctrl+C.
2. Do NOT stop the precision demo engine.
3. Put this ZIP in TradeAI_System root.
4. Run:
       Expand-Archive -Path .\TradeAI_API_PrecisionDemo_v3.zip -DestinationPath . -Force
5. Start:
       py -m TradeAI.api.tradeai_api

Test:
       curl.exe http://127.0.0.1:8765/health
       curl.exe http://127.0.0.1:8765/status
       curl.exe http://127.0.0.1:8765/positions
