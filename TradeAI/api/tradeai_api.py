from __future__ import annotations

from fastapi import FastAPI, Query

from TradeAI.api.state_service import (
    get_account,
    get_health,
    get_performance,
    get_positions,
    get_signals,
    get_status,
    get_symbols,
    get_trades,
)


app = FastAPI(
    title="TradeAI Local API",
    description=(
        "Read-only localhost bridge for NIRA and the active TradeAI precision demo engine."
    ),
    version="4.0.0",
)


@app.get("/")
def root() -> dict:
    return {
        "service": "TradeAI Local API",
        "version": "4.0.0",
        "engine": "PRECISION_DEMO",
        "read_only": True,
        "endpoints": [
            "/health",
            "/status",
            "/positions",
            "/account",
            "/signals",
            "/signals?symbol=EURUSD",
            "/trades",
            "/performance",
            "/symbols",
            "/docs",
        ],
    }


@app.get("/health")
def health() -> dict:
    return get_health()


@app.get("/status")
def status() -> dict:
    return get_status()


@app.get("/positions")
def positions() -> dict:
    return get_positions()


@app.get("/account")
def account() -> dict:
    return get_account()


@app.get("/signals")
def signals(symbol: str | None = None) -> dict:
    return get_signals(symbol)


@app.get("/symbols")
def symbols() -> dict:
    return get_symbols()


@app.get("/trades")
def trades(
    limit: int = Query(default=50, ge=1, le=500),
    symbol: str | None = None,
    include_open: bool = True,
) -> dict:
    return get_trades(
        limit=limit,
        symbol=symbol,
        include_open=include_open,
    )


@app.get("/performance")
def performance() -> dict:
    return get_performance()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "TradeAI.api.tradeai_api:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        access_log=True,
    )


if __name__ == "__main__":
    main()
