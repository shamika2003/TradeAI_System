from __future__ import annotations

import sys
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parent.parent
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from TradeAI.config.settings import (  # noqa: E402
    BROKER_PROFILE_PATH,
    COMMISSION_PER_LOT,
    SIMULATED_SLIPPAGE_PIPS,
    SYMBOLS,
)
from shared.tradeai_core.broker_profile import capture_mt5_broker_profile  # noqa: E402


def run() -> None:
    try:
        import MetaTrader5 as mt5
    except Exception as exc:
        raise RuntimeError("MetaTrader5 Python package is not available") from exc

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"MT5 account unavailable: {mt5.last_error()}")

        for symbol in SYMBOLS:
            info = mt5.symbol_info(symbol)
            if info is None:
                raise RuntimeError(f"Broker does not expose required symbol: {symbol}")
            if not getattr(info, "visible", True):
                mt5.symbol_select(symbol, True)

        profile = capture_mt5_broker_profile(
            mt5,
            SYMBOLS,
            path=BROKER_PROFILE_PATH,
            commission_per_lot=COMMISSION_PER_LOT,
            slippage_pips=SIMULATED_SLIPPAGE_PIPS,
        )

        missing = [s for s in SYMBOLS if s not in profile.get("symbols", {})]
        if missing:
            raise RuntimeError("Profile capture missing symbols: " + ", ".join(missing))

        print(f"BROKER PROFILE SAVED: {BROKER_PROFILE_PATH}")
        for symbol in SYMBOLS:
            spec = profile["symbols"][symbol]
            print(
                f"{symbol}: min={spec['volume_min']} step={spec['volume_step']} "
                f"spread={spec['spread_pips']:.3f}p tick_value={spec['trade_tick_value']}"
            )
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"BROKER PROFILE CAPTURE FAILED: {exc}")
        raise SystemExit(1)
