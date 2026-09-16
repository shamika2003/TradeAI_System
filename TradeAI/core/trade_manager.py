# filename: core/trade_manager.py

import csv
import os
from datetime import datetime, timezone

from analytics.logger import log
from config.settings import (
    BREAK_EVEN_TRIGGER_PIPS,
    MAX_HOLD_BARS,
    MAX_HOLD_MINUTES,
    TRAILING_TRIGGER_PIPS,
    USE_BREAK_EVEN,
    USE_MAX_HOLD,
    USE_TRAILING_STOP,
)


class TradeManager:
    def __init__(self, executor):
        self.executor = executor
        self.active_trades = {}
        self.file = "data/trade_manager_positions.csv"
        os.makedirs("data", exist_ok=True)
        self._save_positions()
        log("INFO | Trade Manager initialized")

    def _save_positions(self):
        with open(self.file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "symbol", "direction", "entry", "lot",
                "stop_loss", "take_profit", "status"
            ])
            for trade in self.active_trades.values():
                writer.writerow([
                    datetime.now(), trade["symbol"], trade["direction"],
                    trade["entry"], trade["lot"], trade["sl"], trade["tp"], "OPEN"
                ])

    @staticmethod
    def _coerce_datetime(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            # MT5 position.time is Unix seconds.
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            pass
        try:
            import pandas as pd
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts.to_pydatetime()
        except Exception:
            return None

    def register_trade(self, position):
        try:
            if position is None:
                return False

            if isinstance(position, dict):
                trade = {
                    "symbol": position.get("symbol"),
                    "direction": position.get("type"),
                    "entry": position.get("entry_price"),
                    "lot": position.get("volume"),
                    "sl": position.get("stop_loss"),
                    "tp": position.get("take_profit"),
                    "open_time": self._coerce_datetime(position.get("open_time")),
                    "bars_held": 0,
                    "be_done": False,
                }
            else:
                trade = {
                    "symbol": position.symbol,
                    "direction": position.type,
                    "entry": position.price_open,
                    "lot": position.volume,
                    "sl": position.sl,
                    "tp": position.tp,
                    "open_time": self._coerce_datetime(getattr(position, "time", None)),
                    "bars_held": 0,
                    "be_done": False,
                }

            symbol = trade["symbol"]
            if symbol is None:
                return False
            self.active_trades[symbol] = trade
            self._save_positions()
            log(f"INFO | Trade Manager tracking {symbol}")
            return True
        except Exception as e:
            log(f"ERROR | Register trade {e}")
            return False

    def remove_trade(self, symbol):
        if symbol in self.active_trades:
            del self.active_trades[symbol]
            self._save_positions()
            log(f"INFO | Trade Manager removed {symbol}")
            return True
        return False

    def sync_positions(self):
        positions = self.executor.get_all_positions()
        active_symbols = set()
        for position in positions:
            symbol = position.get("symbol") if isinstance(position, dict) else position.symbol
            if symbol:
                active_symbols.add(symbol)
                if symbol not in self.active_trades:
                    self.register_trade(position)
        for symbol in list(self.active_trades.keys()):
            if symbol not in active_symbols:
                self.remove_trade(symbol)

    def _max_hold_reached(self, trade, position, candle_time=None):
        if not USE_MAX_HOLD:
            return False

        # Paper/backtest update() is called once after each execution candle.
        if isinstance(position, dict):
            trade["bars_held"] = int(trade.get("bars_held", 0)) + 1
            return trade["bars_held"] >= int(MAX_HOLD_BARS)

        # LIVE: use elapsed wall-clock minutes because MT5 gives broker position time.
        opened = self._coerce_datetime(trade.get("open_time"))
        now = self._coerce_datetime(candle_time) or datetime.now()
        if opened is None:
            return False
        return (now - opened).total_seconds() >= float(MAX_HOLD_MINUTES) * 60.0

    def update(self, symbol, atr=None, candle_time=None):
        if symbol not in self.active_trades:
            return False

        trade = self.active_trades[symbol]
        position = self.executor.get_position(symbol)
        if position is None:
            return False

        current = position.get("current_price") if isinstance(position, dict) else position.price_current
        if current is None:
            return False

        current = float(current)
        entry = float(trade["entry"])
        direction = trade["direction"]
        pip = float(self.executor.pip_size(symbol))
        if pip <= 0:
            return False

        if direction == "BUY":
            profit_pips = (current - entry) / pip
        else:
            profit_pips = (entry - current) / pip

        log(f"INFO | MANAGER {symbol} profit={profit_pips:.1f} pips")

        # Maximum holding period is checked after the candle's SL/TP processing.
        if self._max_hold_reached(trade, position, candle_time=candle_time):
            log(f"INFO | MAX HOLD EXIT {symbol} bars={trade.get('bars_held', 0)}")
            return bool(
                self.executor.close_position(
                    symbol,
                    reason="MAX_HOLD",
                    candle_time=candle_time,
                )
            )

        if USE_BREAK_EVEN and profit_pips >= float(BREAK_EVEN_TRIGGER_PIPS) and not trade["be_done"]:
            new_sl = entry
            log(
                f"INFO | BREAK EVEN triggered {symbol} "
                f"profit={profit_pips:.1f} pips entry={entry:.5f} new_sl={new_sl:.5f}"
            )
            success = self.executor.modify_position(symbol, stop_loss=new_sl)
            if success:
                trade["sl"] = new_sl
                trade["be_done"] = True
                log(f"INFO | BREAK EVEN SUCCESS {symbol} SL={new_sl:.5f}")
            else:
                log(f"ERROR | BREAK EVEN FAILED {symbol} requested_sl={new_sl:.5f}")

        if USE_TRAILING_STOP and profit_pips >= float(TRAILING_TRIGGER_PIPS):
            if atr is None:
                log(f"WARNING | Missing ATR trailing {symbol}")
            else:
                atr = float(atr)
                if atr > 0:
                    if direction == "BUY":
                        new_sl = current - atr
                        if new_sl > float(trade["sl"]):
                            if self.executor.modify_position(symbol, stop_loss=new_sl):
                                trade["sl"] = new_sl
                    else:
                        new_sl = current + atr
                        if new_sl < float(trade["sl"]):
                            if self.executor.modify_position(symbol, stop_loss=new_sl):
                                trade["sl"] = new_sl

        self._save_positions()
        return True
