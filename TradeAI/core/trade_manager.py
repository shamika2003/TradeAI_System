# filename: core/trade_manager.py

import csv
import math
import os
from datetime import datetime, timezone

from analytics.logger import log
from shared.tradeai_core.management_policy import opportunity_thesis_exit
from config.settings import (
    AI_DEFENSIVE_EXIT_ADVERSE_R,
    AI_DEFENSIVE_EXIT_MIN_BARS,
    AI_DEFENSIVE_EXIT_SIGNAL_MULTIPLIER,
    BREAK_EVEN_BUFFER_PIPS,
    BREAK_EVEN_MIN_PROFIT_MONEY,
    BREAK_EVEN_TRIGGER_R,
    MAX_HOLD_BARS,
    MAX_HOLD_MINUTES,
    PROFIT_LOCK_R,
    PROFIT_LOCK_TRIGGER_R,
    TRAILING_ATR_MULTIPLIER,
    TRAILING_TRIGGER_R,
    USE_AI_DEFENSIVE_EXIT,
    USE_BREAK_EVEN,
    USE_MAX_HOLD,
    USE_PROFIT_LOCK,
    USE_TRAILING_STOP,
    USE_OPPORTUNITY_THESIS_EXIT,
    THESIS_EXIT_MIN_BARS,
    THESIS_EXIT_ADVERSE_R,
    THESIS_STALE_BARS,
    THESIS_STALE_MAX_R,
    THESIS_OWN_PROBABILITY_FRACTION,
    THESIS_OWN_EV_FLOOR,
    THESIS_OPPOSITE_EV_MARGIN,
    THESIS_OPPOSITE_PROBABILITY_MARGIN,
    THESIS_SETUP_FRACTION,
)


class TradeManager:
    """
    Production trade lifecycle manager.

    Stage 3 changes:
        - management thresholds are measured in R, not fixed pips
        - break-even includes configured commission + a small pip buffer
        - trailing starts from an R threshold and uses ATR distance
        - losing trades can be closed early only on a strong opposite model edge
        - MT5 numeric position directions are normalized correctly

    R is based on the ORIGINAL entry-to-stop price distance. It never shrinks
    after stop modifications, so management decisions remain stable.
    """

    def __init__(self, executor):
        self.executor = executor
        self.active_trades = {}
        self.file = "data/trade_manager_positions.csv"
        os.makedirs("data", exist_ok=True)
        self._save_positions()
        log("INFO | Trade Manager initialized | R-based management enabled")

    # =====================================================
    # PERSISTED STATUS SNAPSHOT
    # =====================================================

    def _save_positions(self):
        with open(self.file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time",
                "symbol",
                "direction",
                "entry",
                "lot",
                "initial_sl",
                "stop_loss",
                "take_profit",
                "bars_held",
                "be_done",
                "status",
            ])
            for trade in self.active_trades.values():
                writer.writerow([
                    datetime.now(),
                    trade["symbol"],
                    trade["direction"],
                    trade["entry"],
                    trade["lot"],
                    trade.get("initial_sl"),
                    trade["sl"],
                    trade["tp"],
                    trade.get("bars_held", 0),
                    trade.get("be_done", False),
                    "OPEN",
                ])

    # =====================================================
    # NORMALIZATION HELPERS
    # =====================================================

    @staticmethod
    def _coerce_datetime(value):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            # MT5 position.time is Unix seconds.
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(
                    float(value),
                    tz=timezone.utc,
                ).replace(tzinfo=None)
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

    @staticmethod
    def _normalize_direction(value):
        """
        Normalize both TradeAI strings and MT5 numeric position types.

        MetaTrader 5 uses 0=BUY and 1=SELL for position.type. The previous
        manager treated any non-"BUY" value as SELL, which could invert live
        management logic for BUY positions.
        """
        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            numeric = int(value)
            if numeric == 0:
                return "BUY"
            if numeric == 1:
                return "SELL"

        text = str(value).strip().upper()
        if text in {"BUY", "0", "POSITION_TYPE_BUY", "ORDER_TYPE_BUY"}:
            return "BUY"
        if text in {"SELL", "1", "POSITION_TYPE_SELL", "ORDER_TYPE_SELL"}:
            return "SELL"
        return None

    @staticmethod
    def _finite_float(value, default=None):
        try:
            number = float(value)
            if math.isfinite(number):
                return number
        except (TypeError, ValueError):
            pass
        return default

    # =====================================================
    # POSITION REGISTRATION / SYNC
    # =====================================================

    def register_trade(self, position):
        try:
            if position is None:
                return False

            if isinstance(position, dict):
                direction = self._normalize_direction(position.get("type"))
                entry = self._finite_float(position.get("entry_price"))
                lot = self._finite_float(position.get("volume"))
                stop = self._finite_float(position.get("stop_loss"))
                take_profit = self._finite_float(position.get("take_profit"))
                trade = {
                    "symbol": position.get("symbol"),
                    "direction": direction,
                    "entry": entry,
                    "lot": lot,
                    "sl": stop,
                    "initial_sl": stop,
                    "tp": take_profit,
                    "open_time": self._coerce_datetime(position.get("open_time")),
                    "bars_held": 0,
                    "be_done": False,
                }
            else:
                direction = self._normalize_direction(getattr(position, "type", None))
                entry = self._finite_float(getattr(position, "price_open", None))
                lot = self._finite_float(getattr(position, "volume", None))
                stop = self._finite_float(getattr(position, "sl", None))
                take_profit = self._finite_float(getattr(position, "tp", None))
                trade = {
                    "symbol": getattr(position, "symbol", None),
                    "direction": direction,
                    "entry": entry,
                    "lot": lot,
                    "sl": stop,
                    "initial_sl": stop,
                    "tp": take_profit,
                    "open_time": self._coerce_datetime(getattr(position, "time", None)),
                    "bars_held": 0,
                    "be_done": False,
                }

            symbol = trade["symbol"]
            if (
                symbol is None
                or trade["direction"] not in ("BUY", "SELL")
                or trade["entry"] is None
                or trade["lot"] is None
                or trade["lot"] <= 0
                or trade["sl"] is None
            ):
                log(f"ERROR | Trade Manager invalid position registration: {symbol}")
                return False

            initial_risk_price = abs(trade["entry"] - trade["initial_sl"])
            if not math.isfinite(initial_risk_price) or initial_risk_price <= 0:
                log(f"ERROR | Trade Manager invalid initial risk {symbol}")
                return False

            trade["initial_risk_price"] = initial_risk_price
            self.active_trades[symbol] = trade
            self._save_positions()
            log(
                f"INFO | Trade Manager tracking {symbol} "
                f"direction={trade['direction']} initial_risk={initial_risk_price:.8f}"
            )
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
            symbol = position.get("symbol") if isinstance(position, dict) else getattr(position, "symbol", None)
            if symbol:
                active_symbols.add(symbol)
                if symbol not in self.active_trades:
                    self.register_trade(position)
        for symbol in list(self.active_trades.keys()):
            if symbol not in active_symbols:
                self.remove_trade(symbol)

    # =====================================================
    # R / COST HELPERS
    # =====================================================

    @staticmethod
    def _r_multiple(trade, current):
        risk_distance = float(trade.get("initial_risk_price", 0.0) or 0.0)
        if risk_distance <= 0:
            return 0.0

        entry = float(trade["entry"])
        if trade["direction"] == "BUY":
            return (float(current) - entry) / risk_distance
        return (entry - float(current)) / risk_distance

    def _commission_money(self, symbol, trade, position):
        if isinstance(position, dict):
            exact = self._finite_float(position.get("commission"))
            if exact is not None and exact >= 0:
                return exact

        try:
            return abs(
                float(
                    self.executor.commission_for_lot(
                        symbol,
                        float(trade["lot"]),
                    )
                )
            )
        except Exception:
            return 0.0

    def _cost_adjusted_break_even(self, symbol, trade, position, pip):
        """
        Return a stop price that covers the configured commission plus a small
        safety buffer. Entry spread/slippage is already embedded in entry_price.
        """
        entry = float(trade["entry"])
        lot = float(trade["lot"])
        commission = self._commission_money(symbol, trade, position)

        commission_pips = 0.0
        min_profit_pips = 0.0
        try:
            pip_value = float(
                self.executor.pip_value_per_lot(
                    symbol,
                    price=entry,
                )
            )
            per_pip = pip_value * lot
            if math.isfinite(per_pip) and per_pip > 0:
                commission_pips = commission / per_pip
                min_profit_pips = max(0.0, float(BREAK_EVEN_MIN_PROFIT_MONEY)) / per_pip
        except Exception:
            commission_pips = 0.0
            min_profit_pips = 0.0

        # The extra component is whichever is larger: the ordinary pip buffer
        # or enough price movement to survive 2-decimal P/L rounding with a
        # small positive cash result. This prevents nominal BE trades from
        # appearing as $0.00 after commission.
        positive_buffer_pips = max(
            max(0.0, float(BREAK_EVEN_BUFFER_PIPS)),
            max(0.0, min_profit_pips),
        )
        offset_pips = max(0.0, commission_pips) + positive_buffer_pips
        offset_price = offset_pips * pip

        if trade["direction"] == "BUY":
            return entry + offset_price
        return entry - offset_price

    # =====================================================
    # DEFENSIVE AI EXIT
    # =====================================================

    @staticmethod
    def _prediction_number(prediction, key, default=0.0):
        try:
            value = float(prediction.get(key, default))
            if math.isfinite(value):
                return value
        except Exception:
            pass
        return float(default)

    def _opposite_model_edge(self, trade, prediction):
        if not isinstance(prediction, dict):
            return False
        if not bool(prediction.get("policy_enabled", True)):
            return False

        signal = self._prediction_number(prediction, "signal", 0.0)
        confidence = self._prediction_number(prediction, "confidence", 0.0)
        min_confidence = self._prediction_number(prediction, "min_confidence", 1.0)
        signal_threshold = abs(
            self._prediction_number(prediction, "signal_threshold", 1.0)
        )
        required_signal = signal_threshold * max(
            0.0,
            float(AI_DEFENSIVE_EXIT_SIGNAL_MULTIPLIER),
        )

        if confidence < min_confidence:
            return False

        if trade["direction"] == "BUY":
            return signal <= -required_signal
        return signal >= required_signal

    def _should_defensive_exit(self, trade, r_multiple, prediction):
        # Stage 9: prefer the richer directional opportunity thesis. The rule
        # is shared with validation so early exits are not a runtime-only trick.
        if USE_OPPORTUNITY_THESIS_EXIT and isinstance(prediction, dict) and (
            "buy_expected_r" in prediction or "sell_expected_r" in prediction
        ):
            should_exit, reason = opportunity_thesis_exit(
                direction=trade["direction"],
                r_multiple=r_multiple,
                bars_held=int(trade.get("bars_held", 0)),
                prediction=prediction,
                min_bars=THESIS_EXIT_MIN_BARS,
                adverse_r=THESIS_EXIT_ADVERSE_R,
                stale_bars=THESIS_STALE_BARS,
                stale_max_r=THESIS_STALE_MAX_R,
                own_probability_fraction=THESIS_OWN_PROBABILITY_FRACTION,
                own_ev_floor=THESIS_OWN_EV_FLOOR,
                opposite_ev_margin=THESIS_OPPOSITE_EV_MARGIN,
                opposite_probability_margin=THESIS_OPPOSITE_PROBABILITY_MARGIN,
                setup_fraction=THESIS_SETUP_FRACTION,
            )
            trade["last_thesis_exit_reason"] = reason
            return should_exit

        if not USE_AI_DEFENSIVE_EXIT:
            return False
        if int(trade.get("bars_held", 0)) < int(AI_DEFENSIVE_EXIT_MIN_BARS):
            return False
        if r_multiple > -abs(float(AI_DEFENSIVE_EXIT_ADVERSE_R)):
            return False
        return self._opposite_model_edge(trade, prediction)

    # =====================================================
    # MAX HOLD
    # =====================================================

    def _max_hold_reached(self, trade, position, candle_time=None):
        if not USE_MAX_HOLD:
            return False

        # PAPER/BACKTEST update() is called once at the decision point before
        # the next execution candle. One increment therefore represents one
        # completed held candle.
        if isinstance(position, dict):
            trade["bars_held"] = int(trade.get("bars_held", 0)) + 1
            return trade["bars_held"] >= int(MAX_HOLD_BARS)

        # LIVE: use elapsed wall-clock minutes because MT5 gives broker time.
        opened = self._coerce_datetime(trade.get("open_time"))
        now = self._coerce_datetime(candle_time) or datetime.now()
        if opened is None:
            return False
        return (now - opened).total_seconds() >= float(MAX_HOLD_MINUTES) * 60.0

    # =====================================================
    # MAIN UPDATE
    # =====================================================

    def update(self, symbol, atr=None, candle_time=None, prediction=None):
        if symbol not in self.active_trades:
            return False

        trade = self.active_trades[symbol]
        position = self.executor.get_position(symbol)
        if position is None:
            return False

        current = (
            position.get("current_price")
            if isinstance(position, dict)
            else getattr(position, "price_current", None)
        )
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

        r_multiple = self._r_multiple(trade, current)

        log(
            f"INFO | MANAGER {symbol} direction={direction} "
            f"profit={profit_pips:.1f}p R={r_multiple:+.2f} "
            f"bars={trade.get('bars_held', 0)}"
        )

        # -------------------------------------------------
        # Holding period
        # -------------------------------------------------
        if self._max_hold_reached(trade, position, candle_time=candle_time):
            log(f"INFO | MAX HOLD EXIT {symbol} bars={trade.get('bars_held', 0)}")
            return bool(
                self.executor.close_position(
                    symbol,
                    reason="MAX_HOLD",
                    candle_time=candle_time,
                )
            )

        # -------------------------------------------------
        # Defensive losing-trade exit
        # -------------------------------------------------
        if self._should_defensive_exit(trade, r_multiple, prediction):
            signal = self._prediction_number(prediction, "signal", 0.0)
            confidence = self._prediction_number(prediction, "confidence", 0.0)
            log(
                f"INFO | AI DEFENSIVE EXIT {symbol} R={r_multiple:+.2f} "
                f"signal={signal:+.4f} confidence={confidence:.4f}"
            )
            return bool(
                self.executor.close_position(
                    symbol,
                    reason=str(trade.get("last_thesis_exit_reason") or "AI_DEFENSIVE_EXIT"),
                    candle_time=candle_time,
                )
            )

        # -------------------------------------------------
        # Cost-adjusted break-even
        # -------------------------------------------------
        cost_be = self._cost_adjusted_break_even(
            symbol,
            trade,
            position,
            pip,
        )

        if (
            USE_BREAK_EVEN
            and r_multiple >= float(BREAK_EVEN_TRIGGER_R)
            and not trade["be_done"]
        ):
            new_sl = cost_be
            log(
                f"INFO | BREAK EVEN triggered {symbol} "
                f"R={r_multiple:.2f} entry={entry:.5f} new_sl={new_sl:.5f}"
            )
            success = self.executor.modify_position(symbol, stop_loss=new_sl)
            if success:
                trade["sl"] = new_sl
                trade["be_done"] = True
                log(f"INFO | BREAK EVEN SUCCESS {symbol} SL={new_sl:.5f}")
            else:
                log(f"WARNING | BREAK EVEN RETRY LATER {symbol} requested_sl={new_sl:.5f}")

        # -------------------------------------------------
        # Positive-R profit lock
        # -------------------------------------------------
        if USE_PROFIT_LOCK and r_multiple >= float(PROFIT_LOCK_TRIGGER_R):
            risk_distance = float(trade.get("initial_risk_price", 0.0) or 0.0)
            if risk_distance > 0:
                lock_distance = risk_distance * max(0.0, float(PROFIT_LOCK_R))
                current_sl = float(trade["sl"])
                if direction == "BUY":
                    new_sl = max(entry + lock_distance, cost_be)
                    if new_sl > current_sl:
                        if self.executor.modify_position(symbol, stop_loss=new_sl):
                            trade["sl"] = new_sl
                            trade["be_done"] = True
                            log(
                                f"INFO | PROFIT LOCK {symbol} "
                                f"R={r_multiple:.2f} locked={float(PROFIT_LOCK_R):.2f}R "
                                f"SL={new_sl:.5f}"
                            )
                else:
                    new_sl = min(entry - lock_distance, cost_be)
                    if new_sl < current_sl:
                        if self.executor.modify_position(symbol, stop_loss=new_sl):
                            trade["sl"] = new_sl
                            trade["be_done"] = True
                            log(
                                f"INFO | PROFIT LOCK {symbol} "
                                f"R={r_multiple:.2f} locked={float(PROFIT_LOCK_R):.2f}R "
                                f"SL={new_sl:.5f}"
                            )

        # -------------------------------------------------
        # R-triggered ATR trailing
        # -------------------------------------------------
        if USE_TRAILING_STOP and r_multiple >= float(TRAILING_TRIGGER_R):
            if atr is None:
                log(f"WARNING | Missing ATR trailing {symbol}")
            else:
                atr_value = self._finite_float(atr)
                if atr_value is not None and atr_value > 0:
                    trail_distance = atr_value * max(0.0, float(TRAILING_ATR_MULTIPLIER))
                    current_sl = float(trade["sl"])

                    if direction == "BUY":
                        new_sl = current - trail_distance
                        if trade.get("be_done"):
                            new_sl = max(new_sl, cost_be)
                        if new_sl > current_sl:
                            if self.executor.modify_position(symbol, stop_loss=new_sl):
                                trade["sl"] = new_sl
                                log(
                                    f"INFO | TRAIL SUCCESS {symbol} "
                                    f"R={r_multiple:.2f} SL={new_sl:.5f}"
                                )
                    else:
                        new_sl = current + trail_distance
                        if trade.get("be_done"):
                            new_sl = min(new_sl, cost_be)
                        if new_sl < current_sl:
                            if self.executor.modify_position(symbol, stop_loss=new_sl):
                                trade["sl"] = new_sl
                                log(
                                    f"INFO | TRAIL SUCCESS {symbol} "
                                    f"R={r_multiple:.2f} SL={new_sl:.5f}"
                                )

        self._save_positions()
        return True
