# filename: core/risk_manager.py

import math
import time

from analytics.logger import log

from config.settings import (
    RISK_PERCENT,
    MAX_ACTUAL_RISK_PERCENT,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    USE_ATR_STOPS,
    MAX_DRAWDOWN_PERCENT,
    MAX_DAILY_LOSS_PERCENT,
)


class RiskManager:

    def __init__(
            self,
            executor,
            max_open_trades=1,
            max_total_trades=1,
            cooldown_seconds=300,
            risk_percent=None
    ):

        self.executor = executor

        self.max_open_trades = max_open_trades

        self.max_total_trades = max_total_trades

        self.cooldown_seconds = cooldown_seconds

        self.risk_percent = (
            float(RISK_PERCENT)
            if risk_percent is None
            else float(risk_percent)
        )

        if self.risk_percent <= 0:
            raise ValueError("risk_percent must be positive")

        self.last_trade_time = {}

        self.locked_symbols = set()

        log(
            f"INFO | Risk Manager initialized "
            f"risk={self.risk_percent:.3f}%"
        )


    # =====================================================
    # POSITION HELPERS
    # =====================================================

    def get_symbol_positions(
            self,
            symbol
    ):

        try:

            return self.executor.get_symbol_positions(
                symbol
            )

        except Exception as e:

            log(
                f"ERROR | Symbol positions {symbol}: {e}"
            )

            return []


    def get_all_positions(self):

        try:

            return self.executor.get_all_positions()

        except Exception as e:

            log(
                f"ERROR | All positions: {e}"
            )

            return []


    # =====================================================
    # TIME
    # =====================================================

    def _time(
            self,
            candle_time=None
    ):

        if candle_time is not None:

            try:

                return candle_time.timestamp()

            except Exception:

                pass

        return time.time()


    # =====================================================
    # RISK CAPITAL SOURCE
    # =====================================================

    def _risk_balance(self):
        getter = getattr(self.executor, "get_risk_balance", None)
        if callable(getter):
            return float(getter())
        return float(self.executor.get_balance())

    def _risk_equity(self):
        getter = getattr(self.executor, "get_risk_equity", None)
        if callable(getter):
            return float(getter())
        return float(self.executor.get_equity())

    def _risk_high_watermark(self):
        getter = getattr(self.executor, "get_risk_high_watermark", None)
        if callable(getter):
            return float(getter())
        return float(self.executor.initial_capital)

    def _record_rejection(self, reason):
        recorder = getattr(self.executor, "record_risk_rejection", None)
        if callable(recorder):
            try:
                recorder(reason)
            except Exception:
                pass

    def _max_actual_risk_percent(self):
        # Calibrated risk is 0.30%. Never let broker lot rounding silently
        # turn it into a 1%+ trade. A 25% tolerance is enough for volume steps.
        return min(
            float(MAX_ACTUAL_RISK_PERCENT),
            self.risk_percent * 1.25
        )


    # =====================================================
    # RISK MONEY
    # =====================================================

    def get_allowed_risk_money(self):

        balance = self._risk_balance()

        return (
            balance *
            self.risk_percent /
            100.0
        )


    # =====================================================
    # CALCULATE LOT
    # =====================================================

    def calculate_lot(
            self,
            symbol,
            stop_loss_distance,
            price=None
    ):

        try:

            stop_loss_distance = float(
                stop_loss_distance
            )

            if stop_loss_distance <= 0:
                return None

            pip = float(
                self.executor.pip_size(
                    symbol
                )
            )

            if pip <= 0:
                return None

            stop_pips = (
                stop_loss_distance /
                pip
            )

            if stop_pips <= 0:
                return None

            pip_value_per_lot = float(
                self.executor.pip_value_per_lot(
                    symbol,
                    price=price
                )
            )

            if pip_value_per_lot <= 0:
                return None

            allowed_risk = (
                self.get_allowed_risk_money()
            )

            raw_lot = (
                allowed_risk /
                (
                    stop_pips *
                    pip_value_per_lot
                )
            )

            # -------------------------------------------------
            # BROKER MINIMUM-LOT COMPATIBILITY
            # -------------------------------------------------
            #
            # Do not discard a valid signal only because the exact
            # calculated lot is slightly below the broker minimum.
            #
            # We may TRY the broker minimum lot, but it is approved
            # only after calculating the real stop-loss risk and
            # proving it remains inside the calibrated tolerance.
            #
            # This preserves safe opportunities without silently
            # turning a 0.30% strategy into a much larger-risk trade.
            # -------------------------------------------------

            lot_request = raw_lot
            used_minimum_fallback = False

            limits_getter = getattr(
                self.executor,
                "get_lot_limits",
                None
            )

            if callable(limits_getter):

                limits = limits_getter(
                    symbol
                )

                if limits is not None:

                    min_lot = float(
                        limits.get(
                            "min",
                            0.0
                        )
                    )

                    if (
                        min_lot > 0
                        and
                        raw_lot < min_lot
                    ):

                        lot_request = min_lot
                        used_minimum_fallback = True

            lot = self.executor.normalize_lot(
                symbol,
                lot_request
            )

            if lot is None:

                self._record_rejection(
                    "LOT_BELOW_BROKER_MINIMUM"
                )

                log(
                    f"DEBUG | LOW BALANCE LOT REJECTED "
                    f"{symbol} "
                    f"risk_balance="
                    f"${self._risk_balance():.2f} "
                    f"allowed_risk="
                    f"${allowed_risk:.2f} "
                    f"stop="
                    f"{stop_pips:.1f}p "
                    f"raw_lot="
                    f"{raw_lot:.6f}"
                )

                return None

            # -------------------------------------------------
            # ACTUAL RISK AFTER BROKER NORMALIZATION
            # -------------------------------------------------

            actual_risk = (
                stop_pips *
                pip_value_per_lot *
                lot
            )

            balance = self._risk_balance()

            if balance <= 0:
                return None

            actual_risk_percent = (
                actual_risk /
                balance
            ) * 100.0

            # -------------------------------------------------
            # SAFETY GATE
            # -------------------------------------------------

            if (
                actual_risk_percent >
                self._max_actual_risk_percent()
            ):

                self._record_rejection(
                    "MIN_LOT_TOO_RISKY"
                )

                log(
                    f"WARNING | MIN LOT TOO RISKY "
                    f"{symbol} "
                    f"lot={lot} "
                    f"actual_risk="
                    f"${actual_risk:.2f} "
                    f"actual_risk_pct="
                    f"{actual_risk_percent:.2f}% "
                    f"max="
                    f"{self._max_actual_risk_percent():.3f}% "
                    f"stop="
                    f"{stop_pips:.1f}p"
                )

                return None

            if used_minimum_fallback:

                log(
                    f"INFO | BROKER MIN LOT SAFE "
                    f"{symbol} "
                    f"raw_lot="
                    f"{raw_lot:.6f} "
                    f"broker_lot="
                    f"{lot:.6f} "
                    f"actual_risk="
                    f"${actual_risk:.2f} "
                    f"actual_risk_pct="
                    f"{actual_risk_percent:.3f}% "
                    f"max="
                    f"{self._max_actual_risk_percent():.3f}%"
                )

            log(
                f"DEBUG | LOT APPROVED "
                f"{symbol} "
                f"balance="
                f"${balance:.2f} "
                f"allowed="
                f"${allowed_risk:.2f} "
                f"stop="
                f"{stop_pips:.1f}p "
                f"raw="
                f"{raw_lot:.6f} "
                f"final="
                f"{lot:.6f} "
                f"actual_risk="
                f"${actual_risk:.2f} "
                f"actual_risk_pct="
                f"{actual_risk_percent:.3f}%"
            )

            return lot

        except Exception as e:

            log(
                f"ERROR | calculate_lot {symbol}: {e}"
            )

            return None


    # =====================================================
    # RISK CHECK
    # =====================================================

    def check_risk_amount(
            self,
            symbol,
            lot,
            stop_distance,
            price=None
    ):

        try:

            pip = float(
                self.executor.pip_size(
                    symbol
                )
            )

            if pip <= 0:
                return False


            stop_pips = (
                float(stop_distance) /
                pip
            )


            pip_value = float(
                self.executor.pip_value_per_lot(
                    symbol,
                    price=price
                )
            )


            risk = (
                stop_pips *
                pip_value *
                float(lot)
            )


            balance = self._risk_balance()


            if balance <= 0:
                return False


            risk_percent = (
                risk /
                balance
            ) * 100.0


            if (
                risk_percent >
                self._max_actual_risk_percent()
            ):

                self._record_rejection("RISK_AMOUNT_EXCEEDED")

                log(
                    f"WARNING | RISK REJECTED "
                    f"{symbol} "
                    f"lot={lot} "
                    f"risk=${risk:.2f} "
                    f"risk_pct={risk_percent:.2f}% "
                    f"max={self._max_actual_risk_percent():.3f}%"
                )

                return False


            return True


        except Exception as e:

            log(
                f"ERROR | check_risk_amount {symbol}: {e}"
            )

            return False


    # =====================================================
    # ACCOUNT HEALTH
    # =====================================================

    def check_account_health(self):

        try:

            equity = self._risk_equity()
            high_watermark = self._risk_high_watermark()

            if equity <= 0 or high_watermark <= 0:
                return False

            drawdown = max(
                0.0,
                (high_watermark - equity) / high_watermark
            )

            if drawdown >= (float(MAX_DRAWDOWN_PERCENT) / 100.0):
                self._record_rejection("MAX_DRAWDOWN_BLOCK")
                log(
                    f"WARNING | ACCOUNT PROTECTION "
                    f"peak_drawdown={drawdown * 100:.2f}% "
                    f"max={MAX_DRAWDOWN_PERCENT:.2f}%"
                )
                return False

            daily_checker = getattr(
                self.executor,
                "is_daily_loss_limit_hit",
                None
            )

            if callable(daily_checker) and daily_checker(
                float(MAX_DAILY_LOSS_PERCENT)
            ):
                self._record_rejection("DAILY_LOSS_BLOCK")
                log(
                    f"WARNING | DAILY LOSS PROTECTION "
                    f"max={MAX_DAILY_LOSS_PERCENT:.2f}%"
                )
                return False

            return True

        except Exception as e:

            log(
                f"ERROR | Account health: {e}"
            )

            return False


    # =====================================================
    # SL / TP
    # =====================================================

    def calculate_stop_targets(
            self,
            symbol,
            direction,
            price,
            df
    ):

        try:

            if not USE_ATR_STOPS:

                return None, None


            if (
                df is None
                or
                "atr" not in df.columns
            ):

                return None, None


            atr = float(
                df["atr"].iloc[-1]
            )


            if not math.isfinite(atr):

                return None, None


            if atr <= 0:

                return None, None


            sl_distance = (
                atr *
                float(ATR_SL_MULTIPLIER)
            )


            tp_distance = (
                atr *
                float(ATR_TP_MULTIPLIER)
            )


            if direction == "BUY":

                sl = (
                    float(price) -
                    sl_distance
                )

                tp = (
                    float(price) +
                    tp_distance
                )


            elif direction == "SELL":

                sl = (
                    float(price) +
                    sl_distance
                )

                tp = (
                    float(price) -
                    tp_distance
                )


            else:

                return None, None


            # -------------------------------------------------
            # Use executor price precision when possible.
            # -------------------------------------------------

            info = self.executor._get_symbol_info(
                symbol
            )


            if info is not None:

                digits = int(
                    getattr(
                        info,
                        "digits",
                        5
                    )
                )

            else:

                digits = (
                    3
                    if "JPY" in symbol.upper()
                    else 5
                )


            return (
                round(sl, digits),
                round(tp, digits)
            )


        except Exception as e:

            log(
                f"ERROR | SL/TP {symbol}: {e}"
            )

            return None, None


    # =====================================================
    # CAN TRADE
    # =====================================================

    def can_trade(
            self,
            symbol,
            candle_time=None
    ):

        now = self._time(
            candle_time
        )


        # -------------------------------------------------
        # Cooldown
        # -------------------------------------------------

        last = self.last_trade_time.get(
            symbol,
            0
        )


        if (
            now -
            last
        ) < self.cooldown_seconds:

            return False


        # -------------------------------------------------
        # Account protection
        # -------------------------------------------------

        if not self.check_account_health():

            return False


        # -------------------------------------------------
        # Symbol lock
        # -------------------------------------------------

        if symbol in self.locked_symbols:

            return False


        # -------------------------------------------------
        # One position per symbol
        # -------------------------------------------------

        if len(
            self.get_symbol_positions(symbol)
        ) > 0:

            return False


        # -------------------------------------------------
        # Global position limit
        # -------------------------------------------------

        positions = self.get_all_positions()


        if len(positions) >= self.max_open_trades:

            return False


        if len(positions) >= self.max_total_trades:

            return False


        return True


    # =====================================================
    # REGISTER OPEN
    # =====================================================

    def register_trade_open(
            self,
            symbol,
            candle_time=None
    ):

        self.last_trade_time[symbol] = (
            self._time(
                candle_time
            )
        )

        self.locked_symbols.add(
            symbol
        )


    # =====================================================
    # REGISTER CLOSE
    # =====================================================

    def register_trade_close(
            self,
            symbol,
            candle_time=None
    ):

        self.locked_symbols.discard(
            symbol
        )

        self.last_trade_time[symbol] = (
            self._time(
                candle_time
            )
        )


    # =====================================================
    # CLEANUP
    # =====================================================

    def cleanup_closed_symbols(self):

        active = set()

        for position in (
            self.get_all_positions()
        ):

            if isinstance(
                    position,
                    dict
            ):

                symbol = position.get(
                    "symbol"
                )

            else:

                symbol = getattr(
                    position,
                    "symbol",
                    None
                )


            if symbol:

                active.add(
                    symbol
                )


        for symbol in list(
                self.locked_symbols
        ):

            if symbol not in active:

                self.locked_symbols.discard(
                    symbol
                )


    # =====================================================
    # SIGNAL FILTER
    # =====================================================

    def allow_signal(
            self,
            signal_value,
            threshold
    ):

        if signal_value is None:

            return False

        return (
            abs(
                float(signal_value)
            ) >=
            float(threshold)
        )
