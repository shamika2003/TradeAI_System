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
    MAX_PORTFOLIO_RISK_PERCENT,
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
        # Never let broker lot rounding silently turn the calibrated budget into
        # a much larger trade. A 25% tolerance is reserved for volume steps.
        return min(
            float(MAX_ACTUAL_RISK_PERCENT),
            self.risk_percent * 1.25
        )

    @staticmethod
    def _position_value(position, *names, default=None):
        if isinstance(position, dict):
            for name in names:
                if name in position and position.get(name) is not None:
                    return position.get(name)
            return default
        for name in names:
            value = getattr(position, name, None)
            if value is not None:
                return value
        return default

    @staticmethod
    def _position_direction(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            if int(value) == 0:
                return "BUY"
            if int(value) == 1:
                return "SELL"
        text = str(value).strip().upper()
        if text in {"BUY", "0", "POSITION_TYPE_BUY", "ORDER_TYPE_BUY"}:
            return "BUY"
        if text in {"SELL", "1", "POSITION_TYPE_SELL", "ORDER_TYPE_SELL"}:
            return "SELL"
        return None

    def get_open_position_risk_money(self):
        """Conservative aggregate loss-at-current-SL for open positions.

        Risk is measured from the actual entry to the current stop. Stops that
        have already moved beyond entry contribute only commission, so profit
        protection immediately frees portfolio risk capacity.
        """
        total = 0.0
        for position in self.get_all_positions():
            try:
                symbol = str(self._position_value(position, "symbol", default="") or "")
                direction = self._position_direction(
                    self._position_value(position, "type", "direction")
                )
                entry = float(self._position_value(position, "entry_price", "price_open"))
                stop = float(self._position_value(position, "stop_loss", "sl"))
                lot = float(self._position_value(position, "volume", "lot"))
                if not symbol or direction not in {"BUY", "SELL"} or lot <= 0 or stop <= 0:
                    continue

                if direction == "BUY":
                    loss_distance = max(0.0, entry - stop)
                else:
                    loss_distance = max(0.0, stop - entry)

                pip = float(self.executor.pip_size(symbol))
                pip_value = float(self.executor.pip_value_per_lot(symbol, price=entry))
                price_risk = (loss_distance / pip) * pip_value * lot if pip > 0 else 0.0

                commission = 0.0
                exact_commission = self._position_value(position, "commission")
                if exact_commission is not None:
                    commission = abs(float(exact_commission))
                else:
                    getter = getattr(self.executor, "commission_for_lot", None)
                    if callable(getter):
                        commission = abs(float(getter(symbol, lot)))

                total += max(0.0, price_risk) + max(0.0, commission)
            except Exception:
                # Unknown positions fail conservatively at the admission layer
                # through the global position limit; do not crash live trading.
                continue
        return float(total)

    def get_portfolio_risk_headroom_money(self):
        balance = self._risk_balance()
        if balance <= 0:
            return 0.0
        cap = balance * float(MAX_PORTFOLIO_RISK_PERCENT) / 100.0
        return max(0.0, cap - self.get_open_position_risk_money())


    # =====================================================
    # RISK MONEY
    # =====================================================

    def get_allowed_risk_money(self):
        balance = self._risk_balance()
        if balance <= 0:
            return 0.0

        per_trade = balance * self.risk_percent / 100.0
        portfolio_headroom = self.get_portfolio_risk_headroom_money()
        allowed = min(per_trade, portfolio_headroom)

        if allowed <= 0:
            self._record_rejection("PORTFOLIO_RISK_LIMIT")

        return max(0.0, allowed)


    # =====================================================
    # CALCULATE LOT
    # =====================================================

    def calculate_lot(
            self,
            symbol,
            stop_loss_distance=None,
            price=None,
            direction=None,
            stop_loss=None
    ):

        """
        Calculate a broker-compatible lot from the NET loss at SL.

        Preferred Stage 2 call:
            price=requested/candle price
            direction=BUY/SELL
            stop_loss=absolute stop price

        In that mode the executor supplies its expected executable
        entry and commission, so spread/slippage/commission are part
        of the risk budget. The old distance-only path remains for
        compatibility with existing tests/tools.
        """

        try:

            net_risk_mode = (
                price is not None
                and direction is not None
                and stop_loss is not None
            )

            estimate = None

            if net_risk_mode:

                estimator = getattr(
                    self.executor,
                    "estimate_stop_loss_risk",
                    None
                )

                if not callable(estimator):
                    return None

                estimate = estimator(
                    symbol,
                    direction,
                    price,
                    stop_loss,
                    1.0
                )

                risk_per_lot = float(
                    estimate["net_risk"]
                )

                stop_pips = float(
                    estimate["stop_pips"]
                )

                pip_value_per_lot = float(
                    estimate["pip_value_per_lot"]
                )

                estimated_entry = float(
                    estimate["entry_price"]
                )

            else:

                if stop_loss_distance is None:
                    return None

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
                    stop_loss_distance
                    / pip
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

                risk_per_lot = (
                    stop_pips
                    * pip_value_per_lot
                )

                estimated_entry = (
                    float(price)
                    if price is not None
                    else 0.0
                )

            if (
                not math.isfinite(risk_per_lot)
                or
                risk_per_lot <= 0
            ):
                return None

            allowed_risk = (
                self.get_allowed_risk_money()
            )

            if allowed_risk <= 0:
                return None

            raw_lot = (
                allowed_risk
                / risk_per_lot
            )

            # -------------------------------------------------
            # BROKER MINIMUM-LOT COMPATIBILITY
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

            # Some executors/tests expose only normalize_lot() and do not
            # provide get_lot_limits(). If normalization rounds the request
            # upward, treat it as a minimum/step fallback for rejection
            # telemetry instead of mislabelling it as a generic risk breach.
            if (
                lot is not None
                and float(lot) > float(raw_lot) + 1e-12
            ):
                used_minimum_fallback = True

            if lot is None:

                self._record_rejection(
                    "LOT_BELOW_BROKER_MINIMUM"
                )

                log(
                    f"DEBUG | LOW BALANCE LOT REJECTED "
                    f"{symbol} "
                    f"risk_balance=${self._risk_balance():.2f} "
                    f"allowed_risk=${allowed_risk:.2f} "
                    f"stop={stop_pips:.2f}p "
                    f"raw_lot={raw_lot:.6f}"
                )

                return None

            # -------------------------------------------------
            # ACTUAL NET RISK AFTER BROKER NORMALIZATION
            # -------------------------------------------------

            if net_risk_mode:

                actual_estimate = (
                    self.executor.estimate_stop_loss_risk(
                        symbol,
                        direction,
                        price,
                        stop_loss,
                        lot
                    )
                )

                actual_risk = float(
                    actual_estimate["net_risk"]
                )

                price_risk = float(
                    actual_estimate["price_risk"]
                )

                commission = float(
                    actual_estimate["commission"]
                )

            else:

                actual_risk = (
                    risk_per_lot
                    * lot
                )

                price_risk = actual_risk
                commission = 0.0

            balance = self._risk_balance()

            if balance <= 0:
                return None

            actual_risk_percent = (
                actual_risk
                / balance
            ) * 100.0

            if (
                actual_risk_percent
                > self._max_actual_risk_percent()
            ):

                self._record_rejection(
                    "MIN_LOT_TOO_RISKY"
                    if used_minimum_fallback
                    else "RISK_AMOUNT_EXCEEDED"
                )

                log(
                    f"WARNING | LOT RISK REJECTED "
                    f"{symbol} "
                    f"lot={lot:.6f} "
                    f"net_risk=${actual_risk:.2f} "
                    f"risk_pct={actual_risk_percent:.3f}% "
                    f"max={self._max_actual_risk_percent():.3f}%"
                )

                return None

            if used_minimum_fallback:

                log(
                    f"INFO | BROKER MIN LOT SAFE "
                    f"{symbol} "
                    f"raw_lot={raw_lot:.6f} "
                    f"broker_lot={lot:.6f} "
                    f"net_risk=${actual_risk:.2f} "
                    f"risk_pct={actual_risk_percent:.3f}%"
                )

            log(
                f"DEBUG | LOT APPROVED "
                f"{symbol} "
                f"balance=${balance:.2f} "
                f"allowed=${allowed_risk:.2f} "
                f"entry={estimated_entry:.6f} "
                f"stop={stop_pips:.2f}p "
                f"raw={raw_lot:.6f} "
                f"final={lot:.6f} "
                f"price_risk=${price_risk:.2f} "
                f"commission=${commission:.2f} "
                f"net_risk=${actual_risk:.2f} "
                f"risk_pct={actual_risk_percent:.3f}%"
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
            stop_distance=None,
            price=None,
            direction=None,
            stop_loss=None
    ):

        try:

            net_risk_mode = (
                price is not None
                and direction is not None
                and stop_loss is not None
            )

            if net_risk_mode:

                estimate = (
                    self.executor.estimate_stop_loss_risk(
                        symbol,
                        direction,
                        price,
                        stop_loss,
                        lot
                    )
                )

                risk = float(
                    estimate["net_risk"]
                )

            else:

                if stop_distance is None:
                    return False

                pip = float(
                    self.executor.pip_size(
                        symbol
                    )
                )

                if pip <= 0:
                    return False

                stop_pips = (
                    float(stop_distance)
                    / pip
                )

                pip_value = float(
                    self.executor.pip_value_per_lot(
                        symbol,
                        price=price
                    )
                )

                risk = (
                    stop_pips
                    * pip_value
                    * float(lot)
                )

            balance = self._risk_balance()

            if balance <= 0:
                return False

            risk_percent = (
                risk
                / balance
            ) * 100.0

            if (
                risk_percent
                > self._max_actual_risk_percent()
            ):

                self._record_rejection(
                    "RISK_AMOUNT_EXCEEDED"
                )

                log(
                    f"WARNING | RISK REJECTED "
                    f"{symbol} "
                    f"lot={float(lot):.6f} "
                    f"net_risk=${risk:.2f} "
                    f"risk_pct={risk_percent:.3f}% "
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