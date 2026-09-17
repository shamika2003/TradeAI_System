# filename: execution/base_executor.py

from abc import ABC, abstractmethod
import math

from shared.tradeai_core.production_economics import fallback_pip_value_per_lot
from shared.tradeai_core.broker_profile import (
    get_execution_costs,
    load_broker_profile,
    symbol_info_from_profile,
)


class BaseExecutor(ABC):

    """
    Common execution interface for:

        PAPER
        BACKTEST
        LIVE MT5

    The strategy must not care which executor is being used.

    All executors expose:

        account state
        position state
        symbol specifications
        pip size
        pip value
        trade execution
        position modification
        trade closing
    """

    def __init__(self, capital: float = 10.0):

        self.initial_capital = float(capital)

        self.balance = float(capital)

        self.equity = float(capital)

        # Active positions.
        #
        # PAPER/BACKTEST:
        #     actual simulated positions
        #
        # LIVE:
        #     local snapshots
        #
        self.positions = {}

        # Completed trades.
        self.trade_history = []

        # TradeManager reference.
        self.trade_manager = None

        # Optional MT5 broker snapshot shared with PAPER/BACKTEST.
        self._broker_profile = None
        self.reload_broker_profile()


    # =====================================================
    # TRADE MANAGER
    # =====================================================

    def register_trade_manager(self, manager):

        self.trade_manager = manager


    def unregister_trade_manager(self):

        self.trade_manager = None


    # =====================================================
    # POSITION API
    # =====================================================

    def has_open_trade(self, symbol):

        return self.get_position(symbol) is not None


    def get_position(self, symbol):

        return self.positions.get(symbol)


    def get_symbol_positions(self, symbol):

        position = self.get_position(symbol)

        if position is None:
            return []

        return [position]


    def get_all_positions(self):

        return list(
            self.positions.values()
        )


    def get_open_count(self):

        return len(
            self.positions
        )


    # =====================================================
    # ACCOUNT API
    # =====================================================

    def get_balance(self):

        return float(
            self.balance
        )


    def get_equity(self):

        return float(
            self.equity
        )


    def reset(self):

        self.balance = float(
            self.initial_capital
        )

        self.equity = float(
            self.initial_capital
        )

        self.positions.clear()

        self.trade_history.clear()


    # =====================================================
    # SYMBOL INFORMATION
    # =====================================================

    def reload_broker_profile(self):

        try:
            from config.settings import (
                BROKER_PROFILE_PATH,
                USE_BROKER_PROFILE,
            )

            if not USE_BROKER_PROFILE:
                self._broker_profile = None
                return None

            self._broker_profile = load_broker_profile(
                BROKER_PROFILE_PATH
            )

        except Exception:
            self._broker_profile = None

        return self._broker_profile


    def _get_symbol_info(self, symbol):

        """
        LIVE MT5 overrides this.

        PAPER/BACKTEST consumes the last broker snapshot so
        digits, tick value and volume rules match the broker.
        """

        return symbol_info_from_profile(
            self._broker_profile,
            symbol,
        )


    def _get_symbol_tick(self, symbol):

        """
        LIVE MT5 overrides this when a current bid/ask tick
        is available. PAPER/BACKTEST normally estimates
        execution from the requested historical price.
        """

        return None


    # =====================================================
    # EXECUTION / RISK ESTIMATION
    # =====================================================

    def estimate_entry_price(
            self,
            symbol,
            direction,
            requested_price
    ):

        """
        Best pre-order estimate of the executable entry price.

        LIVE uses the current broker ask/bid when available.
        PAPER/BACKTEST overrides this to include the exact
        simulated spread and slippage used by open_trade().
        """

        requested = float(requested_price)
        direction = str(direction).upper()

        if direction not in ("BUY", "SELL"):
            raise ValueError(
                f"Invalid direction: {direction}"
            )

        try:
            tick = self._get_symbol_tick(
                symbol
            )

            if tick is not None:
                if direction == "BUY":
                    value = float(
                        getattr(tick, "ask", 0.0)
                    )
                else:
                    value = float(
                        getattr(tick, "bid", 0.0)
                    )

                if math.isfinite(value) and value > 0:
                    return value

        except Exception:
            pass

        return requested


    def estimate_stop_loss_risk(
            self,
            symbol,
            direction,
            requested_price,
            stop_loss,
            lot
    ):

        """
        Estimate NET monetary loss if the stop is hit.

        The estimate intentionally includes:
            - expected executable entry price
            - entry-to-stop price loss
            - configured/broker commission estimate

        This is the quantity RiskManager must size against,
        rather than the candle-close-to-stop distance.
        """

        lot = float(lot)

        if not math.isfinite(lot) or lot <= 0:
            raise ValueError("lot must be positive")

        entry = float(
            self.estimate_entry_price(
                symbol,
                direction,
                requested_price
            )
        )

        stop = self.normalize_price(
            symbol,
            stop_loss,
        )

        pip = float(
            self.pip_size(symbol)
        )

        if not math.isfinite(pip) or pip <= 0:
            raise ValueError("invalid pip size")

        stop_pips = abs(
            entry - stop
        ) / pip

        pip_value = float(
            self.pip_value_per_lot(
                symbol,
                price=entry
            )
        )

        if (
            not math.isfinite(pip_value)
            or
            pip_value <= 0
        ):
            raise ValueError(
                "invalid pip value"
            )

        price_risk = (
            stop_pips
            * pip_value
            * lot
        )

        commission = abs(
            float(
                self.commission_for_lot(
                    symbol,
                    lot
                )
            )
        )

        net_risk = (
            price_risk
            + commission
        )

        return {
            "entry_price": entry,
            "stop_loss": stop,
            "stop_pips": stop_pips,
            "pip_value_per_lot": pip_value,
            "price_risk": price_risk,
            "commission": commission,
            "net_risk": net_risk,
        }


    def execution_costs(self, symbol):

        from config.settings import (
            DEFAULT_SPREAD_PIPS,
            SIMULATED_SLIPPAGE_PIPS,
            COMMISSION_PER_LOT,
        )

        return get_execution_costs(
            self._broker_profile,
            symbol,
            default_spread_pips=DEFAULT_SPREAD_PIPS,
            default_slippage_pips=SIMULATED_SLIPPAGE_PIPS,
            default_commission_per_lot=COMMISSION_PER_LOT,
        )


    def spread_pips(self, symbol):
        return float(
            self.execution_costs(symbol)["spread_pips"]
        )


    def slippage_pips(self, symbol):
        return float(
            self.execution_costs(symbol)["slippage_pips"]
        )


    def minimum_stop_distance(self, symbol):

        info = self._get_symbol_info(symbol)
        if info is None:
            return 0.0

        try:
            point = float(getattr(info, "point", 0.0))
            stops = int(getattr(info, "trade_stops_level", 0) or 0)
            freeze = int(getattr(info, "trade_freeze_level", 0) or 0)
            if point <= 0:
                return 0.0
            return max(stops, freeze) * point
        except Exception:
            return 0.0


    def normalize_price(self, symbol, price):

        value = float(price)
        info = self._get_symbol_info(symbol)
        if info is None:
            return value

        try:
            digits = int(getattr(info, "digits", 0) or 0)
            tick_size = float(
                getattr(info, "trade_tick_size", 0.0)
                or getattr(info, "point", 0.0)
            )
            if tick_size > 0:
                value = round(value / tick_size) * tick_size
            if digits > 0:
                value = round(value, digits)
        except Exception:
            pass

        return value


    # =====================================================
    # PIP SIZE
    # =====================================================

    def pip_size(self, symbol):

        """
        Standard FX pip size.

        LIVE:
            Derived from MT5 symbol digits.

        PAPER/BACKTEST:
            Uses standard FX fallback.
        """

        try:

            info = self._get_symbol_info(
                symbol
            )

            if info is not None:

                point = float(
                    getattr(
                        info,
                        "point",
                        0.0
                    )
                )

                digits = int(
                    getattr(
                        info,
                        "digits",
                        0
                    )
                )

                if point > 0:

                    if digits in (3, 5):

                        return point * 10.0

                    return point

        except Exception:

            pass


        # ---------------------------------------------
        # Standard FX fallback
        # ---------------------------------------------

        symbol_upper = symbol.upper()

        if "JPY" in symbol_upper:

            return 0.01

        return 0.0001


    # =====================================================
    # TICK SIZE
    # =====================================================

    def tick_size(self, symbol):

        try:

            info = self._get_symbol_info(
                symbol
            )

            if info is not None:

                value = float(
                    getattr(
                        info,
                        "trade_tick_size",
                        0.0
                    )
                )

                if value > 0:

                    return value

        except Exception:

            pass


        return self.pip_size(symbol)


    # =====================================================
    # TICK VALUE
    # =====================================================

    def tick_value_per_lot(
            self,
            symbol,
            price=None
    ):

        try:

            info = self._get_symbol_info(
                symbol
            )

            if info is not None:

                value = float(
                    getattr(
                        info,
                        "trade_tick_value",
                        0.0
                    )
                )

                if value > 0:

                    return value

        except Exception:

            pass


        return self.pip_value_per_lot(
            symbol,
            price
        ) / (
            self.pip_size(symbol) /
            self.tick_size(symbol)
        )


    # =====================================================
    # PIP VALUE
    # =====================================================

    def pip_value_per_lot(
            self,
            symbol,
            price=None
    ):

        """
        Monetary value of one FX pip for one standard lot.

        LIVE:
            Uses actual MT5 tick size + tick value.

        PAPER/BACKTEST:
            Uses the standard FX approximation.

        NOTE:
        This is deliberately centralized so that Paper and
        Live use the same formula.
        """

        try:

            info = self._get_symbol_info(
                symbol
            )

            if info is not None:

                tick_size = float(
                    getattr(
                        info,
                        "trade_tick_size",
                        0.0
                    )
                )

                tick_value = float(
                    getattr(
                        info,
                        "trade_tick_value",
                        0.0
                    )
                )

                if (
                    tick_size > 0
                    and
                    tick_value > 0
                ):

                    pip = self.pip_size(
                        symbol
                    )

                    return (
                        tick_value *
                        (
                            pip /
                            tick_size
                        )
                    )

        except Exception:

            pass


        # ---------------------------------------------
        # Historical/PAPER USD-account fallback.
        #
        # Unlike the old fixed $10 fallback, this correctly
        # converts USD-base pairs such as USDJPY and USDCNH
        # using the current price when it is available.
        # ---------------------------------------------

        return fallback_pip_value_per_lot(
            symbol,
            price=price
        )


    # =====================================================
    # VOLUME LIMITS
    # =====================================================

    def get_lot_limits(self, symbol):

        """
        Return:

            {
                "min": ...,
                "max": ...,
                "step": ...
            }

        LIVE overrides this with MT5 broker data.

        PAPER/BACKTEST uses configuration fallback.
        """

        info = self._get_symbol_info(symbol)

        if info is not None:
            try:
                minimum = float(getattr(info, "volume_min", 0.0))
                maximum = float(getattr(info, "volume_max", 0.0))
                step = float(getattr(info, "volume_step", 0.0))
                if minimum > 0 and maximum > 0 and step > 0:
                    return {
                        "min": minimum,
                        "max": maximum,
                        "step": step,
                    }
            except Exception:
                pass

        from config.settings import (
            MIN_LOT,
            MAX_LOT,
            LOT_STEP
        )

        return {
            "min": float(MIN_LOT),
            "max": float(MAX_LOT),
            "step": float(LOT_STEP)
        }


    # =====================================================
    # LOT NORMALIZATION
    # =====================================================

    def normalize_lot(
            self,
            symbol,
            lot
    ):

        try:

            if lot is None:
                return None

            lot = float(lot)

            if not math.isfinite(lot):
                return None

            if lot <= 0:
                return None


            limits = self.get_lot_limits(
                symbol
            )

            if limits is None:
                return None


            minimum = float(
                limits["min"]
            )

            maximum = float(
                limits["max"]
            )

            step = float(
                limits["step"]
            )


            if (
                minimum <= 0
                or
                maximum <= 0
                or
                step <= 0
            ):

                return None


            # Never exceed maximum.

            lot = min(
                lot,
                maximum
            )


            # IMPORTANT:
            #
            # We do NOT increase a lot below the minimum.
            #
            # This prevents a $10 account from accidentally
            # becoming a 10% / 20% / 30% risk account.

            if lot < minimum:

                return None


            # Align downward.

            steps = math.floor(
                (
                    lot -
                    minimum
                ) /
                step
                +
                1e-9
            )


            normalized = (
                minimum +
                steps * step
            )


            normalized = max(
                minimum,
                min(
                    normalized,
                    maximum
                )
            )


            # Determine decimals.

            text = (
                f"{step:.10f}"
                .rstrip("0")
            )

            if "." in text:

                decimals = len(
                    text.split(".")[1]
                )

            else:

                decimals = 0


            normalized = round(
                normalized,
                decimals
            )


            if normalized < minimum:
                return None

            if normalized > maximum:
                return None


            return normalized


        except Exception:

            return None


    # =====================================================
    # COMMISSION
    # =====================================================

    def commission_for_lot(
            self,
            symbol,
            lot
    ):

        rate = float(
            self.execution_costs(symbol)["commission_per_lot"]
        )

        return (
            rate *
            float(lot)
        )


    # =====================================================
    # EMERGENCY CLOSE
    # =====================================================

    def close_all_positions(self):

        for symbol in list(
                self.positions.keys()
        ):

            try:

                self.close_position(
                    symbol,
                    reason="SYSTEM_CLOSE"
                )

            except Exception:

                pass


    # =====================================================
    # BROKER SYNC
    # =====================================================

    def check_closed_trades(self):

        return None


    # =====================================================
    # REQUIRED API
    # =====================================================

    @abstractmethod
    def open_trade(
            self,
            symbol,
            direction,
            price,
            lot,
            stop_loss,
            take_profit,
            candle_time=None
    ):

        pass


    @abstractmethod
    def update_price(
            self,
            symbol,
            price,
            candle_time=None
    ):

        pass


    @abstractmethod
    def close_position(
            self,
            *args,
            **kwargs
    ):

        pass


    @abstractmethod
    def modify_position(
            self,
            symbol,
            stop_loss=None,
            take_profit=None
    ):

        pass
