# filename: execution/base_executor.py

from abc import ABC, abstractmethod
import math

from shared.tradeai_core.production_economics import fallback_pip_value_per_lot


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

    def _get_symbol_info(self, symbol):

        """
        LIVE MT5 overrides this.

        PAPER/BACKTEST can override this if symbol
        specifications are available from the historical
        dataset.
        """

        return None


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

        from config.settings import (
            COMMISSION_PER_LOT
        )

        return (
            float(COMMISSION_PER_LOT) *
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
