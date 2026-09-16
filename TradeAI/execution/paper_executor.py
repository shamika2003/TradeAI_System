# filename: execution/paper_executor.py


from datetime import datetime


from analytics.logger import log
from analytics.trade_logger import TradeLogger


from execution.base_executor import BaseExecutor


from config.settings import (
    SYMBOLS,
    TRADE_LOT,
    COMMISSION_PER_LOT,
    DEFAULT_SPREAD_PIPS,
    SIMULATED_SLIPPAGE_PIPS
)


class PaperExecutor(BaseExecutor):


    """
    Paper / Backtest execution engine.

    Responsibilities:

        - Simulated entry execution
        - Spread
        - Slippage
        - Position tracking
        - OHLC candle execution
        - TP / SL execution
        - Commission
        - Trade history
        - Balance / equity management


    IMPORTANT:

        Historical candles are supplied by ReplayEngine.

        Predictor NEVER sees the future execution candle.

        PaperExecutor receives that candle separately
        through update_candle().
    """


    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
            self,
            capital=1000
    ):


        super().__init__(
            capital
        )


        self.trade_logger = TradeLogger()


        log(
            f"INFO | Paper Executor initialized "
            f"capital=${capital:.2f}"
        )


    # =====================================================
    # SLIPPAGE
    # =====================================================

    def slippage_price(
            self,
            symbol,
            price,
            direction
    ):


        """
        Apply simulated entry slippage.

        BUY:

            price moves upward.

        SELL:

            price moves downward.
        """


        slippage = (

            float(SIMULATED_SLIPPAGE_PIPS) *

            self.pip_size(symbol)

        )


        if direction == "BUY":

            return price + slippage


        return price - slippage


    # =====================================================
    # OPEN TRADE
    # =====================================================

    def open_trade(
            self,
            symbol,
            direction,
            price,
            lot=TRADE_LOT,
            stop_loss=None,
            take_profit=None,
            candle_time=None
    ):


        try:


            # -------------------------------------------------
            # SYMBOL VALIDATION
            # -------------------------------------------------

            if symbol not in SYMBOLS:

                log(
                    f"WARNING | Invalid symbol {symbol}"
                )

                return False


            # -------------------------------------------------
            # ONE POSITION PER SYMBOL
            # -------------------------------------------------

            if self.has_open_trade(symbol):

                log(
                    f"WARNING | Trade already exists "
                    f"{symbol}"
                )

                return False


            # -------------------------------------------------
            # SL / TP REQUIRED
            # -------------------------------------------------

            if (
                stop_loss is None
                or
                take_profit is None
            ):

                log(
                    f"ERROR | Missing SL/TP "
                    f"{symbol}"
                )

                return False


            # -------------------------------------------------
            # LOT VALIDATION
            # -------------------------------------------------

            lot = float(lot)


            if lot <= 0:

                log(
                    f"ERROR | Invalid lot "
                    f"{symbol} lot={lot}"
                )

                return False


            # -------------------------------------------------
            # ORIGINAL MARKET PRICE
            # -------------------------------------------------

            requested_price = float(
                price
            )


            # -------------------------------------------------
            # SPREAD
            # -------------------------------------------------

            spread = (

                DEFAULT_SPREAD_PIPS *

                self.pip_size(symbol)

            )


            # -------------------------------------------------
            # APPLY HALF SPREAD
            #
            # BUY:
            #     buy at higher price
            #
            # SELL:
            #     sell at lower price
            # -------------------------------------------------

            if direction == "BUY":

                entry = (

                    requested_price +

                    spread / 2

                )

            elif direction == "SELL":

                entry = (

                    requested_price -

                    spread / 2

                )

            else:

                log(
                    f"ERROR | Invalid direction "
                    f"{direction}"
                )

                return False


            # -------------------------------------------------
            # SLIPPAGE
            # -------------------------------------------------

            entry = self.slippage_price(

                symbol,

                entry,

                direction

            )


            # -------------------------------------------------
            # VALIDATE SL / TP DIRECTION
            # -------------------------------------------------

            stop_loss = float(
                stop_loss
            )

            take_profit = float(
                take_profit
            )


            if direction == "BUY":

                if stop_loss >= entry:

                    log(
                        f"ERROR | Invalid BUY SL "
                        f"{symbol} "
                        f"entry={entry} "
                        f"sl={stop_loss}"
                    )

                    return False


                if take_profit <= entry:

                    log(
                        f"ERROR | Invalid BUY TP "
                        f"{symbol} "
                        f"entry={entry} "
                        f"tp={take_profit}"
                    )

                    return False


            else:

                if stop_loss <= entry:

                    log(
                        f"ERROR | Invalid SELL SL "
                        f"{symbol} "
                        f"entry={entry} "
                        f"sl={stop_loss}"
                    )

                    return False


                if take_profit >= entry:

                    log(
                        f"ERROR | Invalid SELL TP "
                        f"{symbol} "
                        f"entry={entry} "
                        f"tp={take_profit}"
                    )

                    return False


            # -------------------------------------------------
            # COMMISSION
            # -------------------------------------------------

            commission = (

                COMMISSION_PER_LOT *

                lot

            )


            # -------------------------------------------------
            # POSITION
            # -------------------------------------------------

            position = {

                "symbol":
                symbol,

                "type":
                direction,

                "volume":
                lot,

                "entry_price":
                entry,

                "requested_price":
                requested_price,

                "current_price":
                entry,

                "stop_loss":
                stop_loss,

                "take_profit":
                take_profit,

                "exit_price":
                None,

                "profit":
                0.0,

                "gross_profit":
                0.0,

                "pips":
                0.0,

                "commission":
                commission,

                "open_time":
                (
                    candle_time
                    if candle_time
                    else datetime.now()
                ),

                "close_time":
                None,

                "exit_reason":
                None,

                "status":
                "OPEN"

            }


            # -------------------------------------------------
            # STORE POSITION
            # -------------------------------------------------

            self.positions[symbol] = position


            # -------------------------------------------------
            # TRADE MANAGER
            # -------------------------------------------------

            if self.trade_manager:

                self.trade_manager.register_trade(

                    position

                )


            # -------------------------------------------------
            # LOG
            # -------------------------------------------------

            log(

                f"INFO | PAPER OPEN "
                f"{symbol} "
                f"{direction} "
                f"ENTRY={entry:.5f} "
                f"SL={stop_loss:.5f} "
                f"TP={take_profit:.5f} "
                f"LOT={lot:.3f}"

            )


            return True


        except Exception as e:


            log(

                f"ERROR | Paper open "
                f"{symbol}: {e}"

            )


            return False


    # =====================================================
    # UPDATE PRICE
    # =====================================================

    def update_price(
            self,
            symbol,
            price,
            candle_time=None
    ):


        """
        Tick / single-price update.

        Used when a single market price is available.

        Backtest OHLC execution should use update_candle().
        """


        if symbol not in self.positions:

            return False


        try:


            price = float(
                price
            )


            position = self.positions[
                symbol
            ]


            position[
                "current_price"
            ] = price


            # -------------------------------------------------
            # UPDATE UNREALIZED P/L
            # -------------------------------------------------

            gross_profit = self.calculate_profit(

                symbol,

                position["entry_price"],

                price,

                position["type"],

                position["volume"]

            )


            position[
                "gross_profit"
            ] = gross_profit


            position[
                "profit"
            ] = round(

                gross_profit -

                position["commission"],

                2

            )


            # -------------------------------------------------
            # CHECK TP / SL
            # -------------------------------------------------

            return self.check_exit(

                symbol,

                price,

                candle_time

            )


        except Exception as e:


            log(

                f"ERROR | Paper update price "
                f"{symbol}: {e}"

            )


            return False


    # =====================================================
    # UPDATE CANDLE
    # =====================================================

    def update_candle(
            self,
            symbol,
            candle,
            candle_time=None
    ):


        """
        Execute one complete OHLC candle.

        This is the primary execution method for
        PAPER / BACKTEST mode.


        BUY:

            low <= SL
                -> STOP LOSS

            high >= TP
                -> TAKE PROFIT


        SELL:

            high >= SL
                -> STOP LOSS

            low <= TP
                -> TAKE PROFIT


        If both SL and TP are touched inside the
        same candle, STOP LOSS wins.

        This is conservative because OHLC data
        does not contain intrabar order.
        """


        if symbol not in self.positions:

            return False


        try:


            position = self.positions[
                symbol
            ]


            # -------------------------------------------------
            # READ OHLC
            # -------------------------------------------------

            open_price = float(
                candle["open"]
            )

            high = float(
                candle["high"]
            )

            low = float(
                candle["low"]
            )

            close = float(
                candle["close"]
            )


            # -------------------------------------------------
            # BASIC OHLC VALIDATION
            # -------------------------------------------------

            if high < low:

                log(

                    f"ERROR | Invalid candle "
                    f"{symbol} "
                    f"high={high} "
                    f"low={low}"

                )

                return False


            # -------------------------------------------------
            # CANDLE TIME
            # -------------------------------------------------

            if candle_time is None:

                candle_time = candle.get(
                    "time"
                )


            if candle_time is None:

                candle_time = datetime.now()


            # -------------------------------------------------
            # BUY
            # -------------------------------------------------

            if position["type"] == "BUY":


                sl_hit = (

                    low <=

                    position["stop_loss"]

                )


                tp_hit = (

                    high >=

                    position["take_profit"]

                )


                # ---------------------------------------------
                # BOTH HIT
                # ---------------------------------------------

                if sl_hit and tp_hit:

                    log(

                        f"WARNING | PAPER "
                        f"AMBIGUOUS CANDLE "
                        f"{symbol} BUY "
                        f"SL and TP both touched "
                        f"-> STOP LOSS"

                    )


                    return self._close_at_price(

                        symbol=symbol,

                        exit_price=position[
                            "stop_loss"
                        ],

                        reason="STOP_LOSS",

                        candle_time=candle_time

                    )


                # ---------------------------------------------
                # STOP LOSS
                # ---------------------------------------------

                if sl_hit:

                    return self._close_at_price(

                        symbol=symbol,

                        exit_price=position[
                            "stop_loss"
                        ],

                        reason="STOP_LOSS",

                        candle_time=candle_time

                    )


                # ---------------------------------------------
                # TAKE PROFIT
                # ---------------------------------------------

                if tp_hit:

                    return self._close_at_price(

                        symbol=symbol,

                        exit_price=position[
                            "take_profit"
                        ],

                        reason="TAKE_PROFIT",

                        candle_time=candle_time

                    )


            # -------------------------------------------------
            # SELL
            # -------------------------------------------------

            else:


                sl_hit = (

                    high >=

                    position["stop_loss"]

                )


                tp_hit = (

                    low <=

                    position["take_profit"]

                )


                # ---------------------------------------------
                # BOTH HIT
                # ---------------------------------------------

                if sl_hit and tp_hit:

                    log(

                        f"WARNING | PAPER "
                        f"AMBIGUOUS CANDLE "
                        f"{symbol} SELL "
                        f"SL and TP both touched "
                        f"-> STOP LOSS"

                    )


                    return self._close_at_price(

                        symbol=symbol,

                        exit_price=position[
                            "stop_loss"
                        ],

                        reason="STOP_LOSS",

                        candle_time=candle_time

                    )


                # ---------------------------------------------
                # STOP LOSS
                # ---------------------------------------------

                if sl_hit:

                    return self._close_at_price(

                        symbol=symbol,

                        exit_price=position[
                            "stop_loss"
                        ],

                        reason="STOP_LOSS",

                        candle_time=candle_time

                    )


                # ---------------------------------------------
                # TAKE PROFIT
                # ---------------------------------------------

                if tp_hit:

                    return self._close_at_price(

                        symbol=symbol,

                        exit_price=position[
                            "take_profit"
                        ],

                        reason="TAKE_PROFIT",

                        candle_time=candle_time

                    )


            # -------------------------------------------------
            # NO EXIT
            #
            # Mark position at candle close.
            # -------------------------------------------------

            position[
                "current_price"
            ] = close


            gross_profit = self.calculate_profit(

                symbol,

                position["entry_price"],

                close,

                position["type"],

                position["volume"]

            )


            position[
                "gross_profit"
            ] = gross_profit


            position[
                "profit"
            ] = round(

                gross_profit -

                position["commission"],

                2

            )


            # -------------------------------------------------
            # EQUITY = BALANCE + UNREALIZED NET P/L
            # -------------------------------------------------

            self.equity = (

                self.balance +

                position["profit"]

            )


            return False


        except Exception as e:


            log(

                f"ERROR | Paper update candle "
                f"{symbol}: {e}"

            )


            return False


    # =====================================================
    # CHECK EXIT
    # =====================================================

    def check_exit(
            self,
            symbol,
            price,
            candle_time=None
    ):


        """
        Check a single market price against SL/TP.

        For OHLC backtesting use update_candle()
        because it can inspect both high and low.
        """


        if symbol not in self.positions:

            return False


        position = self.positions[
            symbol
        ]


        price = float(
            price
        )


        if position["type"] == "BUY":


            if price <= position[
                    "stop_loss"
            ]:

                return self._close_at_price(

                    symbol=symbol,

                    exit_price=position[
                        "stop_loss"
                    ],

                    reason="STOP_LOSS",

                    candle_time=candle_time

                )


            if price >= position[
                    "take_profit"
            ]:

                return self._close_at_price(

                    symbol=symbol,

                    exit_price=position[
                        "take_profit"
                    ],

                    reason="TAKE_PROFIT",

                    candle_time=candle_time

                )


        else:


            if price >= position[
                    "stop_loss"
            ]:

                return self._close_at_price(

                    symbol=symbol,

                    exit_price=position[
                        "stop_loss"
                    ],

                    reason="STOP_LOSS",

                    candle_time=candle_time

                )


            if price <= position[
                    "take_profit"
            ]:

                return self._close_at_price(

                    symbol=symbol,

                    exit_price=position[
                        "take_profit"
                    ],

                    reason="TAKE_PROFIT",

                    candle_time=candle_time

                )


        return False


    # =====================================================
    # CLOSE AT EXACT PRICE
    # =====================================================

    def _close_at_price(
            self,
            symbol,
            exit_price,
            reason,
            candle_time=None
    ):


        if symbol not in self.positions:

            return False


        position = self.positions[
            symbol
        ]


        exit_price = float(
            exit_price
        )


        # -------------------------------------------------
        # FINAL PRICE
        # -------------------------------------------------

        position[
            "current_price"
        ] = exit_price


        position[
            "exit_price"
        ] = exit_price


        # -------------------------------------------------
        # GROSS PROFIT
        # -------------------------------------------------

        gross_profit = self.calculate_profit(

            symbol,

            position["entry_price"],

            exit_price,

            position["type"],

            position["volume"]

        )


        position[
            "gross_profit"
        ] = gross_profit


        # -------------------------------------------------
        # NET PROFIT
        # -------------------------------------------------

        net_profit = (

            gross_profit -

            position["commission"]

        )


        position[
            "profit"
        ] = round(

            net_profit,

            2

        )


        # -------------------------------------------------
        # PIPS
        # -------------------------------------------------

        pip = self.pip_size(
            symbol
        )


        if position["type"] == "BUY":

            difference = (

                exit_price -

                position["entry_price"]

            )

        else:

            difference = (

                position["entry_price"] -

                exit_price

            )


        position[
            "pips"
        ] = round(

            difference / pip,

            2

        )


        # -------------------------------------------------
        # CLOSE
        # -------------------------------------------------

        return self.close_position(

            symbol=symbol,

            reason=reason,

            candle_time=candle_time

        )


    # =====================================================
    # CLOSE POSITION
    # =====================================================

    def close_position(
            self,
            symbol,
            reason="AI_CLOSE",
            candle_time=None
    ):


        if symbol not in self.positions:

            return False


        # -------------------------------------------------
        # REMOVE ACTIVE POSITION
        # -------------------------------------------------

        position = self.positions.pop(
            symbol
        )


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        position[
            "status"
        ] = "CLOSED"


        position[
            "exit_reason"
        ] = reason


        position[
            "close_time"
        ] = (

            candle_time

            if candle_time

            else datetime.now()

        )


        # -------------------------------------------------
        # FALLBACK EXIT PRICE
        # -------------------------------------------------

        if position[
            "exit_price"
        ] is None:

            position[
                "exit_price"
            ] = position[
                "current_price"
            ]


        # -------------------------------------------------
        # FINAL BALANCE
        # -------------------------------------------------

        self.balance += position[
            "profit"
        ]


        self.equity = self.balance


        # -------------------------------------------------
        # TRADE HISTORY
        # -------------------------------------------------

        self.trade_history.append(

            position.copy()

        )


        # -------------------------------------------------
        # TRADE LOGGER
        # -------------------------------------------------

        try:

            self.trade_logger.save_trade(
                position
            )

        except Exception as e:

            log(

                f"WARNING | Trade logger "
                f"failed {symbol}: {e}"

            )


        # -------------------------------------------------
        # TRADE MANAGER
        # -------------------------------------------------

        if self.trade_manager:

            self.trade_manager.remove_trade(
                symbol
            )


        # -------------------------------------------------
        # LOG
        # -------------------------------------------------

        log(

            f"INFO | PAPER CLOSE "
            f"{symbol} "
            f"{reason} "
            f"entry={position['entry_price']:.5f} "
            f"exit={position['exit_price']:.5f} "
            f"pips={position['pips']:.2f} "
            f"gross={position['gross_profit']:.2f} "
            f"commission={position['commission']:.2f} "
            f"net={position['profit']:.2f} "
            f"balance={self.balance:.2f}"

        )


        return True


    # =====================================================
    # PROFIT CALCULATION
    # =====================================================

    def calculate_profit(
            self,
            symbol,
            entry,
            current,
            direction,
            lot
    ):


        pip = self.pip_size(
            symbol
        )


        if direction == "BUY":

            difference = (

                current -

                entry

            )

        else:

            difference = (

                entry -

                current

            )


        pips = (

            difference /

            pip

        )


        pip_value = (

            self.pip_value_per_lot(
                symbol,
                price=entry
            )

            *

            lot

        )


        profit = (

            pips *

            pip_value

        )


        return profit


    # =====================================================
    # MODIFY POSITION
    # =====================================================

    def modify_position(
            self,
            symbol,
            stop_loss=None,
            take_profit=None
    ):


        if symbol not in self.positions:

            return False


        position = self.positions[
            symbol
        ]


        # -------------------------------------------------
        # MODIFY SL
        # -------------------------------------------------

        if stop_loss is not None:

            position[
                "stop_loss"
            ] = float(
                stop_loss
            )


        # -------------------------------------------------
        # MODIFY TP
        # -------------------------------------------------

        if take_profit is not None:

            position[
                "take_profit"
            ] = float(
                take_profit
            )


        log(

            f"INFO | PAPER MODIFY "
            f"{symbol} "
            f"SL={position['stop_loss']} "
            f"TP={position['take_profit']}"

        )


        return True
