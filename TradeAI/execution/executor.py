# filename: execution/executor.py


import json
import math
import os
import time

from pathlib import Path

import MetaTrader5 as mt5 

from datetime import datetime, timedelta

from analytics.logger import log
from analytics.trade_logger import TradeLogger

from execution.base_executor import BaseExecutor

from config.settings import (
    SYMBOLS,
    TRADE_LOT,
    MT5_MAGIC,
    DEVIATION,
    MODE,
    DEMO_FORWARD_CAPITAL,
    DEMO_FORWARD_STATE_PATH,
    DEMO_FORWARD_REQUIRE_DEMO_ACCOUNT,
    DEMO_FORWARD_MIN_TRADES,
    DEMO_FORWARD_MAX_DD_PERCENT,
    DEMO_FORWARD_MIN_PROFIT_FACTOR,
    DEMO_FORWARD_MIN_RETURN_PERCENT,
    BROKER_PROFILE_PATH,
    COMMISSION_PER_LOT,
    SIMULATED_SLIPPAGE_PIPS,
)
from shared.tradeai_core.demo_forward import DemoForwardLedger
from shared.tradeai_core.broker_profile import capture_mt5_broker_profile


class BrainExecutor(BaseExecutor):

    """
    LIVE MT5 execution engine.

    Responsibilities:

        - Real MT5 execution
        - Broker account synchronization
        - Broker lot rules
        - Dynamic risk-based lot sizing
        - SL / TP validation
        - Position tracking
        - Broker-side close detection
        - Closed trade reconciliation
        - Trade history
        - TradeManager synchronization

    Important:

        MT5 is the source of truth for live account state.

        PaperExecutor is the source of truth for simulated
        account state.

        Strategy / risk / trade lifecycle should remain shared.
    """


    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
            self,
            capital=1000
    ):

        # -------------------------------------------------
        # Base initialization
        # -------------------------------------------------

        super().__init__(
            capital
        )


        # -------------------------------------------------
        # MT5 configuration
        # -------------------------------------------------

        self.magic = MT5_MAGIC
        self.forward_ledger = None

        # -------------------------------------------------
        # ACCOUNT STATUS / EVENT LOGGING
        # -------------------------------------------------
        #
        # bot.log is an event log, not a five-second telemetry dump.
        # Identical rounded account snapshots are therefore not
        # appended repeatedly.
        #
        # The current broker account values are still written on
        # every refresh to runtime_account_status.json, replacing
        # the previous snapshot atomically.
        # -------------------------------------------------

        self._last_account_log_snapshot = None
        self._account_status_path = (
            Path(DEMO_FORWARD_STATE_PATH).parent /
            "runtime_account_status.json"
        )


        # -------------------------------------------------
        # Trade logger
        # -------------------------------------------------

        self.trade_logger = TradeLogger()


        # -------------------------------------------------
        # Synchronize real account
        # -------------------------------------------------

        if not self.refresh_account():

            raise RuntimeError(
                "Unable to read MT5 account information"
            )


        # -------------------------------------------------
        # DEMO FORWARD SAFETY + SHADOW CAPITAL
        # -------------------------------------------------

        if MODE == "DEMO_FORWARD":

            account = mt5.account_info()

            if account is None:
                raise RuntimeError("Unable to verify MT5 account mode")

            if DEMO_FORWARD_REQUIRE_DEMO_ACCOUNT:
                demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
                if int(getattr(account, "trade_mode", -1)) != int(demo_mode):
                    raise RuntimeError(
                        "DEMO_FORWARD REFUSED: connected MT5 account is not DEMO. "
                        "No order will be sent to a real-money account."
                    )

            self.forward_ledger = DemoForwardLedger(
                path=DEMO_FORWARD_STATE_PATH,
                initial_balance=DEMO_FORWARD_CAPITAL,
                min_trades=DEMO_FORWARD_MIN_TRADES,
                max_dd_percent=DEMO_FORWARD_MAX_DD_PERCENT,
                min_profit_factor=DEMO_FORWARD_MIN_PROFIT_FACTOR,
                min_return_percent=DEMO_FORWARD_MIN_RETURN_PERCENT,
            )

            self.initial_capital = self.forward_ledger.initial_balance

            summary = self.forward_ledger.summary()
            log(
                f"INFO | DEMO FORWARD SAFE "
                f"shadow_balance=${summary['virtual_balance']:.2f} "
                f"closed={summary['closed_trades']} "
                f"risk_gate_account=DEMO"
            )

        else:
            # Real LIVE mode uses broker balance as its capital reference.
            self.initial_capital = self.balance


        # -------------------------------------------------
        # SNAPSHOT BROKER EXECUTION RULES FOR BACKTEST PARITY
        # -------------------------------------------------

        try:
            profile = capture_mt5_broker_profile(
                mt5,
                SYMBOLS,
                path=BROKER_PROFILE_PATH,
                commission_per_lot=COMMISSION_PER_LOT,
                slippage_pips=SIMULATED_SLIPPAGE_PIPS,
            )
            self.reload_broker_profile()
            log(
                f"INFO | Broker profile captured "
                f"symbols={len(profile.get('symbols', {}))} "
                f"path={BROKER_PROFILE_PATH}"
            )
        except Exception as e:
            log(
                f"WARNING | Broker profile capture failed: {e}"
            )


        log(
            f"INFO | MT5 Executor initialized "
            f"broker_balance=${self.balance:.2f} "
            f"broker_equity=${self.equity:.2f}"
        )


    # =====================================================
    # SYMBOL INFORMATION
    # =====================================================

    def _get_symbol_info(
            self,
            symbol
    ):

        try:

            info = mt5.symbol_info(
                symbol
            )

            if info is None:

                log(
                    f"ERROR | Unable to read symbol info "
                    f"{symbol}"
                )

                return None


            return info


        except Exception as e:

            log(
                f"ERROR | _get_symbol_info "
                f"{symbol}: {e}"
            )

            return None


    def _get_symbol_tick(
            self,
            symbol
    ):

        try:

            tick = mt5.symbol_info_tick(
                symbol
            )

            if tick is None:

                log(
                    f"ERROR | Unable to read symbol tick "
                    f"{symbol}"
                )

                return None


            return tick


        except Exception as e:

            log(
                f"ERROR | _get_symbol_tick "
                f"{symbol}: {e}"
            )

            return None


    # =====================================================
    # ACCOUNT SYNCHRONIZATION
    # =====================================================

    def _write_runtime_account_status(
            self,
            account
    ):

        """
        Persist only the latest broker/shadow account snapshot.

        This file is intentionally overwritten atomically rather
        than appended, so live telemetry does not flood bot.log.
        """

        try:

            payload = {
                "updated_at": datetime.now().astimezone().isoformat(),
                "mode": MODE,
                "broker": {
                    "balance": float(
                        account.balance
                    ),
                    "equity": float(
                        account.equity
                    ),
                    "free_margin": float(
                        account.margin_free
                    ),
                    "currency": str(
                        getattr(
                            account,
                            "currency",
                            ""
                        )
                    ),
                },
            }


            if self.forward_ledger is not None:

                summary = self.forward_ledger.summary()

                payload["shadow"] = {
                    "initial_balance": float(
                        summary[
                            "initial_balance"
                        ]
                    ),
                    "virtual_balance": float(
                        summary[
                            "virtual_balance"
                        ]
                    ),
                    "net_profit": float(
                        summary[
                            "net_profit"
                        ]
                    ),
                    "closed_trades": int(
                        summary[
                            "closed_trades"
                        ]
                    ),
                    "max_drawdown_percent": float(
                        summary[
                            "max_drawdown_percent"
                        ]
                    ),
                }


            self._account_status_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            temp_path = (
                self._account_status_path.parent /
                (
                    self._account_status_path.name +
                    ".tmp"
                )
            )


            temp_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True
                ),
                encoding="utf-8"
            )


            os.replace(
                temp_path,
                self._account_status_path
            )


        except Exception as e:

            # Telemetry must never stop trading.
            log(
                f"WARNING | account status write failed: {e}"
            )


    def refresh_account(
            self
    ):

        try:

            account = mt5.account_info()


            if account is None:

                log(
                    f"ERROR | MT5 account_info failed "
                    f"{mt5.last_error()}"
                )

                return False


            self.balance = float(
                account.balance
            )

            self.equity = float(
                account.equity
            )


            free_margin = float(
                account.margin_free
            )


            # Always maintain a single latest-value telemetry file.
            self._write_runtime_account_status(
                account
            )


            # -------------------------------------------------
            # EVENT-ORIENTED ACCOUNT LOGGING
            # -------------------------------------------------
            #
            # Compare exactly what bot.log displays (2 decimals).
            # If nothing visible changed, do not append another line.
            #
            # This keeps account changes easy to see while the live
            # status file continues to refresh every loop.
            # -------------------------------------------------

            snapshot = (
                round(
                    self.balance,
                    2
                ),
                round(
                    self.equity,
                    2
                ),
                round(
                    free_margin,
                    2
                ),
            )


            if (
                self._last_account_log_snapshot
                !=
                snapshot
            ):

                previous = (
                    self._last_account_log_snapshot
                )

                if previous is None:

                    prefix = (
                        "DEBUG | ACCOUNT"
                    )

                else:

                    prefix = (
                        "INFO | ACCOUNT CHANGED"
                    )


                log(
                    f"{prefix} "
                    f"balance=${self.balance:.2f} "
                    f"equity=${self.equity:.2f} "
                    f"free_margin=${free_margin:.2f}"
                )


                self._last_account_log_snapshot = (
                    snapshot
                )


            return True


        except Exception as e:

            log(
                f"ERROR | refresh_account {e}"
            )

            return False


    # =====================================================
    # RISK CAPITAL / DEMO FORWARD LEDGER
    # =====================================================

    def get_risk_balance(self):
        if self.forward_ledger is not None:
            return float(self.forward_ledger.virtual_balance)
        return float(self.balance)

    def get_risk_equity(self):
        # With one-position-at-a-time gating, realized shadow balance is the
        # conservative risk reference between entries.
        if self.forward_ledger is not None:
            return float(self.forward_ledger.virtual_balance)
        return float(self.equity)

    def get_risk_high_watermark(self):
        if self.forward_ledger is not None:
            return float(self.forward_ledger.high_watermark)
        return float(self.initial_capital)

    def is_daily_loss_limit_hit(self, max_percent):
        if self.forward_ledger is None:
            return False
        return self.forward_ledger.daily_loss_percent() >= float(max_percent)

    def record_risk_rejection(self, reason):
        if self.forward_ledger is not None:
            self.forward_ledger.record_rejection(reason)

    def get_forward_summary(self):
        if self.forward_ledger is None:
            return None
        return self.forward_ledger.summary()


    # =====================================================
    # POSITION CHECK
    # =====================================================

    def has_open_trade(
            self,
            symbol
    ):

        try:

            return (
                self.get_position(symbol)
                is not None
            )


        except Exception as e:

            log(
                f"ERROR | has_open_trade "
                f"{symbol}: {e}"
            )

            return False


    def get_position(
            self,
            symbol
    ):

        try:

            positions = mt5.positions_get(
                symbol=symbol
            )


            if not positions:

                return None


            for position in positions:

                if position.magic == self.magic:

                    return position


            return None


        except Exception as e:

            log(
                f"ERROR | get_position "
                f"{symbol}: {e}"
            )

            return None


    def get_symbol_positions(
            self,
            symbol
    ):

        try:

            positions = mt5.positions_get(
                symbol=symbol
            )


            if not positions:

                return []


            return [

                position

                for position in positions

                if position.magic == self.magic

            ]


        except Exception as e:

            log(
                f"ERROR | get_symbol_positions "
                f"{symbol}: {e}"
            )

            return []


    def get_all_positions(
            self
    ):

        try:

            positions = mt5.positions_get()


            if not positions:

                return []


            return [

                position

                for position in positions

                if position.magic == self.magic

            ]


        except Exception as e:

            log(
                f"ERROR | get_all_positions "
                f"{e}"
            )

            return []


    # =====================================================
    # BROKER LOT RULES
    # =====================================================

    def get_lot_limits(
            self,
            symbol
    ):

        try:

            info = mt5.symbol_info(
                symbol
            )


            if info is None:

                log(
                    f"ERROR | Unable to read symbol info "
                    f"{symbol}"
                )

                return None


            min_lot = float(
                info.volume_min
            )

            max_lot = float(
                info.volume_max
            )

            step = float(
                info.volume_step
            )


            if (
                min_lot <= 0
                or
                max_lot <= 0
                or
                step <= 0
            ):

                log(
                    f"ERROR | Invalid broker lot rules "
                    f"{symbol} "
                    f"min={min_lot} "
                    f"max={max_lot} "
                    f"step={step}"
                )

                return None


            return {

                "min": min_lot,
                "max": max_lot,
                "step": step

            }


        except Exception as e:

            log(
                f"ERROR | get_lot_limits "
                f"{symbol}: {e}"
            )

            return None


    # =====================================================
    # NORMALIZE LOT
    # =====================================================

    def normalize_lot(
            self,
            symbol,
            lot
    ):

        try:

            if lot is None:

                return None


            lot = float(
                lot
            )


            if not math.isfinite(lot):

                return None


            if lot <= 0:

                return None


            limits = self.get_lot_limits(
                symbol
            )


            if limits is None:

                return None


            min_lot = limits["min"]
            max_lot = limits["max"]
            step = limits["step"]


            # -------------------------------------------------
            # Never exceed broker maximum
            # -------------------------------------------------

            if lot > max_lot:

                log(
                    f"DEBUG | Lot capped "
                    f"{symbol} "
                    f"{lot:.6f} -> "
                    f"{max_lot:.6f}"
                )

                lot = max_lot


            # -------------------------------------------------
            # Never force a $10-$50 account above broker
            # minimum volume.
            #
            # If calculated risk is below broker minimum,
            # reject the trade instead.
            # -------------------------------------------------

            if lot < min_lot:

                log(
                    f"DEBUG | Lot below broker minimum "
                    f"{symbol} "
                    f"requested={lot:.6f} "
                    f"minimum={min_lot:.6f}"
                )

                return None


            # -------------------------------------------------
            # Align to broker volume step
            # -------------------------------------------------

            steps = math.floor(
                (lot - min_lot) / step
                + 1e-9
            )


            normalized = (
                min_lot +
                steps * step
            )


            normalized = max(
                min_lot,
                min(
                    normalized,
                    max_lot
                )
            )


            # -------------------------------------------------
            # Determine decimal precision
            # -------------------------------------------------

            step_text = (
                f"{step:.10f}"
                .rstrip("0")
            )


            if "." in step_text:

                decimals = len(
                    step_text.split(".")[1]
                )

            else:

                decimals = 0


            normalized = round(
                normalized,
                decimals
            )


            if normalized < min_lot:

                return None


            if normalized > max_lot:

                return None


            return normalized


        except Exception as e:

            log(
                f"ERROR | normalize_lot "
                f"{symbol}: {e}"
            )

            return None


    # =====================================================
    # POSITION -> LOCAL SNAPSHOT
    # =====================================================

    def _position_to_dict(
            self,
            symbol,
            position=None
    ):

        local = self.positions.get(
            symbol,
            {}
        )


        if position is None:

            position = self.get_position(
                symbol
            )


        if position is None and not local:

            return None


        if position is not None:

            if position.type == mt5.POSITION_TYPE_BUY:

                direction = "BUY"

            else:

                direction = "SELL"


            return {

                "symbol":
                position.symbol,

                "type":
                direction,

                "volume":
                float(position.volume),

                "entry_price":
                float(position.price_open),

                "current_price":
                float(position.price_current),

                "stop_loss":
                float(position.sl)
                if position.sl
                else None,

                "take_profit":
                float(position.tp)
                if position.tp
                else None,

                "open_time":
                datetime.fromtimestamp(
                    position.time
                ),

                "ticket":
                position.ticket

            }


        return local.copy()


    # =====================================================
    # FIND CLOSED DEALS
    # =====================================================

    def _get_closed_deals(
            self,
            symbol,
            snapshot,
            close_time=None
    ):

        try:

            start = snapshot.get(
                "open_time"
            )


            if not isinstance(
                    start,
                    datetime
            ):

                start = (
                    datetime.now()
                    -
                    timedelta(days=7)
                )


            end = (

                close_time

                if isinstance(
                    close_time,
                    datetime
                )

                else datetime.now()

            )


            ticket = snapshot.get(
                "ticket"
            )


            for attempt in range(5):

                deals = mt5.history_deals_get(
                    start,
                    end
                )


                if deals:

                    matching = []


                    for deal in deals:

                        if deal.symbol != symbol:

                            continue


                        if deal.magic != self.magic:

                            continue


                        if (
                            ticket is not None
                            and
                            deal.position != ticket
                        ):

                            continue


                        matching.append(
                            deal
                        )


                    if matching:

                        return matching


                if attempt < 4:

                    log(
                        f"DEBUG | Waiting for MT5 "
                        f"trade history "
                        f"{symbol} "
                        f"attempt={attempt + 1}/5"
                    )

                    time.sleep(1)


            return []


        except Exception as e:

            log(
                f"ERROR | _get_closed_deals "
                f"{symbol}: {e}"
            )

            return []


    # =====================================================
    # ARCHIVE CLOSED TRADE
    # =====================================================

    def _archive_closed_trade(
            self,
            symbol,
            snapshot,
            exit_reason="MT5_CLOSE",
            close_time=None
    ):

        try:

            if snapshot is None:

                return False


            deals = self._get_closed_deals(
                symbol,
                snapshot,
                close_time
            )


            if not deals:

                log(
                    f"DEBUG | No matching MT5 history "
                    f"{symbol}"
                )

                return False


            # -------------------------------------------------
            # MT5 deal accounting
            #
            # profit  = trading P/L
            # commission = broker commission
            # swap = overnight financing
            #
            # Net result must reflect what actually happened
            # to the live account.
            # -------------------------------------------------

            profit = 0.0
            commission = 0.0
            swap = 0.0

            exit_price = None

            close_deals = []


            for deal in deals:

                deal_entry = getattr(
                    deal,
                    "entry",
                    None
                )


                # Keep exit deals for final result.
                #
                # Some brokers may expose slightly different
                # deal entry values, therefore profit-bearing
                # matching deals are retained safely.
                if (
                    deal_entry
                    in (
                        mt5.DEAL_ENTRY_OUT,
                        mt5.DEAL_ENTRY_OUT_BY,
                        mt5.DEAL_ENTRY_INOUT
                    )
                ):

                    close_deals.append(
                        deal
                    )


                profit += float(
                    getattr(
                        deal,
                        "profit",
                        0.0
                    )
                )


                commission += float(
                    getattr(
                        deal,
                        "commission",
                        0.0
                    )
                )


                swap += float(
                    getattr(
                        deal,
                        "swap",
                        0.0
                    )
                )


            # -------------------------------------------------
            # Prefer the final closing deal price.
            # -------------------------------------------------

            if close_deals:

                exit_price = (
                    close_deals[-1].price
                )

            else:

                exit_price = (
                    deals[-1].price
                )


            net_profit = (
                profit
                +
                commission
                +
                swap
            )


            end = (

                close_time

                if isinstance(
                    close_time,
                    datetime
                )

                else datetime.now()

            )


            trade = {

                "symbol":
                symbol,

                "type":
                snapshot.get(
                    "type"
                ),

                "entry_price":
                snapshot.get(
                    "entry_price"
                ),

                "exit_price":
                exit_price,

                "volume":
                snapshot.get(
                    "volume"
                ),

                "profit":
                round(
                    net_profit,
                    2
                ),

                "gross_profit":
                round(
                    profit,
                    2
                ),

                "commission":
                round(
                    commission,
                    2
                ),

                "swap":
                round(
                    swap,
                    2
                ),

                "open_time":
                snapshot.get(
                    "open_time"
                ),

                "close_time":
                end,

                "exit_reason":
                exit_reason,

                "ticket":
                snapshot.get(
                    "ticket"
                )

            }


            self.trade_history.append(
                trade
            )


            self.trade_logger.save_trade(
                trade
            )

            if self.forward_ledger is not None:
                self.forward_ledger.record_trade(trade)


            log(
                f"INFO | MT5 SAVED "
                f"{symbol} "
                f"net={net_profit:.2f} "
                f"gross={profit:.2f} "
                f"commission={commission:.2f} "
                f"swap={swap:.2f} "
                f"exit={exit_price}"
            )


            return True


        except Exception as e:

            log(
                f"ERROR | archive "
                f"{symbol}: {e}"
            )

            return False


    # =====================================================
    # OPEN TRADE
    # =====================================================

    def open_trade(
            self,
            symbol,
            direction,
            price,
            lot=None,
            stop_loss=None,
            take_profit=None,
            candle_time=None
    ):

        try:

            # =================================================
            # VALIDATION
            # =================================================

            if symbol not in SYMBOLS:

                log(
                    f"ERROR | Symbol not allowed "
                    f"{symbol}"
                )

                return False


            if self.has_open_trade(
                    symbol
            ):

                log(
                    f"DEBUG | Trade already open "
                    f"{symbol}"
                )

                return False


            if stop_loss is None:

                log(
                    f"ERROR | Missing stop loss "
                    f"{symbol}"
                )

                return False


            if take_profit is None:

                log(
                    f"ERROR | Missing take profit "
                    f"{symbol}"
                )

                return False


            # =================================================
            # SYMBOL
            # =================================================

            symbol_info = mt5.symbol_info(
                symbol
            )


            if symbol_info is None:

                log(
                    f"ERROR | Unable to read symbol "
                    f"{symbol}"
                )

                return False


            tick = mt5.symbol_info_tick(
                symbol
            )


            if tick is None:

                log(
                    f"ERROR | Unable to read tick "
                    f"{symbol}"
                )

                return False


            # =================================================
            # EXECUTION PRICE
            # =================================================

            if direction == "BUY":

                order_type = (
                    mt5.ORDER_TYPE_BUY
                )

                execution_price = float(
                    tick.ask
                )


            elif direction == "SELL":

                order_type = (
                    mt5.ORDER_TYPE_SELL
                )

                execution_price = float(
                    tick.bid
                )


            else:

                log(
                    f"ERROR | Invalid direction "
                    f"{symbol} {direction}"
                )

                return False


            # =================================================
            # BROKER PRICE RULES
            # =================================================

            digits = int(
                symbol_info.digits
            )

            point = float(
                symbol_info.point
            )

            stops_level = int(
                symbol_info.trade_stops_level
            )

            freeze_level = int(
                symbol_info.trade_freeze_level
            )

            tick_size = float(
                getattr(
                    symbol_info,
                    "trade_tick_size",
                    point
                )
            )


            if tick_size <= 0:

                tick_size = point


            minimum_stop_points = max(
                stops_level,
                freeze_level
            )


            minimum_stop_distance = (
                minimum_stop_points
                *
                point
            )


            # =================================================
            # DYNAMIC LOT
            # =================================================

            if lot is None:

                distance = abs(
                    execution_price
                    -
                    float(stop_loss)
                )


                lot = self.calculate_lot(
                    symbol,
                    distance
                )


            if lot is None:

                log(
                    f"ERROR | Risk manager returned "
                    f"no valid lot "
                    f"{symbol}"
                )

                return False


            lot = float(
                lot
            )


            if lot <= 0:

                return False


            requested_lot = lot


            lot = self.normalize_lot(
                symbol,
                lot
            )


            if lot is None:

                log(
                    f"INFO | Trade rejected because "
                    f"risk lot is below broker minimum "
                    f"{symbol} "
                    f"requested={requested_lot:.6f}"
                )

                return False


            # =================================================
            # SL / TP
            # =================================================

            stop_loss = float(
                stop_loss
            )

            take_profit = float(
                take_profit
            )


            # =================================================
            # BUY VALIDATION
            # =================================================

            if direction == "BUY":

                if stop_loss >= execution_price:

                    log(
                        f"ERROR | Invalid BUY SL "
                        f"{symbol}"
                    )

                    return False


                if take_profit <= execution_price:

                    log(
                        f"ERROR | Invalid BUY TP "
                        f"{symbol}"
                    )

                    return False


                if (
                    execution_price
                    -
                    stop_loss
                ) < minimum_stop_distance:

                    log(
                        f"ERROR | BUY SL too close "
                        f"{symbol}"
                    )

                    return False


                if (
                    take_profit
                    -
                    execution_price
                ) < minimum_stop_distance:

                    log(
                        f"ERROR | BUY TP too close "
                        f"{symbol}"
                    )

                    return False


            # =================================================
            # SELL VALIDATION
            # =================================================

            elif direction == "SELL":

                if stop_loss <= execution_price:

                    log(
                        f"ERROR | Invalid SELL SL "
                        f"{symbol}"
                    )

                    return False


                if take_profit >= execution_price:

                    log(
                        f"ERROR | Invalid SELL TP "
                        f"{symbol}"
                    )

                    return False


                if (
                    stop_loss
                    -
                    execution_price
                ) < minimum_stop_distance:

                    log(
                        f"ERROR | SELL SL too close "
                        f"{symbol}"
                    )

                    return False


                if (
                    execution_price
                    -
                    take_profit
                ) < minimum_stop_distance:

                    log(
                        f"ERROR | SELL TP too close "
                        f"{symbol}"
                    )

                    return False


            # =================================================
            # PRICE NORMALIZATION
            # =================================================

            execution_price = round(
                execution_price,
                digits
            )

            stop_loss = round(
                stop_loss,
                digits
            )

            take_profit = round(
                take_profit,
                digits
            )


            # =================================================
            # ORDER REQUEST
            # =================================================

            request = {

                "action":
                mt5.TRADE_ACTION_DEAL,

                "symbol":
                symbol,

                "volume":
                lot,

                "type":
                order_type,

                "price":
                execution_price,

                "sl":
                stop_loss,

                "tp":
                take_profit,

                "deviation":
                DEVIATION,

                "magic":
                self.magic,

                "comment":
                "TRADE AI BOT",

                "type_time":
                mt5.ORDER_TIME_GTC,

                "type_filling":
                mt5.ORDER_FILLING_IOC

            }


            # =================================================
            # PRE-FLIGHT CHECK
            # =================================================

            check = mt5.order_check(
                request
            )


            if check is None:

                log(
                    f"ERROR | MT5 order_check "
                    f"{symbol} "
                    f"error={mt5.last_error()}"
                )

                return False


            if check.retcode != 0:

                log(
                    f"ERROR | MT5 ORDER CHECK FAILED "
                    f"{symbol} "
                    f"retcode={check.retcode} "
                    f"comment={check.comment}"
                )

                return False


            # =================================================
            # SEND
            # =================================================

            result = mt5.order_send(
                request
            )


            if result is None:

                log(
                    f"ERROR | MT5 order_send None "
                    f"{symbol} "
                    f"error={mt5.last_error()}"
                )

                return False


            if result.retcode != (
                mt5.TRADE_RETCODE_DONE
            ):

                log(
                    f"ERROR | MT5 rejected "
                    f"{symbol} "
                    f"retcode={result.retcode} "
                    f"comment={result.comment}"
                )

                return False


            # =================================================
            # WAIT FOR POSITION
            # =================================================

            time.sleep(
                1
            )


            position = self.get_position(
                symbol
            )


            if position is None:

                log(
                    f"ERROR | MT5 order succeeded but "
                    f"position was not found "
                    f"{symbol}"
                )

                self.refresh_account()

                return False


            # =================================================
            # USE ACTUAL BROKER VALUES
            # =================================================

            actual_entry = float(
                position.price_open
            )

            actual_volume = float(
                position.volume
            )

            actual_ticket = position.ticket

            actual_open_time = (
                datetime.fromtimestamp(
                    position.time
                )
            )


            # =================================================
            # LOCAL POSITION
            # =================================================

            self.positions[symbol] = {

                "symbol":
                symbol,

                "type":
                direction,

                "volume":
                actual_volume,

                "entry_price":
                actual_entry,

                "current_price":
                float(
                    position.price_current
                ),

                "stop_loss":
                float(
                    position.sl
                )
                if position.sl
                else None,

                "take_profit":
                float(
                    position.tp
                )
                if position.tp
                else None,

                "ticket":
                actual_ticket,

                "open_time":
                actual_open_time,

                "profit":
                0.0,

                "status":
                "OPEN"

            }


            # =================================================
            # TRADE MANAGER
            # =================================================

            if self.trade_manager:

                self.trade_manager.register_trade(
                    self.positions[symbol]
                )


            # =================================================
            # REFRESH REAL ACCOUNT
            # =================================================

            self.refresh_account()


            log(
                f"INFO | MT5 OPEN "
                f"{symbol} "
                f"{direction} "
                f"lot={actual_volume} "
                f"entry={actual_entry}"
            )


            return True


        except Exception as e:

            log(
                f"ERROR | open_trade "
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
        MT5 manages SL/TP itself.

        Python's responsibility is to synchronize local
        state with the broker.
        """

        try:

            self.check_closed_trades()


            position = self.get_position(
                symbol
            )


            if position is not None:

                if symbol in self.positions:

                    self.positions[
                        symbol
                    ]["current_price"] = float(
                        position.price_current
                    )

                    self.positions[
                        symbol
                    ]["profit"] = float(
                        position.profit
                    )


                return True


            return False


        except Exception as e:

            log(
                f"ERROR | update_price "
                f"{symbol}: {e}"
            )

            return False


    # =====================================================
    # CLOSE POSITION
    # =====================================================

    def close_position(
            self,
            position,
            reason="AI_CLOSE",
            candle_time=None
    ):

        try:

            # -------------------------------------------------
            # Resolve symbol
            # -------------------------------------------------

            if isinstance(
                    position,
                    str
            ):

                symbol = position

                mt5_position = (
                    self.get_position(
                        symbol
                    )
                )

                snapshot = (
                    self.positions.get(
                        symbol
                    )
                )


            else:

                symbol = getattr(
                    position,
                    "symbol",
                    None
                )

                mt5_position = position

                snapshot = (
                    self.positions.get(
                        symbol
                    )
                )


            if symbol is None:

                return False


            # -------------------------------------------------
            # Already closed by broker
            # -------------------------------------------------

            if mt5_position is None:

                snapshot = (

                    snapshot

                    or

                    self._position_to_dict(
                        symbol
                    )

                )


                if snapshot is None:

                    return False


                archived = (
                    self._archive_closed_trade(
                        symbol,
                        snapshot,
                        exit_reason=reason,
                        close_time=(
                            candle_time
                            if candle_time
                            else datetime.now()
                        )
                    )
                )


                if archived:

                    self.positions.pop(
                        symbol,
                        None
                    )


                    if self.trade_manager:

                        self.trade_manager.remove_trade(
                            symbol
                        )


                    self.refresh_account()


                return archived


            # -------------------------------------------------
            # Closing market price
            # -------------------------------------------------

            tick = mt5.symbol_info_tick(
                symbol
            )


            if tick is None:

                return False


            if (
                mt5_position.type
                ==
                mt5.POSITION_TYPE_BUY
            ):

                order_type = (
                    mt5.ORDER_TYPE_SELL
                )

                close_price = float(
                    tick.bid
                )

            else:

                order_type = (
                    mt5.ORDER_TYPE_BUY
                )

                close_price = float(
                    tick.ask
                )


            # -------------------------------------------------
            # CLOSE REQUEST
            # -------------------------------------------------

            request = {

                "action":
                mt5.TRADE_ACTION_DEAL,

                "symbol":
                symbol,

                "volume":
                float(
                    mt5_position.volume
                ),

                "type":
                order_type,

                "position":
                mt5_position.ticket,

                "price":
                close_price,

                "deviation":
                DEVIATION,

                "magic":
                self.magic,

                "comment":
                reason,

                "type_time":
                mt5.ORDER_TIME_GTC,

                "type_filling":
                mt5.ORDER_FILLING_IOC

            }


            result = mt5.order_send(
                request
            )


            if (
                result is None
                or
                result.retcode
                !=
                mt5.TRADE_RETCODE_DONE
            ):

                log(
                    f"ERROR | Close failed "
                    f"{symbol} "
                    f"retcode="
                    f"{getattr(result, 'retcode', None)} "
                    f"comment="
                    f"{getattr(result, 'comment', None)}"
                )

                return False


            # -------------------------------------------------
            # Wait for broker history
            # -------------------------------------------------

            time.sleep(
                0.5
            )


            snapshot = (

                snapshot

                or

                self._position_to_dict(
                    symbol,
                    mt5_position
                )

            )


            archived = (
                self._archive_closed_trade(
                    symbol,
                    snapshot,
                    exit_reason=reason,
                    close_time=(
                        candle_time
                        if candle_time
                        else datetime.now()
                    )
                )
            )


            if archived:

                self.positions.pop(
                    symbol,
                    None
                )


                if self.trade_manager:

                    self.trade_manager.remove_trade(
                        symbol
                    )


                self.refresh_account()


                log(
                    f"INFO | MT5 CLOSED "
                    f"{symbol} "
                    f"reason={reason}"
                )


                return True


            # -------------------------------------------------
            # The broker already closed the position, but
            # history may still be propagating.
            #
            # Keep local state so check_closed_trades()
            # can finish reconciliation.
            # -------------------------------------------------

            log(
                f"DEBUG | MT5 close sent but "
                f"history not ready "
                f"{symbol}"
            )


            return False


        except Exception as e:

            log(
                f"ERROR | close_position "
                f"{symbol}: {e}"
            )

            return False


    # =====================================================
    # CHECK BROKER-SIDE CLOSED TRADES
    # =====================================================

    def check_closed_trades(
            self
    ):

        try:

            changed = False


            for symbol, trade in list(
                    self.positions.items()
            ):

                # -------------------------------------------------
                # Still open at broker
                # -------------------------------------------------

                if self.has_open_trade(
                        symbol
                ):

                    continue


                # -------------------------------------------------
                # Broker closed it
                # -------------------------------------------------

                log(
                    f"INFO | Broker closed detected "
                    f"{symbol}"
                )


                snapshot = (
                    self._position_to_dict(
                        symbol,
                        None
                    )
                )


                if snapshot is None:

                    snapshot = trade


                archived = (
                    self._archive_closed_trade(
                        symbol,
                        snapshot,
                        exit_reason="BROKER_CLOSE",
                        close_time=datetime.now()
                    )
                )


                if archived:

                    self.positions.pop(
                        symbol,
                        None
                    )


                    if self.trade_manager:

                        self.trade_manager.remove_trade(
                            symbol
                        )


                    changed = True


                    log(
                        f"INFO | BROKER CLOSE "
                        f"ARCHIVED {symbol}"
                    )


                else:

                    log(
                        f"DEBUG | Waiting for broker "
                        f"history {symbol}"
                    )


            # -------------------------------------------------
            # Account is always refreshed after checking
            # broker-side position changes.
            # -------------------------------------------------

            if changed:

                self.refresh_account()


            return changed


        except Exception as e:

            log(
                f"ERROR | check_closed_trades "
                f"{e}"
            )

            return False


    # =====================================================
    # MODIFY POSITION
    # =====================================================

    def modify_position(
            self,
            symbol,
            stop_loss=None,
            take_profit=None
    ):

        try:

            position = self.get_position(
                symbol
            )


            if position is None:

                log(
                    f"WARNING | Modify skipped "
                    f"{symbol}: position not found"
                )

                return False


            info = mt5.symbol_info(
                symbol
            )


            if info is None:

                return False


            tick = mt5.symbol_info_tick(
                symbol
            )


            if tick is None:

                return False


            digits = int(
                info.digits
            )

            point = float(
                info.point
            )

            tick_size = float(
                info.trade_tick_size
            )

            stops_level = int(
                info.trade_stops_level
            )

            freeze_level = int(
                info.trade_freeze_level
            )


            if tick_size <= 0:

                tick_size = point


            minimum_distance = (
                max(
                    stops_level,
                    freeze_level
                )
                *
                point
            )


            if (
                position.type
                ==
                mt5.POSITION_TYPE_BUY
            ):

                market_price = float(
                    tick.bid
                )

            else:

                market_price = float(
                    tick.ask
                )


            new_sl = (

                stop_loss

                if stop_loss is not None

                else position.sl

            )


            new_tp = (

                take_profit

                if take_profit is not None

                else position.tp

            )


            # -------------------------------------------------
            # Tick-size normalization
            # -------------------------------------------------

            if (
                new_sl is not None
                and
                new_sl > 0
            ):

                new_sl = (
                    round(
                        float(new_sl)
                        /
                        tick_size
                    )
                    *
                    tick_size
                )


            if (
                new_tp is not None
                and
                new_tp > 0
            ):

                new_tp = (
                    round(
                        float(new_tp)
                        /
                        tick_size
                    )
                    *
                    tick_size
                )


            if new_sl is not None:

                new_sl = round(
                    new_sl,
                    digits
                )


            if new_tp is not None:

                new_tp = round(
                    new_tp,
                    digits
                )


            # -------------------------------------------------
            # Validate SL
            # -------------------------------------------------

            if (
                new_sl is not None
                and
                new_sl > 0
            ):

                if (
                    position.type
                    ==
                    mt5.POSITION_TYPE_BUY
                ):

                    if new_sl >= (
                        market_price
                        -
                        minimum_distance
                    ):

                        log(
                            f"WARNING | Invalid BUY SL "
                            f"{symbol}"
                        )

                        return False


                else:

                    if new_sl <= (
                        market_price
                        +
                        minimum_distance
                    ):

                        log(
                            f"WARNING | Invalid SELL SL "
                            f"{symbol}"
                        )

                        return False


            # -------------------------------------------------
            # Validate TP
            # -------------------------------------------------

            if (
                new_tp is not None
                and
                new_tp > 0
            ):

                if (
                    position.type
                    ==
                    mt5.POSITION_TYPE_BUY
                ):

                    if new_tp <= (
                        market_price
                        +
                        minimum_distance
                    ):

                        log(
                            f"WARNING | Invalid BUY TP "
                            f"{symbol}"
                        )

                        return False


                else:

                    if new_tp >= (
                        market_price
                        -
                        minimum_distance
                    ):

                        log(
                            f"WARNING | Invalid SELL TP "
                            f"{symbol}"
                        )

                        return False


            # -------------------------------------------------
            # MT5 SL/TP modification
            # -------------------------------------------------

            request = {

                "action":
                mt5.TRADE_ACTION_SLTP,

                "symbol":
                symbol,

                "position":
                position.ticket,

                "sl":
                new_sl
                if new_sl is not None
                else 0.0,

                "tp":
                new_tp
                if new_tp is not None
                else 0.0,

                "magic":
                self.magic,

                "comment":
                "AI TRAILING MODIFY"

            }


            result = mt5.order_send(
                request
            )


            if result is None:

                log(
                    f"ERROR | Modify failed "
                    f"{symbol}: "
                    f"{mt5.last_error()}"
                )

                return False


            if result.retcode != (
                mt5.TRADE_RETCODE_DONE
            ):

                log(
                    f"ERROR | Modify failed "
                    f"{symbol} "
                    f"retcode={result.retcode} "
                    f"comment={result.comment}"
                )

                return False


            # -------------------------------------------------
            # Update local state
            # -------------------------------------------------

            if symbol in self.positions:

                self.positions[
                    symbol
                ]["stop_loss"] = new_sl

                self.positions[
                    symbol
                ]["take_profit"] = new_tp


            log(
                f"INFO | MT5 MODIFY "
                f"{symbol} "
                f"SL={new_sl} "
                f"TP={new_tp}"
            )


            return True


        except Exception as e:

            log(
                f"ERROR | modify_position "
                f"{symbol}: {e}"
            )

            return False
