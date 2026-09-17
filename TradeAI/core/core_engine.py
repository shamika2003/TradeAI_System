# filename: core/core_engine.py

from analytics.logger import log

from core.predictor import Predictor
from core.trade_entry import TradeEntryEngine
from core.trade_manager import TradeManager
from core.risk_manager import RiskManager

from config.settings import (
    SYMBOLS,
    MAX_OPEN_TRADES,
    MAX_TOTAL_TRADES,
    COOLDOWN_SECONDS
)


class CoreEngine:

    """
    Central trading pipeline.

    Architecture:

        MARKET DATA
             ↓
        CoreEngine
             ↓
        Existing trade?
          /       \
        YES       NO
         ↓         ↓
    Execution   Predictor
         ↓         ↓
       SL/TP     Entry
         ↓         ↓
       Close     Trade
                   ↓
              Executor

    BACKTEST/PAPER:

        ReplayEngine provides the current future candle
        ONLY to the execution layer.

        The predictor receives ONLY historical candles.

    LIVE:

        ReplayEngine is None.

        MT5 manages broker-side SL/TP.
    """


    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(
            self,
            executor,
            replay=None
    ):

        self.executor = executor

        # -------------------------------------------------
        # ReplayEngine
        #
        # None in LIVE mode.
        # ReplayEngine instance in PAPER/BACKTEST mode.
        # -------------------------------------------------

        self.replay = replay


        # =================================================
        # AI PREDICTORS + CALIBRATED PRODUCTION RISK
        # =================================================

        self.predictors = {

            symbol:

            Predictor()

            for symbol in SYMBOLS

        }

        if not self.predictors:
            raise RuntimeError("No production predictors configured")

        first_predictor = next(iter(self.predictors.values()))
        risk_policy = first_predictor.get_risk_policy()
        calibrated_risk_percent = float(risk_policy["risk_percent"])


        # =================================================
        # RISK
        # =================================================

        self.risk = RiskManager(

            executor=self.executor,

            max_open_trades=MAX_OPEN_TRADES,

            max_total_trades=MAX_TOTAL_TRADES,

            cooldown_seconds=COOLDOWN_SECONDS,

            risk_percent=calibrated_risk_percent

        )


        # =================================================
        # TRADE MANAGER
        # =================================================

        self.trade_manager = TradeManager(

            executor=self.executor

        )


        self.executor.register_trade_manager(

            self.trade_manager

        )


        # =================================================
        # ENTRY
        # =================================================

        self.entry = TradeEntryEngine(

            executor=self.executor,

            risk_manager=self.risk,

            config=self._load_config()

        )


        # =================================================
        # STATE
        # =================================================

        self.synced = False

        self.last_prediction_candle = {}
        self.last_management_prediction_candle = {}
        self.management_prediction_cache = {}


        log(
            "INFO | Core Engine Ready"
        )


    # =====================================================
    # CONFIG
    # =====================================================

    def _load_config(
            self
    ):

        import config.settings as config

        return config


    # =====================================================
    # RESET SYMBOL
    # =====================================================

    def reset_symbol(
            self,
            symbol
    ):

        self.entry.reset_symbol(
            symbol
        )


    # =====================================================
    # GET CURRENT REPLAY CANDLE
    # =====================================================

    def _get_replay_candle(
            self,
            symbol
    ):

        """
        Return the future candle from ReplayEngine.

        IMPORTANT:

        This candle is NEVER passed to Predictor.

        It exists only for execution simulation.
        """

        if self.replay is None:

            return None


        try:

            return self.replay.get_current_candle(
                symbol
            )

        except Exception as e:

            log(
                f"ERROR | Replay candle "
                f"{symbol}: {e}"
            )

            return None


    # =====================================================
    # MANAGEMENT PREDICTION
    # =====================================================

    def _get_management_prediction(
            self,
            symbol,
            df,
            candle_time
    ):

        """
        Get one causal model decision per candle while a trade is open.

        In backtest mode ``df`` contains only candles completed before the
        replay execution candle. The result can therefore be used to manage
        the position before that future candle is processed.
        """

        if df is None or df.empty:
            return None

        cached_candle = self.last_management_prediction_candle.get(symbol)
        if cached_candle == candle_time:
            return self.management_prediction_cache.get(symbol)

        predictor = self.predictors.get(symbol)
        if predictor is None:
            return None

        try:
            prediction = predictor.predict(df, symbol)
        except TypeError:
            prediction = predictor.predict(df, None, symbol)
        except Exception as e:
            log(f"WARNING | Management prediction {symbol}: {e}")
            prediction = None

        self.last_management_prediction_candle[symbol] = candle_time
        self.management_prediction_cache[symbol] = prediction

        return prediction


    # =====================================================
    # MANAGE PAPER/BACKTEST TRADE
    # =====================================================

    def _manage_paper_trade(
            self,
            symbol,
            candle_time
    ):

        """
        Process the actual replay candle against an
        existing PAPER position.

        This is deliberately separated from prediction.

        Predictor:

            historical candles only

        Executor:

            current future candle
        """

        if self.replay is None:

            return False


        # -------------------------------------------------
        # Get future candle
        # -------------------------------------------------

        candle = self._get_replay_candle(
            symbol
        )


        if candle is None:

            return False


        # -------------------------------------------------
        # Executor must support candle execution
        # -------------------------------------------------

        if not hasattr(
                self.executor,
                "update_candle"
        ):

            return False


        # -------------------------------------------------
        # Execute candle
        # -------------------------------------------------

        closed = self.executor.update_candle(

            symbol=symbol,

            candle=candle,

            candle_time=candle_time

        )


        # -------------------------------------------------
        # Trade closed
        # -------------------------------------------------

        if closed:

            self.risk.register_trade_close(

                symbol,

                candle_time

            )


            self.reset_symbol(
                symbol
            )


            log(
                f"INFO | PAPER TRADE CLOSED "
                f"{symbol}"
            )


            return True


        return False


    # =====================================================
    # MANAGE LIVE TRADE
    # =====================================================

    def _manage_live_trade(
            self,
            symbol,
            current_atr,
            prediction=None,
            candle_time=None
    ):

        """
        Live MT5 trade management.

        MT5 itself handles broker-side SL/TP.

        TradeManager handles:

            - Break-even
            - Trailing stop
            - local position management
        """

        # -------------------------------------------------
        # Check broker/executor state
        # -------------------------------------------------

        if not self.executor.has_open_trade(
                symbol
        ):

            return False


        # -------------------------------------------------
        # MT5 executor synchronizes closed trades
        # -------------------------------------------------

        if hasattr(
                self.executor,
                "check_closed_trades"
        ):

            try:

                self.executor.check_closed_trades()

            except Exception as e:

                log(
                    f"WARNING | Closed trade "
                    f"sync {symbol}: {e}"
                )


        # -------------------------------------------------
        # Trade manager
        # -------------------------------------------------

        self.trade_manager.update(

            symbol,

            current_atr,

            candle_time=candle_time,

            prediction=prediction

        )

        # TradeManager may close a position because MAX_HOLD was reached.
        if not self.executor.has_open_trade(symbol):
            self.risk.register_trade_close(symbol)
            self.reset_symbol(symbol)

        return True


    # =====================================================
    # PROCESS
    # =====================================================

    def process(
            self,
            symbol,
            df,
            candle_time=None
    ):

        try:

            # =================================================
            # SYNC EXECUTOR POSITIONS ONCE
            # =================================================

            if not self.synced:

                self.trade_manager.sync_positions()

                self.synced = True


            # =================================================
            # VALIDATE DATA
            # =================================================

            if df is None or df.empty:

                return


            # =================================================
            # CURRENT ATR
            # =================================================

            current_atr = None


            if "atr" in df.columns:

                try:

                    current_atr = float(
                        df["atr"].iloc[-1]
                    )

                except Exception:

                    current_atr = None


            # =================================================
            # EXISTING TRADE
            # =================================================

            if self.executor.has_open_trade(
                    symbol
            ):

                # -------------------------------------------------
                # Causal management prediction
                # -------------------------------------------------
                #
                # While a trade is open we still ask the model for its
                # current directional edge. In replay mode df ends BEFORE
                # the current execution candle, so this cannot see the
                # future candle.
                # -------------------------------------------------

                management_prediction = self._get_management_prediction(
                    symbol=symbol,
                    df=df,
                    candle_time=candle_time
                )

                # -------------------------------------------------
                # PAPER / BACKTEST
                # -------------------------------------------------
                #
                # Management happens BEFORE the future execution candle.
                # This lets a stop moved from the last completed candle
                # protect the next candle instead of being applied one bar
                # too late.
                # -------------------------------------------------

                if self.replay is not None:

                    self.trade_manager.update(
                        symbol,
                        current_atr,
                        candle_time=candle_time,
                        prediction=management_prediction
                    )

                    # Manager may close because of MAX_HOLD or an AI
                    # defensive exit before the future candle executes.
                    if not self.executor.has_open_trade(symbol):
                        self.risk.register_trade_close(symbol, candle_time)
                        self.reset_symbol(symbol)
                        return

                    closed = self._manage_paper_trade(
                        symbol=symbol,
                        candle_time=candle_time
                    )

                    if closed:
                        return

                    return

                # -------------------------------------------------
                # LIVE MT5
                # -------------------------------------------------

                self._manage_live_trade(
                    symbol=symbol,
                    current_atr=current_atr,
                    prediction=management_prediction,
                    candle_time=candle_time
                )

                return


            # =================================================
            # NEW CANDLE VALIDATION
            # =================================================
            #
            # This is especially important in LIVE mode.
            #
            # The main loop may run every few seconds while the
            # strategy works on M5 candles.
            #
            # Therefore prediction must happen once per candle.
            #

            if candle_time is None:

                log(
                    f"DEBUG | No candle time "
                    f"for {symbol}"
                )

                return


            last_candle = (
                self.last_prediction_candle.get(
                    symbol
                )
            )


            if last_candle == candle_time:

                return


            # -------------------------------------------------
            # Mark candle as processed BEFORE prediction.
            #
            # Prevents duplicate model calls.
            # -------------------------------------------------

            self.last_prediction_candle[symbol] = (
                candle_time
            )


            # =================================================
            # PREDICTOR
            # =================================================

            predictor = self.predictors.get(
                symbol
            )


            if predictor is None:

                log(
                    f"WARNING | No predictor "
                    f"for {symbol}"
                )

                return


            try:

                prediction = predictor.predict(

                    df,

                    symbol

                )

            except TypeError:

                # Compatibility with older Predictor API.

                prediction = predictor.predict(

                    df,

                    None,

                    symbol

                )


            if prediction is None:

                return


            # =================================================
            # STRUCTURED MODEL TELEMETRY FOR THE DESKTOP UI
            # =================================================
            # Keep the model decision tied to its symbol.  Older UI builds tried
            # to infer symbol ownership by zipping the last N generic prediction
            # log lines against SYMBOLS; that becomes wrong as soon as one symbol
            # skips a candle/model call.  The UI now prefers this explicit record.
            try:
                signal_value = float(prediction.get("signal", 0.0) or 0.0)
                confidence_value = float(prediction.get("confidence", 0.0) or 0.0)
                min_confidence = float(prediction.get("min_confidence", 0.0) or 0.0)
                signal_threshold = float(prediction.get("signal_threshold", 0.0) or 0.0)
                policy_enabled = bool(prediction.get("policy_enabled", True))
                p_buy = float(prediction.get("p_buy", 0.0) or 0.0)
                p_hold = float(prediction.get("p_hold", 0.0) or 0.0)
                p_sell = float(prediction.get("p_sell", 0.0) or 0.0)
                log(
                    f"INFO | MODEL_STATE | symbol={symbol} "
                    f"signal={signal_value:+.6f} "
                    f"confidence={confidence_value:.6f} "
                    f"min_confidence={min_confidence:.6f} "
                    f"signal_threshold={signal_threshold:.6f} "
                    f"enabled={1 if policy_enabled else 0} "
                    f"p_buy={p_buy:.6f} p_hold={p_hold:.6f} p_sell={p_sell:.6f} "
                    f"candle={candle_time}"
                )
            except Exception as telemetry_error:
                log(f"WARNING | Model telemetry failed {symbol}: {telemetry_error}")


            # =================================================
            # ENTRY
            # =================================================

            self.entry.check_entry(

                symbol=symbol,

                prediction=prediction,

                df=df,

                candle_time=candle_time

            )


        except Exception as e:

            log(

                f"ERROR | Core process "
                f"{symbol}: {e}"

            )
