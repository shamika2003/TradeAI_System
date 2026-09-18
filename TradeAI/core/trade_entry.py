from shared.tradeai_core.decision_policy import SIGNAL_EPISODE_RESET_BARS
from analytics.logger import log


class TradeEntryEngine:


    def __init__(
            self,
            executor,
            risk_manager,
            config
    ):

        self.executor = executor

        self.risk = risk_manager

        self.config = config

        # prevent duplicate entries on same candle
        self.last_signal = {}

        # One execution attempt per continuous structural signal episode.
        # Repeated adjacent BUY/SELL predictions are the same opportunity, not
        # fresh independent trades. The episode resets only after several HOLD
        # bars (or on a direct direction reversal).
        self.signal_episode_direction = {}
        self.signal_episode_hold_bars = {}

        log(
            "INFO | Trade Entry Engine initialized"
        )



    # =====================================================
    # DECODE AI PREDICTION
    # =====================================================

    def decode_prediction(
            self,
            prediction
    ):
        if prediction is None:
            return None

        # Stage 9 opportunity models make an explicit eligibility decision.
        # This avoids converting every weak signed score into a trade.
        if "trade_eligible" in prediction:
            if prediction.get("policy_enabled", True) is False:
                return None
            if not bool(prediction.get("trade_eligible", False)):
                return None
            direction = str(prediction.get("direction") or "").upper()
            if direction not in ("BUY", "SELL"):
                return None
            log(
                f"DEBUG | Opportunity {direction} "
                f"P={float(prediction.get('confidence', 0.0)):.3f} "
                f"EV={float(prediction.get('selected_expected_r', 0.0)):+.3f}R"
            )
            return direction

        # Legacy fallback for old tests/artifacts.
        signal = prediction.get("signal")
        confidence = prediction.get("confidence", 0)
        if signal is None:
            return None
        min_confidence = float(prediction.get("min_confidence", self.config.MIN_CONFIDENCE))
        signal_threshold = float(prediction.get("signal_threshold", self.config.SIGNAL_THRESHOLD))
        if prediction.get("policy_enabled", True) is False or confidence < min_confidence:
            return None
        if abs(float(signal)) < signal_threshold:
            return None
        return "BUY" if float(signal) > 0 else "SELL"




    # =====================================================
    # CHECK ENTRY
    # =====================================================

    def check_entry(
            self,
            symbol,
            prediction,
            df,
            candle_time=None
    ):


        try:


            direction = self.decode_prediction(
                prediction
            )


            if direction is None:
                hold_bars = int(self.signal_episode_hold_bars.get(symbol, 0)) + 1
                self.signal_episode_hold_bars[symbol] = hold_bars
                if hold_bars >= int(SIGNAL_EPISODE_RESET_BARS):
                    self.signal_episode_direction.pop(symbol, None)
                return False

            active_episode = self.signal_episode_direction.get(symbol)
            if active_episode == direction:
                self.signal_episode_hold_bars[symbol] = 0
                log(f"DEBUG | Same setup episode blocked {symbol} {direction}")
                return False

            # A direct reversal is a genuinely different setup. Otherwise this
            # records the first eligible bar of a new episode even if broker
            # risk later rejects it, matching offline candidate de-clustering.
            self.signal_episode_direction[symbol] = direction
            self.signal_episode_hold_bars[symbol] = 0



            if df is None or df.empty:

                return False




            price = float(
                df["close"].iloc[-1]
            )




            # =================================================
            # DUPLICATE CANDLE CHECK
            # =================================================

            current_signal = (
                direction,
                candle_time
            )


            if self.last_signal.get(symbol) == current_signal:


                log(
                    f"DEBUG | Duplicate signal blocked {symbol}"
                )


                return False





            # =================================================
            # RISK CHECK
            # =================================================


            if not self.risk.can_trade(
                symbol,
                candle_time
            ):


                log(
                    f"DEBUG | Risk blocked {symbol}"
                )


                return False






# =================================================
# BASE SL TP
# =================================================

            stop_loss, take_profit = (

                self.risk.calculate_stop_targets(

                    symbol,

                    direction,

                    price,

                    df

                )

            )


            if (
                stop_loss is None
                or
                take_profit is None
            ):

                log(
                    f"ERROR | Missing SL TP {symbol}"
                )

                return False





            # =================================================
            # RISK BASED LOT SIZE
            # =================================================


            stop_distance = abs(
                price - stop_loss
            )


            lot = self.risk.calculate_lot(

                symbol,

                price=price,

                direction=direction,

                stop_loss=stop_loss

            )


            # =================================================
            # VALIDATE LOT BEFORE RISK CHECK
            # =================================================

            if lot is None or lot <= 0:

                log(
                    f"DEBUG | No valid broker-compatible lot "
                    f"{symbol}"
                )

                return False


            # =================================================
            # FINAL RISK AMOUNT CHECK
            # =================================================

            if not self.risk.check_risk_amount(

                    symbol,

                    lot,

                    stop_distance,

                    price=price,

                    direction=direction,

                    stop_loss=stop_loss

            ):

                log(
                    f"DEBUG | Risk rejected {symbol}"
                )

                return False



            if (
                stop_loss is None
                or
                take_profit is None
            ):


                log(
                    f"ERROR | Missing SL TP {symbol}"
                )

                return False







            # =================================================
            # SEND TO EXECUTOR
            # =================================================


            result = self.executor.open_trade(

                symbol=symbol,

                direction=direction,

                price=price,

                lot=lot,

                stop_loss=stop_loss,

                take_profit=take_profit,

                candle_time=candle_time

            )





            if not result:

                log(
                    f"ERROR | Executor rejected {symbol}"
                )

                return False







            # remember this candle signal

            self.last_signal[symbol] = current_signal





            # risk lock

            self.risk.register_trade_open(

                symbol,

                candle_time

            )





            log(

                f"INFO | ENTRY SUCCESS "
                f"{symbol} "
                f"{direction} "
                f"SL={stop_loss:.5f} "
                f"TP={take_profit:.5f}"

            )



            return True





        except Exception as e:


            log(

                f"ERROR | Trade entry failed "
                f"{symbol}: {e}"

            )


            return False





    # =====================================================
    # RESET SYMBOL AFTER CLOSE
    # =====================================================

    def reset_symbol(
            self,
            symbol
    ):


        if symbol in self.last_signal:

            del self.last_signal[symbol]


            log(
                f"DEBUG | Signal reset {symbol}"
            )
