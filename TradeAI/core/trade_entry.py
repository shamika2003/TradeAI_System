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


        signal = prediction.get(
            "signal"
        )


        confidence = prediction.get(
            "confidence",
            0
        )


        log(
            f"DEBUG | Prediction "
            f"signal={signal:.6f} "
            f"confidence={confidence:.2f}"
            if signal is not None
            else
            "DEBUG | Empty prediction"
        )



        if signal is None:
            return None



        # calibrated per-symbol policy thresholds are supplied by Predictor.
        # Fall back to config only for compatibility/testing.
        min_confidence = float(
            prediction.get(
                "min_confidence",
                self.config.MIN_CONFIDENCE
            )
        )

        signal_threshold = float(
            prediction.get(
                "signal_threshold",
                self.config.SIGNAL_THRESHOLD
            )
        )

        if prediction.get("policy_enabled", True) is False:
            return None

        if confidence < min_confidence:

            log(
                f"DEBUG | Low confidence "
                f"{confidence:.2f} < {min_confidence:.2f}"
            )

            return None



        # movement / probability-edge filter

        if abs(signal) < signal_threshold:

            return None



        if signal > 0:

            return "BUY"


        return "SELL"





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

                return False



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
