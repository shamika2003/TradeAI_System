# filename: signal_engine.py

import numpy as np
from analytics.logger import log
from config.settings import SIGNAL_THRESHOLD


# =====================================================
# SIGNAL ENGINE (ML → TRADE SIGNAL CONVERTER)
# =====================================================

class SignalState:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = None


# =====================================================
# MAIN DECISION FUNCTION
# =====================================================
def decide_signal(signal_value: float):
    """
    Converts ML output into trading action.
    """

    try:

        if signal_value is None:
            return SignalState.HOLD

        signal_value = float(signal_value)

        # -----------------------------
        # NO TRADE ZONE (NOISE FILTER)
        # -----------------------------
        if abs(signal_value) < SIGNAL_THRESHOLD:
            return SignalState.HOLD

        # -----------------------------
        # BUY SIGNAL
        # -----------------------------
        if signal_value > 0:
            return SignalState.BUY

        # -----------------------------
        # SELL SIGNAL
        # -----------------------------
        if signal_value < 0:
            return SignalState.SELL

        return SignalState.HOLD

    except Exception as e:
        log(f"ERROR | Signal engine failed: {e}")
        return SignalState.HOLD