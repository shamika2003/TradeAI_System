"""EXAMPLE ONLY: moving average crossover, NOT A TRADEAI TRAINED MODEL."""
from __future__ import annotations

from ..broker import Signal
from ..runner import Observation


def decide(observation: Observation) -> Signal | None:
    prices = [bar.close for bar in observation.completed_m5]
    if len(prices) < 31:
        return None
    a = sum(prices[-10:]) / 10
    b = sum(prices[-30:]) / 30
    a_prev = sum(prices[-11:-1]) / 10
    b_prev = sum(prices[-31:-1]) / 30
    if a_prev <= b_prev and a > b:
        return Signal("BUY", 0.01, sl_pips=15, tp_pips=25, note="DEMO_MA_CROSS_NOT_TRADEAI")
    if a_prev >= b_prev and a < b:
        return Signal("SELL", 0.01, sl_pips=15, tp_pips=25, note="DEMO_MA_CROSS_NOT_TRADEAI")
    return None
