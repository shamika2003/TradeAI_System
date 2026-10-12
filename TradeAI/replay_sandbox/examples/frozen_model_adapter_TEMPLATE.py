"""TEMPLATE ONLY. Must be adapted to the REAL frozen model and signal/manager API.

IMPORTANT: supply the real TradeAI model artifacts, feature schema, thresholds,
manager and execution policy before connecting. Never fit/retrain during replay.
"""
from ..broker import Signal
from ..runner import Observation


def decide(observation: Observation) -> Signal | None:
    # Available causal history: observation.completed_m5 / completed_h1.
    # These collections never include the current candle's high/low/close.
    # TODO: Call the original TradeAI feature builder with AS-OF timestamps.
    # TODO: Load only frozen model+threshold artifacts, verify hashes/freeze time.
    # TODO: Run the original entry AND manager policies, not a hand-made substitute.
    # TODO: Convert a validated existing-engine order intention to Signal(...).
    raise NotImplementedError("Integrate the verified frozen TradeAI model/manager first")
