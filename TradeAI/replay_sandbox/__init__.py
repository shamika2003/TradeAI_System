"""Offline accelerated historical replay sandbox. NEVER connects to MT5 order API."""
from .broker import Signal
from .runner import Observation, run_replay

__all__ = ["Signal", "Observation", "run_replay"]
