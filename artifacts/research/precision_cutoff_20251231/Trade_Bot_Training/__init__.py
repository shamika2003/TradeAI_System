"""TradeAI Direction Model V1 training package.

This package is intentionally limited to:
- market data collection
- causal feature generation
- next-M5 directional labels
- model training / calibration
- model-only evaluation and inference

Trade entry, position sizing, stop-loss, take-profit and trade management
belong to later TradeAI layers and are deliberately excluded here.
"""