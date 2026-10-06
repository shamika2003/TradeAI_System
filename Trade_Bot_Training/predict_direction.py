from __future__ import annotations

import argparse

from Trade_Bot_Training.data_collector import MarketDataCollector
from Trade_Bot_Training.direction_config import SYMBOLS
from Trade_Bot_Training.direction_model import DirectionModel
from Trade_Bot_Training.feature_engine import build_feature_frame, model_feature_columns


def predict_latest(symbol: str) -> None:
    symbol = str(symbol).upper()
    if symbol not in SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    collector = MarketDataCollector()
    try:
        m5 = collector.fetch_recent_closed(symbol, "M5", 500)
        h1 = collector.fetch_recent_closed(symbol, "H1", 300)
    finally:
        collector.disconnect()

    frame = build_feature_frame(m5, h1)
    feature_columns = model_feature_columns(frame)
    valid = frame.dropna(subset=feature_columns)
    if valid.empty:
        raise RuntimeError("Not enough recent history to build a valid feature row.")

    latest = valid.tail(1).copy()
    model = DirectionModel(symbol)
    prediction = model.predict_proba(latest).iloc[0]
    row = latest.iloc[0]

    print()
    print("=" * 78)
    print(f"TRADEAI DIRECTION MODEL | {symbol}")
    print("=" * 78)
    print(f"Last M5 bar open : {row['time']}")
    print(f"Decision time    : {row['decision_time']}")
    print(f"Current close    : {row['close']}")
    print(f"Forecast horizon : {int(prediction['horizon_minutes'])} minutes")
    print(f"P(DOWN)          : {prediction['p_down']:.4f}")
    print(f"P(UP)            : {prediction['p_up']:.4f}")
    print(f"Direction        : {prediction['direction']}")
    print(f"Confidence       : {prediction['confidence']:.4f}")
    print(f"Expected move    : {prediction['expected_move_atr']:+.4f} ATR")
    print(f"Expected move    : {prediction['expected_move_bps']:+.3f} bps")
    print()
    print("MODEL OUTPUT ONLY - NO TRADE DECISION WAS MADE.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("symbol", choices=SYMBOLS)
    args = parser.parse_args()
    predict_latest(args.symbol)


if __name__ == "__main__":
    main()
