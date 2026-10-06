from __future__ import annotations

import argparse

from Trade_Bot_Training.precision_config import SYMBOLS
from Trade_Bot_Training.precision_dataset_builder import build_all
from Trade_Bot_Training.train_precision_model import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeAI selective precision direction model")
    parser.add_argument("command", choices=["build", "train", "all", "predict"])
    parser.add_argument("--symbol", choices=SYMBOLS, default="EURUSD")
    args = parser.parse_args()

    if args.command == "build":
        build_all()
    elif args.command == "train":
        train_all()
    elif args.command == "all":
        build_all()
        train_all()
    elif args.command == "predict":
        from Trade_Bot_Training.predict_precision import predict_latest
        predict_latest(args.symbol)


if __name__ == "__main__":
    main()
