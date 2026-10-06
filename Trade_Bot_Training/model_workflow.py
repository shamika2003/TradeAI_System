from __future__ import annotations

import argparse

from Trade_Bot_Training.data_collector import collect_all
from Trade_Bot_Training.dataset_builder import build_all
from Trade_Bot_Training.direction_config import SYMBOLS
from Trade_Bot_Training.predict_direction import predict_latest
from Trade_Bot_Training.train_direction_model import train_all
from Trade_Bot_Training.validate_direction_model import validate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradeAI active executable direction-model workflow"
    )
    parser.add_argument(
        "command",
        choices=["collect", "build", "train", "validate", "all", "predict"],
    )
    parser.add_argument("--symbol", choices=SYMBOLS, default="EURUSD")
    args = parser.parse_args()

    if args.command == "collect":
        collect_all()
    elif args.command == "build":
        build_all()
    elif args.command == "train":
        train_all()
    elif args.command == "validate":
        validate_all()
    elif args.command == "all":
        collect_all()
        build_all()
        train_all()
        validate_all()
    elif args.command == "predict":
        predict_latest(args.symbol)


if __name__ == "__main__":
    main()
