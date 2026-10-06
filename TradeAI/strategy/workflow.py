from __future__ import annotations

import argparse

from TradeAI.strategy.build_signal_policy import build_all_policies
from TradeAI.strategy.evaluate_signal_policy import evaluate_all_policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradeAI precision-entry signal workflow"
    )
    parser.add_argument(
        "command",
        choices=["build", "evaluate", "all"],
    )
    args = parser.parse_args()

    if args.command == "build":
        build_all_policies()
    elif args.command == "evaluate":
        evaluate_all_policies()
    else:
        build_all_policies()
        evaluate_all_policies()


if __name__ == "__main__":
    main()
