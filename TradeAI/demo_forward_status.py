# filename: TradeAI/demo_forward_status.py

from pathlib import Path
import sys


def _add_system_root() -> Path:
    """
    Make TradeAI_System importable no matter where this script is launched from.

    Expected layout:
        TradeAI_System/
            shared/tradeai_core/
            TradeAI/demo_forward_status.py
    """
    here = Path(__file__).resolve()

    for parent in here.parents:
        if (parent / "shared" / "tradeai_core").is_dir():
            root_text = str(parent)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return parent

    raise RuntimeError(
        "TradeAI_System root not found. "
        "Expected shared/tradeai_core beside the TradeAI folder."
    )


SYSTEM_ROOT = _add_system_root()

from config.settings import (  # noqa: E402
    DEMO_FORWARD_CAPITAL,
    DEMO_FORWARD_STATE_PATH,
    DEMO_FORWARD_MIN_TRADES,
    DEMO_FORWARD_MAX_DD_PERCENT,
    DEMO_FORWARD_MIN_PROFIT_FACTOR,
    DEMO_FORWARD_MIN_RETURN_PERCENT,
)
from shared.tradeai_core.demo_forward import DemoForwardLedger  # noqa: E402


def main():
    ledger = DemoForwardLedger(
        path=DEMO_FORWARD_STATE_PATH,
        initial_balance=DEMO_FORWARD_CAPITAL,
        min_trades=DEMO_FORWARD_MIN_TRADES,
        max_dd_percent=DEMO_FORWARD_MAX_DD_PERCENT,
        min_profit_factor=DEMO_FORWARD_MIN_PROFIT_FACTOR,
        min_return_percent=DEMO_FORWARD_MIN_RETURN_PERCENT,
    )

    s = ledger.summary()
    pf = s["profit_factor"]
    pf_text = "INF" if pf == float("inf") else f"{pf:.3f}"

    print("\n" + "=" * 80)
    print("TRADEAI DEMO FORWARD GATE STATUS")
    print("=" * 80)
    print(f"Virtual account : ${s['initial_balance']:.2f} -> ${s['virtual_balance']:.2f}")
    print(f"Net / return    : ${s['net_profit']:+.2f} ({s['return_percent']:+.1f}%)")
    print(f"Closed trades   : {s['closed_trades']} / {s['min_trades_required']} required")
    print(f"Win rate        : {s['win_rate_percent']:.1f}%")
    print(f"Profit factor   : {pf_text} / >= {s['min_profit_factor_required']:.2f}")
    print(f"Max drawdown    : {s['max_drawdown_percent']:.2f}% / <= {s['max_dd_percent_allowed']:.2f}%")
    print(f"Daily loss now  : {s['daily_loss_percent']:.2f}%")
    print(f"By symbol       : {s['by_symbol']}")
    print(f"Risk rejections : {s['rejections']}")
    print("-" * 80)
    print("GATE            : " + ("PASS" if s["accepted"] else "COLLECTING / NOT YET PASSED"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
