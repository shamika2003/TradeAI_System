from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TradeAI"))

from core.risk_manager import RiskManager
from execution.paper_executor import PaperExecutor


def _fallback_executor():
    """Stage-2 unit tests intentionally exercise config fallback economics.

    A captured broker_profile.json may exist after Stage 4. Clear it here so
    these deterministic tests keep the original 0.001 fallback lot step.
    """
    ex = PaperExecutor(capital=150.0)
    ex._broker_profile = None
    return ex


def test_paper_expected_entry_matches_actual_fill():
    ex = _fallback_executor()

    requested = 160.545
    expected = ex.estimate_entry_price(
        "USDJPY",
        "SELL",
        requested,
    )

    # MT5 historical bars are BID-side. A SELL enters on BID,
    # so only the configured 0.2 pip adverse slippage applies.
    assert abs(expected - 160.543) < 1e-9

    ok = ex.open_trade(
        symbol="USDJPY",
        direction="SELL",
        price=requested,
        lot=0.023,
        stop_loss=160.562,
        take_profit=160.510,
    )

    assert ok is True
    position = ex.get_position("USDJPY")
    assert abs(position["entry_price"] - expected) < 1e-12


def test_150_account_sizes_from_net_stop_loss_not_candle_distance():
    ex = _fallback_executor()
    rm = RiskManager(
        executor=ex,
        risk_percent=0.30,
    )

    lot = rm.calculate_lot(
        "USDJPY",
        price=160.545,
        direction="SELL",
        stop_loss=160.562,
    )

    # The old candle-close-only method produced about 0.042 lots.
    # With BID-side MT5 parity plus slippage + commission, 0.023 lots
    # keeps the expected stop loss inside the $0.45 risk budget.
    assert lot == 0.023
    assert lot < 0.042

    estimate = ex.estimate_stop_loss_risk(
        "USDJPY",
        "SELL",
        160.545,
        160.562,
        lot,
    )

    assert estimate["entry_price"] == expected_entry(ex)
    assert estimate["net_risk"] <= 0.45 + 1e-9
    assert estimate["price_risk"] > 0
    assert estimate["commission"] > 0


def expected_entry(ex):
    return ex.estimate_entry_price(
        "USDJPY",
        "SELL",
        160.545,
    )


def test_final_risk_gate_uses_net_risk():
    ex = _fallback_executor()
    rm = RiskManager(
        executor=ex,
        risk_percent=0.30,
    )

    assert rm.check_risk_amount(
        "USDJPY",
        0.023,
        price=160.545,
        direction="SELL",
        stop_loss=160.562,
    ) is True

    # The historical 0.042 lot still exceeds the $0.45 net-risk budget,
    # and must now be rejected by the pre-trade gate.
    assert rm.check_risk_amount(
        "USDJPY",
        0.042,
        price=160.545,
        direction="SELL",
        stop_loss=160.562,
    ) is False


def test_actual_paper_stop_loss_stays_inside_150_account_risk_budget():
    ex = _fallback_executor()
    rm = RiskManager(
        executor=ex,
        risk_percent=0.30,
    )

    lot = rm.calculate_lot(
        "USDJPY",
        price=160.545,
        direction="SELL",
        stop_loss=160.562,
    )

    assert lot == 0.023

    assert ex.open_trade(
        symbol="USDJPY",
        direction="SELL",
        price=160.545,
        lot=lot,
        stop_loss=160.562,
        take_profit=160.510,
    ) is True

    assert ex.update_candle(
        "USDJPY",
        {
            "open": 160.540,
            "high": 160.570,
            "low": 160.530,
            "close": 160.560,
        },
    ) is True

    trade = ex.trade_history[-1]
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["profit"] >= -0.45
    assert ex.get_balance() >= 149.55