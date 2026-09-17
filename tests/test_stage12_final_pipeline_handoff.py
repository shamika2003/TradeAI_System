from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "Trade_Bot_Training" / "final_clean_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("tradeai_final_clean_validation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage4_drawdown_failure_is_diagnostic_not_final_blocker():
    module = _load_module()
    report = {
        "acceptance_pass": False,
        "acceptance_gates": {
            "150.0": {
                "normal": {
                    "net_profit_positive": True,
                    "profit_factor": True,
                    "max_drawdown": False,
                    "trade_count": True,
                },
                "stress": {
                    "net_profit_nonnegative": True,
                    "profit_factor": True,
                    "max_drawdown": False,
                    "trade_count": True,
                },
            }
        },
    }

    diag = module._stage4_diagnostic(report)

    assert diag["acceptance_pass"] is False
    assert diag["failed_checks"] == [
        "150.0.normal.max_drawdown",
        "150.0.stress.max_drawdown",
    ]


def test_stage4_pass_has_no_failed_checks():
    module = _load_module()
    report = {
        "acceptance_pass": True,
        "acceptance_gates": {
            "150.0": {
                "normal": {"max_drawdown": True},
                "stress": {"max_drawdown": True},
            }
        },
    }

    diag = module._stage4_diagnostic(report)

    assert diag == {"acceptance_pass": True, "failed_checks": []}
