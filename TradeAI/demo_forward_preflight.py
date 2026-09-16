# filename: TradeAI/demo_forward_preflight.py

from pathlib import Path
import sys

import MetaTrader5 as mt5


def _add_system_root() -> Path:

    here = Path(__file__).resolve()

    for parent in here.parents:

        if (
            parent /
            "shared" /
            "tradeai_core"
        ).is_dir():

            root_text = str(
                parent
            )

            if root_text not in sys.path:
                sys.path.insert(
                    0,
                    root_text
                )

            return parent

    raise RuntimeError(
        "TradeAI_System root not found. "
        "Expected shared/tradeai_core beside the TradeAI folder."
    )


SYSTEM_ROOT = _add_system_root()


from config.settings import (  # noqa: E402
    DEMO_FORWARD_CAPITAL,
    DEMO_FORWARD_REQUIRE_DEMO_ACCOUNT,
    MAX_ACTUAL_RISK_PERCENT,
    MODEL_PATH,
    SYMBOLS,
)

from core.predictor import Predictor  # noqa: E402
from market.mt5_feed import initialize_mt5  # noqa: E402


def _pip_size(info):

    point = float(
        getattr(
            info,
            "point",
            0.0
        )
        or
        0.0
    )

    digits = int(
        getattr(
            info,
            "digits",
            0
        )
        or
        0
    )

    if point <= 0:
        return 0.0

    if digits in (3, 5):
        return point * 10.0

    return point


def _pip_value_per_lot(info):

    tick_size = float(
        getattr(
            info,
            "trade_tick_size",
            0.0
        )
        or
        0.0
    )

    tick_value = float(
        getattr(
            info,
            "trade_tick_value",
            0.0
        )
        or
        0.0
    )

    pip = _pip_size(
        info
    )

    if (
        tick_size <= 0
        or
        tick_value <= 0
        or
        pip <= 0
    ):
        return 0.0

    return (
        tick_value *
        (
            pip /
            tick_size
        )
    )


def main():

    print(
        "\n" +
        "=" * 80
    )

    print(
        "TRADEAI DEMO FORWARD PREFLIGHT — NO ORDERS ARE SENT"
    )

    print(
        "=" * 80
    )

    try:

        initialize_mt5()

        account = mt5.account_info()

        if account is None:
            raise RuntimeError(
                f"MT5 account_info failed: {mt5.last_error()}"
            )

        demo_mode = getattr(
            mt5,
            "ACCOUNT_TRADE_MODE_DEMO",
            0
        )

        is_demo = (
            int(
                getattr(
                    account,
                    "trade_mode",
                    -1
                )
            )
            ==
            int(
                demo_mode
            )
        )

        print(
            f"Account login   : {account.login}"
        )

        print(
            f"Server          : {account.server}"
        )

        print(
            f"Broker balance  : ${account.balance:.2f}"
        )

        print(
            f"Currency        : {account.currency}"
        )

        print(
            f"Account mode    : "
            f"{'DEMO' if is_demo else 'NOT DEMO'}"
        )

        print(
            f"Shadow capital  : "
            f"${DEMO_FORWARD_CAPITAL:.2f}"
        )

        print(
            f"Model artifact  : {MODEL_PATH}"
        )

        if (
            DEMO_FORWARD_REQUIRE_DEMO_ACCOUNT
            and
            not is_demo
        ):

            raise RuntimeError(
                "PREFLIGHT FAILED: DEMO_FORWARD is hard-blocked "
                "on non-demo accounts."
            )

        predictor = Predictor()

        risk = predictor.get_risk_policy()

        risk_percent = float(
            risk[
                "risk_percent"
            ]
        )

        max_actual_risk_percent = min(
            float(
                MAX_ACTUAL_RISK_PERCENT
            ),
            risk_percent * 1.25
        )

        target_risk_money = (
            DEMO_FORWARD_CAPITAL *
            risk_percent /
            100.0
        )

        max_risk_money = (
            DEMO_FORWARD_CAPITAL *
            max_actual_risk_percent /
            100.0
        )

        print(
            f"Calibrated risk : "
            f"{risk_percent:.3f}% per trade"
        )

        print(
            f"Risk money      : "
            f"target=${target_risk_money:.4f} "
            f"max=${max_risk_money:.4f} "
            f"({max_actual_risk_percent:.3f}%)"
        )

        print(
            "\nBroker volume rules + maximum safe stop "
            "when using the broker minimum lot:"
        )

        for symbol in SYMBOLS:

            info = mt5.symbol_info(
                symbol
            )

            if info is None:
                raise RuntimeError(
                    f"Missing broker symbol: {symbol}"
                )

            min_lot = float(
                info.volume_min
            )

            pip_value = (
                _pip_value_per_lot(
                    info
                )
            )

            if (
                min_lot > 0
                and
                pip_value > 0
            ):

                max_safe_stop_pips = (
                    max_risk_money /
                    (
                        min_lot *
                        pip_value
                    )
                )

                safe_text = (
                    f"{max_safe_stop_pips:.2f} pips"
                )

            else:

                safe_text = "UNKNOWN"

            print(
                f"  {symbol:<8} "
                f"min={info.volume_min:g} "
                f"step={info.volume_step:g} "
                f"max={info.volume_max:g} "
                f"digits={info.digits} "
                f"| max-safe-stop={safe_text}"
            )

        print(
            "\nPREFLIGHT PASS — demo account + model policies are valid."
        )

        print(
            "Broker-minimum fallback is enabled: a 0.01 lot order "
            "is allowed only when its real SL risk stays within "
            "the calibrated tolerance."
        )

        print(
            "No trade was placed by this script."
        )

        print(
            "=" * 80 +
            "\n"
        )

    finally:

        mt5.shutdown()


if __name__ == "__main__":

    main()
