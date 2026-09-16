# filename: test_paper_executor.py

from execution.paper_executor import PaperExecutor


def test_buy_tp():

    executor = PaperExecutor(
        capital=10.0
    )

    opened = executor.open_trade(

        symbol="EURUSD",

        direction="BUY",

        price=1.10000,

        lot=0.001,

        stop_loss=1.09900,

        take_profit=1.10100,

        candle_time="TEST"

    )

    assert opened

    entry = executor.get_position(
        "EURUSD"
    )["entry_price"]

    print(
        f"BUY TP entry={entry}"
    )

    closed = executor.update_candle(

        "EURUSD",

        {
            "open": 1.10000,
            "high": 1.10150,
            "low": 1.10000,
            "close": 1.10120
        },

        candle_time="TEST"

    )

    assert closed

    trade = executor.trade_history[-1]

    assert trade["exit_reason"] == "TAKE_PROFIT"

    assert trade["exit_price"] == 1.10100

    print(
        "PASS | BUY TP"
    )


def test_buy_sl():

    executor = PaperExecutor(
        capital=10.0
    )

    opened = executor.open_trade(

        symbol="EURUSD",

        direction="BUY",

        price=1.10000,

        lot=0.001,

        stop_loss=1.09900,

        take_profit=1.10100,

        candle_time="TEST"

    )

    assert opened

    closed = executor.update_candle(

        "EURUSD",

        {
            "open": 1.10000,
            "high": 1.10050,
            "low": 1.09850,
            "close": 1.09950
        },

        candle_time="TEST"

    )

    assert closed

    trade = executor.trade_history[-1]

    assert trade["exit_reason"] == "STOP_LOSS"

    assert trade["exit_price"] == 1.09900

    print(
        "PASS | BUY SL"
    )


def test_sell_tp():

    executor = PaperExecutor(
        capital=10.0
    )

    opened = executor.open_trade(

        symbol="EURUSD",

        direction="SELL",

        price=1.10000,

        lot=0.001,

        stop_loss=1.10100,

        take_profit=1.09900,

        candle_time="TEST"

    )

    assert opened

    closed = executor.update_candle(

        "EURUSD",

        {
            "open": 1.10000,
            "high": 1.10020,
            "low": 1.09850,
            "close": 1.09920
        },

        candle_time="TEST"

    )

    assert closed

    trade = executor.trade_history[-1]

    assert trade["exit_reason"] == "TAKE_PROFIT"

    assert trade["exit_price"] == 1.09900

    print(
        "PASS | SELL TP"
    )


def test_sell_sl():

    executor = PaperExecutor(
        capital=10.0
    )

    opened = executor.open_trade(

        symbol="EURUSD",

        direction="SELL",

        price=1.10000,

        lot=0.001,

        stop_loss=1.10100,

        take_profit=1.09900,

        candle_time="TEST"

    )

    assert opened

    closed = executor.update_candle(

        "EURUSD",

        {
            "open": 1.10000,
            "high": 1.10150,
            "low": 1.09950,
            "close": 1.10120
        },

        candle_time="TEST"

    )

    assert closed

    trade = executor.trade_history[-1]

    assert trade["exit_reason"] == "STOP_LOSS"

    assert trade["exit_price"] == 1.10100

    print(
        "PASS | SELL SL"
    )


def test_both_hit_uses_sl():

    executor = PaperExecutor(
        capital=10.0
    )

    opened = executor.open_trade(

        symbol="EURUSD",

        direction="BUY",

        price=1.10000,

        lot=0.001,

        stop_loss=1.09900,

        take_profit=1.10100,

        candle_time="TEST"

    )

    assert opened

    closed = executor.update_candle(

        "EURUSD",

        {
            "open": 1.10000,
            "high": 1.10200,
            "low": 1.09800,
            "close": 1.10050
        },

        candle_time="TEST"

    )

    assert closed

    trade = executor.trade_history[-1]

    assert trade["exit_reason"] == "STOP_LOSS"

    assert trade["exit_price"] == 1.09900

    print(
        "PASS | BOTH HIT -> STOP LOSS"
    )


if __name__ == "__main__":

    test_buy_tp()

    test_buy_sl()

    test_sell_tp()

    test_sell_sl()

    test_both_hit_uses_sl()

    print()
    print("==============================")
    print("ALL PAPER EXECUTION TESTS PASS")
    print("==============================")