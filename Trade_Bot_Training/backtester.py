# filename: backtester.py

import numpy as np
from config_model import FUTURE_PERIOD, SPREAD_COST, SIGNAL_THRESHOLD 


def run_backtest(df, predictions):

    df = df.reset_index(drop=True)
    predictions = np.array(predictions)

    if len(df) != len(predictions):
        raise RuntimeError("Backtester: prediction length mismatch")

    print("\n" + "═" * 80)
    print("📉 STRATEGY BACKTEST ENGINE")
    print("═" * 80)
    print(f"📊 Samples     : {len(df):,}")
    print(f"🧠 Signals     : {len(predictions):,}")
    print(f"🎯 Threshold   : {SIGNAL_THRESHOLD}")
    print(f"💸 Spread cost : {SPREAD_COST}")
    print("─" * 80)

    # =========================================================
    # SCALE SIGNALS
    # =========================================================
    signals = np.nan_to_num(predictions)
    signals = signals / (np.std(signals) + 1e-9)

    pnl = []

    N = len(signals) - FUTURE_PERIOD - 1

    print(f"⚙️ Backtest horizon: {max(N, 0):,}")
    print("🚀 Running simulation...\n")

    active_trades = 0

    for i in range(max(N, 0)):

        signal = signals[i]

        # trade filter
        if abs(signal) < SIGNAL_THRESHOLD:
            continue

        active_trades += 1

        entry_price = df["close"].iloc[i + 1]
        exit_price = df["close"].iloc[i + 1 + FUTURE_PERIOD]

        price_return = (exit_price - entry_price) / entry_price

        position_size = np.clip(abs(signal) / 2.5, 0, 1)

        spread = SPREAD_COST

        if signal > 0:
            trade_return = position_size * (price_return - spread)
        else:
            trade_return = position_size * (-price_return - spread)

        pnl.append(trade_return)

        if len(pnl) % 5000 == 0:
            print(f"📊 Trades processed: {len(pnl):,}")

    pnl = np.array(pnl)

    print(f"\n📊 Active trade signals: {active_trades:,}")

    if len(pnl) == 0:
        print("\n⚠️ No trades executed")
        print("═" * 80)
        return {}

    # =========================================================
    # METRICS
    # =========================================================
    equity = np.cumsum(pnl)

    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max

    total_return = equity[-1]
    win_rate = np.mean(pnl > 0)

    avg_return = np.mean(pnl)
    std_return = np.std(pnl) + 1e-9

    sharpe = avg_return / std_return
    sharpe_annual = sharpe * np.sqrt(288 * 252)

    max_drawdown = np.min(drawdown)

    trade_count = len(pnl)
    trade_frequency = trade_count / max(N, 1)

    # =========================================================
    # REPORT
    # =========================================================
    print("\n" + "═" * 80)
    print("📊 BACKTEST PERFORMANCE REPORT")
    print("═" * 80)

    print(f"💰 Total Return     : {total_return:.6f}")
    print(f"📈 Win Rate         : {win_rate:.4f}")
    print(f"⚖️ Sharpe Ratio     : {sharpe:.6f}")
    print(f"📊 Annual Sharpe    : {sharpe_annual:.6f}")
    print(f"📉 Max Drawdown     : {max_drawdown:.6f}")
    print(f"🔁 Trades           : {trade_count:,}")
    print(f"⏱ Trade Frequency   : {trade_frequency:.6f}")

    print("═" * 80)
    print("✔ Backtest completed successfully")
    print("═" * 80)

    return {
        "pnl": pnl,
        "total_return": float(total_return),
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "annual_sharpe": float(sharpe_annual),
        "max_drawdown": float(max_drawdown),
        "trade_count": int(trade_count),
        "trade_frequency": float(trade_frequency)
    }