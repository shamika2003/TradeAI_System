# filename: performance.py

import os
import csv
import math

from datetime import datetime
from analytics.logger import log

# =====================================================
# PERFORMANCE ANALYZER
# =====================================================
class PerformanceAnalyzer:


    def __init__(self, report_folder="reports", initial_balance=1000):

        self.report_folder = report_folder
        self.initial_balance = initial_balance
        os.makedirs(self.report_folder, exist_ok=True)

    # =====================================================
    # SAVE TRADE CSV
    # =====================================================
    def save_trade_history(self, trades):

        if not trades:
            return

        path = os.path.join( self.report_folder, "trade_history.csv")

        keys = trades[0].keys()

        with open(path, "w", newline="") as f:

            writer = csv.DictWriter(f, fieldnames=keys)

            writer.writeheader()

            writer.writerows(trades)

        log(f"INFO | Trade history saved {path}")

    # =====================================================
    # MAX DRAWDOWN
    # =====================================================
    def calculate_drawdown(self, trades):

        balance = self.initial_balance
        peak = balance
        max_dd = 0
        max_dd_percent = 0
        curve = []

        for trade in trades:

            balance += trade["profit"]

            curve.append(balance)

            if balance > peak:
                peak = balance

            dd = peak - balance

            dd_percent = ( dd / peak) * 100

            if dd > max_dd:
                max_dd = dd

            if dd_percent > max_dd_percent:
                max_dd_percent = dd_percent

        return (
            max_dd,
            max_dd_percent,
            curve
        )

    # =====================================================
    # SHARPE STYLE METRIC
    # =====================================================
    def calculate_sharpe(self, trades):

        returns = [
            t["profit"]
            for t in trades
        ]

        if len(returns) < 2:
            return 0

        avg = sum(returns) / len(returns)

        variance = sum(
            (x - avg) ** 2

            for x in returns

        ) / (len(returns)-1)

        std = math.sqrt(variance)

        if std == 0:
            return 0

        return avg / std

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================
    def analyze(self, trades):

        if not trades:
            log("WARNING | No trades found")
            return

        total = len(trades)

        wins = [t for t in trades if t["profit"] > 0]

        losses = [t for t in trades if t["profit"] <= 0]

        win_count = len(wins)

        loss_count = len(losses)

        win_rate = (win_count / total) * 100

        # -----------------------------
        # PROFIT
        # -----------------------------
        net_profit = sum(t["profit"] for t in trades)

        gross_profit = sum(t["profit"] for t in wins)

        gross_loss = abs(sum(t["profit"] for t in losses))

        if gross_loss == 0:
            profit_factor = 999

        else:
            profit_factor = (gross_profit / gross_loss)

        # -----------------------------
        # AVERAGES
        # -----------------------------
        average_trade = (net_profit / total)

        average_win = (gross_profit / win_count
            if win_count else 0
        )

        average_loss = (gross_loss / loss_count
            if loss_count else 0
        )

        # Risk reward
        if average_loss != 0:
            risk_reward = (average_win / average_loss)

        else:
            risk_reward = 999

        # Expectancy
        expectancy = ((win_rate/100) * average_win - (1-win_rate/100) * average_loss)

        # -----------------------------
        # DRAWDOWN
        # -----------------------------
        max_dd, max_dd_percent, curve = (
            self.calculate_drawdown(trades)
        )

        # -----------------------------
        # STREAKS
        # -----------------------------
        max_win_streak = 0

        max_loss_streak = 0

        win_streak = 0

        loss_streak = 0

        for t in trades:
            if t["profit"] > 0:
                win_streak += 1
                loss_streak = 0

            else:
                loss_streak += 1
                win_streak = 0

            max_win_streak = max(max_win_streak, win_streak)

            max_loss_streak = max(max_loss_streak, loss_streak)

        # -----------------------------
        # FINAL BALANCE
        # -----------------------------
        final_balance = (self.initial_balance + net_profit)

        total_return = ( net_profit / self.initial_balance) * 100

        # -----------------------------
        # RECOVERY FACTOR
        # -----------------------------
        if max_dd > 0:
            recovery_factor = (net_profit / max_dd)

        else:
            recovery_factor = 999

        sharpe = self.calculate_sharpe(trades)

        # -----------------------------
        # BEST / WORST
        # -----------------------------
        best_trade = max(trades, key=lambda x:x["profit"])

        worst_trade = min(trades, key=lambda x:x["profit"])

        report = f"""

=====================================
 AI TRADING BACKTEST REPORT
=====================================

Generated:
{datetime.now()}

INITIAL BALANCE:
${self.initial_balance:.2f}

FINAL BALANCE:
${final_balance:.2f}

TOTAL RETURN:
{total_return:.2f}%


-------------------------------------
TRADE STATISTICS
-------------------------------------

Total Trades:
{total}

Winning Trades:
{win_count}

Losing Trades:
{loss_count}

Win Rate:
{win_rate:.2f}%


-------------------------------------
PROFIT ANALYSIS
-------------------------------------

Net Profit:
${net_profit:.2f}

Gross Profit:
${gross_profit:.2f}

Gross Loss:
${gross_loss:.2f}

Profit Factor:
{profit_factor:.2f}

Average Trade:
${average_trade:.2f}

Average Win:
${average_win:.2f}

Average Loss:
${average_loss:.2f}

Risk Reward:
1 : {risk_reward:.2f}

Expectancy:
${expectancy:.2f}


-------------------------------------
RISK METRICS
-------------------------------------

Maximum Drawdown:
${max_dd:.2f}

Maximum Drawdown:
{max_dd_percent:.2f}%

Recovery Factor:
{recovery_factor:.2f}

Sharpe Ratio:
{sharpe:.2f}


-------------------------------------
STREAKS
-------------------------------------

Best Win Streak:
{max_win_streak}

Worst Loss Streak:
{max_loss_streak}


-------------------------------------
TRADE EXTREMES
-------------------------------------

Best Trade:

Symbol:
{best_trade['symbol']}

Type:
{best_trade['type']}

Profit:
${best_trade['profit']:.2f}


Worst Trade:

Symbol:
{worst_trade['symbol']}

Type:
{worst_trade['type']}

Profit:
${worst_trade['profit']:.2f}

=====================================

"""

        path = os.path.join(self.report_folder, "performance_report.txt")

        with open(path, "w") as f:
            f.write(report)

        self.save_trade_history(trades)

        log("INFO | Performance report generated")