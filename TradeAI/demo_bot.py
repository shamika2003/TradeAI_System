# filename: demo_bot.py

import atexit
import csv
import json
import math
import signal
import time
from datetime import datetime, timezone

import joblib


from config.settings import (

    SYMBOLS,
    DATA_PATH,

    USE_OFFLINE,
    MODE,
    DEMO_FORWARD_CAPITAL,

    DEFAULT_CAPITAL,

    HISTORY_SIZE,

    BACKTEST_DELAY,

    LIVE_INTERVAL,
    REPORT_DIR,
    MODEL_PATH,
    TIMEFRAME,
    BACKTEST_START_DATE,
    BACKTEST_END_DATE

)


from analytics.logger import log

from core.feature_engine import FeatureTransformer, FEATURE_NAMES

from core.core_engine import CoreEngine

from core.command_control import CommandControl
from core.runtime_control import EngineInstanceLock

from shared.tradeai_core.model_contract import (
    assert_backtest_is_out_of_sample,
    validate_model_artifact,
)
from shared.tradeai_core.target_definition import TARGET_VERSION



# =====================================================
# CONDITIONAL IMPORTS
# =====================================================

if USE_OFFLINE:

    from execution.paper_executor import PaperExecutor

    from market.replay_engine import ReplayEngine

    from analytics.performance import PerformanceAnalyzer


else:

    from execution.executor import BrainExecutor

    from market.mt5_feed import (
        initialize_mt5,
        get_mtf_data
    )



# =====================================================
# GLOBAL CONTROL
# =====================================================

running = True
backtest_training_window = None



def stop_bot(sig, frame):

    global running

    running = False

    log(
        "INFO | Shutdown requested"
    )



signal.signal(
    signal.SIGINT,
    stop_bot
)


signal.signal(
    signal.SIGTERM,
    stop_bot
)




# =====================================================
# CREATE H1 FROM M5
# =====================================================

def create_h1(df):

    try:

        h1 = (

            df

            .set_index(
                "time"
            )

            .resample(
                "1h"
            )

            .agg(
                {

                    "open":
                    "first",

                    "high":
                    "max",

                    "low":
                    "min",

                    "close":
                    "last",

                    "tick_volume":
                    "sum",

                    "spread":
                    "mean",

                    "real_volume":
                    "sum"

                }
            )

            .dropna()

            .reset_index()

        )


        return h1



    except Exception as e:


        log(
            f"ERROR | H1 creation failed {e}"
        )


        return None





# =====================================================
# BACKTEST FEATURE FRAME
# =====================================================

def prepare_backtest_features(df_m5, symbol):
    """Use the canonical precomputed features stored in market_dataset.csv.

    The training dataset is already built by the shared point-in-time-safe
    FeatureTransformer. Re-running the multi-timeframe merge on that dataframe
    duplicates h1_* columns (for example h1_trend_strength_x/_y) and breaks the
    production feature contract. Predictor selects FEATURE_NAMES only, so target
    columns remain harmless and are not exposed to the model.
    """
    if df_m5 is None or df_m5.empty:
        return None

    missing = [name for name in FEATURE_NAMES if name not in df_m5.columns]
    if missing:
        log(
            f"ERROR | Backtest dataset is missing canonical features for {symbol}: "
            f"{missing[:6]}{'...' if len(missing) > 6 else ''}"
        )
        return None

    df = df_m5.copy()
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=list(FEATURE_NAMES)).reset_index(drop=True)
    if df.empty:
        return None
    df["symbol"] = symbol
    return df


# =====================================================
# REPORT FUNCTION
# =====================================================

def create_report(executor):


    if not USE_OFFLINE:

        getter = getattr(executor, "get_forward_summary", None)
        if callable(getter):
            summary = getter()
            if summary is not None:
                pf = summary["profit_factor"]
                pf_text = "INF" if pf == float("inf") else f"{pf:.3f}"
                log(
                    f"INFO | DEMO FORWARD STATUS "
                    f"${summary['initial_balance']:.2f} -> "
                    f"${summary['virtual_balance']:.2f} "
                    f"return={summary['return_percent']:+.1f}% "
                    f"trades={summary['closed_trades']}/"
                    f"{summary['min_trades_required']} "
                    f"PF={pf_text} "
                    f"DD={summary['max_drawdown_percent']:.2f}% "
                    f"accepted={summary['accepted']}"
                )
                return

        log("INFO | Forward report unavailable")
        return



    log(
        "INFO | Creating backtest report"
    )


    analyzer = PerformanceAnalyzer(

        report_folder=str(REPORT_DIR),
        initial_balance=DEFAULT_CAPITAL

    )


    analyzer.analyze(

        executor.trade_history

    )


    log(
        "INFO | Report completed"
    )






# =====================================================
# RUNTIME BACKTEST ARTIFACT
# =====================================================

def _iso_utc():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_runtime_backtest_summary(executor, replay):
    """Persist the actual desktop-triggered backtest result for the UI."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trades = list(getattr(executor, "trade_history", []) or [])
    start_balance = float(DEFAULT_CAPITAL)
    end_balance = float(getattr(executor, "balance", start_balance) or start_balance)

    profits = []
    balances = [start_balance]
    by_symbol = {}
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    balance = start_balance
    win_values = []
    loss_values = []
    by_side = {"BUY": {"trades": 0, "profit": 0.0, "wins": 0}, "SELL": {"trades": 0, "profit": 0.0, "wins": 0}}

    for trade in trades:
        try:
            profit = float(trade.get("profit", 0.0) or 0.0)
        except Exception:
            profit = 0.0
        profits.append(profit)
        balance += profit
        balances.append(balance)
        symbol = str(trade.get("symbol", "?") or "?")
        by_symbol[symbol] = by_symbol.get(symbol, 0.0) + profit
        side = str(trade.get("type", trade.get("direction", "")) or "").upper()
        if side in by_side:
            by_side[side]["trades"] += 1
            by_side[side]["profit"] += profit
            if profit > 0:
                by_side[side]["wins"] += 1
        if profit > 0:
            wins += 1
            gross_profit += profit
            win_values.append(profit)
        elif profit < 0:
            gross_loss += abs(profit)
            loss_values.append(profit)

    peak = start_balance
    max_dd_pct = 0.0
    drawdown_curve = []
    for value in balances:
        peak = max(peak, value)
        dd = max(0.0, (peak - value) / peak * 100.0) if peak > 0 else 0.0
        drawdown_curve.append(dd)
        max_dd_pct = max(max_dd_pct, dd)

    total = len(trades)
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return_pct = ((end_balance / start_balance) - 1.0) * 100.0 if start_balance > 0 else 0.0

    recent = []
    for trade in trades[-20:][::-1]:
        recent.append({
            "open_time": str(trade.get("open_time", "") or ""),
            "close_time": str(trade.get("close_time", "") or ""),
            "symbol": str(trade.get("symbol", "") or ""),
            "direction": str(trade.get("type", trade.get("direction", "")) or ""),
            "lot": float(trade.get("volume", trade.get("lot", 0.0)) or 0.0),
            "profit": float(trade.get("profit", 0.0) or 0.0),
            "exit_reason": str(trade.get("exit_reason", "") or ""),
            "confidence": trade.get("confidence"),
        })

    summary = {
        "version": "tradeai_runtime_backtest_v1",
        "generated_utc": _iso_utc(),
        "mode": "BACKTEST",
        "start_date": BACKTEST_START_DATE,
        "end_date": BACKTEST_END_DATE,
        "training_window": backtest_training_window,
        "start_balance": start_balance,
        "end_balance": end_balance,
        "net_profit": sum(profits),
        "return_percent": return_pct,
        "total_trades": total,
        "wins": wins,
        "losses": max(0, total - wins),
        "win_rate": (wins / total) * 100.0 if total else 0.0,
        "profit_factor": None if math.isinf(pf) else pf,
        "profit_factor_infinite": bool(math.isinf(pf)),
        "max_drawdown_percent": max_dd_pct,
        "curve": balances[-2000:],
        "drawdown_curve": drawdown_curve[-2000:],
        "average_win": (sum(win_values) / len(win_values)) if win_values else 0.0,
        "average_loss": (sum(loss_values) / len(loss_values)) if loss_values else 0.0,
        "expectancy": (sum(profits) / total) if total else 0.0,
        "by_side": by_side,
        "by_symbol": by_symbol,
        "best_symbol": max(by_symbol, key=by_symbol.get) if by_symbol else "—",
        "worst_symbol": min(by_symbol, key=by_symbol.get) if by_symbol else "—",
        "recent_trades": recent,
        "replay_progress": replay.progress() if replay is not None else {},
    }

    path = REPORT_DIR / "runtime_backtest_summary.json"
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    temp.replace(path)

    history_dir = REPORT_DIR / "backtest_runs"
    history_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_json = history_dir / f"backtest_{run_stamp}.json"
    run_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    trade_path = REPORT_DIR / "runtime_backtest_trades.csv"
    if trades:
        keys = sorted({key for trade in trades for key in trade.keys()})
        with trade_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for trade in trades:
                writer.writerow({key: trade.get(key) for key in keys})
        history_csv = history_dir / f"backtest_{run_stamp}_trades.csv"
        with history_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for trade in trades:
                writer.writerow({key: trade.get(key) for key in keys})

    log(
        f"INFO | Runtime backtest summary saved "
        f"trades={total} return={return_pct:+.2f}% DD={max_dd_pct:.2f}%"
    )
    return summary


# =====================================================
# MAIN
# =====================================================

def main():

    global running, backtest_training_window
    running = True
    backtest_training_window = None

    # Shared artifacts/control files make concurrent TradeAI engines unsafe.
    # This also prevents DEMO_FORWARD + BACKTEST from running at the same time.
    instance_guard = EngineInstanceLock(MODE)
    try:
        instance_guard.acquire()
    except Exception as exc:
        log(f"ERROR | TradeAI launch blocked: {exc}")
        return
    atexit.register(instance_guard.release)

    controller = CommandControl(mode=MODE)
    controller.start(initial_state="STARTING", note="Initializing TradeAI")

    log(
        f"INFO | Starting Trade AI mode={MODE} pid={instance_guard.pid}"
    )

    # =================================================
    # BACKTEST PROVENANCE GUARD
    # =================================================
    # A runtime historical backtest is allowed only when the model was trained
    # with a dedicated cutoff at or before the requested backtest start.
    # This blocks the previous in-sample mistake where a full-history model was
    # evaluated on dates it had already learned from.
    if MODE == "BACKTEST":
        try:
            artifact = joblib.load(MODEL_PATH)
            validate_model_artifact(
                artifact,
                expected_symbols=SYMBOLS,
                expected_timeframe=f"M{TIMEFRAME}",
                expected_target_version=TARGET_VERSION,
            )
            backtest_training_window = assert_backtest_is_out_of_sample(
                artifact,
                backtest_start=BACKTEST_START_DATE,
                backtest_end=BACKTEST_END_DATE,
            )
            log(
                f"INFO | BACKTEST PROVENANCE OK "
                f"fit_end={backtest_training_window['fit_end']} "
                f"safe_from={backtest_training_window['backtest_safe_from']} "
                f"requested={BACKTEST_START_DATE}->{BACKTEST_END_DATE}"
            )
        except Exception as exc:
            message = str(exc)
            log(f"ERROR | {message}")
            controller.shutdown("ERROR", message)
            return



    # =================================================
    # EXECUTOR
    # =================================================

    if USE_OFFLINE:


        executor = PaperExecutor(

            capital=DEFAULT_CAPITAL

        )


        log(
            "INFO | PAPER EXECUTOR ENABLED"
        )



    else:


        if not initialize_mt5():


            log(
                "ERROR | MT5 initialization failed"
            )
            controller.shutdown("ERROR", "MT5 initialization failed")

            return

        # =================================================
        # VERIFY MT5 ACCOUNT
        # =================================================

        import MetaTrader5 as mt5

        account = mt5.account_info()


        if account is None:

            message = f"Unable to read MT5 account {mt5.last_error()}"
            log(
                f"ERROR | {message}"
            )
            controller.shutdown("ERROR", message)

            return


        log(
            f"INFO | MT5 ACCOUNT "
            f"login={account.login} "
            f"server={account.server} "
            f"balance=${account.balance:.2f} "
            f"equity=${account.equity:.2f} "
            f"currency={account.currency}"
        )


        executor = BrainExecutor(

            capital=DEFAULT_CAPITAL

        )


        log(
            "INFO | MT5 EXECUTOR ENABLED"
        )





    # =================================================
    # DATA ENGINE
    # =================================================

    if USE_OFFLINE:

        replay = ReplayEngine(

            file_path=DATA_PATH,

            symbols=SYMBOLS,

            history_size=HISTORY_SIZE

        )

        log(
            "INFO | CSV BACKTEST MODE"
        )

    else:

        replay = None

        if MODE == "DEMO_FORWARD":
            log(
                f"INFO | DEMO FORWARD MT5 MODE "
                f"shadow_capital=${DEMO_FORWARD_CAPITAL:.2f}"
            )
        else:
            log(
                "INFO | LIVE MT5 MODE"
            )


    # =================================================
    # CORE ENGINE
    # =================================================
    #
    # IMPORTANT:
    #
    # ReplayEngine must be created BEFORE CoreEngine
    # so CoreEngine can use the current future candle
    # for PAPER/BACKTEST execution.
    # =================================================

    core = CoreEngine(

        executor=executor,

        replay=replay

    )


    transformer = FeatureTransformer()





    log(
        "INFO | Systems Ready"
    )





    # =================================================
    # COMMAND CONTROL
    # =================================================

    controller.set_state("RUNNING", "Systems ready")
    controller.set_details(
        model_ready=True,
        data_feed="REPLAY" if USE_OFFLINE else "MT5",
        mt5_connected=not USE_OFFLINE,
        last_cycle_utc=_iso_utc(),
        cycle_count=0,
        last_symbol="",
        current_candle="",
        last_error="",
    )


    log(
        "INFO | Control plane online: stop / pause / resume / report / status"
    )





    # =====================================================
    # OFFLINE BACKTEST
    # =====================================================

    if USE_OFFLINE:

        while running and controller.running:

            try:
                progress_map = replay.progress()
                progress_values = list(progress_map.values())
                progress_avg = sum(progress_values) / len(progress_values) if progress_values else 0.0
                controller.set_details(
                    backtest_progress=round(progress_avg, 2),
                    replay_time=str(replay.get_current_time() or ""),
                    trades=len(getattr(executor, "trade_history", []) or []),
                    balance=float(getattr(executor, "balance", DEFAULT_CAPITAL) or DEFAULT_CAPITAL),
                    backtest_start=BACKTEST_START_DATE,
                    backtest_end=BACKTEST_END_DATE,
                    data_feed="REPLAY",
                    model_ready=True,
                    mt5_connected=False,
                    last_cycle_utc=_iso_utc(),
                )
            except Exception:
                pass

            # =============================================
            # REPORT
            # =============================================

            if controller.report_requested:

                controller.report_requested = False

                create_report(
                    executor
                )


            # =============================================
            # PAUSE
            # =============================================

            if controller.paused:

                if controller.wait(1):
                    break

                continue


            # =============================================
            # GET NEXT MARKET SNAPSHOT
            # =============================================

            market = replay.next_market_snapshot()


            if market is None:

                log(
                    "INFO | CSV completed"
                )

                if MODE == "BACKTEST":
                    create_report(executor)
                    summary = _write_runtime_backtest_summary(executor, replay)
                    controller.replace_details({
                        "backtest_progress": 100.0,
                        "replay_time": str(replay.get_current_time() or ""),
                        "trades": int(summary.get("total_trades", 0)),
                        "balance": float(summary.get("end_balance", DEFAULT_CAPITAL)),
                        "return_percent": float(summary.get("return_percent", 0.0)),
                        "drawdown_percent": float(summary.get("max_drawdown_percent", 0.0)),
                        "backtest_start": BACKTEST_START_DATE,
                        "backtest_end": BACKTEST_END_DATE,
                    })
                    controller.shutdown("COMPLETED", "Backtest completed successfully")
                    break

                while running and controller.running:

                    time.sleep(1)


                    if controller.report_requested:

                        controller.report_requested = False

                        create_report(
                            executor
                        )

                break


            # =============================================
            # PROCESS EACH SYMBOL
            # =============================================

            for symbol, df_m5 in market.items():

                if not running or controller.should_stop():
                    break

                try:

                    # -----------------------------------------
                    # HISTORICAL DATA
                    # -----------------------------------------

                    if df_m5 is None:

                        continue


                    if len(df_m5) < HISTORY_SIZE:

                        continue


                    # -----------------------------------------
                    # CURRENT FUTURE CANDLE
                    # -----------------------------------------

                    current_candle = replay.get_current_candle(
                        symbol
                    )


                    if current_candle is None:

                        log(
                            f"WARNING | No current candle "
                            f"for {symbol}"
                        )

                        continue


                    candle_time = current_candle["time"]


                    # -----------------------------------------
                    # CANONICAL BACKTEST FEATURES
                    # -----------------------------------------
                    # market_dataset.csv already contains the exact causal
                    # Stage-5 feature columns used to train the model. Do NOT
                    # rebuild MTF features here or h1_* columns are duplicated.

                    df = prepare_backtest_features(
                        df_m5,
                        symbol
                    )

                    if df is None or df.empty:
                        continue

                    # -----------------------------------------
                    # CORE
                    # -----------------------------------------

                    core.process(

                        symbol=symbol,

                        df=df,

                        candle_time=candle_time

                    )


                except Exception as e:

                    message = f"{symbol}: {e}"
                    log(f"ERROR | {message}")
                    try:
                        controller.set_details(last_error=message, last_symbol=symbol, last_cycle_utc=_iso_utc())
                    except Exception:
                        pass


            # =============================================
            # BACKTEST SPEED
            # =============================================

            if controller.wait(BACKTEST_DELAY):
                break






    # =====================================================
    # LIVE MT5 MODE
    # =====================================================

    else:

        cycle_count = 0
        last_symbol = ""
        last_candle = ""
        last_error = ""

        while running and controller.running:
            cycle_started = time.perf_counter()
            cycle_count += 1

            # =========================================
            # UPDATE ACCOUNT STATE
            # =========================================

            if hasattr(
                executor,
                "refresh_account"
            ):

                executor.refresh_account()


            # =========================================
            # CHECK CLOSED MT5 TRADES
            # =========================================

            if hasattr(
                executor,
                "check_closed_trades"
            ):

                executor.check_closed_trades()

            if controller.report_requested:
                controller.report_requested = False
                create_report(executor)
            
            

            # =========================================
            # COMMANDS
            # =========================================

            if controller.paused:
                controller.set_details(
                    model_ready=True,
                    data_feed="MT5",
                    mt5_connected=True,
                    cycle_count=cycle_count,
                    last_cycle_utc=_iso_utc(),
                    last_symbol=last_symbol,
                    current_candle=last_candle,
                    last_error=last_error,
                )
                if controller.wait(1):
                    break
                continue





            # =========================================
            # MARKET LOOP
            # =========================================

            for symbol in SYMBOLS:

                if not running or controller.should_stop():
                    break

                try:



                    df_m5, df_h1 = get_mtf_data(
                        symbol,
                        should_stop=lambda: (not running) or controller.should_stop(),
                    )



                    if (

                        df_m5 is None

                        or

                        df_h1 is None

                    ):

                        continue





                    df = transformer.build_multi_timeframe_features(

                        df_m5,

                        df_h1

                    )



                    if df is None or df.empty:

                        continue





                    candle_value = df["time"].iloc[-1]
                    core.process(
                        symbol,
                        df,
                        candle_time=candle_value
                    )
                    last_symbol = symbol
                    last_candle = str(candle_value)





                except Exception as e:
                    last_error = f"{symbol}: {e}"
                    log(f"ERROR | {last_error}")

            if not running or controller.should_stop():
                break

            loop_ms = (time.perf_counter() - cycle_started) * 1000.0
            try:
                account_state = getattr(executor, "account_info", None)
                controller.set_details(
                    model_ready=True,
                    data_feed="MT5",
                    mt5_connected=True,
                    cycle_count=cycle_count,
                    last_cycle_utc=_iso_utc(),
                    last_symbol=last_symbol,
                    current_candle=last_candle,
                    loop_ms=round(loop_ms, 1),
                    last_error=last_error,
                )
            except Exception:
                pass

            if controller.wait(LIVE_INTERVAL):
                break





    # =====================================================
    # SHUTDOWN
    # =====================================================

    if controller.state != "COMPLETED":
        controller.shutdown("STOPPED", "Bot stopped")

    if not USE_OFFLINE:
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except Exception:
            pass

    instance_guard.release()

    log(
        "INFO | Bot stopped"
    )






# =====================================================
# START
# =====================================================

if __name__ == "__main__":

    main()
