from __future__ import annotations

import argparse
import contextlib
import io
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Trade_Bot_Training.precision_config import (
    ACTION_CONFIDENCE,
    HORIZON_MINUTES,
    SYMBOLS as MODEL_SYMBOLS,
)
from TradeAI.analytics.logger import log
from TradeAI.precision_demo_engine_base import (
    PrecisionDemoEngine as _BasePrecisionDemoEngine,
)


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
ROOT_LOG_PATH = SYSTEM_ROOT / "TRADEAI_DEMO.log"

DEFAULT_SIMULATED_CAPITAL = 100.0


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _root_log(message: str) -> None:
    ROOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with ROOT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} | {message}\n")


def _parse_symbols(raw: str) -> list[str]:
    if not raw.strip():
        return list(MODEL_SYMBOLS)

    return [
        part.strip().upper()
        for part in raw.split(",")
        if part.strip()
    ]


class PrecisionDemoEngine(_BasePrecisionDemoEngine):
    """Working precision demo engine with a $100 low-balance simulation layer.

    The broker DEMO account balance is deliberately ignored for TradeAI capital
    measurement. Position size never exceeds the broker's minimum lot.
    """

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        poll_seconds: float = 1.0,
        simulated_capital: float = DEFAULT_SIMULATED_CAPITAL,
    ) -> None:
        super().__init__(
            symbols=symbols,
            poll_seconds=poll_seconds,
        )

        requested = _finite_float(simulated_capital)
        if requested is None or requested <= 0.0:
            raise ValueError("simulated_capital must be greater than zero.")

        previous = _finite_float(
            self.execution_state.get("simulated_start_capital")
        )

        # Lock starting capital once a managed trade exists so forward results
        # cannot be accidentally rebased halfway through the test.
        has_trade_history = bool(
            self.execution_state.get("open_positions")
            or self.execution_state.get("completed")
        )

        if (
            has_trade_history
            and previous is not None
            and previous > 0.0
        ):
            self.simulated_start_capital = float(previous)
        else:
            self.simulated_start_capital = float(requested)

        self._current_entry_action: str | None = None

        self.execution_state["simulated_start_capital"] = (
            self.simulated_start_capital
        )
        self.execution_state["sizing_mode"] = (
            "LOW_BALANCE_BROKER_MINIMUM_LOT"
        )
        self.execution_state["root_log"] = str(ROOT_LOG_PATH)

        self.state["simulated_start_capital"] = (
            self.simulated_start_capital
        )
        self.state["root_log"] = str(ROOT_LOG_PATH)

        self._refresh_virtual_state()

    # ------------------------------------------------------------------
    # Quiet terminal
    # ------------------------------------------------------------------

    def connect(self) -> None:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            super().connect()

        for line in capture.getvalue().splitlines():
            if line.strip():
                _root_log(f"MT5 | {line.strip()}")

    def disconnect(self) -> None:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            super().disconnect()

        for line in capture.getvalue().splitlines():
            if line.strip():
                _root_log(f"MT5 | {line.strip()}")

    def _print_decision(
        self,
        row: dict[str, Any],
    ) -> None:
        _root_log(
            "MODEL_STATE | "
            f"symbol={row['symbol']} "
            f"action={row['action']} "
            f"tradeable={int(bool(row['tradeable']))} "
            f"confidence={float(row['confidence']):.4f} "
            f"p_down={float(row['p_down']):.4f} "
            f"p_hold={float(row['p_hold']):.4f} "
            f"p_up={float(row['p_up']):.4f} "
            f"bar={row['bar_time']}"
        )

    def _resolve_pending(
        self,
        symbol: str,
        history: pd.DataFrame,
    ) -> None:
        capture = io.StringIO()

        with contextlib.redirect_stdout(capture):
            super()._resolve_pending(
                symbol,
                history,
            )

        for line in capture.getvalue().splitlines():
            if line.strip():
                _root_log(
                    f"FORWARD_AUDIT | {line.strip()}"
                )

    def print_startup(self) -> None:
        enabled = [
            symbol
            for symbol in self.symbols
            if self.models[
                symbol
            ].model.historical_gate_pass
        ]

        _root_log(
            "ENGINE_START | "
            f"mode=DEMO_ONLY "
            f"virtual_start={self.simulated_start_capital:.2f} "
            f"lot_mode=BROKER_MINIMUM "
            f"confidence_gate={ACTION_CONFIDENCE:.2f} "
            f"horizon_minutes={HORIZON_MINUTES} "
            f"symbols={','.join(enabled)}"
        )

        print(
            "TradeAI DEMO running quietly | "
            f"virtual=${self.simulated_start_capital:.2f} | "
            f"log={ROOT_LOG_PATH.name} | Ctrl+C to stop"
        )

    def print_demo_preflight(
        self,
        result: dict[str, Any],
    ) -> None:
        virtual = self._virtual_account_snapshot()

        print(
            "DEMO VERIFIED | "
            f"broker_equity=${float(result.get('equity') or 0.0):.2f} | "
            f"virtual_equity=${virtual['equity']:.2f} | "
            "lot=broker minimum"
        )
        print(f"LOG: {ROOT_LOG_PATH}")

    # ------------------------------------------------------------------
    # Low-balance virtual account
    # ------------------------------------------------------------------

    def _virtual_account_snapshot(
        self,
    ) -> dict[str, float]:
        realized = 0.0

        completed = self.execution_state.get(
            "completed",
            {},
        )

        if isinstance(completed, dict):
            for record in completed.values():
                if not isinstance(record, dict):
                    continue

                profit = _finite_float(
                    record.get("broker_profit")
                )

                if profit is not None:
                    realized += profit

        floating = 0.0
        reserved_margin = 0.0

        if self.mt5 is not None:
            for position in self._managed_positions():
                profit = _finite_float(
                    getattr(
                        position,
                        "profit",
                        None,
                    )
                )

                if profit is not None:
                    floating += profit

                symbol = str(
                    getattr(
                        position,
                        "symbol",
                        "",
                    )
                    or ""
                ).upper()

                volume = _finite_float(
                    getattr(
                        position,
                        "volume",
                        None,
                    )
                )

                if (
                    not symbol
                    or volume is None
                    or volume <= 0.0
                ):
                    continue

                tick = self.mt5.symbol_info_tick(
                    symbol
                )

                if tick is None:
                    continue

                position_type = int(
                    getattr(
                        position,
                        "type",
                        -1,
                    )
                )

                if position_type == int(
                    self.mt5.POSITION_TYPE_BUY
                ):
                    order_type = int(
                        self.mt5.ORDER_TYPE_BUY
                    )
                    price = _finite_float(
                        getattr(
                            tick,
                            "ask",
                            None,
                        )
                    )

                elif position_type == int(
                    self.mt5.POSITION_TYPE_SELL
                ):
                    order_type = int(
                        self.mt5.ORDER_TYPE_SELL
                    )
                    price = _finite_float(
                        getattr(
                            tick,
                            "bid",
                            None,
                        )
                    )

                else:
                    continue

                if price is None or price <= 0:
                    continue

                try:
                    margin = self.mt5.order_calc_margin(
                        order_type,
                        symbol,
                        float(volume),
                        float(price),
                    )
                except Exception:
                    margin = None

                margin = _finite_float(
                    margin
                )

                if (
                    margin is not None
                    and margin > 0.0
                ):
                    reserved_margin += margin

        balance = (
            self.simulated_start_capital
            + realized
        )

        equity = (
            balance
            + floating
        )

        free_margin = (
            equity
            - reserved_margin
        )

        return {
            "start_capital": float(
                self.simulated_start_capital
            ),
            "realized_profit": float(
                realized
            ),
            "floating_profit": float(
                floating
            ),
            "balance": float(
                balance
            ),
            "equity": float(
                equity
            ),
            "reserved_margin": float(
                reserved_margin
            ),
            "free_margin": float(
                free_margin
            ),
            "return_percent": float(
                (
                    balance
                    - self.simulated_start_capital
                )
                / self.simulated_start_capital
                * 100.0
            ),
        }

    def _refresh_virtual_state(
        self,
    ) -> dict[str, float]:
        snapshot = (
            self._virtual_account_snapshot()
        )

        self.execution_state[
            "low_balance"
        ] = snapshot

        self.execution_state[
            "simulated_start_capital"
        ] = self.simulated_start_capital

        self.state[
            "low_balance"
        ] = snapshot

        self.state[
            "simulated_start_capital"
        ] = self.simulated_start_capital

        self._save_execution_state()

        return snapshot

    # ------------------------------------------------------------------
    # Lot sizing
    # ------------------------------------------------------------------

    def _minimum_volume(
        self,
        symbol: str,
    ) -> float:
        # Keep the already-working broker minimum-volume logic.
        volume = super()._minimum_volume(
            symbol
        )

        # The base executor calls _minimum_volume immediately before sending
        # a BUY/SELL. We set this context in _execute_tradeable_signal.
        action = str(
            self._current_entry_action
            or ""
        ).upper()

        if action not in {
            "BUY",
            "SELL",
        }:
            return volume

        if self.mt5 is None:
            raise RuntimeError(
                "MT5 is not connected."
            )

        tick = self.mt5.symbol_info_tick(
            symbol
        )

        if tick is None:
            raise RuntimeError(
                f"{symbol}: live tick unavailable "
                "for low-balance margin check."
            )

        if action == "BUY":
            order_type = int(
                self.mt5.ORDER_TYPE_BUY
            )
            price = _finite_float(
                getattr(
                    tick,
                    "ask",
                    None,
                )
            )
        else:
            order_type = int(
                self.mt5.ORDER_TYPE_SELL
            )
            price = _finite_float(
                getattr(
                    tick,
                    "bid",
                    None,
                )
            )

        if price is None or price <= 0.0:
            raise RuntimeError(
                f"{symbol}: invalid market price "
                "for margin check."
            )

        required_margin = (
            self.mt5.order_calc_margin(
                order_type,
                symbol,
                float(volume),
                float(price),
            )
        )

        required_margin = _finite_float(
            required_margin
        )

        if (
            required_margin is None
            or required_margin < 0.0
        ):
            raise RuntimeError(
                f"{symbol}: broker could not "
                "calculate required margin."
            )

        virtual = (
            self._virtual_account_snapshot()
        )

        if virtual[
            "equity"
        ] <= 0.0:
            raise RuntimeError(
                "LOW_BALANCE_STOP: "
                "virtual account equity is depleted."
            )

        if required_margin > max(
            0.0,
            virtual[
                "free_margin"
            ],
        ):
            raise RuntimeError(
                "LOW_BALANCE_MARGIN_BLOCK: "
                f"required=${required_margin:.2f}, "
                f"virtual_free="
                f"${virtual['free_margin']:.2f}."
            )

        _root_log(
            "LOW_BALANCE_ENTRY_CHECK | "
            f"symbol={symbol} "
            f"action={action} "
            f"volume={volume:g} "
            f"required_margin={required_margin:.2f} "
            f"virtual_balance={virtual['balance']:.2f} "
            f"virtual_equity={virtual['equity']:.2f} "
            f"virtual_free={virtual['free_margin']:.2f}"
        )

        return volume

    def _execute_tradeable_signal(
        self,
        row: dict[str, Any],
    ) -> None:
        self._current_entry_action = str(
            row.get(
                "action",
                "",
            )
        ).upper()

        capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(
                capture
            ):
                super()._execute_tradeable_signal(
                    row
                )
        finally:
            self._current_entry_action = None

        for line in capture.getvalue().splitlines():
            if line.strip():
                _root_log(
                    f"EXECUTION | {line.strip()}"
                )

        self._refresh_virtual_state()

    def _close_record(
        self,
        decision_id: str,
        record: dict[str, Any],
        *,
        reason: str,
    ) -> bool:
        capture = io.StringIO()

        with contextlib.redirect_stdout(
            capture
        ):
            result = super()._close_record(
                decision_id,
                record,
                reason=reason,
            )

        for line in capture.getvalue().splitlines():
            if line.strip():
                _root_log(
                    f"EXECUTION | {line.strip()}"
                )

        virtual = (
            self._refresh_virtual_state()
        )

        _root_log(
            "LOW_BALANCE_UPDATE | "
            f"balance={virtual['balance']:.2f} "
            f"equity={virtual['equity']:.2f} "
            f"realized={virtual['realized_profit']:+.2f} "
            f"return={virtual['return_percent']:+.3f}%"
        )

        return result

    def _event(
        self,
        event: str,
        **kwargs,
    ) -> None:
        # Keep all existing CSV event logging.
        super()._event(
            event,
            **kwargs,
        )

        row = kwargs.get("row")
        record = kwargs.get("record")
        source: dict[str, Any] = {}

        if isinstance(
            record,
            dict,
        ):
            source.update(
                record
            )

        if isinstance(
            row,
            dict,
        ):
            source.update(
                row
            )

        _root_log(
            "EXECUTION_EVENT | "
            f"event={event} "
            f"symbol={source.get('symbol', '')} "
            f"action={source.get('action', '')} "
            f"confidence={source.get('confidence')} "
            f"volume={source.get('volume')} "
            f"ticket={source.get('position_ticket')} "
            f"note={kwargs.get('note', '')}"
        )

    # ------------------------------------------------------------------
    # Publish virtual state without console noise
    # ------------------------------------------------------------------

    def _publish_state(
        self,
        row: dict[str, Any] | None = None,
    ) -> None:
        self.state[
            "simulated_start_capital"
        ] = self.simulated_start_capital

        self.state[
            "root_log"
        ] = str(
            ROOT_LOG_PATH
        )

        if self.mt5 is not None:
            self.state[
                "low_balance"
            ] = self._virtual_account_snapshot()

        super()._publish_state(
            row
        )

    # ------------------------------------------------------------------
    # Quiet continuous runner
    # ------------------------------------------------------------------

    def run_demo_forever(
        self,
    ) -> None:
        self.print_startup()
        self.connect()

        try:
            preflight = self.demo_preflight()

            _root_log(
                "PREFLIGHT_PASS | "
                f"broker_balance={preflight.get('balance')} "
                f"broker_equity={preflight.get('equity')} "
                f"virtual_start="
                f"{self.simulated_start_capital:.2f}"
            )

            self.reconcile_execution_state()
            self._refresh_virtual_state()

            # Preserve the base engine's restart/idempotency behavior:
            # never replay an already-processed closed M5 candle.
            for symbol in self.symbols:
                try:
                    self.process_symbol(
                        symbol,
                        force=False,
                    )
                except Exception as exc:
                    _root_log(
                        f"ENGINE_ERROR | "
                        f"symbol={symbol} "
                        f"error={exc}"
                    )
                    log(
                        f"ERROR | PRECISION_DEMO | "
                        f"symbol={symbol} | {exc}"
                    )

            self._publish_state()

            while True:
                any_processed = False

                for symbol in self.symbols:
                    try:
                        processed = (
                            self.process_symbol(
                                symbol,
                                force=False,
                            )
                        )

                        any_processed = (
                            any_processed
                            or processed
                        )

                    except Exception as exc:
                        _root_log(
                            f"ENGINE_ERROR | "
                            f"symbol={symbol} "
                            f"error={exc}"
                        )
                        log(
                            f"ERROR | PRECISION_DEMO | "
                            f"symbol={symbol} | {exc}"
                        )

                if any_processed:
                    self._refresh_virtual_state()
                    self._publish_state()

                time.sleep(
                    self.poll_seconds
                )

        except KeyboardInterrupt:
            open_count = len(
                self.execution_state.get(
                    "open_positions",
                    {},
                )
            )

            _root_log(
                "ENGINE_STOP_REQUESTED | "
                f"managed_open_positions="
                f"{open_count}"
            )

            print(
                "TradeAI DEMO stopped | "
                f"open_positions={open_count} | "
                f"log={ROOT_LOG_PATH.name}"
            )

        finally:
            self._refresh_virtual_state()
            self._publish_state()
            self.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TradeAI precision DEMO low-balance "
            "auto-execution engine"
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify DEMO account and low-balance "
            "configuration without sending an order."
        ),
    )

    parser.add_argument(
        "--close-all",
        action="store_true",
        help=(
            "Close only positions owned by the "
            "precision-demo engine."
        ),
    )

    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_SIMULATED_CAPITAL,
        help=(
            "Virtual starting capital. "
            "Default: 100. Use 150 for a $150 test."
        ),
    )

    parser.add_argument(
        "--symbols",
        default="",
    )

    args = parser.parse_args()

    engine = PrecisionDemoEngine(
        symbols=_parse_symbols(
            args.symbols
        ),
        poll_seconds=args.poll,
        simulated_capital=args.capital,
    )

    if args.check:
        engine.connect()

        try:
            result = (
                engine.demo_preflight()
            )

            engine.print_demo_preflight(
                result
            )

            engine.reconcile_execution_state()
            virtual = (
                engine._refresh_virtual_state()
            )

            print(
                "PREFLIGHT PASS - no order sent | "
                f"virtual=${virtual['equity']:.2f}"
            )

        finally:
            engine.disconnect()

        return

    if args.close_all:
        # The base close-all is intentionally retained because it already
        # closes only positions carrying the precision-demo magic/tag.
        capture = io.StringIO()

        with contextlib.redirect_stdout(
            capture
        ):
            engine.close_all_managed_positions()

        for line in capture.getvalue().splitlines():
            if line.strip():
                _root_log(
                    f"CLOSE_ALL | {line.strip()}"
                )

        print(
            f"Close-all finished | log={ROOT_LOG_PATH.name}"
        )
        return

    engine.run_demo_forever()


if __name__ == "__main__":
    main()
