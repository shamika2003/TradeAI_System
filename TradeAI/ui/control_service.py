from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime_control import RuntimeControlPlane


TRADEAI_DIR = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = TRADEAI_DIR.parent
SETTINGS_PATH = TRADEAI_DIR / "config" / "settings.py"
DEMO_BOT = TRADEAI_DIR / "demo_bot.py"
VALIDATION_PATH = SYSTEM_ROOT / "artifacts" / "reports" / "stage5_risk_validation.json"


@dataclass
class EngineState:
    state: str = "OFFLINE"
    online: bool = False
    paused: bool = False
    pid: int | None = None
    mode: str = "UNKNOWN"
    heartbeat_age: float | None = None
    started_at: str = ""
    last_command: str = ""
    note: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None
    uptime_seconds: float | None = None


class EngineControlService:
    """Desktop process controller for TradeAI's non-live operating modes.

    The dashboard never creates manual orders.  LIVE remains intentionally blocked
    from desktop launch; DEMO_FORWARD, PAPER, and BACKTEST use the same bot entry
    point with a durable heartbeat/control plane.
    """

    HEARTBEAT_TIMEOUT = 4.0
    UI_STARTABLE_MODES = {"DEMO_FORWARD", "PAPER", "BACKTEST"}
    TERMINAL_STATES = {"STOPPED", "COMPLETED", "ERROR"}

    def __init__(self) -> None:
        self._last_process: subprocess.Popen | None = None
        self._last_launch_mode: str | None = None

    @staticmethod
    def _safe_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _parse_utc(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            return None

    @classmethod
    def _uptime_seconds(cls, started_at: str) -> float | None:
        started = cls._parse_utc(str(started_at or ""))
        if started is None:
            return None
        return max(0.0, (datetime.now(timezone.utc) - started).total_seconds())

    @staticmethod
    def _read_setting(name: str, default: str = "") -> str:
        try:
            text = SETTINGS_PATH.read_text(encoding="utf-8")
            m = re.search(rf"^{re.escape(name)}\s*=\s*([^#\r\n]+)", text, re.M)
            if not m:
                return default
            raw = m.group(1).strip()
            if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                return raw[1:-1]
            return raw
        except Exception:
            return default

    def mode(self) -> str:
        return self._read_setting("MODE", "UNKNOWN").upper()

    def model_accepted(self) -> bool:
        report = self._safe_json(VALIDATION_PATH)
        return bool(report.get("acceptance_pass", False))

    def demo_account_check(self) -> tuple[bool, str]:
        if self.mode() != "DEMO_FORWARD":
            return True, "Not required for this mode"
        try:
            import MetaTrader5 as mt5  # type: ignore
            if mt5.terminal_info() is None and not mt5.initialize():
                return False, "MT5 terminal unavailable"
            account = mt5.account_info()
            if account is None:
                return False, "MT5 account unavailable"
            demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
            is_demo = int(getattr(account, "trade_mode", -1)) == int(demo_mode)
            return (True, "Demo account verified") if is_demo else (False, "Real-money account detected")
        except Exception as exc:
            return False, f"Account check failed: {exc}"

    def open_position_count(self) -> int:
        try:
            import MetaTrader5 as mt5  # type: ignore
            if mt5.terminal_info() is None and not mt5.initialize():
                return 0
            positions = mt5.positions_get()
            return len(positions) if positions is not None else 0
        except Exception:
            return 0

    def _local_process_exit_code(self) -> int | None:
        if self._last_process is None:
            return None
        return self._last_process.poll()

    def status(self) -> EngineState:
        raw = RuntimeControlPlane.read_status()
        configured_mode = self.mode()
        process_code = self._local_process_exit_code()
        process_running = self._last_process is not None and process_code is None

        if not raw:
            if process_running:
                return EngineState(
                    state="STARTING",
                    online=True,
                    pid=self._last_process.pid,
                    mode=self._last_launch_mode or configured_mode,
                    note="Waiting for engine heartbeat",
                )
            if self._last_process is not None and process_code is not None:
                return EngineState(
                    state="ERROR",
                    online=False,
                    pid=self._last_process.pid,
                    mode=self._last_launch_mode or configured_mode,
                    note=f"Engine process exited before publishing a heartbeat (code {process_code}). Check Runtime Console.",
                    exit_code=process_code,
                )
            return EngineState(mode=configured_mode, exit_code=process_code)

        heartbeat = self._parse_utc(str(raw.get("heartbeat_utc", "") or ""))
        age = None
        if heartbeat is not None:
            age = max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())

        state = str(raw.get("state", "OFFLINE") or "OFFLINE").upper()
        raw_pid = raw.get("pid")
        try:
            pid = int(raw_pid) if raw_pid is not None else None
        except Exception:
            pid = None

        details = raw.get("details", {}) if isinstance(raw.get("details"), dict) else {}
        mode = str(raw.get("mode") or self._last_launch_mode or configured_mode).upper()
        note = str(raw.get("note", "") or "")
        fresh = age is not None and age <= self.HEARTBEAT_TIMEOUT

        # Explicit terminal states stay visible after the process exits.  This is
        # especially important for BACKTEST COMPLETE and startup ERROR diagnostics.
        if state in self.TERMINAL_STATES:
            return EngineState(
                state=state,
                online=False,
                paused=False,
                pid=pid,
                mode=mode,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note=note,
                details=details,
                exit_code=process_code,
            )

        if fresh:
            return EngineState(
                state=state,
                online=True,
                paused=bool(raw.get("paused", False)),
                pid=pid,
                mode=mode,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note=note,
                details=details,
                exit_code=process_code,
            )

        # We launched a local process and it is still alive, but its heartbeat is
        # not ready yet.  Surface STARTING instead of incorrectly showing OFFLINE.
        if process_running:
            return EngineState(
                state="STARTING",
                online=True,
                paused=False,
                pid=self._last_process.pid,
                mode=self._last_launch_mode or configured_mode,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note="Waiting for a fresh engine heartbeat",
                details=details,
            )

        # A process that exited without publishing a clean terminal state should
        # be diagnosed rather than leaving the UI stuck on STARTING forever.
        if self._last_process is not None and process_code is not None and state in {"STARTING", "RUNNING", "PAUSED", "STOPPING"}:
            return EngineState(
                state="ERROR",
                online=False,
                paused=False,
                pid=pid or self._last_process.pid,
                mode=self._last_launch_mode or mode,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note=f"Engine process exited unexpectedly (code {process_code}). Check Runtime Console.",
                details=details,
                exit_code=process_code,
            )

        return EngineState(
            state="OFFLINE",
            online=False,
            pid=pid,
            mode=configured_mode,
            heartbeat_age=age,
            started_at=str(raw.get("started_at", "") or ""),
            last_command=str(raw.get("last_command", "") or ""),
            note=note,
            details=details,
            exit_code=process_code,
        )

    def preflight(self, mode_override: str | None = None) -> tuple[bool, str]:
        mode = str(mode_override or self.mode()).upper()
        if mode not in self.UI_STARTABLE_MODES:
            if mode == "LIVE":
                return False, "LIVE mode cannot be started from the desktop control panel"
            return False, f"Unsupported mode: {mode}"
        if not DEMO_BOT.exists():
            return False, "TradeAI/demo_bot.py was not found"
        if mode == "DEMO_FORWARD" and not self.model_accepted():
            return False, "Stage 5 production acceptance is not PASS"
        if mode == "DEMO_FORWARD":
            ok, message = self.demo_account_check()
            if not ok:
                return False, message
        return True, "Control interlocks passed"

    def _launch(self, *, mode_override: str | None = None, extra_env: dict[str, str] | None = None) -> tuple[bool, str]:
        current = self.status()
        if current.online:
            return False, f"TradeAI is already {current.state.lower()} (PID {current.pid or '—'})"

        mode = str(mode_override or self.mode()).upper()
        ok, message = self.preflight(mode)
        if not ok:
            return False, message

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

        try:
            RuntimeControlPlane.clear_status()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            if mode_override:
                env["TRADEAI_MODE_OVERRIDE"] = mode
            if extra_env:
                env.update({str(k): str(v) for k, v in extra_env.items()})

            self._last_process = subprocess.Popen(
                [sys.executable, str(DEMO_BOT)],
                cwd=str(TRADEAI_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                creationflags=creationflags,
            )
            self._last_launch_mode = mode
            return True, f"TradeAI {mode} launch requested (PID {self._last_process.pid})"
        except Exception as exc:
            return False, f"Unable to start TradeAI: {exc}"

    def start_engine(self) -> tuple[bool, str]:
        return self._launch()

    def start_backtest(
        self,
        *,
        start_date: str,
        end_date: str,
        capital: float,
        symbols: list[str] | None = None,
    ) -> tuple[bool, str]:
        extra = {
            "TRADEAI_BACKTEST_START_DATE": str(start_date),
            "TRADEAI_BACKTEST_END_DATE": str(end_date),
            "TRADEAI_START_BALANCE": str(float(capital)),
        }
        if symbols:
            extra["TRADEAI_SYMBOLS_OVERRIDE"] = ",".join(str(s).strip().upper() for s in symbols if str(s).strip())
        return self._launch(mode_override="BACKTEST", extra_env=extra)

    def stop(self) -> tuple[bool, str]:
        state = self.status()
        if not state.online:
            return False, "TradeAI engine is offline"

        # During early startup the engine-side control listener may not exist yet.
        # A process launched by this UI can still be terminated safely here.
        if state.state == "STARTING" and self._last_process is not None and self._last_process.poll() is None:
            try:
                self._last_process.terminate()
                return True, "STARTING process termination requested"
            except Exception as exc:
                return False, f"Unable to terminate starting process: {exc}"

        return self.command("stop")

    def command(self, command: str) -> tuple[bool, str]:
        state = self.status()
        if not state.online:
            return False, "TradeAI engine is offline"
        try:
            RuntimeControlPlane.issue_command(command, source="tradeai-desktop-ui")
            return True, f"{command.upper()} command sent"
        except Exception as exc:
            return False, f"Command failed: {exc}"
