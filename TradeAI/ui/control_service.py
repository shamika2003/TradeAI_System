from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
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
    session_id: str = ""
    heartbeat_age: float | None = None
    started_at: str = ""
    last_command: str = ""
    note: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None
    uptime_seconds: float | None = None


class EngineControlService:
    """UI controller for a detachable, single-instance TradeAI engine.

    The UI is the only supported launcher. After launch, the engine may continue
    running when the UI closes if the user explicitly chooses that option. A later
    UI session re-attaches through the durable status/command/instance files.
    """

    HEARTBEAT_TIMEOUT = 4.0
    UI_STARTABLE_MODES = {"DEMO_FORWARD", "PAPER", "BACKTEST"}
    TERMINAL_STATES = {"STOPPED", "COMPLETED", "ERROR"}

    def __init__(self) -> None:
        self._last_process: subprocess.Popen | None = None
        self._last_launch_mode: str | None = None
        self._last_session_id: str = ""

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
            match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\r\n]+)", text, re.M)
            if not match:
                return default
            raw = match.group(1).strip()
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
        state = self.status()
        if state.mode not in {"DEMO_FORWARD", "LIVE"}:
            return 0
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
        active_instance = RuntimeControlPlane.active_instance(clean_stale=True)
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
                    session_id=self._last_session_id,
                    note="Waiting for engine heartbeat",
                )
            if active_instance:
                return EngineState(
                    state="STARTING",
                    online=True,
                    pid=active_instance.get("pid"),
                    mode=str(active_instance.get("mode") or configured_mode),
                    session_id=str(active_instance.get("session_id", "") or ""),
                    started_at=str(active_instance.get("started_at", "") or ""),
                    note="Existing engine detected; waiting for heartbeat",
                )
            if self._last_process is not None and process_code is not None:
                return EngineState(
                    state="ERROR",
                    online=False,
                    pid=self._last_process.pid,
                    mode=self._last_launch_mode or configured_mode,
                    session_id=self._last_session_id,
                    note=f"Engine process exited before publishing a heartbeat (code {process_code}). Check Runtime Console.",
                    exit_code=process_code,
                )
            return EngineState(mode=configured_mode, exit_code=process_code)

        heartbeat = self._parse_utc(str(raw.get("heartbeat_utc", "") or ""))
        age = None if heartbeat is None else max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())

        state = str(raw.get("state", "OFFLINE") or "OFFLINE").upper()
        raw_pid = raw.get("pid")
        try:
            pid = int(raw_pid) if raw_pid is not None else None
        except Exception:
            pid = None

        details = raw.get("details", {}) if isinstance(raw.get("details"), dict) else {}
        mode = str(raw.get("mode") or self._last_launch_mode or configured_mode).upper()
        session_id = str(raw.get("session_id") or active_instance.get("session_id") or self._last_session_id or "")
        note = str(raw.get("note", "") or "")
        fresh = age is not None and age <= self.HEARTBEAT_TIMEOUT
        active_pid = active_instance.get("pid") if active_instance else None
        pid_alive = bool(pid and RuntimeControlPlane.pid_alive(pid))
        active_matches = bool(active_pid and pid and int(active_pid) == int(pid))

        if state in self.TERMINAL_STATES and not pid_alive:
            return EngineState(
                state=state,
                online=False,
                paused=False,
                pid=pid,
                mode=mode,
                session_id=session_id,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note=note,
                details=details,
                exit_code=process_code,
            )

        # Fresh heartbeat + live instance is authoritative even if this dashboard
        # did not launch the process. This is the re-attach path after UI reopen.
        if fresh and (pid_alive or active_matches):
            return EngineState(
                state=state,
                online=True,
                paused=bool(raw.get("paused", False)),
                pid=pid,
                mode=mode,
                session_id=session_id,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note=note,
                details=details,
                exit_code=process_code,
            )

        if process_running:
            return EngineState(
                state="STARTING",
                online=True,
                paused=False,
                pid=self._last_process.pid,
                mode=self._last_launch_mode or configured_mode,
                session_id=self._last_session_id,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note="Waiting for a fresh engine heartbeat",
                details=details,
            )

        if active_instance:
            return EngineState(
                state="STARTING" if state == "OFFLINE" else state,
                online=True,
                paused=bool(raw.get("paused", False)),
                pid=int(active_instance.get("pid") or 0) or pid,
                mode=str(active_instance.get("mode") or mode),
                session_id=str(active_instance.get("session_id") or session_id),
                heartbeat_age=age,
                started_at=str(active_instance.get("started_at") or raw.get("started_at", "") or ""),
                last_command=str(raw.get("last_command", "") or ""),
                note="Existing engine detected; heartbeat is stale",
                details=details,
            )

        return EngineState(
            state="OFFLINE",
            online=False,
            pid=pid,
            mode=configured_mode,
            session_id=session_id,
            heartbeat_age=age,
            started_at=str(raw.get("started_at", "") or ""),
            last_command=str(raw.get("last_command", "") or ""),
            note=note,
            details=details,
            exit_code=process_code,
        )

    def preflight(self, mode_override: str | None = None) -> tuple[bool, str]:
        mode = str(mode_override or self.mode()).upper()

        active = RuntimeControlPlane.active_instance(clean_stale=True)
        if active:
            return False, (
                f"TradeAI engine already running "
                f"(PID {active.get('pid') or 'unknown'}, mode {active.get('mode') or 'UNKNOWN'}). "
                "This dashboard will attach to it instead of launching another engine."
            )

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

        session_id = secrets.token_hex(16)
        launch_token = secrets.token_urlsafe(32)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        try:
            RuntimeControlPlane.clear_status()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["TRADEAI_UI_LAUNCH_TOKEN"] = launch_token
            env["TRADEAI_ENGINE_SESSION_ID"] = session_id
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
                start_new_session=(os.name != "nt"),
            )
            self._last_launch_mode = mode
            self._last_session_id = session_id
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

    def command(self, command: str) -> tuple[bool, str]:
        state = self.status()
        if not state.online:
            return False, "TradeAI engine is offline"
        try:
            RuntimeControlPlane.issue_command(
                command,
                source="tradeai-desktop-ui",
                target_session_id=state.session_id,
            )
            return True, f"{command.upper()} command sent"
        except Exception as exc:
            return False, f"Command failed: {exc}"

    def stop(self, timeout_seconds: float = 7.0) -> tuple[bool, str]:
        """Stop and verify the real engine PID is gone before reporting success."""
        state = self.status()
        pid = state.pid
        if not state.online or not pid:
            return False, "TradeAI engine is offline"

        # Ask the engine to unwind cleanly first so reports/locks/status are finalized.
        try:
            RuntimeControlPlane.issue_command(
                "stop",
                source="tradeai-desktop-ui",
                target_session_id=state.session_id,
            )
        except Exception as exc:
            return False, f"Unable to send STOP command: {exc}"

        deadline = time.monotonic() + max(1.0, float(timeout_seconds))
        while time.monotonic() < deadline:
            if not RuntimeControlPlane.pid_alive(pid):
                return True, "TradeAI engine stopped"
            time.sleep(0.1)

        # A blocking MT5 IPC call can prevent the graceful command from returning.
        # At this point the user explicitly asked to stop, so terminate the exact
        # engine PID recorded by the active instance lock.
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.kill(int(pid), signal.SIGTERM)
        except Exception as exc:
            return False, f"STOP requested, but PID {pid} could not be terminated: {exc}"

        for _ in range(30):
            if not RuntimeControlPlane.pid_alive(pid):
                RuntimeControlPlane.active_instance(clean_stale=True)
                return True, "TradeAI engine stopped"
            time.sleep(0.1)

        return False, f"TradeAI engine PID {pid} is still running"

    def detach(self) -> tuple[bool, str]:
        """Leave a healthy engine running while this dashboard closes."""
        state = self.status()
        if not state.online:
            return False, "No running TradeAI engine to keep in background"
        return True, f"TradeAI {state.mode} continues in background (PID {state.pid})"
