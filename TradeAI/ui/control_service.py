from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime_control import (
    RuntimeControlPlane,
    UI_OWNER_PID_ENV,
    UI_SESSION_ID_ENV,
    pid_alive,
    terminate_pid,
)


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
    owner_pid: int | None = None
    session_id: str = ""


class EngineControlService:
    """UI-owned TradeAI engine controller.

    The engine is intentionally subordinate to this desktop process:
    - demo_bot.py receives an unguessable dashboard session ID and owner PID;
    - direct/background demo_bot.py launches are rejected by the engine;
    - STOP waits for the actual PID to exit and force-terminates only as fallback;
    - closing the owning UI stops its engine instead of leaving it orphaned.
    """

    HEARTBEAT_TIMEOUT = 4.0
    STOP_GRACE_SECONDS = 3.0
    TERMINATE_GRACE_SECONDS = 1.25
    UI_STARTABLE_MODES = {"DEMO_FORWARD", "PAPER", "BACKTEST"}
    TERMINAL_STATES = {"STOPPED", "COMPLETED", "ERROR"}

    def __init__(self) -> None:
        self._last_process: subprocess.Popen | None = None
        self._last_launch_mode: str | None = None
        self._owner_pid = os.getpid()
        self._session_id = secrets.token_urlsafe(24)

    @property
    def session_id(self) -> str:
        return self._session_id

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

    def _local_process_alive(self) -> bool:
        return self._last_process is not None and self._last_process.poll() is None

    def _instance_is_owned_by_this_ui(self, active: dict[str, Any]) -> bool:
        if not active:
            return False
        return (
            str(active.get("session_id", "") or "") == self._session_id
            and int(active.get("owner_pid", 0) or 0) == self._owner_pid
        )

    def _cleanup_orphan_instance(self) -> tuple[bool, str]:
        """Remove an engine whose owning dashboard no longer exists.

        The instance file is project-local, so a PID referenced here belongs to this
        TradeAI installation's engine lock rather than an arbitrary Python process.
        """
        active = RuntimeControlPlane.active_instance(clean_stale=True)
        if not active:
            return True, "No active engine"

        active_pid = int(active.get("pid", 0) or 0)
        owner_pid = int(active.get("owner_pid", 0) or 0)
        session_id = str(active.get("session_id", "") or "")

        if self._instance_is_owned_by_this_ui(active):
            return False, f"TradeAI is already running (PID {active_pid})"

        # A live owner means another dashboard window/session owns this engine.
        if owner_pid > 0 and pid_alive(owner_pid):
            return False, (
                f"Another TradeAI dashboard owns engine PID {active_pid} "
                f"(dashboard PID {owner_pid}). Close/stop it there first."
            )

        # Legacy engines have no owner/session metadata. They are not allowed in
        # the UI-owned architecture, so terminate the project-local orphan.
        if active_pid > 0 and pid_alive(active_pid):
            if not terminate_pid(active_pid, force=True):
                return False, f"Unable to terminate orphan TradeAI engine PID {active_pid}"

        RuntimeControlPlane.active_instance(clean_stale=True)
        RuntimeControlPlane.clear_status()
        RuntimeControlPlane.clear_command()
        label = "legacy/unowned" if not session_id else "orphaned"
        return True, f"Cleaned {label} TradeAI engine state"

    def status(self) -> EngineState:
        raw = RuntimeControlPlane.read_status()
        active_instance = RuntimeControlPlane.active_instance(clean_stale=True)
        configured_mode = self.mode()
        process_code = self._local_process_exit_code()
        local_running = self._local_process_alive()

        if not raw:
            if local_running:
                return EngineState(
                    state="STARTING",
                    online=True,
                    pid=self._last_process.pid if self._last_process else None,
                    mode=self._last_launch_mode or configured_mode,
                    note="Waiting for engine heartbeat",
                    owner_pid=self._owner_pid,
                    session_id=self._session_id,
                )
            if active_instance:
                active_pid = int(active_instance.get("pid", 0) or 0) or None
                return EngineState(
                    state="STARTING",
                    online=bool(active_pid and pid_alive(active_pid)),
                    pid=active_pid,
                    mode=str(active_instance.get("mode") or configured_mode),
                    started_at=str(active_instance.get("started_at", "") or ""),
                    note="Engine process exists; waiting for heartbeat",
                    owner_pid=int(active_instance.get("owner_pid", 0) or 0) or None,
                    session_id=str(active_instance.get("session_id", "") or ""),
                )
            if self._last_process is not None and process_code is not None:
                return EngineState(
                    state="STOPPED" if process_code == 0 else "ERROR",
                    online=False,
                    pid=self._last_process.pid,
                    mode=self._last_launch_mode or configured_mode,
                    note=(
                        "Engine process exited"
                        if process_code == 0
                        else f"Engine process exited unexpectedly (code {process_code}). Check Runtime Console."
                    ),
                    exit_code=process_code,
                    owner_pid=self._owner_pid,
                    session_id=self._session_id,
                )
            return EngineState(mode=configured_mode, exit_code=process_code)

        heartbeat = self._parse_utc(str(raw.get("heartbeat_utc", "") or ""))
        age = None
        if heartbeat is not None:
            age = max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())

        state = str(raw.get("state", "OFFLINE") or "OFFLINE").upper()
        try:
            pid = int(raw.get("pid")) if raw.get("pid") is not None else None
        except Exception:
            pid = None

        details = raw.get("details", {}) if isinstance(raw.get("details"), dict) else {}
        mode = str(raw.get("mode") or self._last_launch_mode or configured_mode).upper()
        note = str(raw.get("note", "") or "")
        owner_pid = int(raw.get("owner_pid", 0) or 0) or None
        session_id = str(raw.get("session_id", "") or "")
        fresh = age is not None and age <= self.HEARTBEAT_TIMEOUT
        actual_alive = bool(pid and pid_alive(pid))

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
                owner_pid=owner_pid,
                session_id=session_id,
            )

        # Never call an engine ONLINE from a JSON heartbeat alone. The PID must
        # actually still exist; this removes the UI/background-process mismatch.
        if fresh and actual_alive:
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
                owner_pid=owner_pid,
                session_id=session_id,
            )

        if local_running:
            return EngineState(
                state="STARTING" if state != "STOPPING" else "STOPPING",
                online=True,
                paused=False,
                pid=self._last_process.pid if self._last_process else pid,
                mode=self._last_launch_mode or mode,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                uptime_seconds=self._uptime_seconds(str(raw.get("started_at", "") or "")),
                last_command=str(raw.get("last_command", "") or ""),
                note="Waiting for engine shutdown" if state == "STOPPING" else "Waiting for a fresh engine heartbeat",
                details=details,
                owner_pid=self._owner_pid,
                session_id=self._session_id,
            )

        if not actual_alive and state in {"STARTING", "RUNNING", "PAUSED", "STOPPING"}:
            return EngineState(
                state="STOPPED" if state == "STOPPING" else "ERROR",
                online=False,
                paused=False,
                pid=pid,
                mode=mode,
                heartbeat_age=age,
                started_at=str(raw.get("started_at", "") or ""),
                last_command=str(raw.get("last_command", "") or ""),
                note="Engine process is no longer running" if state == "STOPPING" else "Engine heartbeat exists but process PID is gone",
                details=details,
                exit_code=process_code,
                owner_pid=owner_pid,
                session_id=session_id,
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
            owner_pid=owner_pid,
            session_id=session_id,
        )

    def preflight(self, mode_override: str | None = None) -> tuple[bool, str]:
        mode = str(mode_override or self.mode()).upper()

        ok, message = self._cleanup_orphan_instance()
        if not ok:
            return False, message

        active = RuntimeControlPlane.active_instance(clean_stale=True)
        if active:
            return False, (
                f"Another TradeAI engine is already running "
                f"(PID {active.get('pid') or 'unknown'}, mode {active.get('mode') or 'UNKNOWN'})."
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

        creationflags = 0
        if os.name == "nt":
            # CREATE_NO_WINDOW is enough. Do not create a separate process group:
            # the engine is owned by this UI and must not behave like a detached daemon.
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            RuntimeControlPlane.clear_status()
            RuntimeControlPlane.clear_command()
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env[UI_OWNER_PID_ENV] = str(self._owner_pid)
            env[UI_SESSION_ID_ENV] = self._session_id
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

    def _wait_for_pid_exit(self, pid: int | None, timeout: float) -> bool:
        if not pid:
            return True
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if not pid_alive(pid):
                return True
            time.sleep(0.08)
        return not pid_alive(pid)

    def stop(self) -> tuple[bool, str]:
        state = self.status()
        active = RuntimeControlPlane.active_instance(clean_stale=True)

        pid = None
        if self._local_process_alive() and self._last_process is not None:
            pid = self._last_process.pid
        elif state.pid and pid_alive(state.pid):
            pid = state.pid
        elif active.get("pid") and pid_alive(active.get("pid")):
            pid = int(active.get("pid"))

        if not pid:
            if state.state in self.TERMINAL_STATES or state.state == "OFFLINE":
                return True, "TradeAI engine is already stopped"
            return False, "TradeAI engine process could not be resolved"

        # Do not let one live dashboard kill another dashboard's engine.
        active_owner = int(active.get("owner_pid", 0) or 0) if active else 0
        active_session = str(active.get("session_id", "") or "") if active else ""
        if active and active_owner > 0 and pid_alive(active_owner):
            if active_owner != self._owner_pid or active_session != self._session_id:
                return False, f"Engine PID {pid} belongs to another live TradeAI dashboard"

        mode = state.mode or self._last_launch_mode or self.mode()
        details = dict(state.details or {})

        # Phase 1: graceful stop through the control plane.
        try:
            RuntimeControlPlane.issue_command(
                "stop",
                source="tradeai-desktop-ui",
                session_id=self._session_id,
            )
        except Exception:
            pass

        if self._wait_for_pid_exit(pid, self.STOP_GRACE_SECONDS):
            RuntimeControlPlane.active_instance(clean_stale=True)
            return True, f"TradeAI stopped (PID {pid})"

        # Phase 2: local process terminate, then kill. This is essential for a
        # blocked MT5 IPC call that cannot observe the JSON stop event promptly.
        if self._last_process is not None and self._last_process.pid == pid and self._last_process.poll() is None:
            try:
                self._last_process.terminate()
            except Exception:
                pass
            if not self._wait_for_pid_exit(pid, self.TERMINATE_GRACE_SECONDS):
                try:
                    self._last_process.kill()
                except Exception:
                    terminate_pid(pid, force=True)
        else:
            terminate_pid(pid, force=True)

        dead = self._wait_for_pid_exit(pid, self.TERMINATE_GRACE_SECONDS)
        if not dead:
            return False, f"STOP failed: TradeAI PID {pid} is still alive"

        RuntimeControlPlane.active_instance(clean_stale=True)
        RuntimeControlPlane.publish_terminal_status(
            mode=mode,
            pid=pid,
            state="STOPPED",
            note="Engine force-stopped by owning dashboard after graceful-stop timeout",
            owner_pid=self._owner_pid,
            session_id=self._session_id,
            details=details,
        )
        return True, f"TradeAI stopped and PID {pid} confirmed exited"

    def stop_if_running(self) -> tuple[bool, str]:
        state = self.status()
        active = RuntimeControlPlane.active_instance(clean_stale=True)
        if not state.online and not active and not self._local_process_alive():
            return True, "No TradeAI engine is running"
        return self.stop()

    def command(self, command: str) -> tuple[bool, str]:
        state = self.status()
        if not state.online:
            return False, "TradeAI engine is offline"

        active = RuntimeControlPlane.active_instance(clean_stale=True)
        if active:
            active_session = str(active.get("session_id", "") or "")
            active_owner = int(active.get("owner_pid", 0) or 0)
            if active_session != self._session_id or active_owner != self._owner_pid:
                return False, "This TradeAI engine belongs to another dashboard session"

        try:
            RuntimeControlPlane.issue_command(
                command,
                source="tradeai-desktop-ui",
                session_id=self._session_id,
            )
            return True, f"{command.upper()} command sent"
        except Exception as exc:
            return False, f"Command failed: {exc}"
