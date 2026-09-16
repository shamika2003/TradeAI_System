from __future__ import annotations

import threading
import time
from typing import Any

from analytics.logger import log
from core.runtime_control import RuntimeControlPlane, pid_alive, ui_launch_context


class CommandControl:
    """Engine-side control listener, heartbeat publisher, and UI ownership watchdog."""

    POLL_SECONDS = 0.20
    HEARTBEAT_SECONDS = 0.75

    def __init__(self, mode: str = "UNKNOWN") -> None:
        self.running = True
        self.report_requested = False
        self.paused = False

        ok, reason, owner_pid, session_id = ui_launch_context()
        if not ok:
            raise RuntimeError(reason)

        self._owner_pid = int(owner_pid or 0)
        self._session_id = str(session_id or "")
        self._plane = RuntimeControlPlane(
            mode,
            owner_pid=self._owner_pid,
            session_id=self._session_id,
        )
        self._last_command_id = self._plane.current_command_id()
        self._last_command = ""
        self._state = "STARTING"
        self._note = "Engine initialization"
        self._details: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._finalized = False

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def details(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._details)

    @property
    def session_id(self) -> str:
        return self._session_id

    def should_stop(self) -> bool:
        return self._stop_event.is_set() or not self.running or self._finalized

    def wait(self, seconds: float) -> bool:
        """Interruptible sleep. Returns True if a stop was requested."""
        return self._stop_event.wait(max(0.0, float(seconds)))

    def start(self, initial_state: str = "RUNNING", note: str = "Systems ready") -> None:
        with self._lock:
            self._state = str(initial_state or "RUNNING").upper()
            self._note = str(note or "")
            self._finalized = False
            self.running = True
            self._stop_event.clear()
        self._write_status()

        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self.listen,
            name="TradeAI-ControlPlane",
            daemon=True,
        )
        self._thread.start()

    def set_state(self, state: str, note: str = "") -> None:
        with self._lock:
            self._state = str(state or "UNKNOWN").upper()
            self._note = str(note or "")
            if self._state == "PAUSED":
                self.paused = True
            elif self._state in {"RUNNING", "STARTING"}:
                self.paused = False
        self._write_status()

    def set_details(self, **details: Any) -> None:
        with self._lock:
            self._details.update(details)
        self._write_status()

    def replace_details(self, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._details = dict(details or {})
        self._write_status()

    def heartbeat(self) -> None:
        self._write_status()

    def _write_status(self) -> None:
        with self._lock:
            state = self._state
            paused = self.paused
            last_command_id = self._last_command_id
            last_command = self._last_command
            note = self._note
            details = dict(self._details)
        try:
            with self._write_lock:
                self._plane.write_status(
                    state=state,
                    paused=paused,
                    last_command_id=last_command_id,
                    last_command=last_command,
                    note=note,
                    details=details,
                )
        except Exception as exc:
            log(f"WARNING | Control heartbeat write failed: {exc}")

    def _request_stop(self, note: str, log_message: str) -> None:
        with self._lock:
            if self._finalized or not self.running:
                return
            self.running = False
            self.paused = False
            self._state = "STOPPING"
            self._note = note
            self._stop_event.set()
        log(log_message)
        self._write_status()

    def _handle_command(self, command: str, command_id: int) -> None:
        command = str(command or "").lower().strip()
        if not command:
            return

        with self._lock:
            self._last_command_id = int(command_id or 0)
            self._last_command = command

            if command == "stop":
                self.running = False
                self.paused = False
                self._state = "STOPPING"
                self._note = "Stop requested from dashboard"
                self._stop_event.set()
                log("INFO | Stop command received")

            elif command == "report":
                self.report_requested = True
                self._note = "Report requested"
                log("INFO | Report requested")

            elif command == "pause":
                if self.running:
                    self.paused = True
                    self._state = "PAUSED"
                    self._note = "Strategy processing paused"
                    log("INFO | Bot paused")

            elif command == "resume":
                if self.running:
                    self.paused = False
                    self._state = "RUNNING"
                    self._note = "Strategy processing resumed"
                    log("INFO | Bot resumed")

            elif command == "status":
                self._note = "Status requested"

        self._write_status()

    def listen(self) -> None:
        last_heartbeat = 0.0
        while self.running and not self._finalized:
            # The engine is intentionally owned by the desktop UI. If the UI is
            # closed, crashes, or is force-killed, the engine must not remain in
            # the background managing/fetching indefinitely.
            if not pid_alive(self._owner_pid):
                self._request_stop(
                    "Dashboard process ended; engine shutdown required",
                    "INFO | Dashboard owner ended; stopping TradeAI engine",
                )
                break

            try:
                raw = self._plane.read_command()
                command_id = int(raw.get("command_id", 0) or 0)
                if command_id and command_id != self._last_command_id:
                    command_session = str(raw.get("session_id", "") or "")
                    if command_session != self._session_id:
                        # Consume but reject commands from stale/other dashboard sessions.
                        self._last_command_id = command_id
                    else:
                        self._handle_command(str(raw.get("command", "")), command_id)
            except Exception as exc:
                log(f"WARNING | Control command read failed: {exc}")

            if self.should_stop():
                break

            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_SECONDS:
                self._write_status()
                last_heartbeat = now

            self._stop_event.wait(self.POLL_SECONDS)

    def shutdown(self, final_state: str = "STOPPED", note: str = "") -> None:
        """Publish a terminal state without allowing heartbeat overwrite."""
        with self._lock:
            self._finalized = True
            self.running = False
            self.paused = False
            self._state = str(final_state or "STOPPED").upper()
            self._note = str(note or self._note)
            self._stop_event.set()
        self._write_status()


__all__ = ["CommandControl"]
