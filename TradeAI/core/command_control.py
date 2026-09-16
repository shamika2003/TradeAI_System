from __future__ import annotations

import threading
import time
from typing import Any

from analytics.logger import log
from core.runtime_control import RuntimeControlPlane


class CommandControl:
    """Engine-side control plane listener + heartbeat publisher.

    Older builds listened to ``input()``.  The desktop launcher deliberately
    starts TradeAI with stdin detached, so that listener could never receive UI
    commands and it never produced a heartbeat.  This controller is entirely
    file/control-plane based and therefore works for both console and UI launches.
    """

    POLL_SECONDS = 0.25
    HEARTBEAT_SECONDS = 0.75

    def __init__(self, mode: str = "UNKNOWN") -> None:
        self.running = True
        self.report_requested = False
        self.paused = False

        self._plane = RuntimeControlPlane(mode)
        self._last_command_id = self._plane.current_command_id()
        self._last_command = ""
        self._state = "STARTING"
        self._note = "Engine initialization"
        self._details: dict[str, Any] = {}
        self._lock = threading.RLock()
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

    def start(self, initial_state: str = "RUNNING", note: str = "Systems ready") -> None:
        with self._lock:
            self._state = str(initial_state or "RUNNING").upper()
            self._note = str(note or "")
            self._finalized = False
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
                self._note = "Stop requested from control plane"
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
            try:
                raw = self._plane.read_command()
                command_id = int(raw.get("command_id", 0) or 0)
                if command_id and command_id != self._last_command_id:
                    self._handle_command(str(raw.get("command", "")), command_id)
            except Exception as exc:
                log(f"WARNING | Control command read failed: {exc}")

            now = time.monotonic()
            if now - last_heartbeat >= self.HEARTBEAT_SECONDS:
                self._write_status()
                last_heartbeat = now

            time.sleep(self.POLL_SECONDS)

    def shutdown(self, final_state: str = "STOPPED", note: str = "") -> None:
        """Publish a terminal state without allowing the heartbeat to overwrite it."""
        with self._lock:
            self._finalized = True
            self.running = False
            self.paused = False
            self._state = str(final_state or "STOPPED").upper()
            self._note = str(note or self._note)
        self._write_status()
