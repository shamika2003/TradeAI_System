from __future__ import annotations

import ctypes
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRADEAI_DIR = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = TRADEAI_DIR.parent
REPORT_DIR = SYSTEM_ROOT / "artifacts" / "reports"
COMMAND_PATH = REPORT_DIR / "tradeai_control_command.json"
STATUS_PATH = REPORT_DIR / "tradeai_engine_status.json"
INSTANCE_PATH = REPORT_DIR / "tradeai_engine_instance.json"

UI_OWNER_PID_ENV = "TRADEAI_UI_OWNER_PID"
UI_SESSION_ID_ENV = "TRADEAI_UI_SESSION_ID"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Windows-safe atomic JSON writer with transient sharing-violation retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

        last_error: Exception | None = None
        for attempt in range(12):
            try:
                os.replace(temp, path)
                return
            except PermissionError as exc:
                last_error = exc
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {5, 32}:
                    raise
                last_error = exc
            time.sleep(0.015 * (attempt + 1))

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unable to replace {path}")
    finally:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass


def pid_alive(pid: int | None) -> bool:
    try:
        pid = int(pid or 0)
    except Exception:
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True

    if os.name == "nt":
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return int(exit_code.value) == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def terminate_pid(pid: int | None, *, force: bool = False) -> bool:
    """Terminate a TradeAI child PID. Returns True once the process is gone."""
    try:
        pid = int(pid or 0)
    except Exception:
        return True
    if pid <= 0 or not pid_alive(pid):
        return True
    if pid == os.getpid():
        return False

    try:
        if os.name == "nt":
            PROCESS_TERMINATE = 0x0001
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if not handle:
                return not pid_alive(pid)
            try:
                # Windows has no gentle SIGTERM equivalent for a detached Python
                # process. The engine receives a graceful JSON STOP first; this is
                # only the fallback when it refuses to exit.
                kernel32.TerminateProcess(handle, 1 if force else 0)
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        return not pid_alive(pid)

    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


def ui_launch_context() -> tuple[bool, str, int | None, str]:
    """Validate that demo_bot.py was launched by the desktop dashboard."""
    session_id = str(os.environ.get(UI_SESSION_ID_ENV, "") or "").strip()
    raw_owner = str(os.environ.get(UI_OWNER_PID_ENV, "") or "").strip()

    try:
        owner_pid = int(raw_owner)
    except Exception:
        owner_pid = 0

    if not session_id or owner_pid <= 0:
        return (
            False,
            "TradeAI engine launch blocked: start it from the TradeAI desktop UI.",
            None,
            session_id,
        )
    if not pid_alive(owner_pid):
        return (
            False,
            f"TradeAI engine launch blocked: dashboard owner PID {owner_pid} is not running.",
            owner_pid,
            session_id,
        )
    return True, "UI ownership verified", owner_pid, session_id


class EngineInstanceLock:
    """Cross-process single-engine guard tied to the dashboard session."""

    def __init__(self, mode: str, *, owner_pid: int | None = None, session_id: str = "") -> None:
        self.mode = str(mode or "UNKNOWN").upper()
        self.pid = os.getpid()
        self.owner_pid = int(owner_pid or 0)
        self.session_id = str(session_id or "")
        self.acquired = False

    def acquire(self) -> None:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        for _ in range(4):
            active = RuntimeControlPlane.active_instance(clean_stale=True)
            active_pid = active.get("pid") if active else None
            if active and int(active_pid or 0) != self.pid:
                raise RuntimeError(
                    f"Another TradeAI engine is already running "
                    f"(PID {active_pid or 'unknown'}, mode {active.get('mode', 'UNKNOWN')}). "
                    "Stop it before starting another engine or backtest."
                )

            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            try:
                fd = os.open(str(INSTANCE_PATH), flags)
            except FileExistsError:
                time.sleep(0.05)
                continue

            payload = {
                "version": "tradeai_engine_instance_v2",
                "pid": self.pid,
                "mode": self.mode,
                "owner_pid": self.owner_pid,
                "session_id": self.session_id,
                "started_at": _utc_now(),
            }
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
                self.acquired = True
                return
            except Exception:
                try:
                    INSTANCE_PATH.unlink(missing_ok=True)
                except Exception:
                    pass
                raise

        active = RuntimeControlPlane.active_instance(clean_stale=True)
        raise RuntimeError(
            "TradeAI engine instance lock is busy"
            + (f" (PID {active.get('pid')}, mode {active.get('mode')})" if active else "")
        )

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            raw = _read_json(INSTANCE_PATH)
            if int(raw.get("pid", 0) or 0) == self.pid:
                INSTANCE_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        self.acquired = False


class RuntimeControlPlane:
    """Durable local process control/heartbeat channel shared by engine and UI."""

    VALID_COMMANDS = {"pause", "resume", "report", "stop", "status"}

    def __init__(
        self,
        mode: str = "UNKNOWN",
        *,
        owner_pid: int | None = None,
        session_id: str = "",
    ) -> None:
        self.mode = str(mode or "UNKNOWN").upper()
        self.started_at = _utc_now()
        self.owner_pid = int(owner_pid or 0)
        self.session_id = str(session_id or "")

    @staticmethod
    def current_command_id() -> int:
        try:
            return int(_read_json(COMMAND_PATH).get("command_id", 0) or 0)
        except Exception:
            return 0

    @staticmethod
    def read_command() -> dict[str, Any]:
        return _read_json(COMMAND_PATH)

    @staticmethod
    def issue_command(
        command: str,
        source: str = "desktop-ui",
        *,
        session_id: str = "",
    ) -> int:
        normalized = str(command or "").strip().lower()
        if normalized not in RuntimeControlPlane.VALID_COMMANDS:
            raise ValueError(f"Unsupported engine command: {command}")
        command_id = time.time_ns()
        _atomic_write(
            COMMAND_PATH,
            {
                "version": "tradeai_control_v4",
                "command_id": command_id,
                "command": normalized,
                "source": source,
                "session_id": str(session_id or ""),
                "requested_at": _utc_now(),
            },
        )
        return command_id

    @staticmethod
    def read_status() -> dict[str, Any]:
        return _read_json(STATUS_PATH)

    @staticmethod
    def active_instance(*, clean_stale: bool = True) -> dict[str, Any]:
        if not INSTANCE_PATH.exists():
            return {}
        raw = _read_json(INSTANCE_PATH)
        pid = raw.get("pid")
        if pid and pid_alive(pid):
            return raw

        try:
            age = max(0.0, time.time() - INSTANCE_PATH.stat().st_mtime)
        except Exception:
            age = 999.0
        if not pid and age < 3.0:
            return {
                "pid": None,
                "mode": "STARTING",
                "started_at": "",
                "initializing": True,
            }

        if clean_stale:
            try:
                INSTANCE_PATH.unlink(missing_ok=True)
            except Exception:
                pass
        return {}

    @staticmethod
    def clear_status() -> None:
        try:
            STATUS_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def clear_command() -> None:
        try:
            COMMAND_PATH.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def publish_terminal_status(
        *,
        mode: str,
        pid: int | None,
        state: str,
        note: str,
        owner_pid: int | None = None,
        session_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        _atomic_write(
            STATUS_PATH,
            {
                "version": "tradeai_engine_status_v4",
                "pid": int(pid or 0) or None,
                "mode": str(mode or "UNKNOWN").upper(),
                "state": str(state or "STOPPED").upper(),
                "paused": False,
                "started_at": "",
                "heartbeat_utc": _utc_now(),
                "last_command_id": 0,
                "last_command": "stop",
                "note": str(note or ""),
                "owner_pid": int(owner_pid or 0) or None,
                "session_id": str(session_id or ""),
                "details": details if isinstance(details, dict) else {},
            },
        )

    def write_status(
        self,
        *,
        state: str,
        paused: bool,
        last_command_id: int = 0,
        last_command: str = "",
        note: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_details = details if isinstance(details, dict) else {}
        _atomic_write(
            STATUS_PATH,
            {
                "version": "tradeai_engine_status_v4",
                "pid": os.getpid(),
                "mode": self.mode,
                "state": str(state or "UNKNOWN").upper(),
                "paused": bool(paused),
                "started_at": self.started_at,
                "heartbeat_utc": _utc_now(),
                "last_command_id": int(last_command_id or 0),
                "last_command": str(last_command or ""),
                "note": str(note or ""),
                "owner_pid": self.owner_pid or None,
                "session_id": self.session_id,
                "details": safe_details,
            },
        )


__all__ = [
    "RuntimeControlPlane",
    "EngineInstanceLock",
    "COMMAND_PATH",
    "STATUS_PATH",
    "INSTANCE_PATH",
    "UI_OWNER_PID_ENV",
    "UI_SESSION_ID_ENV",
    "pid_alive",
    "terminate_pid",
    "ui_launch_context",
]
