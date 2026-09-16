from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRADEAI_DIR = Path(__file__).resolve().parent.parent
SYSTEM_ROOT = TRADEAI_DIR.parent
REPORT_DIR = SYSTEM_ROOT / "artifacts" / "reports"
COMMAND_PATH = REPORT_DIR / "tradeai_control_command.json"
STATUS_PATH = REPORT_DIR / "tradeai_engine_status.json"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(temp, path)


class RuntimeControlPlane:
    """Durable local process control/heartbeat channel shared by engine and UI.

    The files in ``artifacts/reports`` are intentionally process-neutral.  The
    dashboard can be closed/reopened without owning the trading loop, and the
    engine can keep publishing its state without depending on console stdin.
    """

    VALID_COMMANDS = {"pause", "resume", "report", "stop", "status"}

    def __init__(self, mode: str = "UNKNOWN") -> None:
        self.mode = str(mode or "UNKNOWN").upper()
        self.started_at = _utc_now()

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
    def issue_command(command: str, source: str = "desktop-ui") -> int:
        normalized = str(command or "").strip().lower()
        if normalized not in RuntimeControlPlane.VALID_COMMANDS:
            raise ValueError(f"Unsupported engine command: {command}")
        command_id = time.time_ns()
        _atomic_write(
            COMMAND_PATH,
            {
                "version": "tradeai_control_v2",
                "command_id": command_id,
                "command": normalized,
                "source": source,
                "requested_at": _utc_now(),
            },
        )
        return command_id

    @staticmethod
    def read_status() -> dict[str, Any]:
        return _read_json(STATUS_PATH)

    @staticmethod
    def clear_status() -> None:
        """Remove a stale status before a new subprocess launch."""
        try:
            STATUS_PATH.unlink(missing_ok=True)
        except Exception:
            pass

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
                "version": "tradeai_engine_status_v2",
                "pid": os.getpid(),
                "mode": self.mode,
                "state": str(state or "UNKNOWN").upper(),
                "paused": bool(paused),
                "started_at": self.started_at,
                "heartbeat_utc": _utc_now(),
                "last_command_id": int(last_command_id or 0),
                "last_command": str(last_command or ""),
                "note": str(note or ""),
                "details": safe_details,
            },
        )
