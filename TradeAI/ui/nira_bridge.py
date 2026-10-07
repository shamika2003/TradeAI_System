from __future__ import annotations

"""Authenticated local client for the shared ELVARA NIRA runtime."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PySide6.QtCore import QThread, Signal


NIRA_BRIDGE_BASE_URL = "http://127.0.0.1:8766"
NIRA_APP_ID = "tradeai"
NIRA_CHAT_URL = f"{NIRA_BRIDGE_BASE_URL}/v1/apps/{NIRA_APP_ID}/chat"
NIRA_HEALTH_URL = f"{NIRA_BRIDGE_BASE_URL}/health"


def _credential_path() -> Path:
    local_app_data = str(os.environ.get("LOCALAPPDATA", "")).strip()
    if not local_app_data:
        local_app_data = str(Path.home() / "AppData" / "Local")
    return (
        Path(local_app_data)
        / "ELVARA"
        / "NIRA"
        / "bridge"
        / "credentials"
        / f"{NIRA_APP_ID}.token"
    )


@dataclass(frozen=True)
class NiraBridgeContext:
    page: str = "overview"
    selected_entity: str = ""


@dataclass(frozen=True)
class NiraBridgeReply:
    app_id: str
    reply: str
    run_id: str
    scope: str


class NiraBridgeError(RuntimeError):
    pass


class NiraBridgeClient:
    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        self.timeout_seconds = max(5.0, min(float(timeout_seconds), 180.0))

    def health(self) -> dict[str, Any]:
        request = Request(
            NIRA_HEALTH_URL,
            method="GET",
            headers={"Accept": "application/json", "Cache-Control": "no-store"},
        )
        payload = self._send_json(request, timeout_seconds=4.0)
        if not bool(payload.get("chatEnabled", False)):
            raise NiraBridgeError("NIRA is running, but embedded chat is not enabled.")
        return payload

    def chat(self, message: str, context: NiraBridgeContext) -> NiraBridgeReply:
        clean_message = str(message or "").strip()
        if not clean_message:
            raise NiraBridgeError("Enter a message for NIRA.")

        token = self._read_token()
        body = json.dumps(
            {
                "message": clean_message,
                "surface": "tradeai_dashboard",
                "page": str(context.page or "overview").strip() or "overview",
                "selectedEntity": str(context.selected_entity or "").strip(),
            },
            ensure_ascii=False,
        ).encode("utf-8")

        request = Request(
            NIRA_CHAT_URL,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {token}",
                "Cache-Control": "no-store",
            },
        )

        payload = self._send_json(request, timeout_seconds=self.timeout_seconds)
        reply = str(payload.get("reply", "") or "").strip()
        if not reply:
            raise NiraBridgeError("NIRA returned an empty response.")

        return NiraBridgeReply(
            app_id=str(payload.get("appId", "") or ""),
            reply=reply,
            run_id=str(payload.get("runId", "") or ""),
            scope=str(payload.get("scope", "") or ""),
        )

    @staticmethod
    def _read_token() -> str:
        path = _credential_path()
        try:
            token = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise NiraBridgeError(
                "NIRA's TradeAI bridge credential does not exist yet. "
                "Start the updated NIRA desktop app once, then try again."
            ) from exc
        except OSError as exc:
            raise NiraBridgeError(
                f"Could not read the local NIRA bridge credential: {exc}"
            ) from exc

        if len(token) < 40:
            raise NiraBridgeError("The local NIRA bridge credential is invalid.")
        return token

    @staticmethod
    def _send_json(request: Request, *, timeout_seconds: float) -> dict[str, Any]:
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(256 * 1024)
        except HTTPError as exc:
            message = ""
            try:
                payload = json.loads(exc.read().decode("utf-8", errors="replace"))
                message = str(payload.get("message", "") or payload.get("error", "") or "")
            except Exception:
                pass
            if exc.code == 401:
                raise NiraBridgeError(
                    message or "TradeAI was not authenticated by the NIRA bridge."
                ) from exc
            raise NiraBridgeError(
                message or f"NIRA bridge returned HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise NiraBridgeError(
                "NIRA is offline or the local bridge at 127.0.0.1:8766 is unavailable."
            ) from exc
        except TimeoutError as exc:
            raise NiraBridgeError("NIRA did not answer before the local timeout.") from exc
        except OSError as exc:
            raise NiraBridgeError(f"NIRA bridge connection failed: {exc}") from exc

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NiraBridgeError("NIRA bridge returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise NiraBridgeError("NIRA bridge returned an unexpected response.")
        return payload


class NiraBridgeWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, client, message, context, parent=None) -> None:
        super().__init__(parent)
        self.client = client
        self.message = message
        self.context = context

    def run(self) -> None:
        try:
            reply = self.client.chat(self.message, self.context)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(reply)


class NiraBridgeProbeWorker(QThread):
    online = Signal(object)
    offline = Signal(str)

    def __init__(self, client, parent=None) -> None:
        super().__init__(parent)
        self.client = client

    def run(self) -> None:
        try:
            health = self.client.health()
        except Exception as exc:
            self.offline.emit(str(exc))
            return
        self.online.emit(health)
