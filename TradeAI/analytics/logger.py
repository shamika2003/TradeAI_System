# filename: TradeAI/analytics/logger.py

from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from config.settings import BOT_LOG
    LOG_FILE = Path(BOT_LOG)
except Exception:
    LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "bot.log"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

if not LOG_FILE.exists():
    LOG_FILE.write_text(
        f"{datetime.now()} | INFO | Logger initialized\n",
        encoding="utf-8",
    )


def _write_log(level: str, message: str):
    timestamp = datetime.now()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {level} | {message}\n")


def info(message: str):
    _write_log("INFO", message)


def warning(message: str):
    _write_log("WARNING", message)


def error(message: str):
    _write_log("ERROR", message)


def debug(message: str):
    _write_log("DEBUG", message)


def log(message: str):
    info(message)
