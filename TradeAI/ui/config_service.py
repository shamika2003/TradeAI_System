from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


TRADEAI_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = TRADEAI_DIR / "config" / "settings.py"


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    kind: str
    section: str
    description: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


FIELDS: tuple[ConfigField, ...] = (
    ConfigField("MODE", "Operating mode", "choice", "Operating", "Baseline engine mode. LIVE can be saved, but the desktop launcher will not start LIVE.", ("DEMO_FORWARD", "PAPER", "BACKTEST", "LIVE")),
    ConfigField("SYMBOLS", "Symbols", "symbols", "Operating", "Comma-separated instruments used by the engine."),
    ConfigField("TIMEFRAME", "Timeframe (minutes)", "int", "Operating", "Primary M5 runtime value is 5.", minimum=1, maximum=1440),
    ConfigField("HISTORY_SIZE", "History size", "int", "Operating", "Bars supplied to the feature pipeline.", minimum=100, maximum=100000),
    ConfigField("LIVE_INTERVAL", "Live loop interval (s)", "float", "Operating", "Sleep between live market loops.", minimum=0.2, maximum=300),
    ConfigField("BACKTEST_DELAY", "Backtest delay (s)", "float", "Operating", "Optional replay delay; 0 is fastest.", minimum=0, maximum=60),

    ConfigField("START_BALANCE", "Paper/backtest capital", "float", "Capital & Limits", "Starting capital for PAPER/BACKTEST.", minimum=0.01, maximum=100000000),
    ConfigField("LOW_BALANCE_MODE", "Low-balance profile", "bool", "Capital & Limits", "Keeps low-capital safety assumptions enabled."),
    ConfigField("LOW_BALANCE_MIN", "Low-balance minimum", "float", "Capital & Limits", minimum=0.01, maximum=100000000),
    ConfigField("LOW_BALANCE_MAX", "Low-balance maximum", "float", "Capital & Limits", minimum=0.01, maximum=100000000),
    ConfigField("MAX_OPEN_POSITIONS", "Max open positions", "int", "Capital & Limits", minimum=1, maximum=100),
    ConfigField("MAX_TOTAL_POSITIONS", "Max total positions", "int", "Capital & Limits", minimum=1, maximum=100),
    ConfigField("COOLDOWN_SECONDS", "Cooldown (s)", "int", "Capital & Limits", minimum=0, maximum=86400),
    ConfigField("MAX_DRAWDOWN_PERCENT", "Max drawdown %", "float", "Capital & Limits", minimum=0, maximum=100),
    ConfigField("MAX_DAILY_LOSS_PERCENT", "Max daily loss %", "float", "Capital & Limits", minimum=0, maximum=100),
    ConfigField("MAX_ACTUAL_RISK_PERCENT", "Max actual risk %", "float", "Capital & Limits", "Hard cap after broker lot rounding.", minimum=0.01, maximum=10),
    ConfigField("MAX_PORTFOLIO_RISK_PERCENT", "Max portfolio risk %", "float", "Capital & Limits", "Aggregate loss-at-stop cap across concurrent positions.", minimum=0.01, maximum=10),

    ConfigField("MIN_LOT", "Minimum lot", "float", "Execution", minimum=0.0001, maximum=1000),
    ConfigField("MAX_LOT", "Maximum lot", "float", "Execution", minimum=0.0001, maximum=1000),
    ConfigField("LOT_STEP", "Lot step", "float", "Execution", minimum=0.0001, maximum=100),
    ConfigField("DEFAULT_LOT", "Default lot", "float", "Execution", minimum=0.0001, maximum=1000),
    ConfigField("USE_ATR_STOPS", "ATR stops", "bool", "Execution"),
    ConfigField("ATR_SL_MULTIPLIER", "ATR SL multiplier", "float", "Execution", minimum=0.01, maximum=100),
    ConfigField("ATR_TP_MULTIPLIER", "ATR TP multiplier", "float", "Execution", minimum=0.01, maximum=100),
    ConfigField("USE_BREAK_EVEN", "Break-even", "bool", "Execution"),
    ConfigField("BREAK_EVEN_TRIGGER_R", "Break-even trigger (R)", "float", "Execution", minimum=0, maximum=100),
    ConfigField("BREAK_EVEN_BUFFER_PIPS", "Break-even buffer (pips)", "float", "Execution", minimum=0, maximum=1000),
    ConfigField("BREAK_EVEN_MIN_PROFIT_MONEY", "Minimum protected profit", "float", "Execution", "Minimum positive cash result when break-even is hit.", minimum=0, maximum=100000),
    ConfigField("USE_PROFIT_LOCK", "Profit lock", "bool", "Execution"),
    ConfigField("PROFIT_LOCK_TRIGGER_R", "Profit-lock trigger (R)", "float", "Execution", minimum=0, maximum=100),
    ConfigField("PROFIT_LOCK_R", "Profit locked (R)", "float", "Execution", minimum=0, maximum=100),
    ConfigField("USE_TRAILING_STOP", "Trailing stop", "bool", "Execution"),
    ConfigField("TRAILING_TRIGGER_R", "Trailing trigger (R)", "float", "Execution", minimum=0, maximum=100),
    ConfigField("TRAILING_ATR_MULTIPLIER", "Trailing ATR multiplier", "float", "Execution", minimum=0, maximum=100),
    ConfigField("USE_AI_DEFENSIVE_EXIT", "AI defensive exit", "bool", "Execution"),
    ConfigField("AI_DEFENSIVE_EXIT_MIN_BARS", "AI exit minimum bars", "int", "Execution", minimum=1, maximum=100000),
    ConfigField("AI_DEFENSIVE_EXIT_ADVERSE_R", "AI exit adverse R", "float", "Execution", minimum=0, maximum=100),
    ConfigField("AI_DEFENSIVE_EXIT_SIGNAL_MULTIPLIER", "AI exit signal multiplier", "float", "Execution", minimum=0, maximum=100),
    ConfigField("USE_MAX_HOLD", "Max-hold exit", "bool", "Execution"),
    ConfigField("MAX_HOLD_BARS", "Max hold bars", "int", "Execution", minimum=1, maximum=100000),
    ConfigField("MAX_HOLD_MINUTES", "Max hold minutes", "int", "Execution", minimum=1, maximum=1000000),
    ConfigField("DEVIATION", "MT5 deviation", "int", "Execution", minimum=0, maximum=10000),
    ConfigField("MT5_MAGIC", "MT5 magic", "int", "Execution", minimum=1, maximum=2147483647),

    ConfigField("BACKTEST_START_DATE", "Start date", "date", "Backtest", "YYYY-MM-DD"),
    ConfigField("BACKTEST_END_DATE", "End date", "date", "Backtest", "YYYY-MM-DD"),
    ConfigField("DEFAULT_SPREAD_PIPS", "Spread (pips)", "float", "Backtest", minimum=0, maximum=1000),
    ConfigField("SIMULATED_SLIPPAGE_PIPS", "Slippage (pips)", "float", "Backtest", minimum=0, maximum=1000),
    ConfigField("COMMISSION_PER_LOT", "Commission / lot", "float", "Backtest", minimum=0, maximum=100000),
    ConfigField("USE_SPREAD_EXECUTION", "Spread execution", "bool", "Backtest"),

    ConfigField("DEMO_FORWARD_CAPITAL", "Shadow capital", "float", "Demo Forward", minimum=0.01, maximum=100000000),
    ConfigField("DEMO_FORWARD_MIN_TRADES", "Minimum trades", "int", "Demo Forward", minimum=1, maximum=10000000),
    ConfigField("DEMO_FORWARD_MAX_DD_PERCENT", "Maximum DD %", "float", "Demo Forward", minimum=0, maximum=100),
    ConfigField("DEMO_FORWARD_MIN_PROFIT_FACTOR", "Minimum profit factor", "float", "Demo Forward", minimum=0, maximum=1000),
    ConfigField("DEMO_FORWARD_MIN_RETURN_PERCENT", "Minimum return %", "float", "Demo Forward", minimum=-100, maximum=100000),

    ConfigField("SIGNAL_THRESHOLD", "Fallback signal threshold", "float", "Compatibility", "Production uses calibrated per-symbol policy when available.", minimum=0, maximum=10),
    ConfigField("MIN_CONFIDENCE", "Fallback minimum confidence", "float", "Compatibility", "Production uses calibrated per-symbol policy when available.", minimum=0, maximum=1),
)

FIELD_MAP = {field.key: field for field in FIELDS}


class TradeAIConfigService:
    """Safe, whitelisted editor for operational values in config/settings.py."""

    def __init__(self, settings_path: Path = SETTINGS_PATH) -> None:
        self.settings_path = Path(settings_path)

    def read(self) -> dict[str, Any]:
        source = self.settings_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(self.settings_path))
        values: dict[str, Any] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in FIELD_MAP or target.id in values:
                continue
            try:
                values[target.id] = ast.literal_eval(node.value)
            except Exception:
                continue
        return values

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError("must be true or false")

    def parse_value(self, field: ConfigField, raw: Any) -> Any:
        if field.kind == "choice":
            value = str(raw).strip().upper()
            if value not in field.choices:
                raise ValueError(f"must be one of {', '.join(field.choices)}")
            return value
        if field.kind == "symbols":
            if isinstance(raw, (list, tuple)):
                symbols = [str(x).strip().upper() for x in raw if str(x).strip()]
            else:
                symbols = [part.strip().upper() for part in str(raw).split(",") if part.strip()]
            if not symbols:
                raise ValueError("at least one symbol is required")
            for symbol in symbols:
                if not re.fullmatch(r"[A-Z0-9._-]+", symbol):
                    raise ValueError(f"invalid symbol {symbol!r}")
            return symbols
        if field.kind == "bool":
            return self._parse_bool(raw)
        if field.kind == "int":
            value = int(str(raw).strip())
        elif field.kind == "float":
            value = float(str(raw).strip())
        elif field.kind == "date":
            value = str(raw).strip()
            date.fromisoformat(value)
            return value
        else:
            return str(raw)

        if field.minimum is not None and value < field.minimum:
            raise ValueError(f"must be >= {field.minimum}")
        if field.maximum is not None and value > field.maximum:
            raise ValueError(f"must be <= {field.maximum}")
        return value

    def validate(self, raw_values: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        errors: list[str] = []

        for key, raw in raw_values.items():
            field = FIELD_MAP.get(key)
            if field is None:
                errors.append(f"{key}: not editable from the UI")
                continue
            try:
                parsed[key] = self.parse_value(field, raw)
            except Exception as exc:
                errors.append(f"{field.label}: {exc}")

        if errors:
            raise ValueError("\n".join(errors))

        merged = self.read()
        merged.update(parsed)

        min_lot = float(merged.get("MIN_LOT", 0.0))
        max_lot = float(merged.get("MAX_LOT", 0.0))
        default_lot = float(merged.get("DEFAULT_LOT", 0.0))
        step = float(merged.get("LOT_STEP", 0.0))
        if not (0 < min_lot <= default_lot <= max_lot):
            raise ValueError("Lot sizes must satisfy MIN_LOT <= DEFAULT_LOT <= MAX_LOT.")
        if step <= 0:
            raise ValueError("LOT_STEP must be greater than zero.")

        low_min = float(merged.get("LOW_BALANCE_MIN", 0.0))
        low_max = float(merged.get("LOW_BALANCE_MAX", 0.0))
        if low_min > low_max:
            raise ValueError("LOW_BALANCE_MIN cannot exceed LOW_BALANCE_MAX.")

        start = str(merged.get("BACKTEST_START_DATE", ""))
        end = str(merged.get("BACKTEST_END_DATE", ""))
        if start and end and date.fromisoformat(start) > date.fromisoformat(end):
            raise ValueError("Backtest start date cannot be after the end date.")

        return parsed

    @staticmethod
    def _python_literal(value: Any) -> str:
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, list):
            return repr(value)
        return repr(value)

    def save(self, raw_values: dict[str, Any]) -> tuple[bool, str]:
        try:
            parsed = self.validate(raw_values)
        except Exception as exc:
            return False, str(exc)

        try:
            source = self.settings_path.read_text(encoding="utf-8")
            for key, value in parsed.items():
                pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=\s*[^\r\n]+")
                replacement = f"{key} = {self._python_literal(value)}"
                source, count = pattern.subn(replacement, source, count=1)
                if count != 1:
                    return False, f"Could not locate {key} in settings.py"

            temp = self.settings_path.with_name(f".{self.settings_path.name}.ui.tmp")
            temp.write_text(source, encoding="utf-8")
            # Parse before replacing the live file.
            ast.parse(source, filename=str(self.settings_path))
            os.replace(temp, self.settings_path)
            return True, "Configuration saved. Engine changes apply on the next engine start."
        except Exception as exc:
            return False, f"Unable to save configuration: {exc}"
