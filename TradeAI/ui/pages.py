from __future__ import annotations

import math
from datetime import datetime

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .components import (
    ElvaraAlertDialog,
    ElvaraButton,
    ElvaraConfirmDialog,
    ElvaraNotice,
    ElvaraSearchField,
    ElvaraStatusBadge,
    ElvaraTable,
)
from .control_service import EngineControlService, EngineState
from .config_service import FIELDS, TradeAIConfigService
from .data_service import REPORT_DIR, RuntimeSnapshot, TradeAIDataService
from .widgets import (
    AMBER,
    BLUE,
    CYAN,
    GREEN,
    RED,
    AreaLineChart,
    CandlestickWidget,
    CompactStat,
    DecisionPipeline,
    HorizontalBarChart,
    MetricCard,
    RiskGauge,
    SectionTitle,
    SignalCard,
)


def _panel(title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame(); frame.setObjectName("GlassPanel")
    layout = QVBoxLayout(frame); layout.setContentsMargins(16, 14, 16, 16); layout.setSpacing(8)
    layout.addWidget(SectionTitle(title, subtitle))
    return frame, layout


def _table(columns: list[str], density: str = "compact") -> ElvaraTable:
    return ElvaraTable(columns, density=density)


def _scroll_shell() -> tuple[QScrollArea, QWidget, QVBoxLayout]:
    scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    host = QWidget(); root = QVBoxLayout(host); root.setContentsMargins(16, 12, 16, 16); root.setSpacing(8)
    scroll.setWidget(host)
    return scroll, host, root


class OverviewPage(QWidget):
    def __init__(self, service: TradeAIDataService, control: EngineControlService, parent=None):
        super().__init__(parent)
        self.service = service
        self.control = control
        self.engine_state = EngineState(mode=self.control.mode())
        self._snapshot: RuntimeSnapshot | None = None
        self.policy = self.service.load_decision_policy()
        self.risk_policy = self.service.load_risk_policy()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, _, root = _scroll_shell()
        outer.addWidget(scroll)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(8)

        # Compact heading + quick engine controls.
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        title_box = QVBoxLayout(); title_box.setSpacing(0)
        eye = QLabel("LIVE OPERATIONS / M5 / AI EXECUTION")
        eye.setObjectName("Eyebrow")
        title = QLabel("Command Center")
        title.setObjectName("PageTitle")
        sub = QLabel("Live market, model gates, portfolio risk and execution telemetry")
        sub.setObjectName("PageSub")
        title_box.addWidget(eye); title_box.addWidget(title); title_box.addWidget(sub)
        title_row.addLayout(title_box)
        title_row.addStretch()

        self.start_btn = ElvaraButton("START AI", variant="primary", size="compact")
        self.pause_btn = ElvaraButton("PAUSE", variant="secondary", size="compact")
        self.resume_btn = ElvaraButton("RESUME", variant="primary", size="compact")
        self.stop_btn = ElvaraButton("STOP", variant="danger", size="compact")
        self.start_btn.clicked.connect(self._start_engine)
        self.pause_btn.clicked.connect(lambda: self._send_command("pause"))
        self.resume_btn.clicked.connect(lambda: self._send_command("resume"))
        self.stop_btn.clicked.connect(self._stop_engine)
        for b in (self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn):
            title_row.addWidget(b, 0, Qt.AlignBottom)

        self.chart_symbol = QComboBox()
        self.chart_symbol.addItems(self.service.symbols)
        self.chart_symbol.setMinimumWidth(100)
        title_row.addWidget(self.chart_symbol, 0, Qt.AlignBottom)
        root.addLayout(title_row)

        # Main B-layout: MARKET | AI DECISION | PORTFOLIO/RISK.
        hero = QGridLayout()
        hero.setSpacing(7)
        hero.setColumnStretch(0, 52)
        hero.setColumnStretch(1, 25)
        hero.setColumnStretch(2, 23)

        market_panel, market_l = _panel("Live Market", "Real MT5 M5 candles · forming candle is display-only")
        market_header = QHBoxLayout(); market_header.setSpacing(7)
        self.market_pair = QLabel("EURUSD / M5"); self.market_pair.setObjectName("Cyan")
        self.market_ohlc = QLabel("O —   H —   L —   C —"); self.market_ohlc.setObjectName("MetricDetail")
        self.market_price = QLabel("SYNCING"); self.market_price.setObjectName("StatusNeutral")
        market_header.addWidget(self.market_pair); market_header.addWidget(self.market_ohlc); market_header.addStretch(); market_header.addWidget(self.market_price)
        market_l.addLayout(market_header)
        self.candle_chart = CandlestickWidget(); self.candle_chart.setMinimumHeight(252)
        market_l.addWidget(self.candle_chart, 1)
        quote_row = QHBoxLayout(); quote_row.setSpacing(5)
        self.quote_labels: dict[str, QLabel] = {}
        for symbol in self.service.symbols:
            q = QLabel(f"{symbol}  —")
            q.setObjectName("StatusNeutral")
            q.setAlignment(Qt.AlignCenter)
            self.quote_labels[symbol] = q
            quote_row.addWidget(q)
        market_l.addLayout(quote_row)
        hero.addWidget(market_panel, 0, 0)

        ai_panel, ai_l = _panel("AI Decision", "Selected symbol · latest production model output")
        ai_head = QHBoxLayout(); ai_head.setSpacing(6)
        self.ai_symbol = QLabel("EURUSD"); self.ai_symbol.setObjectName("SectionTitle")
        self.ai_direction = QLabel("WAIT"); self.ai_direction.setObjectName("SignalHold")
        ai_head.addWidget(self.ai_symbol); ai_head.addStretch(); ai_head.addWidget(self.ai_direction)
        ai_l.addLayout(ai_head)

        ai_stats = QGridLayout(); ai_stats.setSpacing(5)
        self.ai_conf = CompactStat("CONFIDENCE", "—", "production threshold", CYAN)
        self.ai_edge = CompactStat("MODEL EDGE", "—", "probability edge", BLUE)
        ai_stats.addWidget(self.ai_conf, 0, 0); ai_stats.addWidget(self.ai_edge, 0, 1)
        ai_l.addLayout(ai_stats)
        self.pipeline = DecisionPipeline(); ai_l.addWidget(self.pipeline, 1)
        self.ai_reason = QLabel("Waiting for model telemetry")
        self.ai_reason.setObjectName("MetricDetail")
        self.ai_reason.setWordWrap(True)
        ai_l.addWidget(self.ai_reason)
        hero.addWidget(ai_panel, 0, 1)

        portfolio_panel, portfolio_l = _panel("Portfolio / Risk", "DEMO_FORWARD capital source of truth")
        stat_grid = QGridLayout(); stat_grid.setSpacing(5)
        self.forward_stat = CompactStat("FORWARD CAPITAL", "$0.00", "shadow account", GREEN)
        self.pnl_stat = CompactStat("NET P/L", "$0.00", "closed trades", GREEN)
        self.dd_stat = CompactStat("MAX DRAWDOWN", "0.00%", "forward ledger", AMBER)
        risk_pct = float(self.risk_policy.get("risk_percent", 0.0) or 0.0)
        self.risk_stat = CompactStat("RISK / TRADE", f"{risk_pct:.2f}%", "Stage 5 calibrated", CYAN)
        stat_grid.addWidget(self.forward_stat, 0, 0); stat_grid.addWidget(self.pnl_stat, 0, 1)
        stat_grid.addWidget(self.dd_stat, 1, 0); stat_grid.addWidget(self.risk_stat, 1, 1)
        portfolio_l.addLayout(stat_grid)
        self.equity_chart = AreaLineChart(); self.equity_chart.setMinimumHeight(112)
        portfolio_l.addWidget(self.equity_chart, 1)
        portfolio_footer = QHBoxLayout()
        self.broker_value = QLabel("BROKER —"); self.broker_value.setObjectName("MetricDetail")
        self.position_value = QLabel("0 OPEN"); self.position_value.setObjectName("StatusNeutral")
        portfolio_footer.addWidget(self.broker_value); portfolio_footer.addStretch(); portfolio_footer.addWidget(self.position_value)
        portfolio_l.addLayout(portfolio_footer)
        hero.addWidget(portfolio_panel, 0, 2)
        root.addLayout(hero)

        # Bottom operational strip: positions | all-symbol AI | runtime.
        lower = QGridLayout()
        lower.setSpacing(7)
        lower.setColumnStretch(0, 38)
        lower.setColumnStretch(1, 31)
        lower.setColumnStretch(2, 31)

        pos_panel, pos_l = _panel("Open Positions", "Live broker positions")
        self.positions_table = _table(["SYMBOL", "SIDE", "VOL", "ENTRY", "CURRENT", "P/L"])
        self.positions_table.setMinimumHeight(160); self.positions_table.setMaximumHeight(190)
        pos_l.addWidget(self.positions_table)
        lower.addWidget(pos_panel, 0, 0)

        signals_panel, signals_l = _panel("AI Symbol Matrix", "Latest model output against production gates")
        self.signals_table = _table(["SYMBOL", "MODEL", "CONF.", "EDGE", "GATE"])
        self.signals_table.setMinimumHeight(160); self.signals_table.setMaximumHeight(190)
        signals_l.addWidget(self.signals_table)
        lower.addWidget(signals_panel, 0, 1)

        activity_panel, activity_l = _panel("Runtime Stream", "Newest engine events")
        self.activity = QPlainTextEdit(); self.activity.setReadOnly(True); self.activity.setMaximumBlockCount(160)
        self.activity.setMinimumHeight(126); self.activity.setMaximumHeight(154)
        activity_l.addWidget(self.activity)
        lower.addWidget(activity_panel, 0, 2)
        root.addLayout(lower)
        root.addStretch()

        self.chart_symbol.currentTextChanged.connect(self._symbol_changed)
        self.update_engine_state(self.engine_state)

    def _notify_result(self, ok: bool, message: str) -> None:
        if not ok:
            ElvaraAlertDialog.show_alert(self, "TradeAI", message, semantic="danger")

    def _start_engine(self):
        ok, message = self.control.start_engine()
        self._notify_result(ok, message)

    def _send_command(self, command: str):
        ok, message = self.control.command(command)
        self._notify_result(ok, message)

    def _stop_engine(self):
        count = self.control.open_position_count()
        if count > 0:
            confirmed = ElvaraConfirmDialog.confirm(
                self,
                "Stop TradeAI engine?",
                f"{count} MT5 position(s) are still open. Stopping the engine does not close them. "
                "TradeAI strategy processing will stop while broker positions remain open.",
                confirm_text="Stop Engine",
                danger=True,
            )
            if not confirmed:
                return
        ok, message = self.control.stop()
        self._notify_result(ok, message)

    def update_engine_state(self, state: EngineState) -> None:
        self.engine_state = state
        active = state.online and state.state not in {"STARTING", "STOPPING"}
        preflight_ok, _ = self.control.preflight()
        self.start_btn.setEnabled(not state.online and preflight_ok)
        self.pause_btn.setEnabled(active and not state.paused)
        self.resume_btn.setEnabled(active and state.paused)
        self.stop_btn.setEnabled(state.online and state.state != "STOPPING")
        self._refresh_ai_panel()

    def _symbol_changed(self, symbol: str) -> None:
        self.market_pair.setText(f"{symbol} / M5")
        self.ai_symbol.setText(symbol)
        self._refresh_ai_panel()

    def _selected_policy(self) -> dict:
        symbols = self.policy.get("symbols", {}) if isinstance(self.policy.get("symbols"), dict) else {}
        cfg = symbols.get(self.chart_symbol.currentText(), {})
        return cfg if isinstance(cfg, dict) else {}

    def _refresh_ai_panel(self, states: dict | None = None) -> None:
        symbol = self.chart_symbol.currentText()
        states = states or self.service.latest_signal_states()
        st = states.get(symbol, {}) if isinstance(states, dict) else {}
        cfg = self._selected_policy()
        signal = st.get("signal")
        confidence = st.get("confidence")
        min_conf = float(cfg.get("min_confidence", 0.0) or 0.0)
        min_edge = float(cfg.get("signal_threshold", 0.0) or 0.0)
        enabled = bool(cfg.get("enabled", True))
        risk_pct = float(self.risk_policy.get("risk_percent", 0.0) or 0.0)

        if signal is None or confidence is None:
            direction = "WAIT"
            self.ai_conf.set_value("—", f"gate {min_conf*100:.0f}%")
            self.ai_edge.set_value("—", f"gate {min_edge:.3f}")
            self.ai_reason.setText("Waiting for a model output from the current runtime stream")
        else:
            direction = "HOLD"
            if abs(float(signal)) >= min_edge:
                direction = "BUY" if float(signal) > 0 else "SELL"
            conf_ok = float(confidence) >= min_conf
            edge_ok = abs(float(signal)) >= min_edge
            self.ai_conf.set_value(f"{float(confidence)*100:.1f}%", f"gate {min_conf*100:.0f}%", GREEN if conf_ok else RED)
            self.ai_edge.set_value(f"{float(signal):+.3f}", f"gate {min_edge:.3f}", GREEN if edge_ok else RED)
            if direction == "HOLD":
                self.ai_reason.setText("No executable directional edge")
            elif not conf_ok:
                self.ai_reason.setText("Blocked by confidence gate")
            elif not edge_ok:
                self.ai_reason.setText("Blocked by edge gate")
            elif not self.engine_state.online:
                self.ai_reason.setText("Model gate passed; engine is offline")
            else:
                self.ai_reason.setText("Model gates passed; risk and execution constraints remain authoritative")

        self.ai_direction.setText(direction)
        self.ai_direction.setObjectName("SignalBuy" if direction == "BUY" else "SignalSell" if direction == "SELL" else "SignalHold")
        self.ai_direction.style().unpolish(self.ai_direction); self.ai_direction.style().polish(self.ai_direction)
        self.pipeline.set_trade_state(
            None if signal is None else float(signal),
            None if confidence is None else float(confidence),
            min_conf,
            min_edge,
            risk_pct,
            self.engine_state.online and not self.engine_state.paused,
            enabled,
        )

    def update_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._snapshot = snapshot
        cur = snapshot.currency or "USD"
        pnl_color = GREEN if snapshot.net_profit >= 0 else RED
        self.forward_stat.set_value(f"${snapshot.virtual_balance:,.2f}", f"start ${snapshot.initial_virtual_balance:,.2f}", GREEN if snapshot.virtual_balance >= snapshot.initial_virtual_balance else RED)
        self.pnl_stat.set_value(f"${snapshot.net_profit:+,.2f}", f"{snapshot.closed_trades} closed", pnl_color)
        self.dd_stat.set_value(f"{snapshot.drawdown_percent:.2f}%", "forward ledger", GREEN if snapshot.drawdown_percent < 6 else AMBER)
        self.broker_value.setText(f"BROKER  {snapshot.broker_equity:,.2f} {cur}")
        self.position_value.setText(f"{snapshot.open_positions} OPEN")
        self.equity_chart.set_values(snapshot.forward_equity_curve)

        if snapshot.candles:
            self.candle_chart.set_candles(snapshot.candles)
            self.candle_chart.set_live_price(snapshot.chart_bid)
            _, op, hi, lo, close = snapshot.candles[-1]
            self.market_ohlc.setText(f"O {op:.5f}   H {hi:.5f}   L {lo:.5f}   C {close:.5f}")
            shown_price = snapshot.chart_bid if snapshot.chart_bid is not None else close
            spread = "—" if snapshot.chart_spread_pips is None else f"{snapshot.chart_spread_pips:.1f}p"
            self.market_price.setText(f"LIVE  {shown_price:.5f}  ·  {spread}")
            self.market_price.setObjectName("StatusOnline")
            self.market_price.style().unpolish(self.market_price); self.market_price.style().polish(self.market_price)

        for quote in snapshot.quotes:
            label = self.quote_labels.get(quote.symbol)
            if label is None:
                continue
            digits = 3 if "JPY" in quote.symbol.upper() else 5
            price = "—" if quote.bid is None else f"{quote.bid:.{digits}f}"
            spread = "—" if quote.spread_pips is None else f"{quote.spread_pips:.1f}p"
            label.setText(f"{quote.symbol}  {price}  ·  {spread}")

        self.positions_table.setRowCount(len(snapshot.positions))
        for row, pos in enumerate(snapshot.positions):
            digits = 3 if "JPY" in pos.symbol.upper() else 5
            vals = [
                pos.symbol,
                pos.side,
                f"{pos.volume:.3f}".rstrip("0").rstrip("."),
                f"{pos.entry:.{digits}f}",
                f"{pos.current:.{digits}f}",
                f"{pos.profit:+.2f}",
            ]
            for col, value in enumerate(vals):
                item = QTableWidgetItem(value)
                if col == 0: item.setForeground(QColor(CYAN))
                if col == 1: item.setForeground(QColor(GREEN if pos.side == "BUY" else RED))
                if col == 5: item.setForeground(QColor(GREEN if pos.profit >= 0 else RED))
                self.positions_table.setItem(row, col, item)

        states = self.service.latest_signal_states()
        symbols_cfg = self.policy.get("symbols", {}) if isinstance(self.policy.get("symbols"), dict) else {}
        self.signals_table.setRowCount(len(self.service.symbols))
        for row, symbol in enumerate(self.service.symbols):
            st = states.get(symbol, {})
            cfg = symbols_cfg.get(symbol, {}) if isinstance(symbols_cfg, dict) else {}
            signal = st.get("signal"); conf = st.get("confidence")
            min_conf = float(cfg.get("min_confidence", 0.0) or 0.0)
            min_edge = float(cfg.get("signal_threshold", 0.0) or 0.0)
            direction = "WAIT"
            gate = "WAIT"
            if signal is not None and conf is not None:
                if abs(float(signal)) >= min_edge:
                    direction = "BUY" if float(signal) > 0 else "SELL"
                passed = float(conf) >= min_conf and abs(float(signal)) >= min_edge and bool(cfg.get("enabled", True))
                gate = "PASS" if passed else "WAIT"
            vals = [
                symbol,
                direction,
                "—" if conf is None else f"{float(conf)*100:.1f}%",
                "—" if signal is None else f"{float(signal):+.3f}",
                gate,
            ]
            for col, value in enumerate(vals):
                item = QTableWidgetItem(value)
                if col == 0: item.setForeground(QColor(CYAN))
                if col == 1: item.setForeground(QColor(GREEN if value == "BUY" else RED if value == "SELL" else AMBER))
                if col == 4: item.setForeground(QColor(GREEN if value == "PASS" else AMBER))
                self.signals_table.setItem(row, col, item)
        self._refresh_ai_panel(states)

        lines = self.service.tail_log(8)
        self.activity.setPlainText("\n".join(lines))
        self.activity.verticalScrollBar().setValue(self.activity.verticalScrollBar().maximum())

    def refresh(self):
        # Policy files can be regenerated while the UI is open.
        self.policy = self.service.load_decision_policy()
        self.risk_policy = self.service.load_risk_policy()
        risk_pct = float(self.risk_policy.get("risk_percent", 0.0) or 0.0)
        self.risk_stat.set_value(f"{risk_pct:.2f}%", "Stage 5 calibrated")
        self._refresh_ai_panel()




class ControlPage(QWidget):
    def __init__(self, service: TradeAIDataService, control: EngineControlService, parent=None):
        super().__init__(parent)
        self.service = service
        self.control = control
        self._last_state = EngineState(mode=self.control.mode())
        self._snapshot: RuntimeSnapshot | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, _, root = _scroll_shell()
        outer.addWidget(scroll)

        eye = QLabel("ENGINE / CONTROL PLANE")
        eye.setObjectName("Eyebrow")
        title = QLabel("System Control")
        title.setObjectName("PageTitle")
        root.addWidget(eye)
        root.addWidget(title)

        top = QGridLayout()
        top.setSpacing(10)
        self.engine_card = MetricCard("ENGINE", "OFFLINE", "No heartbeat", CYAN, "activity")
        self.mode_card = MetricCard("MODE", self.control.mode(), "configured", BLUE, "control")
        self.pid_card = MetricCard("PID", "—", "no process", BLUE, "terminal")
        self.uptime_card = MetricCard("UPTIME", "—", "session", CYAN, "activity")
        self.heartbeat_card = MetricCard("HEARTBEAT", "OFFLINE", "control link", AMBER, "signals")
        self.cycle_card = MetricCard("LAST CYCLE", "—", "strategy loop", GREEN, "activity")
        for i, card in enumerate((self.engine_card, self.mode_card, self.pid_card, self.uptime_card, self.heartbeat_card, self.cycle_card)):
            top.addWidget(card, i // 3, i % 3)
        root.addLayout(top)

        deck = QFrame()
        deck.setObjectName("ControlDeck")
        dl = QVBoxLayout(deck)
        dl.setContentsMargins(15, 13, 15, 13)
        dl.setSpacing(9)
        dl.addWidget(SectionTitle("Engine Commands", "Process control only — existing broker positions are never closed by these buttons"))

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.start_btn = ElvaraButton("START AI", variant="primary", size="standard")
        self.pause_btn = ElvaraButton("PAUSE", variant="secondary", size="standard")
        self.resume_btn = ElvaraButton("RESUME", variant="primary", size="standard")
        self.report_btn = ElvaraButton("REPORT", variant="ghost", size="standard")
        self.stop_btn = ElvaraButton("STOP", variant="danger", size="standard")
        for b in (self.start_btn, self.pause_btn, self.resume_btn, self.report_btn, self.stop_btn):
            buttons.addWidget(b)
        buttons.addStretch()
        dl.addLayout(buttons)

        self.command_note = ElvaraNotice("Control", "Ready.", "info")
        dl.addWidget(self.command_note)
        root.addWidget(deck)

        mid = QGridLayout()
        mid.setSpacing(10)
        mid.setColumnStretch(0, 5)
        mid.setColumnStretch(1, 5)

        telemetry_panel, telemetry_l = _panel("Runtime Telemetry", "State published by the running engine")
        self.telemetry_table = _table(["CHANNEL", "VALUE", "STATE"])
        self.telemetry_table.setMinimumHeight(225)
        telemetry_l.addWidget(self.telemetry_table)
        mid.addWidget(telemetry_panel, 0, 0)

        health_panel, health_l = _panel("Subsystem Health", "Fast local diagnostics — read only")
        self.health_table = _table(["SUBSYSTEM", "STATE", "DETAIL"])
        self.health_table.setMinimumHeight(225)
        health_l.addWidget(self.health_table)
        mid.addWidget(health_panel, 0, 1)
        root.addLayout(mid)

        guards = QGridLayout()
        guards.setSpacing(10)
        guards.setColumnStretch(0, 6)
        guards.setColumnStretch(1, 4)

        guard_panel, guard_l = _panel("Launch Interlocks", "Required before desktop start")
        self.mode_guard = QLabel()
        self.model_guard = QLabel()
        self.account_guard = QLabel()
        self.start_guard = QLabel()
        for label in (self.mode_guard, self.model_guard, self.account_guard, self.start_guard):
            label.setMinimumHeight(25)
            guard_l.addWidget(label)
        guards.addWidget(guard_panel, 0, 0)

        state_panel, state_l = _panel("Session", "Control-plane identity and last command")
        self.plane_state = QLabel("STATE  OFFLINE")
        self.plane_state.setObjectName("EngineOffline")
        self.started = QLabel("STARTED  —")
        self.started.setObjectName("PageSub")
        self.last_command = QLabel("LAST COMMAND  —")
        self.last_command.setObjectName("PageSub")
        self.session_note = QLabel("—")
        self.session_note.setObjectName("MetricDetail")
        self.session_note.setWordWrap(True)
        state_l.addWidget(self.plane_state)
        state_l.addWidget(self.started)
        state_l.addWidget(self.last_command)
        state_l.addWidget(self.session_note)
        state_l.addStretch()
        guards.addWidget(state_panel, 0, 1)
        root.addLayout(guards)

        self.error_notice = ElvaraNotice("Latest engine error", "No runtime error reported.", "danger")
        self.error_notice.hide()
        root.addWidget(self.error_notice)
        root.addStretch()

        self.start_btn.clicked.connect(self.start_engine)
        self.pause_btn.clicked.connect(lambda: self.send_command("pause"))
        self.resume_btn.clicked.connect(lambda: self.send_command("resume"))
        self.report_btn.clicked.connect(lambda: self.send_command("report"))
        self.stop_btn.clicked.connect(self.stop_engine)
        self.refresh()

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None:
            return "—"
        total = max(0, int(seconds))
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    def _set_guard(self, label: QLabel, title: str, passed: bool, detail: str) -> None:
        label.setObjectName("InterlockPass" if passed else "InterlockFail")
        label.setText(f"{'●' if passed else '×'}  {title}  ·  {detail}")
        label.style().unpolish(label)
        label.style().polish(label)

    def start_engine(self) -> None:
        ok, message = self.control.start_engine()
        self.command_note.set_message(message, semantic="success" if ok else "danger")
        if not ok:
            ElvaraAlertDialog.show_alert(self, "TradeAI Control", message, semantic="danger")
        self.refresh()

    def send_command(self, command: str) -> None:
        ok, message = self.control.command(command)
        self.command_note.set_message(message, semantic="success" if ok else "danger")
        if not ok:
            ElvaraAlertDialog.show_alert(self, "TradeAI Control", message, semantic="danger")
        self.refresh()

    def stop_engine(self) -> None:
        positions = self.control.open_position_count()
        prefix = f"{positions} MT5 position{'s' if positions != 1 else ''} remain open. " if positions else ""
        if ElvaraConfirmDialog.confirm(
            self,
            "Stop TradeAI engine?",
            prefix + "Stopping the process stops strategy management; it does not close broker positions.",
            confirm_text="Stop Engine",
            danger=True,
        ):
            ok, message = self.control.stop()
            self.command_note.set_message(message, semantic="success" if ok else "danger")
            if not ok:
                ElvaraAlertDialog.show_alert(self, "TradeAI Control", message, semantic="danger")
            self.refresh()

    def update_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._snapshot = snapshot
        # Keep health feedback responsive to the same MT5 snapshot used by Command Center.
        if hasattr(self, "health_table"):
            self._refresh_health()

    def _refresh_telemetry(self, state: EngineState) -> None:
        details = state.details or {}
        rows = [
            ("DATA FEED", str(details.get("data_feed", "—")), "PASS" if details.get("data_feed") else "WAIT"),
            ("MODEL", "READY" if details.get("model_ready") else "—", "PASS" if details.get("model_ready") else "WAIT"),
            ("MT5", "CONNECTED" if details.get("mt5_connected") else "OFFLINE", "PASS" if details.get("mt5_connected") else "WAIT"),
            ("LAST SYMBOL", str(details.get("last_symbol") or "—"), "LIVE" if details.get("last_symbol") else "WAIT"),
            ("CANDLE", str(details.get("current_candle") or details.get("replay_time") or "—"), "LIVE" if (details.get("current_candle") or details.get("replay_time")) else "WAIT"),
            ("LOOP TIME", f"{float(details.get('loop_ms', 0.0) or 0.0):.0f} ms" if details.get("loop_ms") is not None else "—", "LIVE" if state.online else "WAIT"),
        ]
        self.telemetry_table.setRowCount(len(rows))
        for row, (name, value, status) in enumerate(rows):
            for col, text in enumerate((name, value, status)):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setForeground(QColor(CYAN))
                elif col == 2:
                    item.setForeground(QColor(GREEN if status in {"PASS", "LIVE"} else AMBER))
                self.telemetry_table.setItem(row, col, item)

    def _refresh_health(self) -> None:
        checks = self.service.system_health()
        # The live snapshot is fresher than a second MT5 initialization attempt.
        if self._snapshot is not None:
            for check in checks:
                if check.get("name") == "MT5":
                    check["ok"] = bool(self._snapshot.mt5_connected)
                    check["detail"] = self._snapshot.terminal_name if self._snapshot.mt5_connected else "Terminal unavailable"
        self.health_table.setRowCount(len(checks))
        for row, check in enumerate(checks):
            ok = bool(check.get("ok"))
            vals = [str(check.get("name", "")), "PASS" if ok else "CHECK", str(check.get("detail", ""))]
            for col, value in enumerate(vals):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setForeground(QColor(CYAN))
                if col == 1:
                    item.setForeground(QColor(GREEN if ok else AMBER))
                self.health_table.setItem(row, col, item)

    def refresh(self) -> None:
        state = self.control.status()
        self._last_state = state
        accent = GREEN if state.online and not state.paused else AMBER if state.online else CYAN
        self.engine_card.set_value(state.state, state.note or ("heartbeat active" if state.online else "not running"), accent)
        self.mode_card.set_value(state.mode or self.control.mode(), "runtime mode" if state.online else "configured mode")
        self.pid_card.set_value(str(state.pid) if state.pid else "—", "active process" if state.online else "no process")
        self.uptime_card.set_value(self._duration(state.uptime_seconds), "process uptime")
        age_text = f"{state.heartbeat_age:.1f}s" if state.heartbeat_age is not None else "—"
        self.heartbeat_card.set_value(age_text, "heartbeat age", GREEN if state.online else AMBER)
        cycle = str((state.details or {}).get("last_cycle_utc") or "—")
        self.cycle_card.set_value(cycle[-12:-1] if cycle not in {"", "—"} else "—", "UTC", GREEN if state.online else CYAN)

        if state.online:
            self.plane_state.setText(f"STATE  {state.state}")
            self.plane_state.setObjectName("EnginePaused" if state.paused else "EngineRunning")
        elif state.state == "ERROR":
            self.plane_state.setText("STATE  ERROR")
            self.plane_state.setObjectName("EngineOffline")
        else:
            self.plane_state.setText(f"STATE  {state.state or 'OFFLINE'}")
            self.plane_state.setObjectName("EngineOffline")
        self.plane_state.style().unpolish(self.plane_state)
        self.plane_state.style().polish(self.plane_state)
        self.started.setText(f"STARTED  {state.started_at or '—'}")
        self.last_command.setText(f"LAST COMMAND  {state.last_command.upper() if state.last_command else '—'}")
        self.session_note.setText(state.note or "No engine note")

        mode = self.control.mode()
        mode_ok = mode in self.control.UI_STARTABLE_MODES
        self._set_guard(self.mode_guard, "MODE", mode_ok, f"{mode} {'allowed' if mode_ok else 'blocked'}")
        model_ok = self.control.model_accepted() if mode == "DEMO_FORWARD" else True
        self._set_guard(self.model_guard, "MODEL", model_ok, "Stage 5 accepted" if model_ok else "acceptance missing")
        account_ok, account_text = self.control.demo_account_check() if mode == "DEMO_FORWARD" else (True, "not required")
        self._set_guard(self.account_guard, "ACCOUNT", account_ok, account_text)
        preflight_ok, preflight_text = self.control.preflight()
        self._set_guard(self.start_guard, "START", preflight_ok, preflight_text)

        active = state.online and state.state not in {"STARTING", "STOPPING"}
        self.start_btn.setEnabled(not state.online and preflight_ok)
        self.pause_btn.setEnabled(active and not state.paused)
        self.resume_btn.setEnabled(active and state.paused)
        self.report_btn.setEnabled(active)
        self.stop_btn.setEnabled(state.online and state.state != "STOPPING")

        self._refresh_telemetry(state)
        self._refresh_health()

        last_error = str((state.details or {}).get("last_error") or "")
        errors = self.service.latest_errors(1)
        error_text = last_error or (errors[-1] if errors else "")
        if error_text:
            self.error_notice.set_message(error_text, semantic="danger")
            self.error_notice.show()
        else:
            self.error_notice.hide()


class SignalsPage(QWidget):
    def __init__(self, service: TradeAIDataService, control: EngineControlService, parent=None):
        super().__init__(parent)
        self.service = service
        self.control = control
        self.policy: dict = {}
        self.risk_policy: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, _, root = _scroll_shell()
        outer.addWidget(scroll)

        eye = QLabel("NIRA / MODEL DECISION TRACE")
        eye.setObjectName("Eyebrow")
        title = QLabel("Signal Intelligence")
        title.setObjectName("PageTitle")
        root.addWidget(eye)
        root.addWidget(title)

        cards = QGridLayout()
        cards.setSpacing(10)
        self.signal_cards: dict[str, SignalCard] = {}
        for i, symbol in enumerate(self.service.symbols):
            card = SignalCard(symbol)
            self.signal_cards[symbol] = card
            cards.addWidget(card, 0, i)
        root.addLayout(cards)

        inspector_panel, inspector_l = _panel("Decision Inspector", "Real model telemetry · prediction is never treated as an order")
        selector = QHBoxLayout()
        selector.addWidget(QLabel("SYMBOL"))
        self.symbol = QComboBox()
        self.symbol.addItems(self.service.symbols)
        self.symbol.setMaximumWidth(150)
        self.symbol.currentTextChanged.connect(lambda _text: self.refresh())
        selector.addWidget(self.symbol)
        selector.addStretch()
        self.decision_badge = ElvaraStatusBadge("WAIT", "neutral")
        selector.addWidget(self.decision_badge)
        inspector_l.addLayout(selector)

        probability_grid = QGridLayout()
        probability_grid.setSpacing(8)
        self.buy_prob = CompactStat("P(BUY)", "—", "model probability", GREEN)
        self.hold_prob = CompactStat("P(HOLD)", "—", "model probability", AMBER)
        self.sell_prob = CompactStat("P(SELL)", "—", "model probability", RED)
        self.conf_stat = CompactStat("CONFIDENCE", "—", "symbol gate", CYAN)
        self.edge_stat = CompactStat("EDGE", "—", "signal gate", BLUE)
        for i, widget in enumerate((self.buy_prob, self.hold_prob, self.sell_prob, self.conf_stat, self.edge_stat)):
            probability_grid.addWidget(widget, 0, i)
        inspector_l.addLayout(probability_grid)

        self.pipeline = DecisionPipeline()
        inspector_l.addWidget(self.pipeline)
        self.reason = ElvaraNotice("Decision", "Waiting for model telemetry.", "info")
        inspector_l.addWidget(self.reason)

        self.gate_table = _table(["GATE", "OBSERVED", "REQUIRED", "RESULT"])
        self.gate_table.setMinimumHeight(220)
        inspector_l.addWidget(self.gate_table)
        root.addWidget(inspector_panel)

        policy_panel, policy_l = _panel("Production Policy", "Promoted Stage 5 symbol thresholds")
        self.table = _table(["SYMBOL", "STATE", "MIN CONF.", "EDGE GATE", "CAL. WIN RATE", "PF (R)"])
        self.table.setMinimumHeight(190)
        policy_l.addWidget(self.table)
        root.addWidget(policy_panel)
        root.addStretch()
        self.refresh()

    @staticmethod
    def _fmt_prob(value) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except Exception:
            return "—"

    def refresh(self) -> None:
        self.policy = self.service.load_decision_policy()
        self.risk_policy = self.service.load_risk_policy()
        symbols_cfg = self.policy.get("symbols", {}) if isinstance(self.policy.get("symbols"), dict) else {}
        states = self.service.latest_signal_states()

        for symbol, card in self.signal_cards.items():
            cfg = symbols_cfg.get(symbol, {}) if isinstance(symbols_cfg, dict) else {}
            st = states.get(symbol, {})
            card.update_state(
                st.get("signal"), st.get("confidence"),
                float(cfg.get("min_confidence", 0) or 0),
                float(cfg.get("signal_threshold", 0) or 0),
                bool(cfg.get("enabled", True)),
            )

        symbol = self.symbol.currentText() or (self.service.symbols[0] if self.service.symbols else "")
        st = states.get(symbol, {})
        cfg = symbols_cfg.get(symbol, {}) if isinstance(symbols_cfg, dict) else {}
        signal = st.get("signal")
        confidence = st.get("confidence")
        min_conf = float(st.get("min_confidence", cfg.get("min_confidence", 0.0)) or 0.0)
        min_edge = float(st.get("signal_threshold", cfg.get("signal_threshold", 0.0)) or 0.0)
        enabled = bool(st.get("enabled", cfg.get("enabled", True)))
        risk_pct = float(self.risk_policy.get("risk_percent", 0.0) or 0.0)
        engine = self.control.status()

        self.buy_prob.set_value(self._fmt_prob(st.get("p_buy")))
        self.hold_prob.set_value(self._fmt_prob(st.get("p_hold")))
        self.sell_prob.set_value(self._fmt_prob(st.get("p_sell")))

        model_ok = signal is not None and confidence is not None
        conf_ok = model_ok and float(confidence) >= min_conf
        edge_ok = model_ok and abs(float(signal)) >= min_edge
        policy_ok = bool(enabled)
        risk_ok = risk_pct > 0
        execution_ready = engine.online and not engine.paused and engine.state == "RUNNING"

        direction = "WAIT"
        if model_ok:
            if abs(float(signal)) >= min_edge:
                direction = "BUY" if float(signal) > 0 else "SELL"
            else:
                direction = "HOLD"

        self.conf_stat.set_value(
            "—" if confidence is None else f"{float(confidence) * 100:.1f}%",
            f"gate {min_conf * 100:.1f}%",
            GREEN if conf_ok else RED if model_ok else CYAN,
        )
        self.edge_stat.set_value(
            "—" if signal is None else f"{float(signal):+.3f}",
            f"gate ±{min_edge:.3f}",
            GREEN if edge_ok else RED if model_ok else BLUE,
        )

        all_model_gates = model_ok and conf_ok and edge_ok and policy_ok and risk_ok
        if direction in {"BUY", "SELL"} and all_model_gates and execution_ready:
            semantic = "success"
            final_text = f"{direction} · READY"
            message = "Model, confidence, edge, policy and risk gates pass. Execution remains subject to live broker/risk-manager constraints."
        elif direction in {"BUY", "SELL"} and all_model_gates:
            semantic = "warning"
            final_text = f"{direction} · STANDBY"
            message = "Model gates pass, but the engine is not currently ready to execute."
        elif not model_ok:
            semantic = "neutral"
            final_text = "WAIT"
            message = "Waiting for the next structured MODEL_STATE record."
        elif not policy_ok:
            semantic = "danger"
            final_text = "BLOCKED"
            message = "Symbol is disabled by the promoted production policy."
        elif not conf_ok:
            semantic = "warning"
            final_text = "BLOCKED"
            message = "Confidence is below the symbol-specific production threshold."
        elif not edge_ok:
            semantic = "warning"
            final_text = "HOLD"
            message = "Directional edge is below the production gate."
        elif not risk_ok:
            semantic = "danger"
            final_text = "BLOCKED"
            message = "No valid calibrated risk policy is available."
        else:
            semantic = "neutral"
            final_text = direction
            message = "No executable decision at this time."

        self.decision_badge.set_semantic(semantic, final_text)
        self.reason.set_message(message, semantic=semantic if semantic in {"success", "warning", "danger"} else "info")
        self.pipeline.set_trade_state(
            None if signal is None else float(signal),
            None if confidence is None else float(confidence),
            min_conf, min_edge, risk_pct,
            execution_ready, enabled,
        )

        gate_rows = [
            ("MODEL", direction if model_ok else "—", "valid output", model_ok),
            ("CONFIDENCE", "—" if confidence is None else f"{float(confidence) * 100:.1f}%", f"≥ {min_conf * 100:.1f}%", bool(conf_ok)),
            ("EDGE", "—" if signal is None else f"{abs(float(signal)):.3f}", f"≥ {min_edge:.3f}", bool(edge_ok)),
            ("POLICY", "ENABLED" if enabled else "DISABLED", "enabled", policy_ok),
            ("RISK", f"{risk_pct:.2f}%", "calibrated", risk_ok),
            ("EXECUTION", engine.state, "RUNNING", execution_ready),
        ]
        self.gate_table.setRowCount(len(gate_rows))
        for row, (gate, observed, required, passed) in enumerate(gate_rows):
            result = "PASS" if passed else "WAIT"
            for col, text in enumerate((gate, observed, required, result)):
                item = QTableWidgetItem(str(text))
                if col == 0:
                    item.setForeground(QColor(CYAN))
                if col == 3:
                    item.setForeground(QColor(GREEN if passed else AMBER))
                self.gate_table.setItem(row, col, item)

        self.table.setRowCount(len(symbols_cfg))
        for row, (sym, policy_cfg) in enumerate(symbols_cfg.items()):
            cal = policy_cfg.get("calibration", {}) if isinstance(policy_cfg, dict) else {}
            vals = [
                sym,
                "ENABLED" if policy_cfg.get("enabled", False) else "DISABLED",
                f"{float(policy_cfg.get('min_confidence', 0)) * 100:.1f}%",
                f"{float(policy_cfg.get('signal_threshold', 0)):.3f}",
                f"{float(cal.get('win_rate', 0)) * 100:.1f}%",
                f"{float(cal.get('profit_factor_r', 0)):.2f}",
            ]
            for col, value in enumerate(vals):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setForeground(QColor(CYAN))
                if col == 1:
                    item.setForeground(QColor(GREEN if policy_cfg.get("enabled", False) else RED))
                self.table.setItem(row, col, item)


class BacktestPage(QWidget):
    def __init__(
        self,
        service: TradeAIDataService,
        control: EngineControlService,
        config: TradeAIConfigService,
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.control = control
        self.config = config

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, _, root = _scroll_shell()
        outer.addWidget(scroll)

        eye = QLabel("REPLAY / PRODUCTION MODEL")
        eye.setObjectName("Eyebrow")
        title = QLabel("Backtest Lab")
        title.setObjectName("PageTitle")
        root.addWidget(eye)
        root.addWidget(title)

        runner, runner_l = _panel("Run", "Real ReplayEngine + CoreEngine + production model + risk/execution rules")
        controls = QGridLayout()
        controls.setSpacing(8)

        self.start_date = QLineEdit()
        self.end_date = QLineEdit()
        self.capital = QLineEdit()
        self.symbol_scope = QComboBox()
        self.symbol_scope.addItem("ALL SYMBOLS")
        self.symbol_scope.addItems(self.service.symbols)
        for edit, placeholder, width in (
            (self.start_date, "YYYY-MM-DD", 145),
            (self.end_date, "YYYY-MM-DD", 145),
            (self.capital, "10.00", 105),
        ):
            edit.setPlaceholderText(placeholder)
            edit.setMaximumWidth(width)
        self.symbol_scope.setMaximumWidth(145)

        controls.addWidget(QLabel("START"), 0, 0)
        controls.addWidget(self.start_date, 0, 1)
        controls.addWidget(QLabel("END"), 0, 2)
        controls.addWidget(self.end_date, 0, 3)
        controls.addWidget(QLabel("CAPITAL"), 0, 4)
        controls.addWidget(self.capital, 0, 5)
        controls.addWidget(QLabel("SCOPE"), 0, 6)
        controls.addWidget(self.symbol_scope, 0, 7)

        self.run_btn = ElvaraButton("RUN", variant="primary", size="standard", icon_name="backtest")
        self.pause_btn = ElvaraButton("PAUSE", variant="secondary", size="standard")
        self.resume_btn = ElvaraButton("RESUME", variant="primary", size="standard")
        self.stop_btn = ElvaraButton("STOP", variant="danger", size="standard")
        self.reload_btn = ElvaraButton("LOAD CONFIG", variant="ghost", size="standard", icon_name="refresh")
        self.run_btn.clicked.connect(self.run_backtest)
        self.pause_btn.clicked.connect(lambda: self.send_backtest_command("pause"))
        self.resume_btn.clicked.connect(lambda: self.send_backtest_command("resume"))
        self.stop_btn.clicked.connect(self.stop_backtest)
        self.reload_btn.clicked.connect(self.load_runner_config)
        for col, button in enumerate((self.run_btn, self.pause_btn, self.resume_btn, self.stop_btn, self.reload_btn), start=8):
            controls.addWidget(button, 0, col)
        controls.setColumnStretch(13, 1)
        runner_l.addLayout(controls)

        status_row = QHBoxLayout()
        self.run_status = ElvaraStatusBadge("IDLE", "neutral")
        self.run_detail = QLabel("Ready")
        self.run_detail.setObjectName("MetricDetail")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setMaximumWidth(320)
        status_row.addWidget(self.run_status)
        status_row.addWidget(self.run_detail, 1)
        status_row.addWidget(self.progress)
        runner_l.addLayout(status_row)
        root.addWidget(runner)

        cards = QGridLayout()
        cards.setSpacing(10)
        self.return_card = MetricCard("RETURN", icon_name="profit", accent=GREEN)
        self.end_card = MetricCard("ENDING BALANCE", icon_name="wallet", accent=CYAN)
        self.win_card = MetricCard("WIN RATE", icon_name="signals", accent=BLUE)
        self.pf_card = MetricCard("PROFIT FACTOR", icon_name="activity", accent=GREEN)
        self.dd_card = MetricCard("MAX DRAWDOWN", icon_name="risk", accent=AMBER)
        self.exp_card = MetricCard("EXPECTANCY", icon_name="activity", accent=CYAN)
        for i, card in enumerate((self.return_card, self.end_card, self.win_card, self.pf_card, self.dd_card, self.exp_card)):
            cards.addWidget(card, i // 3, i % 3)
        root.addLayout(cards)

        source_row = QHBoxLayout()
        self.summary_source = ElvaraStatusBadge("NO RUNTIME RESULT", "neutral")
        self.summary_note = QLabel("Stage 5 validation is used until a desktop run completes.")
        self.summary_note.setObjectName("MetricDetail")
        source_row.addWidget(self.summary_source)
        source_row.addWidget(self.summary_note, 1)
        root.addLayout(source_row)

        charts = QGridLayout()
        charts.setSpacing(10)
        charts.setColumnStretch(0, 5)
        charts.setColumnStretch(1, 3)
        charts.setColumnStretch(2, 2)

        eq_panel, eq_l = _panel("Equity", "Balance after each simulated close")
        self.chart = AreaLineChart()
        self.chart.setMinimumHeight(200)
        eq_l.addWidget(self.chart)
        charts.addWidget(eq_panel, 0, 0)

        dd_panel, dd_l = _panel("Drawdown", "Peak-to-trough percentage")
        self.dd_chart = AreaLineChart()
        self.dd_chart.setMinimumHeight(200)
        dd_l.addWidget(self.dd_chart)
        charts.addWidget(dd_panel, 0, 1)

        sym_panel, sym_l = _panel("Symbol Contribution", "Net P/L")
        self.bars = HorizontalBarChart()
        sym_l.addWidget(self.bars)
        charts.addWidget(sym_panel, 0, 2)
        root.addLayout(charts)

        analytics = QGridLayout()
        analytics.setSpacing(10)
        analytics.setColumnStretch(0, 4)
        analytics.setColumnStretch(1, 6)

        side_panel, side_l = _panel("Long / Short", "Direction-level result split")
        self.side_table = _table(["SIDE", "TRADES", "WINS", "WIN RATE", "NET P/L"])
        self.side_table.setMinimumHeight(145)
        side_l.addWidget(self.side_table)
        analytics.addWidget(side_panel, 0, 0)

        compare_panel, compare_l = _panel("Runtime vs Stage 5", "Latest desktop run compared with the promoted validation artifact")
        self.compare_table = _table(["METRIC", "RUNTIME", "STAGE 5", "DELTA"])
        self.compare_table.setMinimumHeight(145)
        compare_l.addWidget(self.compare_table)
        analytics.addWidget(compare_panel, 0, 1)
        root.addLayout(analytics)

        trade_panel, trade_l = _panel("Recent Trades", "Newest simulated closes")
        self.trade_table = _table(["TIME", "SYMBOL", "SIDE", "LOT", "P/L", "EXIT", "CONF."])
        self.trade_table.setMinimumHeight(200)
        trade_l.addWidget(self.trade_table)
        root.addWidget(trade_panel)

        history_panel, history_l = _panel("Run History", "Archived desktop backtests")
        history_head = QHBoxLayout()
        history_head.addStretch()
        open_history = ElvaraButton("OPEN FOLDER", variant="ghost", size="compact", icon_name="folder")
        open_history.clicked.connect(self.open_history_folder)
        history_head.addWidget(open_history)
        history_l.addLayout(history_head)
        self.history_table = _table(["GENERATED", "RANGE", "CAPITAL", "END", "RETURN", "TRADES", "DD"])
        self.history_table.setMinimumHeight(180)
        history_l.addWidget(self.history_table)
        root.addWidget(history_panel)
        root.addStretch()

        self.load_runner_config()
        self.refresh()

    def load_runner_config(self) -> None:
        try:
            cfg = self.config.read()
            self.start_date.setText(str(cfg.get("BACKTEST_START_DATE", "")))
            self.end_date.setText(str(cfg.get("BACKTEST_END_DATE", "")))
            self.capital.setText(str(cfg.get("START_BALANCE", "10.0")))
        except Exception as exc:
            ElvaraAlertDialog.show_alert(self, "Backtest", f"Could not read configuration: {exc}", semantic="danger")

    def run_backtest(self) -> None:
        try:
            start = self.start_date.text().strip()
            end = self.end_date.text().strip()
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            if start_dt > end_dt:
                raise ValueError("Start date cannot be after end date")
            capital = float(self.capital.text().strip())
            if capital <= 0:
                raise ValueError("Capital must be greater than zero")
        except Exception as exc:
            ElvaraAlertDialog.show_alert(self, "Backtest", str(exc), semantic="danger")
            return

        scope = self.symbol_scope.currentText()
        symbols = None if scope == "ALL SYMBOLS" else [scope]
        ok, message = self.control.start_backtest(start_date=start, end_date=end, capital=capital, symbols=symbols)
        if not ok:
            ElvaraAlertDialog.show_alert(self, "Backtest", message, semantic="danger")
        self.run_detail.setText(message)
        self.refresh()

    def send_backtest_command(self, command: str) -> None:
        state = self.control.status()
        if not state.online or state.mode != "BACKTEST":
            return
        ok, message = self.control.command(command)
        self.run_detail.setText(message)
        if not ok:
            ElvaraAlertDialog.show_alert(self, "Backtest", message, semantic="danger")
        self.refresh()

    def stop_backtest(self) -> None:
        state = self.control.status()
        if not state.online or state.mode != "BACKTEST":
            return
        ok, message = self.control.stop()
        self.run_detail.setText(message)
        if not ok:
            ElvaraAlertDialog.show_alert(self, "Backtest", message, semantic="danger")
        self.refresh()

    def open_history_folder(self) -> None:
        path = REPORT_DIR / "backtest_runs"
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _apply_summary(self, summary: dict) -> None:
        if not summary:
            return
        r = float(summary.get("return_percent", 0.0) or 0.0)
        self.return_card.set_value(f"{r:+.1f}%", f"net ${float(summary.get('net_profit', 0.0) or 0.0):,.2f}", GREEN if r >= 0 else RED)
        self.end_card.set_value(f"${float(summary.get('end_balance', 0.0) or 0.0):,.2f}", f"start ${float(summary.get('start_balance', 0.0) or 0.0):,.2f}")
        self.win_card.set_value(f"{float(summary.get('win_rate', 0.0) or 0.0):.1f}%", f"{int(summary.get('wins', 0) or 0)} / {int(summary.get('total_trades', 0) or 0)} wins")
        pf = summary.get("profit_factor", 0.0)
        pf_text = "∞" if isinstance(pf, float) and math.isinf(pf) else f"{float(pf or 0.0):.2f}"
        self.pf_card.set_value(pf_text, f"best {summary.get('best_symbol', '—')}")
        dd = float(summary.get("max_drawdown_percent", 0.0) or 0.0)
        self.dd_card.set_value(f"{dd:.1f}%", f"worst {summary.get('worst_symbol', '—')}", RED if dd > 10 else AMBER)
        expectancy = float(summary.get("expectancy", 0.0) or 0.0)
        avg_win = float(summary.get("average_win", 0.0) or 0.0)
        avg_loss = float(summary.get("average_loss", 0.0) or 0.0)
        self.exp_card.set_value(f"${expectancy:+.3f}", f"avg win {avg_win:+.2f} / loss {avg_loss:+.2f}", GREEN if expectancy >= 0 else RED)
        self.chart.set_values(list(summary.get("curve", []) or []))
        self.dd_chart.set_values(list(summary.get("drawdown_curve", []) or []))
        self.bars.set_items(sorted((summary.get("by_symbol", {}) or {}).items(), key=lambda x: x[1], reverse=True))

        sides = summary.get("by_side", {}) if isinstance(summary.get("by_side"), dict) else {}
        self.side_table.setRowCount(2)
        for row, side in enumerate(("BUY", "SELL")):
            item = sides.get(side, {}) if isinstance(sides.get(side), dict) else {}
            trades = int(item.get("trades", 0) or 0)
            wins = int(item.get("wins", 0) or 0)
            pnl = float(item.get("profit", 0.0) or 0.0)
            wr = wins / trades * 100.0 if trades else 0.0
            vals = [side, str(trades), str(wins), f"{wr:.1f}%", f"{pnl:+.2f}"]
            for col, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                if col == 0:
                    cell.setForeground(QColor(GREEN if side == "BUY" else RED))
                if col == 4:
                    cell.setForeground(QColor(GREEN if pnl >= 0 else RED))
                self.side_table.setItem(row, col, cell)

        trades = list(summary.get("recent_trades", []) or [])
        self.trade_table.setRowCount(len(trades))
        for row, trade in enumerate(trades):
            profit = float(trade.get("profit", 0.0) or 0.0)
            confidence = trade.get("confidence")
            try:
                conf_text = "—" if confidence in (None, "") else f"{float(confidence) * 100:.1f}%"
            except Exception:
                conf_text = str(confidence)
            vals = [
                str(trade.get("open_time", ""))[-19:],
                str(trade.get("symbol", "")),
                str(trade.get("direction", trade.get("type", ""))),
                str(trade.get("lot", trade.get("volume", ""))),
                f"{profit:+.2f}",
                str(trade.get("exit_reason", "")),
                conf_text,
            ]
            for col, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                if col == 1:
                    cell.setForeground(QColor(CYAN))
                if col == 2:
                    cell.setForeground(QColor(GREEN if val == "BUY" else RED if val == "SELL" else AMBER))
                if col == 4:
                    cell.setForeground(QColor(GREEN if profit >= 0 else RED))
                self.trade_table.setItem(row, col, cell)

    def _refresh_compare(self, runtime: dict, stage5: dict) -> None:
        metrics = [
            ("RETURN", "return_percent", "%"),
            ("WIN RATE", "win_rate", "%"),
            ("PROFIT FACTOR", "profit_factor", ""),
            ("MAX DD", "max_drawdown_percent", "%"),
            ("TRADES", "total_trades", ""),
        ]
        self.compare_table.setRowCount(len(metrics))
        for row, (label, key, suffix) in enumerate(metrics):
            rv = runtime.get(key)
            sv = stage5.get(key)
            try:
                rnum = float(rv or 0.0)
                snum = float(sv or 0.0)
                if math.isinf(rnum):
                    rt = "∞"
                else:
                    rt = f"{rnum:.2f}{suffix}"
                if math.isinf(snum):
                    st = "∞"
                else:
                    st = f"{snum:.2f}{suffix}"
                delta = rnum - snum if math.isfinite(rnum) and math.isfinite(snum) else 0.0
                dt = f"{delta:+.2f}{suffix}" if math.isfinite(delta) else "—"
            except Exception:
                rt, st, dt = str(rv or "—"), str(sv or "—"), "—"
            for col, text in enumerate((label, rt, st, dt)):
                cell = QTableWidgetItem(text)
                if col == 0:
                    cell.setForeground(QColor(CYAN))
                if col == 3 and text != "—":
                    cell.setForeground(QColor(GREEN if not text.startswith("-") else RED))
                self.compare_table.setItem(row, col, cell)

    def _refresh_history(self) -> None:
        runs = self.service.load_backtest_runs(12)
        self.history_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            vals = [
                str(run.get("generated_utc", ""))[:19].replace("T", " "),
                f"{run.get('start_date', '—')} → {run.get('end_date', '—')}",
                f"${float(run.get('start_balance', 0.0) or 0.0):,.2f}",
                f"${float(run.get('end_balance', 0.0) or 0.0):,.2f}",
                f"{float(run.get('return_percent', 0.0) or 0.0):+.1f}%",
                str(int(run.get("total_trades", 0) or 0)),
                f"{float(run.get('max_drawdown_percent', 0.0) or 0.0):.1f}%",
            ]
            for col, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                if col == 4:
                    cell.setForeground(QColor(GREEN if "+" in val and not val.startswith("-") else RED))
                self.history_table.setItem(row, col, cell)

    def refresh(self) -> None:
        state = self.control.status()
        is_backtest = state.mode == "BACKTEST"
        running = state.online and is_backtest
        progress = float(state.details.get("backtest_progress", 0.0) or 0.0) if is_backtest else 0.0
        progress = max(0.0, min(100.0, progress))
        self.progress.setValue(int(progress * 10))
        self.progress.setFormat(f"{progress:.1f}%")

        if running and state.paused:
            self.run_status.set_semantic("warning", "PAUSED")
        elif running and state.state == "STARTING":
            self.run_status.set_semantic("warning", "STARTING")
        elif running:
            self.run_status.set_semantic("success", "RUNNING")
        elif is_backtest and state.state == "COMPLETED":
            self.run_status.set_semantic("success", "COMPLETE")
            self.progress.setValue(1000)
            self.progress.setFormat("100.0%")
        elif is_backtest and state.state == "ERROR":
            self.run_status.set_semantic("danger", "ERROR")
        else:
            self.run_status.set_semantic("neutral", "IDLE")

        details = state.details or {}
        detail_bits = [state.note] if state.note else []
        if is_backtest and details:
            detail_bits.append(f"{int(details.get('trades', 0) or 0)} trades")
            if details.get("balance") is not None:
                detail_bits.append(f"${float(details.get('balance', 0.0) or 0.0):,.2f}")
        self.run_detail.setText(" · ".join(detail_bits) if detail_bits else "Ready")

        self.run_btn.setEnabled(not state.online)
        self.pause_btn.setEnabled(running and not state.paused and state.state not in {"STARTING", "STOPPING"})
        self.resume_btn.setEnabled(running and state.paused)
        self.stop_btn.setEnabled(running)

        runtime = self.service.load_runtime_backtest_summary()
        stage5 = self.service.load_backtest_summary()
        if runtime:
            self.summary_source.set_semantic("success", "LATEST RUNTIME")
            self.summary_note.setText(f"{runtime.get('start_date', '—')} → {runtime.get('end_date', '—')} · {runtime.get('generated_utc', '—')}")
            self._apply_summary(runtime)
            self._refresh_compare(runtime, stage5)
        else:
            self.summary_source.set_semantic("info", "STAGE 5 VALIDATION")
            self.summary_note.setText("No desktop run yet.")
            self._apply_summary(stage5)
            self._refresh_compare({}, stage5)
        self._refresh_history()


class SettingsPage(QWidget):
    def __init__(
        self,
        config: TradeAIConfigService,
        control: EngineControlService,
        service: TradeAIDataService,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.control = control
        self.service = service
        self.editors: dict[str, QWidget] = {}
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, _, root = _scroll_shell()
        outer.addWidget(scroll)

        eye = QLabel("CONFIGURATION / VALIDATED EDITOR")
        eye.setObjectName("Eyebrow")
        title = QLabel("Configuration")
        title.setObjectName("PageTitle")
        root.addWidget(eye)
        root.addWidget(title)

        actions = QHBoxLayout()
        self.save_btn = ElvaraButton("SAVE CONFIG", variant="primary", size="standard")
        self.reload_btn = ElvaraButton("RELOAD", variant="secondary", size="standard", icon_name="refresh")
        self.save_btn.clicked.connect(self.save)
        self.reload_btn.clicked.connect(self.reload)
        actions.addWidget(self.save_btn)
        actions.addWidget(self.reload_btn)
        actions.addStretch()
        self.dirty_badge = ElvaraStatusBadge("SAVED", "success")
        self.engine_badge = ElvaraStatusBadge("ENGINE OFFLINE", "neutral")
        actions.addWidget(self.dirty_badge)
        actions.addWidget(self.engine_badge)
        root.addLayout(actions)

        self.notice = ElvaraNotice(
            "Runtime lifecycle",
            "Configuration is locked while the engine runs. Saved changes load on the next engine start.",
            "info",
        )
        root.addWidget(self.notice)

        policy_panel, policy_l = _panel("Effective Production Policy", "Read-only promoted contracts used by production decision/risk gates")
        policy_grid = QGridLayout()
        self.policy_risk = MetricCard("CALIBRATED RISK", "—", "Stage 5", CYAN, "shield")
        self.policy_symbols = MetricCard("POLICY SYMBOLS", "—", "decision policy", BLUE, "signals")
        self.policy_acceptance = MetricCard("ACCEPTANCE", "—", "production validation", GREEN, "activity")
        self.policy_mode = MetricCard("SAVED MODE", "—", "settings.py", AMBER, "control")
        for i, card in enumerate((self.policy_risk, self.policy_symbols, self.policy_acceptance, self.policy_mode)):
            policy_grid.addWidget(card, 0, i)
        policy_l.addLayout(policy_grid)
        policy_note = QLabel("Fallback signal settings can be edited below, but promoted per-symbol decision gates and the calibrated risk policy remain authoritative when required by the model contract.")
        policy_note.setObjectName("MetricDetail")
        policy_note.setWordWrap(True)
        policy_l.addWidget(policy_note)
        root.addWidget(policy_panel)

        sections: dict[str, list] = {}
        for field in FIELDS:
            sections.setdefault(field.section, []).append(field)

        for section, fields in sections.items():
            panel, layout = _panel(section, "")
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)
            for row, field in enumerate(fields):
                label = QLabel(field.label)
                label.setObjectName("MetricDetail")
                if field.description:
                    label.setToolTip(field.description)

                if field.kind == "choice":
                    editor = QComboBox()
                    editor.addItems(list(field.choices))
                    editor.currentTextChanged.connect(self._mark_dirty)
                elif field.kind == "bool":
                    editor = QComboBox()
                    editor.addItems(["True", "False"])
                    editor.currentTextChanged.connect(self._mark_dirty)
                else:
                    editor = QLineEdit()
                    editor.textChanged.connect(self._mark_dirty)
                    if field.kind == "symbols":
                        editor.setPlaceholderText("EURUSD, GBPUSD, USDJPY")
                    elif field.kind == "date":
                        editor.setPlaceholderText("YYYY-MM-DD")

                editor.setToolTip(field.description)
                self.editors[field.key] = editor
                grid.addWidget(label, row, 0)
                grid.addWidget(editor, row, 1)

                if field.description:
                    detail = QLabel(field.description)
                    detail.setObjectName("Muted")
                    detail.setWordWrap(True)
                    grid.addWidget(detail, row, 2)

            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 2)
            layout.addLayout(grid)
            root.addWidget(panel)

        root.addStretch()
        self.reload()
        self.refresh()

    def _mark_dirty(self, *_args) -> None:
        if self._loading:
            return
        self.dirty_badge.set_semantic("warning", "RESTART REQUIRED")

    def _editor_value(self, key: str):
        editor = self.editors[key]
        if isinstance(editor, QComboBox):
            return editor.currentText()
        if isinstance(editor, QLineEdit):
            return editor.text()
        return ""

    def _refresh_policy_cards(self, values: dict | None = None) -> None:
        values = values or {}
        risk = self.service.load_risk_policy()
        decision = self.service.load_decision_policy()
        validation = self.service.load_validation()
        risk_pct = float(risk.get("risk_percent", 0.0) or 0.0)
        symbols = decision.get("symbols", {}) if isinstance(decision.get("symbols"), dict) else {}
        accepted = bool(validation.get("acceptance_pass", False))
        self.policy_risk.set_value(f"{risk_pct:.2f}%", "promoted policy", GREEN if risk_pct > 0 else RED)
        self.policy_symbols.set_value(str(len(symbols)), "symbol policies")
        self.policy_acceptance.set_value("PASS" if accepted else "CHECK", "Stage 5", GREEN if accepted else AMBER)
        self.policy_mode.set_value(str(values.get("MODE", self.control.mode())), "saved configuration")

    def reload(self) -> None:
        self._loading = True
        try:
            values = self.config.read()
            for field in FIELDS:
                if field.key not in self.editors:
                    continue
                value = values.get(field.key, "")
                editor = self.editors[field.key]
                if field.kind == "symbols" and isinstance(value, list):
                    text = ", ".join(str(x) for x in value)
                elif field.kind == "bool":
                    text = "True" if bool(value) else "False"
                else:
                    text = str(value)
                if isinstance(editor, QComboBox):
                    idx = editor.findText(text)
                    if idx >= 0:
                        editor.setCurrentIndex(idx)
                elif isinstance(editor, QLineEdit):
                    editor.setText(text)
            self._refresh_policy_cards(values)
            self.notice.set_message("Loaded TradeAI/config/settings.py", semantic="info")
            self.dirty_badge.set_semantic("success", "SAVED")
        except Exception as exc:
            self.notice.set_message(f"Unable to read configuration: {exc}", semantic="danger")
        finally:
            self._loading = False

    def save(self) -> None:
        state = self.control.status()
        if state.online:
            ElvaraAlertDialog.show_alert(
                self,
                "Configuration locked",
                "Stop TradeAI before saving. A running process keeps the settings it loaded at startup.",
                semantic="danger",
            )
            return

        raw = {field.key: self._editor_value(field.key) for field in FIELDS}
        ok, message = self.config.save(raw)
        self.notice.set_message(message, semantic="success" if ok else "danger")
        if not ok:
            ElvaraAlertDialog.show_alert(self, "Configuration", message, semantic="danger")
        else:
            self.reload()

    def refresh(self) -> None:
        state = self.control.status()
        self.save_btn.setEnabled(not state.online)
        if state.online:
            self.engine_badge.set_semantic("warning", f"ENGINE {state.state}")
        elif state.state == "ERROR":
            self.engine_badge.set_semantic("danger", "ENGINE ERROR")
        else:
            self.engine_badge.set_semantic("neutral", "ENGINE OFFLINE")


class RiskPage(QWidget):
    def __init__(
        self,
        service: TradeAIDataService,
        config: TradeAIConfigService,
        control: EngineControlService,
        parent=None,
    ):
        super().__init__(parent)
        self.service = service
        self.config = config
        self.control = control
        self._snapshot: RuntimeSnapshot | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll, _, root = _scroll_shell()
        outer.addWidget(scroll)

        eye = QLabel("CAPITAL DEFENSE / LIVE EXPOSURE")
        eye.setObjectName("Eyebrow")
        title = QLabel("Risk Command Center")
        title.setObjectName("PageTitle")
        root.addWidget(eye)
        root.addWidget(title)

        live_grid = QGridLayout()
        live_grid.setSpacing(10)
        self.forward = MetricCard("FORWARD CAPITAL", "$0.00", "shadow account", CYAN, "capital")
        self.floating = MetricCard("FLOATING P/L", "$0.00", "broker positions", GREEN, "profit")
        self.open_risk = MetricCard("OPEN RISK @ SL", "$0.00", "broker-calculated", AMBER, "shield")
        self.risk_usage = MetricCard("RISK DEPLOYED", "0.00%", "of capital", AMBER, "risk")
        self.daily_loss = MetricCard("DAILY LOSS", "0.00%", "ledger", GREEN, "risk")
        self.broker_dd = MetricCard("BROKER EQUITY DD", "0.00%", "balance vs equity", GREEN, "risk")
        for i, card in enumerate((self.forward, self.floating, self.open_risk, self.risk_usage, self.daily_loss, self.broker_dd)):
            live_grid.addWidget(card, i // 3, i % 3)
        root.addLayout(live_grid)

        mid = QGridLayout()
        mid.setSpacing(10)
        mid.setColumnStretch(0, 3)
        mid.setColumnStretch(1, 7)

        gauge_panel, gauge_l = _panel("Drawdown Headroom", "Forward maximum drawdown against the calibrated stress ceiling")
        self.gauge = RiskGauge()
        gauge_l.addWidget(self.gauge)
        self.headroom = QLabel("—")
        self.headroom.setObjectName("MetricDetail")
        gauge_l.addWidget(self.headroom)
        mid.addWidget(gauge_panel, 0, 0)

        policy_panel, policy_l = _panel("Protection Policy", "Effective limits and next-trade calibrated budget")
        policy_grid = QGridLayout()
        self.risk_trade = MetricCard("RISK / TRADE", "—", "Stage 5", CYAN, "shield")
        self.next_budget = MetricCard("NEXT TRADE BUDGET", "—", "capital × risk", BLUE, "capital")
        self.daily_headroom = MetricCard("DAILY LOSS HEADROOM", "—", "configured", GREEN, "risk")
        self.dd_headroom = MetricCard("DRAWDOWN HEADROOM", "—", "configured", GREEN, "risk")
        for i, card in enumerate((self.risk_trade, self.next_budget, self.daily_headroom, self.dd_headroom)):
            policy_grid.addWidget(card, i // 2, i % 2)
        policy_l.addLayout(policy_grid)
        mid.addWidget(policy_panel, 0, 1)
        root.addLayout(mid)

        pos_panel, pos_l = _panel("Position Risk", "Live broker P/L and SL/TP risk economics")
        self.position_table = _table(["SYMBOL", "SIDE", "P/L", "RISK @ SL", "REWARD @ TP", "R:R", "AGE"])
        self.position_table.setMinimumHeight(190)
        pos_l.addWidget(self.position_table)
        root.addWidget(pos_panel)

        val_panel, val_l = _panel("Production Acceptance", "Normal and stress validation gates")
        self.validation_table = _table(["CAPITAL", "NORMAL", "STRESS", "NET PROFIT", "PROFIT FACTOR", "DRAWDOWN", "TRADES"])
        self.validation_table.setMinimumHeight(165)
        val_l.addWidget(self.validation_table)
        root.addWidget(val_panel)
        root.addStretch()

        self.refresh()

    def update_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self._snapshot = snapshot
        self._apply_live(snapshot)

    def _apply_live(self, snapshot: RuntimeSnapshot) -> None:
        risk_policy = self.service.load_risk_policy()
        safety = risk_policy.get("calibration_safety", {}) if isinstance(risk_policy.get("calibration_safety"), dict) else {}
        risk_pct = float(risk_policy.get("risk_percent", 0.0) or 0.0)
        stress_limit = float(safety.get("stress_max_drawdown_percent", 10.0) or 10.0)

        try:
            cfg = self.config.read()
        except Exception:
            cfg = {}
        daily_limit = float(cfg.get("MAX_DAILY_LOSS_PERCENT", 0.0) or 0.0)
        dd_limit = float(cfg.get("MAX_DRAWDOWN_PERCENT", 0.0) or 0.0)

        capital = snapshot.virtual_balance if snapshot.virtual_balance > 0 else snapshot.broker_equity
        self.forward.set_value(f"${snapshot.virtual_balance:,.2f}", f"start ${snapshot.initial_virtual_balance:,.2f}")
        self.floating.set_value(f"${snapshot.floating_profit:+,.2f}", f"{snapshot.open_positions} open", GREEN if snapshot.floating_profit >= 0 else RED)
        self.open_risk.set_value(f"${snapshot.current_risk_money:,.2f}", "risk to broker SL")
        self.risk_usage.set_value(f"{snapshot.risk_usage_percent:.2f}%", "of active capital", AMBER if snapshot.risk_usage_percent > risk_pct * 1.5 and risk_pct > 0 else GREEN)
        self.daily_loss.set_value(f"{snapshot.daily_loss_percent:.2f}%", f"limit {daily_limit:.1f}%", RED if daily_limit and snapshot.daily_loss_percent >= daily_limit else GREEN)
        self.broker_dd.set_value(f"{snapshot.broker_drawdown_percent:.2f}%", "balance vs equity", RED if dd_limit and snapshot.broker_drawdown_percent >= dd_limit else GREEN)

        self.gauge.set_values(snapshot.drawdown_percent, stress_limit, "FORWARD MAX DD")
        self.headroom.setText(f"Stress headroom  {max(0.0, stress_limit - snapshot.drawdown_percent):.2f}%")

        self.risk_trade.set_value(f"{risk_pct:.2f}%", "calibrated production risk")
        self.next_budget.set_value(f"${capital * risk_pct / 100.0:,.3f}" if capital > 0 else "—", "target risk money")
        daily_remaining = max(0.0, daily_limit - snapshot.daily_loss_percent) if daily_limit else 0.0
        dd_remaining = max(0.0, dd_limit - snapshot.drawdown_percent) if dd_limit else 0.0
        self.daily_headroom.set_value(f"{daily_remaining:.2f}%", f"limit {daily_limit:.1f}%", RED if daily_limit and daily_remaining <= daily_limit * 0.15 else GREEN)
        self.dd_headroom.set_value(f"{dd_remaining:.2f}%", f"limit {dd_limit:.1f}%", RED if dd_limit and dd_remaining <= dd_limit * 0.15 else GREEN)

        self.position_table.setRowCount(len(snapshot.positions))
        for row, pos in enumerate(snapshot.positions):
            risk = "—" if pos.risk_to_sl is None else f"${pos.risk_to_sl:,.2f}"
            reward = "—" if pos.reward_to_tp is None else f"${pos.reward_to_tp:,.2f}"
            rr = "—" if pos.reward_risk is None else f"{pos.reward_risk:.2f}R"
            age = "—" if pos.age_minutes is None else (f"{pos.age_minutes / 60:.1f}h" if pos.age_minutes >= 60 else f"{pos.age_minutes:.0f}m")
            vals = [pos.symbol, pos.side, f"{pos.profit:+.2f}", risk, reward, rr, age]
            for col, val in enumerate(vals):
                cell = QTableWidgetItem(val)
                if col == 0:
                    cell.setForeground(QColor(CYAN))
                if col == 1:
                    cell.setForeground(QColor(GREEN if pos.side == "BUY" else RED))
                if col == 2:
                    cell.setForeground(QColor(GREEN if pos.profit >= 0 else RED))
                self.position_table.setItem(row, col, cell)

    def _refresh_validation(self) -> None:
        val = self.service.load_validation()
        gates = val.get("acceptance_gates", {}) if isinstance(val.get("acceptance_gates"), dict) else {}
        rows = []
        for capital, cfg in gates.items():
            if not isinstance(cfg, dict):
                continue
            normal = cfg.get("normal", {}) if isinstance(cfg.get("normal"), dict) else {}
            rows.append((capital, cfg, normal))
        self.validation_table.setRowCount(len(rows))
        for row, (capital, cfg, normal) in enumerate(rows):
            checks = [
                normal.get("net_profit_positive", False),
                normal.get("profit_factor", False),
                normal.get("max_drawdown", False),
                normal.get("trade_count", False),
            ]
            vals = [
                f"${float(capital):.0f}",
                "PASS" if cfg.get("normal_pass") else "FAIL",
                "PASS" if cfg.get("stress_pass") else "FAIL",
                "PASS" if checks[0] else "FAIL",
                "PASS" if checks[1] else "FAIL",
                "PASS" if checks[2] else "FAIL",
                "PASS" if checks[3] else "FAIL",
            ]
            for col, value in enumerate(vals):
                cell = QTableWidgetItem(value)
                cell.setForeground(QColor(GREEN if value == "PASS" else RED if value == "FAIL" else CYAN))
                self.validation_table.setItem(row, col, cell)

    def refresh(self) -> None:
        self._refresh_validation()
        if self._snapshot is not None:
            self._apply_live(self._snapshot)
        else:
            policy = self.service.load_risk_policy()
            safety = policy.get("calibration_safety", {}) if isinstance(policy.get("calibration_safety"), dict) else {}
            selected = float(safety.get("selected_worst_drawdown_percent", 0.0) or 0.0)
            stress = float(safety.get("stress_max_drawdown_percent", 10.0) or 10.0)
            self.gauge.set_values(selected, stress, "CALIBRATED WORST DD")


class ReportsPage(QWidget):
    def __init__(self, service: TradeAIDataService, parent=None):
        super().__init__(parent); self.service=service
        outer=QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        scroll,_,root=_scroll_shell(); outer.addWidget(scroll)
        eye=QLabel("ARTIFACT VAULT / LOCAL"); eye.setObjectName("Eyebrow")
        title=QLabel("Reports & Artifacts"); title.setObjectName("PageTitle")
        sub=QLabel("Validation outputs, forward-state ledgers and policy artifacts generated by TradeAI"); sub.setObjectName("PageSub")
        root.addWidget(eye); root.addWidget(title); root.addWidget(sub)
        action=QHBoxLayout(); action.setSpacing(8)
        open_btn=ElvaraButton("Open Reports Folder", variant="primary", size="standard", icon_name="folder"); open_btn.clicked.connect(self.open_folder)
        refresh_btn=ElvaraButton("Refresh Vault", variant="secondary", size="standard", icon_name="refresh"); refresh_btn.clicked.connect(self.refresh)
        self.search=ElvaraSearchField("Search artifacts...")
        self.search.setMaximumWidth(340)
        self.search.textChanged.connect(lambda _text: self.refresh())
        action.addWidget(open_btn); action.addWidget(refresh_btn); action.addStretch(); action.addWidget(self.search); root.addLayout(action)
        panel,layout=_panel("Artifact Vault","Double-click a file to open it with the default Windows application")
        self.table=_table(["ARTIFACT","FORMAT","SIZE","LAST MODIFIED"]); self.table.doubleClicked.connect(self.open_selected); self.table.setMinimumHeight(330)
        layout.addWidget(self.table); root.addWidget(panel,1); self.refresh()

    def refresh(self) -> None:
        files=self.service.report_files()
        query = self.search.text().strip().lower() if hasattr(self, "search") else ""
        if query:
            files = [path for path in files if query in path.name.lower() or query in path.suffix.lower()]
        self.table.setRowCount(len(files))
        for row,path in enumerate(files):
            vals=[path.name,path.suffix.upper().lstrip("."),f"{path.stat().st_size/1024:,.1f} KB",datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d  %H:%M")]
            for col,v in enumerate(vals):
                item=QTableWidgetItem(v); item.setData(Qt.UserRole,str(path))
                if col==0: item.setForeground(QColor(CYAN))
                self.table.setItem(row,col,item)
    def open_folder(self): REPORT_DIR.mkdir(parents=True,exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(REPORT_DIR)))
    def open_selected(self):
        row=self.table.currentRow(); item=self.table.item(row,0) if row>=0 else None
        if item and item.data(Qt.UserRole): QDesktopServices.openUrl(QUrl.fromLocalFile(item.data(Qt.UserRole)))


class LogsPage(QWidget):
    def __init__(self, service: TradeAIDataService, parent=None):
        super().__init__(parent); self.service=service
        root=QVBoxLayout(self); root.setContentsMargins(18,15,18,20); root.setSpacing(12)
        eye=QLabel("CORE TELEMETRY / READ ONLY"); eye.setObjectName("Eyebrow")
        title=QLabel("Runtime Console"); title.setObjectName("PageTitle")
        sub=QLabel("Raw TradeAI decision telemetry and execution diagnostics"); sub.setObjectName("PageSub")
        root.addWidget(eye); root.addWidget(title); root.addWidget(sub)
        panel,layout=_panel("TRADEAI::BOT.LOG","Live tail · newest events at the bottom")
        header=QHBoxLayout(); badge=ElvaraStatusBadge("● STREAMING", "success"); self.count=QLabel("0 lines"); self.count.setObjectName("MetricDetail")
        header.addWidget(badge); header.addStretch(); header.addWidget(self.count); layout.addLayout(header)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log); root.addWidget(panel,1)
    def refresh(self):
        lines=self.service.tail_log(320); self.count.setText(f"{len(lines)} lines"); self.log.setPlainText("\n".join(lines)); self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
