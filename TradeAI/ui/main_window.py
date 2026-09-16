from __future__ import annotations

from PySide6.QtCore import QPoint, QSettings, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QCloseEvent, QGuiApplication, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizeGrip,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .components import ElvaraNavButton, ElvaraStatusBadge, NiraTopBar
from .control_service import EngineControlService
from .config_service import TradeAIConfigService
from .data_service import RuntimeSnapshot, TradeAIDataService
from .pages import BacktestPage, ControlPage, LogsPage, OverviewPage, ReportsPage, RiskPage, SettingsPage, SignalsPage
from .theme import get_app_style, set_theme_mode, theme_mode
from .widgets import AmbientCanvas, HolographicShell, PulseDot, TradeAILogo


class SnapshotWorker(QThread):
    snapshot_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(self, service: TradeAIDataService, parent=None):
        super().__init__(parent)
        self.service = service
        self._running = True
        self.chart_symbol = "EURUSD"

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                self.snapshot_ready.emit(self.service.get_snapshot(self.chart_symbol))
            except Exception as exc:
                self.worker_error.emit(str(exc))
            for _ in range(10):
                if not self._running:
                    return
                self.msleep(100)


class TitleBar(QFrame):
    theme_requested = Signal()

    def __init__(self, window: "TradeAIMainWindow"):
        super().__init__(window)
        self.window_ref = window
        self._drag_pos: QPoint | None = None
        self.setObjectName("TitleBar")
        self.setFixedHeight(48)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 8, 6)
        layout.setSpacing(8)

        nira = QLabel("N I R A")
        nira.setObjectName("TitleBarTitle")
        product = QLabel("TradeAI")
        product.setObjectName("TitleBarProduct")
        brand = QLabel("// ELVARA")
        brand.setObjectName("TitleBarBrand")
        layout.addWidget(nira)
        sep = QLabel("│")
        sep.setObjectName("BrandSub")
        layout.addWidget(sep)
        layout.addWidget(product)
        layout.addWidget(brand)
        layout.addStretch()

        self.theme_button = QPushButton("HALO")
        self.theme_button.setObjectName("ThemeButton")
        self.theme_button.clicked.connect(self.theme_requested.emit)
        layout.addWidget(self.theme_button)

        for text, obj, cb in (
            ("—", "WindowButton", window.showMinimized),
            ("□", "WindowButton", self._toggle),
            ("×", "WindowCloseButton", window.close),
        ):
            button = QPushButton(text)
            button.setObjectName(obj)
            button.clicked.connect(cb)
            layout.addWidget(button)

    def set_theme_label(self, current_mode: str) -> None:
        self.theme_button.setText("ECLIPSE" if current_mode == "halo" else "HALO")

    def _toggle(self):
        self.window_ref.toggle_safe_maximize()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self.window_ref.manual_maximized:
                event.accept()
                return
            self._drag_pos = event.globalPosition().toPoint() - self.window_ref.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton and not self.window_ref.manual_maximized:
            self.window_ref.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._toggle()
            event.accept()


class TradeAIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradeAI // ELVARA")
        self.setMinimumSize(980, 620)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self._manual_maximized = False
        self._normal_geometry = None
        self._latest_snapshot: RuntimeSnapshot | None = None

        self.settings = QSettings("ELVARA", "TradeAI")
        saved_theme = str(self.settings.value("ui/elvara_theme", "eclipse"))
        self.theme_mode = set_theme_mode(saved_theme)

        self.service = TradeAIDataService()
        self.control = EngineControlService()
        self.config = TradeAIConfigService()

        root = QWidget()
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        base = QVBoxLayout(root)
        base.setContentsMargins(0, 0, 0, 0)
        base.setSpacing(0)

        self.shell = HolographicShell()
        self.shell.setObjectName("HoloShell")
        base.addWidget(self.shell)
        shell_l = QVBoxLayout(self.shell)
        shell_l.setContentsMargins(5, 5, 5, 5)
        shell_l.setSpacing(0)

        # One ambient canvas owns the whole application.  This is what lets the
        # ELVARA celestial background remain visible through the title bar,
        # sidebar, cards and working surfaces instead of being trapped behind
        # only the center page.
        self.canvas = AmbientCanvas()
        shell_l.addWidget(self.canvas)
        canvas_l = QVBoxLayout(self.canvas)
        canvas_l.setContentsMargins(6, 6, 6, 6)
        canvas_l.setSpacing(7)

        self.title_bar = TitleBar(self)
        self.title_bar.theme_requested.connect(self.toggle_theme)
        canvas_l.addWidget(self.title_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(7)
        self.sidebar = self._build_sidebar()
        body.addWidget(self.sidebar)

        workspace = QVBoxLayout()
        workspace.setContentsMargins(0, 0, 0, 0)
        workspace.setSpacing(7)
        workspace.addWidget(self._build_content_header())

        self.stack = QStackedWidget()
        workspace.addWidget(self.stack, 1)
        body.addLayout(workspace, 1)
        canvas_l.addLayout(body, 1)
        canvas_l.addWidget(self._build_statusbar())

        self.overview = OverviewPage(self.service, self.control)
        self.control_page = ControlPage(self.service, self.control)
        self.signals = SignalsPage(self.service, self.control)
        self.backtest = BacktestPage(self.service, self.control, self.config)
        self.risk = RiskPage(self.service, self.config, self.control)
        self.reports = ReportsPage(self.service)
        self.logs = LogsPage(self.service)
        self.settings_page = SettingsPage(self.config, self.control, self.service)
        self.pages = [
            self.overview,
            self.control_page,
            self.signals,
            self.backtest,
            self.risk,
            self.reports,
            self.logs,
            self.settings_page,
        ]
        for page in self.pages:
            self.stack.addWidget(page)
        self.nav_buttons[0].setChecked(True)

        self.worker = SnapshotWorker(self.service, self)
        self.worker.snapshot_ready.connect(self.apply_snapshot)
        self.worker.worker_error.connect(self._worker_error)
        self.overview.chart_symbol.currentTextChanged.connect(self._set_worker_symbol)
        self.worker.start()

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        self.page_timer = QTimer(self)
        self.page_timer.timeout.connect(self._refresh_visible_page)
        self.page_timer.start(3000)

        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self._refresh_engine_status)
        self.control_timer.start(1000)
        self.apply_theme(self.theme_mode)
        self._refresh_engine_status()

        # Full work-area window: visually fullscreen, but Windows taskbar remains usable.
        QTimer.singleShot(0, self.fit_to_available_screen)

    @property
    def manual_maximized(self) -> bool:
        return self._manual_maximized

    def _available_geometry(self):
        screen = self.screen() or QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else None

    def fit_to_available_screen(self) -> None:
        available = self._available_geometry()
        if available is None:
            self.resize(1280, 720)
            return
        self.setGeometry(available)
        self._manual_maximized = True

    def _restore_geometry(self) -> None:
        available = self._available_geometry()
        if available is None:
            return
        width = int(available.width() * 0.90)
        height = int(available.height() * 0.90)
        width = max(min(width, available.width()), min(980, available.width()))
        height = max(min(height, available.height()), min(620, available.height()))
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        self.setGeometry(x, y, width, height)
        self._manual_maximized = False

    def toggle_safe_maximize(self) -> None:
        available = self._available_geometry()
        if available is None:
            return
        if self._manual_maximized:
            self._restore_geometry()
        else:
            self.setGeometry(available)
            self._manual_maximized = True

    def _sidebar_width(self) -> int:
        screen = self._available_geometry()
        if screen is None:
            return 180
        if screen.width() >= 1800:
            return 194
        if screen.width() >= 1400:
            return 184
        return 174

    def _build_sidebar(self) -> QFrame:
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(self._sidebar_width())
        layout = QVBoxLayout(side)
        layout.setContentsMargins(9, 10, 9, 9)
        layout.setSpacing(4)

        brand = QHBoxLayout()
        brand.setSpacing(7)
        brand.addWidget(TradeAILogo(30))
        texts = QVBoxLayout()
        texts.setSpacing(0)
        lockup = QHBoxLayout()
        lockup.setSpacing(6)
        title = QLabel("TRADEAI")
        title.setObjectName("BrandTitle")
        parent_brand = QLabel("// ELVARA")
        parent_brand.setObjectName("BrandAccent")
        lockup.addWidget(title)
        lockup.addWidget(parent_brand, 0, Qt.AlignBottom)
        lockup.addStretch()
        sub = QLabel("NIRA INTELLIGENCE SURFACE")
        sub.setObjectName("BrandSub")
        texts.addLayout(lockup)
        texts.addWidget(sub)
        brand.addLayout(texts, 1)
        layout.addLayout(brand)

        line = QFrame()
        line.setObjectName("SidebarDivider")
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        cap = QLabel("WORKSPACE")
        cap.setObjectName("Eyebrow")
        cap.setContentsMargins(7, 5, 0, 2)
        layout.addWidget(cap)

        self.nav_specs = [
            ("overview", "Command Center", 0),
            ("control", "System Control", 1),
            ("signals", "Signal Intelligence", 2),
            ("backtest", "Backtesting", 3),
            ("risk", "Risk Intelligence", 4),
            ("reports", "Artifact Vault", 5),
            ("terminal", "Runtime Console", 6),
            ("control", "Configuration", 7),
        ]
        self.nav_buttons = []
        for icon, text, idx in self.nav_specs:
            button = ElvaraNavButton(text, icon)
            button.clicked.connect(lambda checked=False, i=idx: self.set_page(i))
            layout.addWidget(button)
            self.nav_buttons.append(button)

        layout.addStretch()

        card = QFrame()
        card.setObjectName("SidebarCoreCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(9, 8, 9, 8)
        cl.setSpacing(4)
        rr = QHBoxLayout()
        self.side_pulse = PulseDot()
        rr.addWidget(self.side_pulse)
        core = QLabel("AI CORE")
        core.setObjectName("CoreLabel")
        rr.addWidget(core)
        rr.addStretch()
        self.core_state = QLabel("READY")
        self.core_state.setObjectName("Positive")
        rr.addWidget(self.core_state)
        cl.addLayout(rr)
        self.core_note = QLabel("Stage 5 policy\nExecution interlocks armed")
        self.core_note.setObjectName("BrandSub")
        cl.addWidget(self.core_note)
        layout.addWidget(card)
        return side

    def _build_content_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("ContentHeader")
        frame.setFixedHeight(46)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(4, 3, 8, 3)
        layout.setSpacing(6)

        # Global NIRA reservation lives immediately below the custom window bar.
        # It is intentionally non-functional until the NIRA runtime is connected.
        self.nira_bar = NiraTopBar()
        self.nira_bar.setMinimumWidth(260)
        layout.addWidget(self.nira_bar, 1)

        self.engine_pill = ElvaraStatusBadge("ENGINE OFFLINE", "danger")
        self.mode_pill = ElvaraStatusBadge("MODE", "accent")
        self.mt5_pill = ElvaraStatusBadge("● MT5 CHECKING", "danger")
        self.symbol_pill = ElvaraStatusBadge("EURUSD / M5", "neutral")
        layout.addWidget(self.engine_pill)
        layout.addWidget(self.mode_pill)
        layout.addWidget(self.mt5_pill)
        layout.addWidget(self.symbol_pill)

        self.spread_label = QLabel("SPREAD —")
        self.spread_label.setObjectName("Eyebrow")
        layout.addWidget(self.spread_label)

        self.clock = QLabel("")
        self.clock.setObjectName("BrandSub")
        layout.addWidget(self.clock)
        return frame

    def _build_statusbar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("BottomBar")
        bar.setFixedHeight(18)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 0, 4, 0)
        layout.setSpacing(10)
        self.bottom_state = QLabel("CORE LINK / INITIALIZING")
        self.bottom_state.setObjectName("BrandSub")
        layout.addWidget(self.bottom_state)
        layout.addStretch()
        mode = QLabel("LOCAL CONTROL PLANE · NO MANUAL ORDER PANEL")
        mode.setObjectName("Eyebrow")
        layout.addWidget(mode)
        layout.addWidget(QSizeGrip(self))
        return bar

    def _update_theme_button(self) -> None:
        if hasattr(self, "title_bar"):
            self.title_bar.set_theme_label(self.theme_mode)

    def toggle_theme(self) -> None:
        target = "halo" if theme_mode() == "eclipse" else "eclipse"
        self.apply_theme(target)

    def apply_theme(self, mode: str | None = None) -> None:
        self.theme_mode = set_theme_mode(mode or self.theme_mode)
        self.settings.setValue("ui/elvara_theme", self.theme_mode)
        style = get_app_style(self.theme_mode)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(style)
        self.setStyleSheet(style)
        self._update_theme_button()

        # Custom-painted cards/backgrounds resolve palette values at paint time.
        # Refreshing here also reloads elvara-dark / elvara-light correctly.
        for widget in [self, *self.findChildren(QWidget)]:
            refresher = getattr(widget, "refresh_theme", None)
            if callable(refresher):
                refresher()
            try:
                widget.style().unpolish(widget)
                widget.style().polish(widget)
            except Exception:
                pass
            widget.update()

        if app is not None:
            app.processEvents()

    def set_page(self, index: int):
        self.stack.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self._refresh_visible_page()

    def _refresh_visible_page(self):
        page = self.stack.currentWidget()
        refresh = getattr(page, "refresh", None)
        if callable(refresh):
            refresh()

    def _set_worker_symbol(self, symbol: str):
        self.worker.chart_symbol = symbol
        self.symbol_pill.setText(f"{symbol} / M5")

    def _update_clock(self):
        self.clock.setText(self.service.formatted_time())

    def _refresh_engine_status(self):
        state = self.control.status()

        if state.online and state.state == "STARTING":
            text, semantic = "ENGINE STARTING", "warning"
        elif state.online and state.paused:
            text, semantic = "ENGINE PAUSED", "warning"
        elif state.online:
            text, semantic = "ENGINE RUNNING", "success"
        elif state.state == "COMPLETED" and state.mode == "BACKTEST":
            text, semantic = "BACKTEST COMPLETE", "success"
        elif state.state == "ERROR":
            text, semantic = "ENGINE ERROR", "danger"
        elif state.state == "STOPPED":
            text, semantic = "ENGINE STOPPED", "neutral"
        else:
            text, semantic = "ENGINE OFFLINE", "danger"

        self.engine_pill.set_semantic(semantic, text)

        if state.online and state.state == "STARTING":
            core_text, core_obj = "STARTING", "Amber"
        elif state.online and state.paused:
            core_text, core_obj = "PAUSED", "Amber"
        elif state.online:
            core_text, core_obj = "RUNNING", "Positive"
        elif state.state == "ERROR":
            core_text, core_obj = "ERROR", "StatusOffline"
        elif state.state == "COMPLETED":
            core_text, core_obj = "READY", "Positive"
        else:
            core_text, core_obj = "READY", "Cyan"
        self.core_state.setText(core_text)
        self.core_state.setObjectName(core_obj)
        self.core_state.style().unpolish(self.core_state)
        self.core_state.style().polish(self.core_state)
        self.side_pulse.set_active(state.online and not state.paused)

        if state.online and state.paused:
            bottom = "CORE LINK / PAUSED"
        elif state.online:
            bottom = f"CORE LINK / {state.mode} ONLINE"
        elif state.state == "COMPLETED" and state.mode == "BACKTEST":
            bottom = "CORE LINK / BACKTEST COMPLETE"
        elif state.state == "ERROR":
            bottom = "CORE LINK / ENGINE ERROR · CHECK RUNTIME CONSOLE"
        else:
            bottom = "CORE LINK / TELEMETRY ONLY"
        self.bottom_state.setText(bottom)

        # Runtime control state is authoritative while an engine session exists.
        # This is especially important for temporary UI-launched BACKTEST runs,
        # where the persisted settings MODE may still be DEMO_FORWARD.
        if state.online or state.state in {"COMPLETED", "ERROR", "STOPPED"}:
            self.mode_pill.setText(state.mode or self.control.mode())

        self.overview.update_engine_state(state)
        current = self.stack.currentWidget()
        if current in {self.control_page, self.backtest, self.settings_page}:
            refresh = getattr(current, "refresh", None)
            if callable(refresh):
                refresh()

    def apply_snapshot(self, snapshot: RuntimeSnapshot):
        self._latest_snapshot = snapshot
        control_state = self.control.status()
        if control_state.online or control_state.state in {"COMPLETED", "ERROR", "STOPPED"}:
            self.mode_pill.setText(control_state.mode or snapshot.mode or "UNKNOWN")
        else:
            self.mode_pill.setText(snapshot.mode or "UNKNOWN")
        self.mt5_pill.set_semantic("success" if snapshot.mt5_connected else "danger", "● MT5 ONLINE" if snapshot.mt5_connected else "● MT5 OFFLINE")
        self.spread_label.setText(
            "SPREAD —" if snapshot.chart_spread_pips is None else f"SPREAD {snapshot.chart_spread_pips:.1f}p"
        )
        self.overview.update_snapshot(snapshot)
        self.control_page.update_snapshot(snapshot)
        self.risk.update_snapshot(snapshot)

    def _worker_error(self, message: str):
        # Keep raw exceptions out of the product chrome. The Runtime Console is
        # the correct place for diagnostics; the global shell only shows state.
        self.mt5_pill.set_semantic("danger", "● DATA ERROR")
        self.bottom_state.setText("CORE LINK / DATA BRIDGE ERROR · CHECK RUNTIME CONSOLE")

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2500)
        super().closeEvent(event)
