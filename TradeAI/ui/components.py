from __future__ import annotations

"""Reusable ELVARA V3 UI building blocks used by TradeAI.

The brand guide separates buttons, inputs, navigation, tables, status/feedback,
and overlays into one component library.  This module gives the desktop product
those same primitives instead of styling one-off widgets page by page.
"""

from PySide6.QtCore import QDateTime, QEvent, QPointF, QRectF, QSize, QTimer, Signal, Qt
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from .icons import svg_icon
from .theme import theme_palette


class ElvaraButton(QPushButton):
    """ELVARA action button: primary, secondary, ghost, danger, or icon."""

    OBJECTS = {
        "primary": "PrimaryButton",
        "secondary": "SecondaryButton",
        "ghost": "GhostButton",
        "danger": "DangerButton",
        "icon": "IconButton",
    }
    HEIGHTS = {"compact": 32, "standard": 40, "large": 48}

    def __init__(
        self,
        text: str = "",
        *,
        variant: str = "secondary",
        size: str = "standard",
        icon_name: str | None = None,
        parent=None,
    ):
        super().__init__(text, parent)
        self.variant = variant if variant in self.OBJECTS else "secondary"
        self.size_name = size if size in self.HEIGHTS else "standard"
        self.icon_name = icon_name
        self._base_text = text
        self._loading = False
        self.setObjectName(self.OBJECTS[self.variant])
        self.setFixedHeight(self.HEIGHTS[self.size_name])
        self.setCursor(Qt.PointingHandCursor)
        if self.variant == "icon":
            self.setFixedWidth(self.HEIGHTS[self.size_name])
        self.refresh_theme()

    def refresh_theme(self) -> None:
        p = theme_palette()
        self.setCursor(Qt.PointingHandCursor if self.isEnabled() else Qt.ArrowCursor)
        if not self.icon_name:
            return
        if not self.isEnabled():
            color = str(p["muted"])
        elif self.variant == "primary":
            color = "#FFFFFF"
        elif self.variant == "danger":
            color = str(p["danger"])
        elif self.variant == "ghost":
            color = str(p["text_soft"])
        else:
            color = str(p["accent2"])
        icon_size = 18 if self.size_name != "compact" else 16
        self.setIcon(svg_icon(self.icon_name, color, icon_size))
        self.setIconSize(QSize(icon_size, icon_size))

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self.refresh_theme()

    def set_loading(self, loading: bool, text: str | None = None) -> None:
        self._loading = bool(loading)
        self.setEnabled(not loading)
        if loading:
            self.setText(text or "Loading...")
        else:
            self.setText(text or self._base_text)


class ElvaraWindowButton(QPushButton):
    """Compact vector-based window control for the frameless TradeAI shell."""

    ICONS = {
        "minimize": "window_minimize",
        "maximize": "window_maximize",
        "restore": "window_restore",
        "close": "window_close",
    }

    def __init__(self, control: str, parent=None):
        super().__init__(parent)
        self.control = control if control in self.ICONS else "minimize"
        self.setObjectName("WindowCloseControl" if self.control == "close" else "WindowControl")
        self.setFixedSize(38, 30)
        self.setFocusPolicy(Qt.NoFocus)
        self.setCursor(Qt.PointingHandCursor)
        self.refresh_theme()

    def set_control(self, control: str) -> None:
        if control not in self.ICONS:
            return
        self.control = control
        self.setObjectName("WindowCloseControl" if control == "close" else "WindowControl")
        self.refresh_theme()
        self.style().unpolish(self)
        self.style().polish(self)

    def refresh_theme(self) -> None:
        p = theme_palette()
        color = str(p["text_soft"])
        self.setIcon(svg_icon(self.ICONS[self.control], color, 15))
        self.setIconSize(QSize(15, 15))



class ElvaraNavButton(QPushButton):
    """40 px navigation row following the ELVARA left-rail anatomy."""

    def __init__(self, text: str, icon_name: str, parent=None):
        super().__init__("  " + text, parent)
        self.icon_name = icon_name
        self.setObjectName("NavButton")
        self.setCheckable(True)
        self.setFixedHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        self.setIcon(svg_icon(self.icon_name, str(theme_palette()["nav_icon"]), 20))


class ElvaraStatusBadge(QLabel):
    OBJECTS = {
        "success": "StatusOnline",
        "danger": "StatusOffline",
        "warning": "StatusWarning",
        "info": "StatusInfo",
        "accent": "StatusMode",
        "neutral": "StatusNeutral",
    }

    def __init__(self, text: str = "", semantic: str = "neutral", parent=None):
        super().__init__(text, parent)
        self.set_semantic(semantic)

    def set_semantic(self, semantic: str, text: str | None = None) -> None:
        if text is not None:
            self.setText(text)
        self.setObjectName(self.OBJECTS.get(semantic, "StatusNeutral"))
        self.style().unpolish(self)
        self.style().polish(self)


class ElvaraSearchField(QLineEdit):
    """Standard 40 px search input with leading search and native clear action."""

    def __init__(self, placeholder: str = "Search...", parent=None):
        super().__init__(parent)
        self.setObjectName("SearchField")
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setFixedHeight(40)
        self._search_action = QAction(self)
        self.addAction(self._search_action, QLineEdit.ActionPosition.LeadingPosition)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        self._search_action.setIcon(svg_icon("search", str(theme_palette()["text_soft"]), 17))


class ElvaraFilterChip(QPushButton):
    def __init__(self, text: str, *, checked: bool = False, parent=None):
        super().__init__(text, parent)
        self.setObjectName("FilterChip")
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)


class ElvaraTable(QTableWidget):
    """ELVARA data table with guide-defined 56 / 44 / 32 px density modes."""

    DENSITY_HEIGHT = {"comfortable": 56, "standard": 44, "compact": 32}

    def __init__(self, columns: list[str], *, density: str = "compact", parent=None):
        super().__init__(0, len(columns), parent)
        self.setHorizontalHeaderLabels(columns)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setFixedHeight(48)
        self.setFocusPolicy(Qt.StrongFocus)
        self.set_density(density)

    def set_density(self, density: str) -> None:
        value = self.DENSITY_HEIGHT.get(density, self.DENSITY_HEIGHT["compact"])
        self.verticalHeader().setDefaultSectionSize(value)


class ElvaraNotice(QFrame):
    """Persistent inline status/feedback surface."""

    OBJECTS = {
        "info": "NoticeInfo",
        "success": "NoticeSuccess",
        "warning": "NoticeWarning",
        "danger": "NoticeDanger",
    }

    def __init__(self, title: str = "", message: str = "", semantic: str = "info", parent=None):
        super().__init__(parent)
        self.setObjectName(self.OBJECTS.get(semantic, "NoticeInfo"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("NoticeTitle")
        self.body_label = QLabel(message)
        self.body_label.setObjectName("NoticeBody")
        self.body_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.body_label)

    def set_message(self, message: str, *, title: str | None = None, semantic: str = "info") -> None:
        if title is not None:
            self.title_label.setText(title)
        self.body_label.setText(message)
        self.setObjectName(self.OBJECTS.get(semantic, "NoticeInfo"))
        self.style().unpolish(self)
        self.style().polish(self)


class ElvaraEmptyState(QFrame):
    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        body = QLabel(message)
        body.setObjectName("PageSub")
        body.setWordWrap(True)
        layout.addStretch()
        layout.addWidget(title_label, 0, Qt.AlignCenter)
        layout.addWidget(body, 0, Qt.AlignCenter)
        layout.addStretch()


class ElvaraAlertDialog(QDialog):
    """Single-action ELVARA alert dialog for info, warning, and error feedback."""

    def __init__(self, title: str, message: str, *, semantic: str = "info", parent=None):
        super().__init__(parent)
        self.setObjectName("ElvaraDialog")
        self.setModal(True)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.setMaximumWidth(620)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        root.addWidget(heading)
        notice = ElvaraNotice("", message, semantic=semantic)
        notice.title_label.hide()
        root.addWidget(notice)

        actions = QHBoxLayout()
        actions.addStretch()
        close = ElvaraButton("Close", variant="primary", size="standard")
        close.clicked.connect(self.accept)
        actions.addWidget(close)
        root.addLayout(actions)

    @classmethod
    def show_alert(cls, parent: QWidget, title: str, message: str, *, semantic: str = "info") -> None:
        cls(title, message, semantic=semantic, parent=parent).exec()


class ElvaraConfirmDialog(QDialog):
    """Focused destructive confirmation using the ELVARA overlay anatomy."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        confirm_text: str = "Confirm",
        danger: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("ElvaraDialog")
        self.setModal(True)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setMaximumWidth(640)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        heading = QLabel(title)
        heading.setObjectName("PageTitle")
        root.addWidget(heading)
        body = QLabel(message)
        body.setObjectName("PageSub")
        body.setWordWrap(True)
        root.addWidget(body)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel = ElvaraButton("Cancel", variant="secondary", size="standard")
        confirm = ElvaraButton(confirm_text, variant="danger" if danger else "primary", size="standard")
        cancel.clicked.connect(self.reject)
        confirm.clicked.connect(self.accept)
        actions.addWidget(cancel)
        actions.addWidget(confirm)
        root.addLayout(actions)

    @classmethod
    def confirm(
        cls,
        parent: QWidget,
        title: str,
        message: str,
        *,
        confirm_text: str = "Confirm",
        danger: bool = False,
    ) -> bool:
        dialog = cls(title, message, confirm_text=confirm_text, danger=danger, parent=parent)
        return dialog.exec() == QDialog.Accepted


class ElvaraCloseChoiceDialog(QDialog):
    """Three-way window-close choice for a running detachable engine."""

    def __init__(self, *, mode: str, pid: int | None, parent=None):
        super().__init__(parent)
        self.setObjectName("ElvaraDialog")
        self.setModal(True)
        self.setWindowTitle("Close TradeAI")
        self.setMinimumWidth(560)
        self.setMaximumWidth(700)
        self.choice = "cancel"

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        heading = QLabel("Close TradeAI")
        heading.setObjectName("PageTitle")
        root.addWidget(heading)

        body = QLabel(
            f"Engine is RUNNING in {str(mode or 'UNKNOWN')} mode"
            + (f" · PID {pid}" if pid else "")
            + ".\n\nChoose whether to keep the trading engine running in the background or stop it before closing the dashboard."
        )
        body.setObjectName("PageSub")
        body.setWordWrap(True)
        root.addWidget(body)

        actions = QHBoxLayout()
        keep = ElvaraButton("KEEP ENGINE RUNNING", variant="primary", size="standard")
        stop = ElvaraButton("STOP ENGINE & CLOSE", variant="danger", size="standard")
        cancel = ElvaraButton("CANCEL", variant="secondary", size="standard")
        keep.clicked.connect(lambda: self._finish("keep"))
        stop.clicked.connect(lambda: self._finish("stop"))
        cancel.clicked.connect(lambda: self._finish("cancel"))
        actions.addWidget(keep)
        actions.addWidget(stop)
        actions.addStretch()
        actions.addWidget(cancel)
        root.addLayout(actions)

    def _finish(self, choice: str) -> None:
        self.choice = choice
        self.accept() if choice != "cancel" else self.reject()

    @classmethod
    def choose(cls, parent: QWidget, *, mode: str, pid: int | None) -> str:
        dialog = cls(mode=mode, pid=pid, parent=parent)
        dialog.exec()
        return dialog.choice


class ElvaraActivePositionCloseDialog(QDialog):
    """Second-stage safety choice when stopping would abandon open MT5 positions."""

    def __init__(self, positions: int, parent=None):
        super().__init__(parent)
        self.setObjectName("ElvaraDialog")
        self.setModal(True)
        self.setWindowTitle("Active position detected")
        self.setMinimumWidth(580)
        self.setMaximumWidth(720)
        self.choice = "cancel"

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(16)

        heading = QLabel("Active position detected")
        heading.setObjectName("PageTitle")
        root.addWidget(heading)

        notice = ElvaraNotice(
            "Engine management is active",
            f"{positions} MT5 position{'s are' if positions != 1 else ' is'} currently open. "
            "Stopping TradeAI stops strategy management but does NOT close broker positions.",
            semantic="warning",
        )
        root.addWidget(notice)

        actions = QHBoxLayout()
        keep = ElvaraButton("KEEP ENGINE RUNNING & CLOSE UI", variant="primary", size="standard")
        stop = ElvaraButton("STOP ENGINE ANYWAY", variant="danger", size="standard")
        cancel = ElvaraButton("CANCEL", variant="secondary", size="standard")
        keep.clicked.connect(lambda: self._finish("keep"))
        stop.clicked.connect(lambda: self._finish("stop"))
        cancel.clicked.connect(lambda: self._finish("cancel"))
        actions.addWidget(keep)
        actions.addWidget(stop)
        actions.addStretch()
        actions.addWidget(cancel)
        root.addLayout(actions)

    def _finish(self, choice: str) -> None:
        self.choice = choice
        self.accept() if choice != "cancel" else self.reject()

    @classmethod
    def choose(cls, parent: QWidget, positions: int) -> str:
        dialog = cls(positions, parent=parent)
        dialog.exec()
        return dialog.choice


class NiraMiniOrb(QWidget):
    """Small animated NIRA presence mark used by the global sidebar dock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self.setFixedSize(34, 34)
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.075) % 6.283185307179586
        self.update()

    def paintEvent(self, event) -> None:
        """An understated, animated NIRA presence glyph (not a second body)."""
        import math

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            palette = theme_palette()
            center = QPointF(self.width() / 2.0, self.height() / 2.0)
            radius = min(self.width(), self.height()) * 0.49
            pulse = 0.5 + 0.5 * math.sin(self._phase * 1.7)

            glow = QRadialGradient(center, radius)
            glow_core = QColor(str(palette["ai"]))
            glow_core.setAlpha(120 + int(35 * pulse))
            glow_edge = QColor(str(palette["accent2"]))
            glow_edge.setAlpha(0)
            glow.setColorAt(0.0, glow_core)
            glow.setColorAt(0.5, QColor(72, 107, 202, 35))
            glow.setColorAt(1.0, glow_edge)
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow)
            painter.drawEllipse(center, radius, radius)

            base = QColor(str(palette["ai"]))
            base.setAlpha(55)
            painter.setBrush(base)
            painter.drawEllipse(center, radius * .51, radius * .51)

            ring = QColor(str(palette["accent2"]))
            ring.setAlpha(220)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(ring, max(1.2, radius * .095)))
            painter.drawArc(
                QRectF(center.x() - radius * .72, center.y() - radius * .72,
                       radius * 1.44, radius * 1.44),
                int(math.degrees(self._phase) * 16), 214 * 16,
            )
            muted_ring = QColor(str(palette["ai"]))
            muted_ring.setAlpha(135)
            painter.setPen(QPen(muted_ring, max(1.0, radius * .065)))
            painter.drawArc(
                QRectF(center.x() - radius * .90, center.y() - radius * .90,
                       radius * 1.80, radius * 1.80),
                -int(math.degrees(self._phase) * 11), 110 * 16,
            )

            core = QRadialGradient(center, radius * .36)
            core.setColorAt(0.0, QColor(str(palette["frost"])))
            core.setColorAt(0.7, QColor(str(palette["accent2"])))
            core.setColorAt(1.0, QColor(str(palette["ai"])))
            painter.setPen(Qt.NoPen)
            painter.setBrush(core)
            painter.drawEllipse(center, radius * .30, radius * .30)
        finally:
            painter.end()


class NiraTopBar(QFrame):
    """One-line entry point to the ONE NIRA runtime, not an in-app chatbot."""

    message_submitted = Signal(str)
    panel_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NiraTopBar")
        self.setFixedHeight(46)
        self._busy = False
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 5, 9, 5)
        root.setSpacing(9)

        self.orb = NiraMiniOrb(self)
        self.orb.setFixedSize(30, 30)
        root.addWidget(self.orb)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("NIRA  /  ASSISTANT")
        title.setObjectName("NiraDockTitle")
        title_box.addWidget(title)
        self.state = QLabel("CONNECTING")
        self.state.setObjectName("NiraDockState")
        title_box.addWidget(self.state)
        root.addLayout(title_box)

        self.input = QLineEdit()
        self.input.setObjectName("NiraTopInput")
        self.input.setPlaceholderText("Ask NIRA about your trading workspace…")
        self.input.setToolTip("Ask the existing NIRA runtime via the local ELVARA bridge")
        self.input.setFixedHeight(34)
        self.input.setMaxLength(12000)
        root.addWidget(self.input, 1)

        self.panel = QPushButton("OPEN CHAT")
        self.panel.setObjectName("NiraChatOpen")
        self.panel.setFixedHeight(34)
        self.panel.setToolTip("Expand NIRA's TradeAI conversation")
        self.panel.setCursor(Qt.PointingHandCursor)
        root.addWidget(self.panel)

        self.send = QPushButton("➜")
        self.send.setObjectName("NiraTopSend")
        self.send.setFixedSize(34, 34)
        self.send.setToolTip("Send message to NIRA")
        self.send.setCursor(Qt.PointingHandCursor)
        root.addWidget(self.send)

        self.input.returnPressed.connect(self._submit)
        self.send.clicked.connect(self._submit)
        self.panel.clicked.connect(self.panel_toggled.emit)

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _submit(self) -> None:
        if self._busy:
            return
        message = self.input.text().strip()
        if not message:
            self.input.setFocus()
            return
        self.set_busy(True)
        self.state.setText("THINKING")
        self.message_submitted.emit(message)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.input.setEnabled(not self._busy)
        self.send.setEnabled(not self._busy)
        self.send.setText("…" if self._busy else "➜")

    def finish_request(self, *, state: str, clear_input: bool = False) -> None:
        self.set_busy(False)
        self.state.setText(state)
        if clear_input:
            self.input.clear()

    def set_bridge_state(self, state: str) -> None:
        if not self._busy:
            self.state.setText(str(state or "").strip().upper() or "OFFLINE")


class NiraComposer(QPlainTextEdit):
    """Enter submits. Shift+Enter makes a new line."""

    submitted = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier | Qt.AltModifier)
        ):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class NiraMessageCard(QFrame):
    """A selectable, word-wrapped conversation turn with a copy action."""

    def __init__(self, speaker: str, message: str, parent=None):
        super().__init__(parent)
        role = {"YOU": "user", "NIRA": "assistant", "BRIDGE": "error"}.get(speaker, "assistant")
        self.setObjectName("NiraMessageCard")
        self.setProperty("role", role)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(5)
        author = QLabel(speaker)
        author.setObjectName("NiraMessageAuthor")
        author.setProperty("role", role)
        header.addWidget(author)
        header.addStretch()
        stamp = QLabel(QDateTime.currentDateTime().toString("hh:mm"))
        stamp.setObjectName("NiraMessageTime")
        header.addWidget(stamp)
        copy = QPushButton("COPY")
        copy.setObjectName("NiraMessageCopy")
        copy.setToolTip("Copy message text")
        copy.setCursor(Qt.PointingHandCursor)
        copy.clicked.connect(lambda: QApplication.clipboard().setText(message))
        header.addWidget(copy)
        layout.addLayout(header)

        body = QLabel(message)
        body.setObjectName("NiraMessageText")
        body.setTextFormat(Qt.MarkdownText if role == "assistant" else Qt.PlainText)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        body.setOpenExternalLinks(False)
        body.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout.addWidget(body)


class NiraSidebarDock(QFrame):
    """TradeAI-scoped UI over the existing authenticated NIRA bridge."""

    close_requested = Signal()
    popout_requested = Signal()
    message_submitted = Signal(str)
    MAX_VISIBLE_MESSAGES = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NiraSidebarDock")
        self.setMinimumWidth(335)
        self.setMaximumWidth(440)
        self._busy = False
        self._floating = False
        self._messages: list[NiraMessageCard] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(13, 14, 13, 13)
        root.setSpacing(12)

        head = QHBoxLayout()
        head.setSpacing(9)
        self.orb = NiraMiniOrb(self)
        self.orb.setFixedSize(36, 36)
        head.addWidget(self.orb)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        heading = QLabel("Ask NIRA")
        heading.setObjectName("NiraChatTitle")
        titles.addWidget(heading)
        subtitle = QLabel("Your intelligence, inside TradeAI")
        subtitle.setObjectName("NiraChatSubtitle")
        titles.addWidget(subtitle)
        head.addLayout(titles, 1)

        self.popout_button = QPushButton("EXPAND")
        self.popout_button.setObjectName("NiraChatTool")
        self.popout_button.setCursor(Qt.PointingHandCursor)
        self.popout_button.setToolTip("Open this conversation in a larger window")
        self.popout_button.clicked.connect(self.popout_requested.emit)
        head.addWidget(self.popout_button)

        close = QPushButton("×")
        close.setObjectName("NiraChatClose")
        close.setFixedSize(28, 28)
        close.setCursor(Qt.PointingHandCursor)
        close.setToolTip("Close this conversation")
        close.clicked.connect(self.close_requested.emit)
        head.addWidget(close)
        root.addLayout(head)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.state = QLabel("CONNECTING")
        self.state.setObjectName("NiraChatStatus")
        self.state.setProperty("connection", "connecting")
        status_row.addWidget(self.state)
        status_row.addStretch()
        self.context = QLabel("Command Center")
        self.context.setObjectName("NiraContextTag")
        status_row.addWidget(self.context)
        root.addLayout(status_row)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("NiraConversationScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content.setObjectName("NiraConversationContent")
        self.turns = QVBoxLayout(content)
        self.turns.setContentsMargins(2, 6, 2, 8)
        self.turns.setSpacing(12)

        self.welcome = QFrame()
        self.welcome.setObjectName("NiraWelcome")
        intro = QVBoxLayout(self.welcome)
        intro.setContentsMargins(16, 17, 16, 17)
        intro.setSpacing(10)
        eyebrow = QLabel("ELVARA  /  NIRA")
        eyebrow.setObjectName("NiraWelcomeEyebrow")
        intro.addWidget(eyebrow)
        headline = QLabel("Clarity for every trading decision.")
        headline.setObjectName("NiraWelcomeHeadline")
        headline.setWordWrap(True)
        intro.addWidget(headline)
        description = QLabel(
            "Ask about live status, signals, positions and performance. "
            "NIRA reads authoritative TradeAI data through the approved connector."
        )
        description.setObjectName("NiraWelcomeDescription")
        description.setWordWrap(True)
        intro.addWidget(description)
        suggestions = (
            ("System overview", "Give me a concise overview of TradeAI's current system status."),
            ("Latest signals", "Explain the latest TradeAI signals and their status."),
            ("Open positions", "Show and explain the currently open TradeAI positions."),
        )
        for label, prompt in suggestions:
            chip = QPushButton("↗  " + label)
            chip.setObjectName("NiraSuggestion")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setToolTip(prompt)
            chip.clicked.connect(lambda _checked=False, text=prompt: self._dispatch(text))
            intro.addWidget(chip)
        self.turns.addWidget(self.welcome)
        self.turns.addStretch(1)
        self.scroll.setWidget(content)
        root.addWidget(self.scroll, 1)

        composer = QFrame()
        composer.setObjectName("NiraComposerShell")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(10, 9, 10, 9)
        composer_layout.setSpacing(7)
        self.input = NiraComposer()
        self.input.setObjectName("NiraComposer")
        self.input.setPlaceholderText("Message NIRA about TradeAI…")
        self.input.setFixedHeight(67)
        self.input.setTabChangesFocus(True)
        self.input.setToolTip("Enter to send · Shift+Enter for a new line")
        composer_layout.addWidget(self.input)
        actions = QHBoxLayout()
        hint = QLabel("ENTER  ↵  SEND   ·   SHIFT+ENTER  NEW LINE")
        hint.setObjectName("NiraComposerHint")
        actions.addWidget(hint)
        actions.addStretch()
        self.send_button = QPushButton("Send  ➜")
        self.send_button.setObjectName("NiraComposerSend")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.clicked.connect(self._submit)
        actions.addWidget(self.send_button)
        composer_layout.addLayout(actions)
        root.addWidget(composer)
        self.input.submitted.connect(self._submit)

        note = QLabel("ONE NIRA RUNTIME    ·    TRADEAI SCOPE    ·    READ ONLY")
        note.setObjectName("NiraSecurityNote")
        note.setAlignment(Qt.AlignCenter)
        root.addWidget(note)

    def set_floating(self, floating: bool) -> None:
        self._floating = bool(floating)
        self.setMaximumWidth(16777215 if floating else 440)
        self.popout_button.setText("DOCK" if floating else "EXPAND")
        self.popout_button.setToolTip(
            "Return NIRA to the dashboard" if floating else "Open NIRA in a larger window"
        )

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.input.setEnabled(not self._busy)
        self.send_button.setEnabled(not self._busy)
        self.send_button.setText("Thinking…" if self._busy else "Send  ➜")

    def set_state(self, state: str) -> None:
        clean = str(state or "").strip().upper() or "OFFLINE"
        self.state.setText("● " + clean)
        self.state.setProperty("connection", clean.lower())
        self.state.style().unpolish(self.state)
        self.state.style().polish(self.state)

    def set_context(self, page: str, selected_entity: str = "") -> None:
        page_text = str(page or "overview").strip().replace("_", " ").title()
        entity = str(selected_entity or "").strip()
        self.context.setText(f"{page_text}  /  {entity}" if entity else page_text)
        self.context.setToolTip("Page and selection are context, not authoritative trading data")

    def _submit(self) -> None:
        self._dispatch(self.input.toPlainText())

    def _dispatch(self, message: str) -> None:
        clean = str(message or "").strip()
        if self._busy or not clean:
            return
        if len(clean) > 12000:
            self._append_message("BRIDGE", "Message exceeds NIRA's 12,000-character limit.")
            return
        self.input.clear()
        self.message_submitted.emit(clean)

    def add_user_message(self, message: str) -> None:
        self._append_message("YOU", message)

    def add_nira_message(self, message: str) -> None:
        self._append_message("NIRA", message)

    def add_error_message(self, message: str) -> None:
        self._append_message("BRIDGE", message)

    def _append_message(self, speaker: str, message: str) -> None:
        clean = str(message or "").strip()
        if not clean:
            return
        self.welcome.hide()
        if len(self._messages) >= self.MAX_VISIBLE_MESSAGES:
            old = self._messages.pop(0)
            self.turns.removeWidget(old.parentWidget())
            old.parentWidget().deleteLater()

        row = QWidget()
        row.setObjectName("NiraTurnRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        card = NiraMessageCard(speaker, clean, row)
        if speaker == "YOU":
            row_layout.addStretch(1)
            row_layout.addWidget(card, 5)
        else:
            row_layout.addWidget(card, 5)
            row_layout.addStretch(1)
        self.turns.insertWidget(self.turns.count() - 1, row)
        self._messages.append(card)
        QTimer.singleShot(0, self._scroll_bottom)

    def _scroll_bottom(self) -> None:
        scroll = self.scroll.verticalScrollBar()
        scroll.setValue(scroll.maximum())
