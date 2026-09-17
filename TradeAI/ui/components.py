from __future__ import annotations

"""Reusable ELVARA V3 UI building blocks used by TradeAI.

The brand guide separates buttons, inputs, navigation, tables, status/feedback,
and overlays into one component library.  This module gives the desktop product
those same primitives instead of styling one-off widgets page by page.
"""

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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
        import math

        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            palette = theme_palette()
            cx = self.width() / 2.0
            cy = self.height() / 2.0
            glow = QRadialGradient(QPointF(cx, cy), 16.0)
            c0 = QColor(str(palette["accent2"])); c0.setAlpha(148)
            c1 = QColor(str(palette["accent"])); c1.setAlpha(62)
            c2 = QColor(str(palette["accent"])); c2.setAlpha(0)
            glow.setColorAt(0.0, c0)
            glow.setColorAt(0.45, c1)
            glow.setColorAt(1.0, c2)
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            p.drawEllipse(QPointF(cx, cy), 16.0, 16.0)

            ring = QColor(str(palette["accent2"])); ring.setAlpha(205)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(ring, 1.2))
            p.drawArc(QRectF(cx - 11.5, cy - 11.5, 23.0, 23.0), int(self._phase * 34 * 16), 226 * 16)

            core = QColor(str(palette["frost"])); core.setAlpha(235)
            p.setPen(Qt.NoPen)
            p.setBrush(core)
            p.drawEllipse(QPointF(cx, cy), 2.0, 2.0)
        finally:
            p.end()


class NiraTopBar(QFrame):
    """Global reserved NIRA command surface shown directly below the title bar.

    This is deliberately a visual reservation only.  It never invents an AI
    response.  When the NIRA runtime is connected later, the shell does not
    need to move again.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NiraTopBar")
        self.setFixedHeight(40)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(7)

        self.orb = NiraMiniOrb(self)
        self.orb.setFixedSize(28, 28)
        root.addWidget(self.orb)

        title = QLabel("NIRA")
        title.setObjectName("NiraDockTitle")
        root.addWidget(title)

        self.state = QLabel("RESERVED")
        self.state.setObjectName("NiraDockState")
        root.addWidget(self.state)

        self.input = QLineEdit()
        self.input.setObjectName("NiraTopInput")
        self.input.setPlaceholderText("Ask NIRA…")
        self.input.setToolTip("NIRA conversational runtime is reserved for a future connection.")
        self.input.setFixedHeight(30)
        root.addWidget(self.input, 1)

        self.send = QPushButton("›")
        self.send.setObjectName("NiraTopSend")
        self.send.setFixedSize(30, 30)
        self.send.setToolTip("NIRA bridge reserved")
        root.addWidget(self.send)

        self.input.returnPressed.connect(self._reserved_action)
        self.send.clicked.connect(self._reserved_action)

        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self._reset_state)

    def _reserved_action(self) -> None:
        self.state.setText("BRIDGE RESERVED")
        self._reset_timer.start(1600)

    def _reset_state(self) -> None:
        self.state.setText("RESERVED")


class NiraSidebarDock(QFrame):
    """Global placeholder for the future NIRA conversational bridge.

    It intentionally does not fabricate assistant replies.  The control remains
    visible everywhere so the eventual NIRA runtime can be connected without
    redesigning the product shell.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("NiraSidebarDock")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(6)
        self.orb = NiraMiniOrb(self)
        head.addWidget(self.orb)

        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel("NIRA")
        title.setObjectName("NiraDockTitle")
        self.state = QLabel("GLOBAL INTERFACE · RESERVED")
        self.state.setObjectName("NiraDockState")
        text.addWidget(title)
        text.addWidget(self.state)
        head.addLayout(text, 1)
        root.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.input = QLineEdit()
        self.input.setObjectName("NiraDockInput")
        self.input.setPlaceholderText("Ask NIRA…")
        self.input.setToolTip("NIRA conversation bridge is reserved for a future runtime connection.")
        self.input.setFixedHeight(31)
        self.send = QPushButton("›")
        self.send.setObjectName("NiraDockSend")
        self.send.setFixedSize(31, 31)
        self.send.setToolTip("Reserved NIRA action")
        row.addWidget(self.input, 1)
        row.addWidget(self.send)
        root.addLayout(row)

        self.input.returnPressed.connect(self._reserved_action)
        self.send.clicked.connect(self._reserved_action)
        self._reset_timer = QTimer(self)
        self._reset_timer.setSingleShot(True)
        self._reset_timer.timeout.connect(self._reset_state)

    def _reserved_action(self) -> None:
        self.state.setText("NIRA BRIDGE · COMING LATER")
        self._reset_timer.start(1800)

    def _reset_state(self) -> None:
        self.state.setText("GLOBAL INTERFACE · RESERVED")
