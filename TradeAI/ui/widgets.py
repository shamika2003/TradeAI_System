from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QRadialGradient,
    QPixmap,
)
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from .icons import svg_icon
from .theme import theme_mode, theme_palette


CYAN = "#00E5FF"
GREEN = "#22D3A7"
RED = "#F87171"
BLUE = "#1468FF"
VIOLET = "#8B5CFF"
AMBER = "#F8BF24"
MUTED = "#94A3B8"


def _role_for_color(color: str | None) -> str | None:
    if not color:
        return None
    mapping = {
        CYAN.upper(): "accent2",
        BLUE.upper(): "accent",
        GREEN.upper(): "success",
        RED.upper(): "danger",
        AMBER.upper(): "warning",
        VIOLET.upper(): "ai",
    }
    return mapping.get(str(color).upper())


def _theme_color(color: str | None, fallback_role: str = "accent") -> str:
    p = theme_palette()
    role = _role_for_color(color)
    return str(p.get(role or fallback_role, color or p[fallback_role]))


class AmbientCanvas(QWidget):
    """Celestial ELVARA working canvas.

    The root brand image is treated as atmosphere, never as a full wallpaper.
    It is cropped into the upper-right field, faded into the UI, and combined
    with code-rendered orbital arcs and a quiet lower horizon.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phase = 0.0
        self._sweep = 0.0
        self._pixmap: QPixmap | None = None
        self._loaded_mode = ""
        self._stars = []
        # Deterministic points so the background never visually jumps.
        seed = 0xE1A4A
        value = seed
        for _ in range(72):
            value = (1103515245 * value + 12345) & 0x7FFFFFFF
            x = (value % 10000) / 10000.0
            value = (1103515245 * value + 12345) & 0x7FFFFFFF
            y = (value % 10000) / 10000.0
            value = (1103515245 * value + 12345) & 0x7FFFFFFF
            radius = 0.35 + (value % 90) / 100.0
            alpha = 22 + (value % 72)
            self._stars.append((x, y, radius, alpha))

        self._timer = QTimer(self)
        self._timer.setInterval(48)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.refresh_theme()

    @staticmethod
    def _system_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _asset_for_mode(self, mode: str) -> QPixmap | None:
        stem = "elvara-light" if mode == "halo" else "elvara-dark"
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = self._system_root() / f"{stem}{ext}"
            if path.exists():
                pix = QPixmap(str(path))
                if not pix.isNull():
                    return pix
        return None

    def refresh_theme(self) -> None:
        mode = theme_mode()
        if mode != self._loaded_mode:
            self._pixmap = self._asset_for_mode(mode)
            self._loaded_mode = mode
        self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.018) % 6.283185307179586
        self._sweep = (self._sweep + 0.00115) % 1.0
        self.update()

    @staticmethod
    def _cover_crop(src: QPixmap, dst_w: int, dst_h: int) -> QPixmap:
        if src.isNull() or dst_w <= 0 or dst_h <= 0:
            return src
        src_ratio = src.width() / max(1, src.height())
        dst_ratio = dst_w / max(1, dst_h)
        if src_ratio > dst_ratio:
            crop_h = src.height()
            crop_w = max(1, int(crop_h * dst_ratio))
            x = max(0, (src.width() - crop_w) // 2)
            y = 0
        else:
            crop_w = src.width()
            crop_h = max(1, int(crop_w / dst_ratio))
            x = 0
            y = max(0, (src.height() - crop_h) // 2)
        return src.copy(x, y, crop_w, crop_h)

    def paintEvent(self, event) -> None:
        colors = theme_palette()
        dark = bool(colors.get("dark", True))
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            r = QRectF(self.rect())

            base = QLinearGradient(0, 0, r.width(), r.height())
            base.setColorAt(0.0, QColor(str(colors["bg1"])))
            base.setColorAt(0.55, QColor(str(colors["bg0"])))
            base.setColorAt(1.0, QColor(str(colors["bg2"])))
            painter.fillRect(r, QBrush(base))

            if self._pixmap is not None and not self._pixmap.isNull():
                image_rect = QRectF(
                    r.width() * 0.36,
                    -r.height() * 0.02,
                    r.width() * 0.68,
                    r.height() * 0.72,
                )
                crop = self._cover_crop(
                    self._pixmap,
                    max(1, int(image_rect.width())),
                    max(1, int(image_rect.height())),
                )
                painter.save()
                painter.setOpacity(0.72 if dark else 0.60)
                painter.drawPixmap(image_rect, crop, QRectF(crop.rect()))
                painter.restore()

                fade_x = QLinearGradient(image_rect.left(), 0, image_rect.right(), 0)
                left = QColor(str(colors["bg0"])); left.setAlpha(255)
                mid = QColor(str(colors["bg0"])); mid.setAlpha(150 if dark else 122)
                clear = QColor(str(colors["bg0"])); clear.setAlpha(0)
                fade_x.setColorAt(0.0, left)
                fade_x.setColorAt(0.25, mid)
                fade_x.setColorAt(0.63, clear)
                fade_x.setColorAt(1.0, clear)
                painter.fillRect(image_rect, QBrush(fade_x))

                fade_y = QLinearGradient(0, image_rect.top(), 0, image_rect.bottom())
                top_clear = QColor(str(colors["bg0"])); top_clear.setAlpha(0)
                bottom = QColor(str(colors["bg0"])); bottom.setAlpha(145 if dark else 110)
                fade_y.setColorAt(0.0, top_clear)
                fade_y.setColorAt(0.64, top_clear)
                fade_y.setColorAt(1.0, bottom)
                painter.fillRect(image_rect, QBrush(fade_y))

            # Soft intelligence bloom.
            glow = QRadialGradient(
                QPointF(r.width() * 0.79, r.height() * 0.07),
                max(r.width(), r.height()) * 0.42,
            )
            c0 = QColor(str(colors["accent"])); c0.setAlpha(42 if dark else 30)
            c1 = QColor(str(colors["accent2"])); c1.setAlpha(16 if dark else 12)
            c2 = QColor(str(colors["accent"])); c2.setAlpha(0)
            glow.setColorAt(0.0, c0)
            glow.setColorAt(0.28, c1)
            glow.setColorAt(1.0, c2)
            painter.fillRect(r, QBrush(glow))

            if dark:
                painter.setPen(Qt.NoPen)
                for sx, sy, radius, alpha in self._stars:
                    star = QColor("#E8F5FF")
                    star.setAlpha(alpha)
                    painter.setBrush(star)
                    painter.drawEllipse(
                        QPointF(sx * r.width(), sy * r.height()),
                        radius,
                        radius,
                    )

            # Orbital arcs are intentionally sparse.
            arc = QColor(str(colors["accent"])); arc.setAlpha(60 if dark else 34)
            painter.setPen(QPen(arc, 1.0))
            painter.drawArc(
                QRectF(r.width() * 0.58, -r.height() * 0.18, r.width() * 0.42, r.height() * 0.44),
                int(192 * 16),
                int(114 * 16),
            )
            arc2 = QColor(str(colors["accent2"])); arc2.setAlpha(38 if dark else 26)
            painter.setPen(QPen(arc2, 1.0))
            painter.drawArc(
                QRectF(r.width() * 0.67, -r.height() * 0.12, r.width() * 0.28, r.height() * 0.32),
                int(210 * 16),
                int(98 * 16),
            )

            # Very faint moving signal sweep.
            sx = self._sweep * r.width()
            sweep = QLinearGradient(sx - 50, 0, sx + 50, 0)
            transparent = QColor(str(colors["accent2"])); transparent.setAlpha(0)
            live = QColor(str(colors["accent2"])); live.setAlpha(20 if dark else 13)
            sweep.setColorAt(0.0, transparent)
            sweep.setColorAt(0.49, transparent)
            sweep.setColorAt(0.50, live)
            sweep.setColorAt(0.51, transparent)
            sweep.setColorAt(1.0, transparent)
            painter.fillRect(QRectF(sx - 50, 0, 100, r.height()), QBrush(sweep))

            # Lower celestial horizon.
            horizon = QPainterPath()
            horizon.moveTo(0, r.height())
            horizon.cubicTo(
                r.width() * 0.20, r.height() * 0.90,
                r.width() * 0.76, r.height() * 0.86,
                r.width(), r.height() * 0.93,
            )
            horizon.lineTo(r.width(), r.height())
            horizon.closeSubpath()
            hg = QLinearGradient(0, r.height() * 0.84, 0, r.height())
            h0 = QColor(str(colors["panel2"])); h0.setAlpha(105 if dark else 86)
            h1 = QColor(str(colors["bg0"])); h1.setAlpha(245 if dark else 168)
            hg.setColorAt(0.0, h0)
            hg.setColorAt(1.0, h1)
            painter.fillPath(horizon, QBrush(hg))
        finally:
            painter.end()


class HolographicShell(QWidget):
    """Transparent application shell.  The AmbientCanvas owns the atmosphere."""

    def paintEvent(self, event) -> None:
        colors = theme_palette()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            border = QColor(str(colors["border_soft"]))
            border.setAlpha(150)
            painter.setPen(QPen(border, 1.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(r, 12, 12)
        finally:
            painter.end()


class CompactStat(QFrame):
    def __init__(self, title: str, value: str = "—", detail: str = "", accent: str = CYAN, parent=None):
        super().__init__(parent)
        self.setObjectName("CompactStat")
        root = QVBoxLayout(self)
        root.setContentsMargins(9, 7, 9, 7)
        root.setSpacing(0)
        self.title = QLabel(title); self.title.setObjectName("MetricLabel")
        self.value = QLabel(value); self.value.setObjectName("CompactValue")
        self.detail = QLabel(detail); self.detail.setObjectName("MetricDetail")
        self._accent = accent
        self._override_color: str | None = None
        root.addWidget(self.title); root.addWidget(self.value); root.addWidget(self.detail)
        self.refresh_theme()

    def refresh_theme(self) -> None:
        color = _theme_color(self._override_color or self._accent, "accent")
        self.value.setStyleSheet(f"color:{color};")

    def set_value(self, value: str, detail: str | None = None, color: str | None = None):
        self.value.setText(value)
        self._override_color = color
        self.refresh_theme()
        if detail is not None:
            self.detail.setText(detail)


class DecisionPipeline(QFrame):
    """Trading-only AI gate visualization; no decorative blob/radar graphics."""

    STEPS = ("MODEL", "CONFIDENCE", "EDGE", "RISK", "EXECUTION")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PipelineFrame")
        root = QVBoxLayout(self)
        root.setContentsMargins(9, 8, 9, 8)
        root.setSpacing(5)
        self.rows = {}
        for index, name in enumerate(self.STEPS):
            row = QFrame(); row.setObjectName("PipelineStep")
            rl = QHBoxLayout(row); rl.setContentsMargins(8, 5, 8, 5); rl.setSpacing(7)
            idx = QLabel(f"0{index+1}"); idx.setObjectName("PipelineIndex")
            label = QLabel(name); label.setObjectName("PipelineName")
            value = QLabel("WAIT"); value.setObjectName("GateWait")
            rl.addWidget(idx); rl.addWidget(label); rl.addStretch(); rl.addWidget(value)
            root.addWidget(row)
            self.rows[name] = value

    def set_step(self, name: str, text: str, state: str = "wait") -> None:
        label = self.rows.get(name)
        if label is None:
            return
        label.setText(text)
        obj = {"pass":"GatePass", "fail":"StatusOffline", "active":"StatusMode"}.get(state, "GateWait")
        label.setObjectName(obj)
        label.style().unpolish(label); label.style().polish(label)

    def set_trade_state(
        self,
        signal: float | None,
        confidence: float | None,
        min_confidence: float,
        min_edge: float,
        risk_percent: float,
        engine_online: bool,
        enabled: bool = True,
    ) -> None:
        if not enabled:
            self.set_step("MODEL", "DISABLED", "fail")
            for step in self.STEPS[1:]: self.set_step(step, "BLOCKED", "fail")
            return
        if signal is None or confidence is None:
            self.set_step("MODEL", "WAITING", "active" if engine_online else "wait")
            self.set_step("CONFIDENCE", f">= {min_confidence*100:.0f}%", "wait")
            self.set_step("EDGE", f">= {min_edge:.3f}", "wait")
            self.set_step("RISK", f"{risk_percent:.2f}%", "pass" if risk_percent > 0 else "wait")
            self.set_step("EXECUTION", "ENGINE ON" if engine_online else "OFFLINE", "active" if engine_online else "wait")
            return
        direction = "BUY" if signal > 0 else "SELL" if signal < 0 else "HOLD"
        self.set_step("MODEL", direction, "pass" if direction != "HOLD" else "wait")
        conf_ok = confidence >= min_confidence
        edge_ok = abs(signal) >= min_edge
        self.set_step("CONFIDENCE", f"{confidence*100:.1f}%", "pass" if conf_ok else "fail")
        self.set_step("EDGE", f"{signal:+.3f}", "pass" if edge_ok else "fail")
        self.set_step("RISK", f"{risk_percent:.2f}%", "pass" if risk_percent > 0 else "wait")
        executable = engine_online and direction != "HOLD" and conf_ok and edge_ok
        self.set_step("EXECUTION", "ELIGIBLE" if executable else ("WAIT" if engine_online else "OFFLINE"), "pass" if executable else "wait")


class TradeAILogo(QWidget):
    def __init__(self, size: int = 38, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = self.width()
        h = self.height()
        grad = QLinearGradient(0, h, w, 0)
        colors = theme_palette()
        grad.setColorAt(0, QColor(str(colors["accent2"])))
        grad.setColorAt(0.58, QColor(str(colors["accent"])))
        grad.setColorAt(1, QColor(str(colors["halo"])))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        unit = w / 8.5
        base = h * 0.77
        for i, height in enumerate((0.26, 0.43, 0.60, 0.78)):
            x = w * 0.10 + i * unit * 1.35
            p.drawRoundedRect(QRectF(x, base - h * height, unit, h * height), 1.6, 1.6)
        pen = QPen(QColor(theme_palette()["accent"]), max(1.5, w * 0.055))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        path = QPainterPath()
        path.moveTo(w * 0.20, h * 0.55)
        path.lineTo(w * 0.43, h * 0.38)
        path.lineTo(w * 0.61, h * 0.45)
        path.lineTo(w * 0.82, h * 0.20)
        p.drawPath(path)
        p.drawLine(QPointF(w * 0.69, h * 0.20), QPointF(w * 0.82, h * 0.20))
        p.drawLine(QPointF(w * 0.82, h * 0.20), QPointF(w * 0.82, h * 0.33))


class StatusPill(QLabel):
    def set_status(self, text: str, online: bool = True) -> None:
        self.setText(text)
        self.setObjectName("StatusOnline" if online else "StatusOffline")
        self.style().unpolish(self)
        self.style().polish(self)


class PulseDot(QWidget):
    """Meaningful live-state motion only.

    The dot is static while the engine is offline and softly pulses only while
    TradeAI is actually active. This keeps ELVARA motion functional instead of
    decorating every surface.
    """

    def __init__(self, color: str = GREEN, parent=None):
        super().__init__(parent)
        self.color_name = color
        self.phase = False
        self.active = False
        self.setFixedSize(15, 15)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(820)

    def set_active(self, active: bool) -> None:
        active = bool(active)
        if self.active == active:
            return
        self.active = active
        self.phase = False
        self.update()

    def _tick(self):
        if not self.active:
            return
        self.phase = not self.phase
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        role = "success" if self.active else "muted"
        c = QColor(str(theme_palette()[role]))
        if self.active:
            halo = QColor(c)
            halo.setAlpha(36 if self.phase else 14)
            p.setPen(Qt.NoPen)
            p.setBrush(halo)
            p.drawEllipse(QRectF(1, 1, 13, 13))
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawEllipse(QRectF(5, 5, 5, 5))


class MetricCard(QFrame):
    def __init__(
        self,
        title: str,
        value: str = "—",
        detail: str = "",
        accent: str = CYAN,
        icon_name: str = "activity",
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setMinimumHeight(72)
        self.setMaximumHeight(82)
        self.accent = accent
        self.icon_name = icon_name

        root = QVBoxLayout(self)
        root.setContentsMargins(9, 7, 9, 7)
        root.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(7)
        icon_box = QFrame()
        icon_box.setObjectName("IconTile")
        icon_box.setFixedSize(27, 27)
        icon_layout = QHBoxLayout(icon_box)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        icon_layout.addWidget(self.icon_label)
        head.addWidget(icon_box)

        label_box = QVBoxLayout()
        label_box.setSpacing(0)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricLabel")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("MetricDetail")
        label_box.addWidget(self.title_label)
        label_box.addWidget(self.detail_label)
        head.addLayout(label_box, 1)

        self.dot = QLabel("●")
        head.addWidget(self.dot, 0, Qt.AlignTop)
        root.addLayout(head)

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        root.addWidget(self.value_label)
        self._override_color: str | None = None
        self.refresh_theme()

    def refresh_theme(self) -> None:
        accent = _theme_color(self.accent, "accent")
        self.icon_label.setPixmap(svg_icon(self.icon_name, accent, 15).pixmap(15, 15))
        self.dot.setStyleSheet(f"color:{accent}; font-size:6pt;")
        if self._override_color:
            value_color = _theme_color(self._override_color, "accent")
            self.value_label.setStyleSheet(f"color:{value_color};")
        else:
            self.value_label.setStyleSheet("")

    def set_value(self, value: str, detail: str | None = None, color: str | None = None) -> None:
        self.value_label.setText(value)
        self._override_color = color
        self.refresh_theme()
        if detail is not None:
            self.detail_label.setText(detail)


class SectionTitle(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        a = QLabel(title)
        a.setObjectName("SectionTitle")
        layout.addWidget(a)
        if subtitle:
            b = QLabel(subtitle)
            b.setObjectName("SectionSub")
            layout.addWidget(b)


class CandlestickWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.candles: list[tuple[float, float, float, float, float]] = []
        self.live_price: float | None = None
        self.setMinimumHeight(250)

    def set_candles(self, candles: list[tuple[float, float, float, float, float]]) -> None:
        # Keep the full 120-candle UI request so the market is visibly zoomed out.
        self.candles = candles[-120:]
        self.update()

    def set_live_price(self, price: float | None) -> None:
        self.live_price = float(price) if price is not None else None
        self.update()

    def paintEvent(self, event) -> None:
        from datetime import datetime

        colors = theme_palette()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            r = self.rect()
            plot = r.adjusted(10, 10, -55, -28)

            if len(self.candles) < 2:
                painter.setPen(QColor(str(colors["chart_text"])))
                painter.drawText(plot, Qt.AlignCenter, "WAITING FOR LIVE MT5 MARKET DATA")
                return

            highs = [c[2] for c in self.candles]
            lows = [c[3] for c in self.candles]
            if self.live_price is not None:
                highs.append(self.live_price)
                lows.append(self.live_price)
            hi, lo = max(highs), min(lows)
            span = max(hi - lo, 1e-9)
            hi += span * 0.075
            lo -= span * 0.075

            grid_color = QColor(str(colors["chart_grid"]))
            grid_color.setAlpha(82 if bool(colors.get("dark", True)) else 105)
            grid = QPen(grid_color, 1)
            painter.setPen(grid)

            for i in range(5):
                y = plot.top() + i * plot.height() / 4
                painter.drawLine(plot.left(), int(y), plot.right(), int(y))
                price = hi - i * (hi - lo) / 4
                painter.setPen(QColor(str(colors["chart_text"])))
                digits = 3 if hi > 20 else 5
                painter.drawText(
                    QRectF(plot.right() + 7, y - 8, 46, 16),
                    Qt.AlignVCenter | Qt.AlignLeft,
                    f"{price:.{digits}f}",
                )
                painter.setPen(grid)

            for i in range(1, 9):
                x = plot.left() + i * plot.width() / 9
                painter.drawLine(int(x), plot.top(), int(x), plot.bottom())

            def y_for(value: float) -> float:
                return plot.bottom() - ((value - lo) / max(hi - lo, 1e-9)) * plot.height()

            n = len(self.candles)
            step = plot.width() / max(n, 1)
            body_w = max(1.6, min(4.8, step * 0.54))

            for idx, (_, op, high, low, close) in enumerate(self.candles):
                x = plot.left() + (idx + 0.5) * step
                up = close >= op
                color = QColor(str(colors["success"] if up else colors["danger"]))
                painter.setPen(QPen(color, 0.9))
                painter.drawLine(QPointF(x, y_for(high)), QPointF(x, y_for(low)))
                top = min(y_for(op), y_for(close))
                bottom = max(y_for(op), y_for(close))
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawRect(QRectF(x - body_w / 2, top, body_w, max(1.2, bottom - top)))

            marker = self.live_price if self.live_price is not None else self.candles[-1][4]
            y = y_for(marker)
            marker_line = QColor(str(colors["accent2"]))
            marker_line.setAlpha(150)
            painter.setPen(QPen(marker_line, 1, Qt.DashLine))
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

            marker_fill = QColor(str(colors["accent"]))
            painter.setBrush(marker_fill)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(QRectF(plot.right() + 4, y - 9, 49, 18), 5, 5)
            painter.setPen(QColor(str(colors["marker_text"])))
            f = painter.font()
            f.setPointSize(7)
            f.setBold(True)
            painter.setFont(f)
            digits = 3 if marker > 20 else 5
            painter.drawText(
                QRectF(plot.right() + 4, y - 9, 49, 18),
                Qt.AlignCenter,
                f"{marker:.{digits}f}",
            )

            painter.setPen(QColor(str(colors["chart_text"])))
            f.setBold(False)
            f.setPointSize(6)
            painter.setFont(f)
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                idx = min(n - 1, int(round((n - 1) * frac)))
                ts = self.candles[idx][0]
                label = datetime.fromtimestamp(ts).strftime("%H:%M")
                x = plot.left() + (idx + 0.5) * step
                painter.drawText(
                    QRectF(x - 23, plot.bottom() + 7, 46, 16),
                    Qt.AlignHCenter | Qt.AlignTop,
                    label,
                )
        finally:
            painter.end()


class AreaLineChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: list[float] = []
        self.setMinimumHeight(135)

    def set_values(self, values: list[float]) -> None:
        self.values = [float(v) for v in values if v is not None][-360:]
        self.update()

    @staticmethod
    def _smooth_path(points: list[QPointF]) -> QPainterPath:
        path = QPainterPath(points[0])
        for index in range(1, len(points)):
            p0 = points[index - 1]
            p1 = points[index]
            cx = (p0.x() + p1.x()) / 2.0
            path.cubicTo(QPointF(cx, p0.y()), QPointF(cx, p1.y()), p1)
        return path

    def paintEvent(self, event) -> None:
        colors = theme_palette()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            r = self.rect()
            plot = r.adjusted(10, 12, -10, -20)

            if len(self.values) < 2:
                painter.setPen(QColor(str(colors["chart_text"])))
                painter.drawText(plot, Qt.AlignCenter, "FORWARD EQUITY TELEMETRY WILL APPEAR HERE")
                return

            lo, hi = min(self.values), max(self.values)
            span = max(hi - lo, max(abs(hi), 1.0) * 0.002)
            lo -= span * 0.14
            hi += span * 0.14

            grid_color = QColor(str(colors["chart_grid"]))
            grid_color.setAlpha(76 if bool(colors.get("dark", True)) else 105)
            painter.setPen(QPen(grid_color, 1))
            for i in range(5):
                y = plot.top() + i * plot.height() / 4
                painter.drawLine(plot.left(), int(y), plot.right(), int(y))

            points: list[QPointF] = []
            for i, value in enumerate(self.values):
                x = plot.left() + i * plot.width() / max(1, len(self.values) - 1)
                y = plot.bottom() - ((value - lo) / max(hi - lo, 1e-9)) * plot.height()
                points.append(QPointF(x, y))

            path = self._smooth_path(points)

            area = QPainterPath(path)
            area.lineTo(points[-1].x(), plot.bottom())
            area.lineTo(points[0].x(), plot.bottom())
            area.closeSubpath()

            fill = QLinearGradient(0, plot.top(), 0, plot.bottom())
            fill_top = QColor(str(colors["chart_fill"]))
            fill_top.setAlpha(66 if bool(colors.get("dark", True)) else 50)
            fill_bottom = QColor(str(colors["chart_fill"]))
            fill_bottom.setAlpha(0)
            fill.setColorAt(0.0, fill_top)
            fill.setColorAt(1.0, fill_bottom)
            painter.fillPath(area, QBrush(fill))

            glow = QColor(str(colors["chart_line"]))
            glow.setAlpha(42)
            painter.setPen(QPen(glow, 6.0))
            painter.drawPath(path)

            painter.setPen(QPen(QColor(str(colors["chart_line"])), 1.8))
            painter.drawPath(path)
            # Deliberately no endpoint dot.  It looked like a chart handle rather
            # than a live financial series in the previous prototype.
        finally:
            painter.end()


class RiskGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.limit = 10.0
        self.label = "DRAWDOWN"
        self.setMinimumSize(150, 135)

    def set_values(self, value: float, limit: float, label: str = "DRAWDOWN") -> None:
        self.value = max(0.0, float(value))
        self.limit = max(.0001, float(limit))
        self.label = label
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        s = min(self.width(), self.height()) - 26
        rect = QRectF((self.width()-s)/2, 8, s, s)
        pen_bg = QPen(QColor(str(theme_palette()["track"])), 12)
        pen_bg.setCapStyle(Qt.RoundCap)
        p.setPen(pen_bg)
        p.drawArc(rect, 225*16, -270*16)
        ratio = min(1.0, self.value/self.limit)
        color = theme_palette()["success"] if ratio < .55 else (theme_palette()["warning"] if ratio < .8 else theme_palette()["danger"])
        pen = QPen(QColor(str(color)), 12)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 225*16, int(-270*16*ratio))
        center = rect.center()
        p.setPen(QColor(str(theme_palette()["text_strong"])))
        f = QFont("Bahnschrift", 18); f.setBold(True); p.setFont(f)
        p.drawText(QRectF(center.x()-70, center.y()-24, 140, 34), Qt.AlignCenter, f"{self.value:.2f}%")
        p.setPen(QColor(str(theme_palette()["text_soft"])))
        f2 = QFont("Inter", 7); f2.setBold(True); p.setFont(f2)
        p.drawText(QRectF(center.x()-80, center.y()+12, 160, 22), Qt.AlignCenter, self.label)
        p.setPen(QColor(str(theme_palette()["chart_text"])))
        p.drawText(QRectF(0, self.height()-24, self.width(), 18), Qt.AlignCenter, f"LIMIT {self.limit:.2f}%")


class HorizontalBarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[tuple[str, float]] = []
        self.setMinimumHeight(155)

    def set_items(self, items: list[tuple[str, float]]) -> None:
        self.items = items
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect().adjusted(12, 10, -12, -10)
        if not self.items:
            p.setPen(QColor(str(theme_palette()["chart_text"]))); p.drawText(r, Qt.AlignCenter, "NO SYMBOL DATA"); return
        max_abs = max(max(abs(v) for _, v in self.items), 1e-9)
        row_h = r.height()/max(len(self.items), 1)
        label_w = 70
        zero_x = r.left()+label_w+(r.width()-label_w)/2
        p.setPen(QPen(QColor(str(theme_palette()["border"])), 1)); p.drawLine(int(zero_x), r.top(), int(zero_x), r.bottom())
        for i, (name, value) in enumerate(self.items):
            y = r.top()+i*row_h
            p.setPen(QColor(str(theme_palette()["chart_text"]))); p.drawText(QRectF(r.left(), y, label_w-8, row_h), Qt.AlignVCenter|Qt.AlignLeft, name)
            width = abs(value)/max_abs * (r.width()-label_w)*.43
            if value >= 0:
                br = QRectF(zero_x, y+row_h*.25, width, row_h*.5)
                brush = QColor(str(theme_palette()["success"]))
            else:
                br = QRectF(zero_x-width, y+row_h*.25, width, row_h*.5)
                brush = QColor(str(theme_palette()["danger"]))
            p.setPen(Qt.NoPen); p.setBrush(brush); p.drawRoundedRect(br, 4, 4)
            p.setPen(QColor(str(theme_palette()["success"] if value >= 0 else theme_palette()["danger"])))
            value_rect = QRectF(zero_x + (width+8 if value >= 0 else -width-62), y, 58, row_h)
            p.drawText(value_rect, Qt.AlignVCenter | (Qt.AlignLeft if value >= 0 else Qt.AlignRight), f"{value:+.2f}")


class SignalCard(QFrame):
    def __init__(self, symbol: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setMinimumHeight(126)
        self.symbol = symbol
        root = QVBoxLayout(self)
        root.setContentsMargins(11, 9, 11, 9)
        root.setSpacing(5)

        head = QHBoxLayout()
        self.symbol_label = QLabel(symbol)
        self.symbol_label.setObjectName("SectionTitle")
        self.signal = QLabel("HOLD")
        self.signal.setObjectName("SignalHold")
        head.addWidget(self.symbol_label)
        head.addStretch()
        head.addWidget(self.signal)
        root.addLayout(head)

        numbers = QHBoxLayout()
        left = QVBoxLayout(); left.setSpacing(1)
        a = QLabel("CONFIDENCE"); a.setObjectName("MetricLabel")
        self.conf = QLabel("—"); self.conf.setObjectName("MetricValue"); self.conf.setStyleSheet("font-size:16pt;")
        left.addWidget(a); left.addWidget(self.conf)
        right = QVBoxLayout(); right.setSpacing(1)
        b = QLabel("MODEL EDGE"); b.setObjectName("MetricLabel")
        self.edge = QLabel("—"); self.edge.setObjectName("MetricValue"); self.edge.setStyleSheet("font-size:16pt;")
        right.addWidget(b); right.addWidget(self.edge)
        numbers.addLayout(left); numbers.addStretch(); numbers.addLayout(right)
        root.addLayout(numbers)

        self.bar = QProgressBar(); self.bar.setRange(0, 1000); self.bar.setTextVisible(False)
        root.addWidget(self.bar)
        footer = QHBoxLayout()
        self.threshold = QLabel("Gate —"); self.threshold.setObjectName("MetricDetail")
        self.gate = QLabel("WAIT"); self.gate.setObjectName("GateWait")
        footer.addWidget(self.threshold); footer.addStretch(); footer.addWidget(self.gate)
        root.addLayout(footer)

    def update_state(self, signal_value: float | None, confidence: float | None, min_conf: float, min_edge: float, enabled: bool = True):
        if signal_value is None or confidence is None:
            self.conf.setText("—"); self.edge.setText("—"); self.bar.setValue(0); self.gate.setText("NO DATA"); return
        direction = "HOLD"
        if abs(signal_value) >= min_edge:
            direction = "BUY" if signal_value > 0 else "SELL"
        self.signal.setText(direction)
        self.signal.setObjectName("SignalBuy" if direction == "BUY" else "SignalSell" if direction == "SELL" else "SignalHold")
        self.signal.style().unpolish(self.signal); self.signal.style().polish(self.signal)
        self.conf.setText(f"{confidence*100:.1f}%")
        self.edge.setText(f"{signal_value:+.3f}")
        self.bar.setValue(int(max(0, min(1000, confidence*1000))))
        passed = enabled and confidence >= min_conf and abs(signal_value) >= min_edge
        self.gate.setText("MODEL GATE PASS" if passed else "WAIT")
        self.gate.setObjectName("GatePass" if passed else "GateWait")
        self.gate.style().unpolish(self.gate); self.gate.style().polish(self.gate)
        reason = f"Conf ≥ {min_conf*100:.0f}%  ·  |Edge| ≥ {min_edge:.3f}"
        self.threshold.setText(reason)
