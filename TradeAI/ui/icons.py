from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


_PATHS = {
    "overview": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "signals": '<path d="M3 12h4l2.4-6 4.2 12 2.4-6H21"/><circle cx="3" cy="12" r="1.2"/><circle cx="21" cy="12" r="1.2"/>',
    "backtest": '<path d="M4 19V5M4 19h16"/><path d="m7 15 4-5 3 2 5-7"/><path d="m16 5 3 0 0 3"/>',
    "risk": '<path d="M12 2.8 20 6v5.6c0 5-3.2 8.5-8 10.1-4.8-1.6-8-5.1-8-10.1V6l8-3.2Z"/><path d="M12 7v6"/><circle cx="12" cy="16.6" r="1"/>',
    "reports": '<path d="M6 2.8h8l4 4V21H6z"/><path d="M14 2.8V7h4M9 11h6M9 15h6M9 19h4"/>',
    "terminal": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="m7 9 3 3-3 3M12.5 15H17"/>',
    "wallet": '<path d="M4 7h15a2 2 0 0 1 2 2v9H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h12"/><path d="M16 11h5v4h-5a2 2 0 0 1 0-4Z"/>',
    "capital": '<path d="m12 3 8 4-8 4-8-4 8-4Z"/><path d="m4 12 8 4 8-4M4 17l8 4 8-4"/>',
    "profit": '<path d="M4 19V5M4 19h16"/><path d="m7 15 4-4 3 2 6-7"/><path d="M16 6h4v4"/>',
    "positions": '<rect x="4" y="7" width="16" height="12" rx="2"/><path d="M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M4 12h16M10 12v2h4v-2"/>',
    "brain": '<path d="M9 4.5A3.5 3.5 0 0 0 5.5 8v1A3.5 3.5 0 0 0 4 15.5 3.5 3.5 0 0 0 9 19v-14.5ZM15 4.5A3.5 3.5 0 0 1 18.5 8v1A3.5 3.5 0 0 1 20 15.5 3.5 3.5 0 0 1 15 19v-14.5Z"/><path d="M9 8H7M9 12H6M15 8h2M15 12h3M9 16H7.5M15 16h1.5"/>',
    "market": '<path d="M4 17V9M8 14V5M12 19V11M16 12V3M20 17V7"/>',
    "shield": '<path d="M12 3 20 6v6c0 4.5-3 7.5-8 9-5-1.5-8-4.5-8-9V6l8-3Z"/><path d="m8.5 12 2.2 2.2 4.8-5"/>',
    "refresh": '<path d="M20 7v5h-5M4 17v-5h5"/><path d="M18.2 9A7 7 0 0 0 6.1 6.1L4 8M5.8 15A7 7 0 0 0 17.9 17.9L20 16"/>',
    "folder": '<path d="M3 6h7l2 2h9v11H3z"/>',
    "activity": '<path d="M3 12h4l2-5 4 10 2-5h6"/>',
    "control": '<circle cx="12" cy="12" r="8"/><path d="M12 3v7M8.5 6.5a7 7 0 1 0 7 0"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/>',
}


def svg_icon(name: str, color: str = "#9BCBFF", size: int = 22) -> QIcon:
    body = _PATHS.get(name, _PATHS["overview"])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'''
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QIcon(pixmap)
