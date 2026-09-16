from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = THIS_DIR.parent
for path in (str(THIS_DIR), str(SYSTEM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from ui.main_window import TradeAIMainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("NIRA · TradeAI")
    app.setOrganizationName("ELVARA")
    app.setStyle("Fusion")

    font = QFont("Inter")
    font.setPointSizeF(9.0)
    app.setFont(font)

    window = TradeAIMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
