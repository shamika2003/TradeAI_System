from __future__ import annotations

"""ELVARA / NIRA visual system for TradeAI.

Two brand themes are supported:
- ECLIPSE: deep celestial working surface.
- HALO: luminous working surface.

The UI is intentionally data-first.  Brand imagery lives in the ambient canvas,
while cards remain translucent enough for the celestial atmosphere to show
through without reducing readability.
"""

_THEME_MODE = "eclipse"

PALETTES: dict[str, dict[str, object]] = {
    "eclipse": {
        "dark": True,
        "name": "ECLIPSE",
        "bg0": "#02060B",
        "bg1": "#06111A",
        "bg2": "#0A1826",
        "shell0": "#02060B",
        "shell1": "#06111A",
        "shell2": "#0A1826",
        "panel0": "#08131E",
        "panel1": "#0B1A29",
        "panel2": "#102437",
        "panel": "#08131E",
        "panel_soft": "#0B1A29",
        "panel_deep": "#06101A",
        "interactive_hover": "#10283E",
        "chrome": "#06101A",
        "sidebar": "#071421",
        "header": "#071421",
        "statusbar": "#050D16",
        "text": "#E8F1FC",
        "text_strong": "#F4F8FF",
        "text_soft": "#B8C9DD",
        "muted": "#778EA8",
        "border": "#24567F",
        "border_soft": "#163750",
        "border_hot": "#56B8FF",
        "border_live": "#29D5F6",
        "accent": "#3B87FF",
        "accent_hover": "#4B95FF",
        "accent_pressed": "#2464C8",
        "accent2": "#29D5F6",
        "ai": "#8A70FF",
        "halo": "#C9D9F2",
        "frost": "#F2F7FF",
        "success": "#2ED0A1",
        "danger": "#EF6175",
        "warning": "#F0B343",
        "info": "#56B8FF",
        "neutral": "#93A8BF",
        "accent_soft": "rgba(59,135,255,36)",
        "accent_soft_hover": "rgba(59,135,255,56)",
        "cyan_soft": "rgba(41,213,246,24)",
        "ai_soft": "rgba(138,112,255,26)",
        "success_soft": "rgba(46,208,161,24)",
        "danger_soft": "rgba(239,97,117,24)",
        "warning_soft": "rgba(240,179,67,24)",
        "info_soft": "rgba(86,184,255,24)",
        "chart_bg": "#07131E",
        "chart_grid": "#1A3A55",
        "chart_text": "#7E94AA",
        "chart_axis": "#24506F",
        "chart_line": "#3B87FF",
        "chart_line2": "#29D5F6",
        "chart_fill": "#3B87FF",
        "chart_compare": "#29D5F6",
        "track": "#102437",
        "nav_icon": "#91A6BC",
        "marker_text": "#FFFFFF",
        "row_odd": "#091725",
        "row_even": "#0B1B2B",
        "row_hover": "#102A42",
        "row_selected": "#123562",
    },
    "halo": {
        "dark": False,
        "name": "HALO",
        "bg0": "#EDF4FB",
        "bg1": "#E4EEF8",
        "bg2": "#D8E7F4",
        "shell0": "#EDF4FB",
        "shell1": "#E7F0F9",
        "shell2": "#DCEAF6",
        "panel0": "#F8FBFF",
        "panel1": "#EFF6FC",
        "panel2": "#E5EFF9",
        "panel": "#F8FBFF",
        "panel_soft": "#EFF6FC",
        "panel_deep": "#E7F1FA",
        "interactive_hover": "#E0EDF9",
        "chrome": "#F7FBFF",
        "sidebar": "#F6FAFE",
        "header": "#F6FAFE",
        "statusbar": "#EDF4FB",
        "text": "#17314B",
        "text_strong": "#09192A",
        "text_soft": "#49657E",
        "muted": "#7C91A6",
        "border": "#9FC3DF",
        "border_soft": "#C7DBEC",
        "border_hot": "#4A93D0",
        "border_live": "#22BAD8",
        "accent": "#2E74EA",
        "accent_hover": "#3F84F0",
        "accent_pressed": "#245FC2",
        "accent2": "#22BAD8",
        "ai": "#826AEC",
        "halo": "#DCE9F6",
        "frost": "#FFFFFF",
        "success": "#169E74",
        "danger": "#E25A6C",
        "warning": "#D7992B",
        "info": "#2E8DD2",
        "neutral": "#71879C",
        "accent_soft": "rgba(46,116,234,30)",
        "accent_soft_hover": "rgba(46,116,234,46)",
        "cyan_soft": "rgba(34,186,216,24)",
        "ai_soft": "rgba(130,106,236,22)",
        "success_soft": "rgba(22,158,116,22)",
        "danger_soft": "rgba(226,90,108,22)",
        "warning_soft": "rgba(215,153,43,22)",
        "info_soft": "rgba(46,141,210,22)",
        "chart_bg": "#F2F7FC",
        "chart_grid": "#C7D9E9",
        "chart_text": "#6E8498",
        "chart_axis": "#A9C2D8",
        "chart_line": "#2E74EA",
        "chart_line2": "#22BAD8",
        "chart_fill": "#2E74EA",
        "chart_compare": "#22BAD8",
        "track": "#D9E7F3",
        "nav_icon": "#5E7690",
        "marker_text": "#FFFFFF",
        "row_odd": "#F7FAFD",
        "row_even": "#EEF5FB",
        "row_hover": "#E4F0FA",
        "row_selected": "#D5E9FB",
    },
}


def normalize_theme(mode: str | None) -> str:
    raw = str(mode or "").strip().lower()
    if raw in {"halo", "light", "luminous", "luminous-halo"}:
        return "halo"
    return "eclipse"


def set_theme_mode(mode: str) -> str:
    global _THEME_MODE
    _THEME_MODE = normalize_theme(mode)
    return _THEME_MODE


def theme_mode() -> str:
    return _THEME_MODE


def theme_palette(mode: str | None = None) -> dict[str, object]:
    return PALETTES[normalize_theme(mode or _THEME_MODE)]


def get_app_style(mode: str | None = None) -> str:
    p = theme_palette(mode)
    dark = bool(p["dark"])

    # Slightly more transparent working surfaces in ECLIPSE.  HALO keeps more
    # opacity so text remains crisp while the light celestial image still reads.
    panel_rgba = "rgba(8,19,30,148)" if dark else "rgba(248,251,255,188)"
    panel_soft_rgba = "rgba(11,26,41,142)" if dark else "rgba(239,246,252,178)"
    panel_deep_rgba = "rgba(6,16,26,184)" if dark else "rgba(231,241,250,210)"
    chrome_rgba = "rgba(6,16,26,218)" if dark else "rgba(247,251,255,218)"
    sidebar_rgba = "rgba(7,20,33,196)" if dark else "rgba(246,250,254,210)"
    header_rgba = "rgba(7,20,33,166)" if dark else "rgba(246,250,254,184)"
    status_rgba = "rgba(5,13,22,205)" if dark else "rgba(237,244,251,212)"
    row_hover_rgba = "rgba(16,42,66,180)" if dark else "rgba(228,240,250,220)"

    return f"""
* {{ outline:none; }}
QWidget {{
    color:{p['text']};
    font-family:"Inter","Segoe UI Variable","Segoe UI";
    font-size:9pt;
    background:transparent;
}}
QMainWindow, QWidget#AppRoot, QWidget#ScrollHost, QWidget#MetricsHost {{
    background:transparent;
}}
QWidget#HoloShell {{ background:transparent; }}

/* =========================================================
   CUSTOM CHROME
   ========================================================= */
QFrame#TitleBar {{
    background:{chrome_rgba};
    border:1px solid {p['border_soft']};
    border-radius:12px;
}}
QFrame#Sidebar {{
    background:{sidebar_rgba};
    border:1px solid {p['border_soft']};
    border-radius:12px;
}}
QFrame#ContentHeader {{
    background:{header_rgba};
    border:1px solid {p['border_soft']};
    border-radius:10px;
}}
QFrame#BottomBar {{
    background:{status_rgba};
    border:1px solid {p['border_soft']};
    border-radius:8px;
}}
QFrame#SidebarDivider {{
    color:{p['border_soft']};
    background:{p['border_soft']};
    max-height:1px;
}}

QLabel#TitleBarTitle {{
    color:{p['text_strong']};
    font-family:"Syne","Segoe UI Semibold";
    font-weight:700;
    font-size:11.5pt;
    letter-spacing:2.7px;
}}
QLabel#TitleBarProduct {{
    color:{p['text_soft']};
    font-size:10pt;
    font-weight:600;
}}
QLabel#TitleBarBrand {{
    color:{p['accent2']};
    font-size:7pt;
    font-weight:650;
    letter-spacing:1px;
}}
QPushButton#ThemeButton {{
    min-width:74px;
    max-width:74px;
    min-height:28px;
    max-height:28px;
    padding:0;
    border-radius:8px;
    background:{panel_soft_rgba};
    color:{p['text_strong']};
    border:1px solid {p['border']};
    font-size:7.4pt;
    font-weight:700;
}}
QPushButton#ThemeButton:hover {{
    border-color:{p['border_hot']};
    background:{p['accent_soft']};
}}
QPushButton#WindowButton, QPushButton#WindowCloseButton {{
    min-width:30px; max-width:30px;
    min-height:26px; max-height:26px;
    padding:0;
    background:transparent;
    color:{p['text_soft']};
    border:1px solid transparent;
    border-radius:7px;
    font-size:9pt;
}}
QPushButton#WindowButton:hover {{
    background:{p['interactive_hover']};
    border-color:{p['border_soft']};
    color:{p['text_strong']};
}}
QPushButton#WindowCloseButton:hover {{
    background:{p['danger']};
    border-color:{p['danger']};
    color:#FFFFFF;
}}

/* =========================================================
   NAVIGATION
   ========================================================= */
QPushButton#NavButton {{
    min-height:38px;
    background:transparent;
    color:{p['text_soft']};
    border:1px solid transparent;
    border-radius:10px;
    padding:0 11px;
    text-align:left;
    font-size:8.3pt;
    font-weight:500;
}}
QPushButton#NavButton:hover {{
    background:{panel_soft_rgba};
    color:{p['text_strong']};
    border-color:{p['border_soft']};
}}
QPushButton#NavButton:checked {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {p['accent']}, stop:1 rgba(59,135,255,165));
    color:#FFFFFF;
    border:1px solid {p['accent2']};
    font-weight:650;
}}
QPushButton#NavButton:focus {{ border-color:{p['accent2']}; }}

/* =========================================================
   DATA SURFACES
   ========================================================= */
QFrame#GlassPanel, QFrame#Panel, QFrame#ControlDeck,
QFrame#MetricCard, QFrame#CompactStat, QFrame#SidebarCoreCard,
QFrame#PipelineFrame, QFrame#PipelineStep {{
    background:{panel_rgba};
    border:1px solid {p['border_soft']};
}}
QFrame#GlassPanel, QFrame#Panel, QFrame#ControlDeck, QFrame#MetricCard {{
    border-radius:12px;
}}
QFrame#CompactStat, QFrame#SidebarCoreCard, QFrame#PipelineFrame, QFrame#PipelineStep {{
    border-radius:9px;
}}
QFrame#CompactStat, QFrame#SidebarCoreCard, QFrame#PipelineFrame {{
    background:{panel_soft_rgba};
}}
QFrame#PipelineStep {{ background:{panel_deep_rgba}; }}
QFrame#IconTile {{
    background:{p['cyan_soft']};
    border:1px solid {p['border_soft']};
    border-radius:8px;
}}

QFrame#NoticeInfo {{ background:{p['info_soft']}; border:1px solid {p['info']}; border-radius:8px; }}
QFrame#NoticeSuccess {{ background:{p['success_soft']}; border:1px solid {p['success']}; border-radius:8px; }}
QFrame#NoticeWarning {{ background:{p['warning_soft']}; border:1px solid {p['warning']}; border-radius:8px; }}
QFrame#NoticeDanger {{ background:{p['danger_soft']}; border:1px solid {p['danger']}; border-radius:8px; }}

/* =========================================================
   TYPE
   ========================================================= */
QLabel#BrandTitle {{
    color:{p['text_strong']};
    font-family:"Syne","Segoe UI Semibold";
    font-size:12.5pt;
    font-weight:700;
    letter-spacing:1px;
}}
QLabel#BrandAccent {{ color:{p['accent2']}; font-size:7pt; font-weight:650; letter-spacing:1px; }}
QLabel#BrandSub {{ color:{p['muted']}; font-size:6.8pt; letter-spacing:.4px; }}
QLabel#CoreLabel {{ color:{p['text_strong']}; font-weight:650; }}
QLabel#Eyebrow {{ color:{p['accent2']}; font-size:6.7pt; font-weight:700; letter-spacing:.85px; }}
QLabel#PageTitle {{
    color:{p['text_strong']};
    font-family:"Syne","Segoe UI Semibold";
    font-size:13.5pt;
    font-weight:650;
    letter-spacing:.3px;
}}
QLabel#PageSub {{ color:{p['text_soft']}; font-size:7.8pt; }}
QLabel#SectionTitle {{ color:{p['text_strong']}; font-size:9pt; font-weight:700; letter-spacing:.35px; }}
QLabel#SectionSub {{ color:{p['muted']}; font-size:6.9pt; }}
QLabel#MetricLabel {{ color:{p['text_soft']}; font-size:6.6pt; font-weight:650; letter-spacing:.35px; }}
QLabel#MetricValue {{
    color:{p['text_strong']};
    font-family:"Bahnschrift","JetBrains Mono","Consolas";
    font-size:13pt;
    font-weight:650;
}}
QLabel#CompactValue {{
    color:{p['text_strong']};
    font-family:"Bahnschrift","JetBrains Mono","Consolas";
    font-size:11pt;
    font-weight:650;
}}
QLabel#MetricDetail {{ color:{p['muted']}; font-size:6.8pt; }}
QLabel#PipelineIndex {{ color:{p['accent2']}; font-family:"Bahnschrift","Consolas"; font-size:6.8pt; font-weight:650; }}
QLabel#PipelineName {{ color:{p['text_strong']}; font-size:7.8pt; font-weight:600; }}
QLabel#Positive {{ color:{p['success']}; font-weight:650; }}
QLabel#Negative {{ color:{p['danger']}; font-weight:650; }}
QLabel#Cyan {{ color:{p['accent2']}; font-weight:650; }}
QLabel#Blue {{ color:{p['accent']}; font-weight:650; }}
QLabel#Amber {{ color:{p['warning']}; font-weight:650; }}
QLabel#NoticeTitle {{ color:{p['text_strong']}; font-weight:650; }}
QLabel#NoticeBody {{ color:{p['text_soft']}; font-size:8pt; }}

/* =========================================================
   BUTTONS
   ========================================================= */
QPushButton#PrimaryButton, QPushButton#SecondaryButton, QPushButton#GhostButton,
QPushButton#DangerButton, QPushButton#IconButton,
QPushButton#ControlStartButton, QPushButton#ControlPauseButton,
QPushButton#ControlResumeButton, QPushButton#ControlStopButton {{
    min-height:30px;
    border-radius:8px;
    padding:0 14px;
    font-size:7.8pt;
    font-weight:600;
}}
QPushButton#PrimaryButton, QPushButton#ControlStartButton, QPushButton#ControlResumeButton {{
    background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {p['accent']}, stop:1 {p['accent_hover']});
    color:#FFFFFF;
    border:1px solid {p['accent2']};
}}
QPushButton#PrimaryButton:hover, QPushButton#ControlStartButton:hover, QPushButton#ControlResumeButton:hover {{
    background:{p['accent_hover']};
    border-color:{p['accent2']};
}}
QPushButton#PrimaryButton:pressed, QPushButton#ControlStartButton:pressed, QPushButton#ControlResumeButton:pressed {{
    background:{p['accent_pressed']};
}}
QPushButton#SecondaryButton, QPushButton#ControlPauseButton {{
    background:{panel_soft_rgba};
    color:{p['text_strong']};
    border:1px solid {p['border']};
}}
QPushButton#SecondaryButton:hover, QPushButton#ControlPauseButton:hover {{
    background:{p['interactive_hover']};
    border-color:{p['border_hot']};
}}
QPushButton#GhostButton {{
    background:transparent;
    color:{p['text_soft']};
    border:1px solid {p['border_soft']};
}}
QPushButton#GhostButton:hover {{
    background:{p['interactive_hover']};
    color:{p['text_strong']};
    border-color:{p['border']};
}}
QPushButton#DangerButton, QPushButton#ControlStopButton {{
    background:{p['danger_soft']};
    color:{p['danger']};
    border:1px solid {p['danger']};
}}
QPushButton#DangerButton:hover, QPushButton#ControlStopButton:hover {{
    background:{p['danger']};
    color:#FFFFFF;
}}
QPushButton#IconButton {{
    min-width:30px; max-width:38px;
    padding:0;
    background:transparent;
    color:{p['text_soft']};
    border:1px solid {p['border_soft']};
}}
QPushButton#IconButton:hover {{
    background:{p['interactive_hover']};
    border-color:{p['border']};
    color:{p['text_strong']};
}}
QPushButton:disabled {{
    color:{p['muted']};
    background:{panel_deep_rgba};
    border-color:{p['border_soft']};
}}

/* =========================================================
   INPUTS
   ========================================================= */
QLineEdit, QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox {{
    min-height:34px;
    background:{panel_deep_rgba};
    color:{p['text_strong']};
    border:1px solid {p['border']};
    border-radius:8px;
    padding:0 10px;
    selection-background-color:{p['accent']};
}}
QLineEdit:hover, QComboBox:hover, QDateEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color:{p['border_hot']};
}}
QLineEdit:focus, QComboBox:focus, QDateEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border:1px solid {p['accent2']};
    background:{panel_soft_rgba};
}}
QComboBox::drop-down {{ border:none; width:22px; }}
QComboBox QAbstractItemView {{
    background:{p['panel']};
    color:{p['text_strong']};
    border:1px solid {p['border']};
    selection-background-color:{p['row_selected']};
}}
QLineEdit#SearchField {{ padding-left:11px; padding-right:26px; }}
QPushButton#FilterChip {{
    min-height:26px;
    background:{panel_deep_rgba};
    color:{p['text_soft']};
    border:1px solid {p['border_soft']};
    border-radius:13px;
    padding:0 10px;
}}
QPushButton#FilterChip:hover {{ border-color:{p['border']}; color:{p['text_strong']}; }}
QPushButton#FilterChip:checked {{ background:{p['accent_soft']}; color:{p['frost']}; border-color:{p['accent']}; }}

/* =========================================================
   STATUS
   ========================================================= */
QLabel#StatusOnline, QLabel#StatusOffline, QLabel#StatusMode, QLabel#StatusNeutral,
QLabel#StatusInfo, QLabel#StatusWarning,
QLabel#SignalBuy, QLabel#SignalSell, QLabel#SignalHold, QLabel#GatePass, QLabel#GateWait {{
    border-radius:9px;
    padding:2px 7px;
    font-size:6.8pt;
    font-weight:650;
}}
QLabel#StatusOnline, QLabel#GatePass, QLabel#SignalBuy {{
    background:{p['success_soft']};
    color:{p['success']};
    border:1px solid rgba(46,208,161,90);
}}
QLabel#StatusOffline, QLabel#SignalSell {{
    background:{p['danger_soft']};
    color:{p['danger']};
    border:1px solid rgba(239,97,117,90);
}}
QLabel#StatusMode {{
    background:{p['ai_soft']};
    color:{p['ai']};
    border:1px solid rgba(138,112,255,85);
}}
QLabel#StatusInfo {{
    background:{p['info_soft']};
    color:{p['info']};
    border:1px solid rgba(86,184,255,85);
}}
QLabel#StatusWarning, QLabel#SignalHold {{
    background:{p['warning_soft']};
    color:{p['warning']};
    border:1px solid rgba(240,179,67,85);
}}
QLabel#StatusNeutral, QLabel#GateWait {{
    background:{panel_deep_rgba};
    color:{p['text_soft']};
    border:1px solid {p['border_soft']};
}}

/* =========================================================
   TABLES
   ========================================================= */
QTableWidget {{
    background:transparent;
    alternate-background-color:{p['row_odd']};
    color:{p['text']};
    border:1px solid {p['border_soft']};
    border-radius:9px;
    gridline-color:transparent;
    selection-background-color:{p['row_selected']};
    selection-color:{p['text_strong']};
    font-size:7.8pt;
}}
QHeaderView::section {{
    min-height:34px;
    background:{panel_deep_rgba};
    color:{p['text_soft']};
    border:none;
    border-bottom:1px solid {p['border']};
    padding:0 10px;
    font-size:6.9pt;
    font-weight:650;
}}
QTableWidget::item {{
    border-bottom:1px solid {p['border_soft']};
    padding:0 9px;
}}
QTableWidget::item:hover {{ background:{row_hover_rgba}; }}

/* =========================================================
   CONSOLE / PROGRESS / DIALOGS
   ========================================================= */
QPlainTextEdit, QTextEdit {{
    background:{panel_deep_rgba};
    border:1px solid {p['border_soft']};
    border-radius:9px;
    color:{p['text']};
    selection-background-color:{p['accent']};
    font-family:"Bahnschrift","JetBrains Mono","Cascadia Mono","Consolas";
    font-size:7.5pt;
    padding:9px;
}}
QProgressBar {{
    border:none;
    border-radius:4px;
    background:{p['track']};
    text-align:center;
    color:{p['text_soft']};
    font-size:6.6pt;
    min-height:7px;
    max-height:7px;
}}
QProgressBar::chunk {{ border-radius:4px; background:{p['accent']}; }}
QToolTip {{
    background:{p['panel']};
    color:{p['text_strong']};
    border:1px solid {p['border']};
    border-radius:6px;
    padding:6px 8px;
}}
QDialog#ElvaraDialog {{
    background:{p['panel']};
    border:1px solid {p['border']};
    border-radius:14px;
}}
QMessageBox {{
    background:{p['panel']};
    color:{p['text']};
}}
QMessageBox QLabel {{ color:{p['text']}; min-width:280px; }}
QMessageBox QPushButton {{
    min-height:30px;
    border-radius:7px;
    padding:0 14px;
    background:{p['panel_soft']};
    color:{p['text_strong']};
    border:1px solid {p['border']};
}}

/* =========================================================
   SCROLLBARS — intentionally narrow
   ========================================================= */
QScrollArea {{ background:transparent; border:none; }}
QScrollArea > QWidget > QWidget {{ background:transparent; }}
QScrollBar:vertical {{
    background:transparent;
    width:5px;
    margin:3px 0;
}}
QScrollBar::handle:vertical {{
    background:rgba(59,135,255,92);
    min-height:30px;
    border-radius:2px;
}}
QScrollBar::handle:vertical:hover {{ background:{p['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height:0px;
    background:transparent;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
QScrollBar:horizontal {{
    background:transparent;
    height:5px;
    margin:0 3px;
}}
QScrollBar::handle:horizontal {{
    background:rgba(59,135,255,92);
    min-width:30px;
    border-radius:2px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width:0px;
    background:transparent;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background:transparent; }}

/* =========================================================
   GLOBAL NIRA RESERVATION BAR
   ========================================================= */
QFrame#NiraTopBar {{
    background:{panel_deep_rgba};
    border:1px solid {p['border_soft']};
    border-radius:10px;
}}
QLabel#NiraDockTitle {{
    color:{p['frost']};
    font-size:9pt;
    font-weight:700;
    letter-spacing:1.4px;
}}
QLabel#NiraDockState {{
    color:{p['ai']};
    font-size:6.2pt;
    font-weight:650;
    letter-spacing:.6px;
}}
QLineEdit#NiraTopInput {{
    min-height:28px; max-height:30px;
    background:{panel_soft_rgba};
    color:{p['text_soft']};
    border:1px solid {p['border_soft']};
    border-radius:8px;
    padding:0 9px;
    font-size:7.5pt;
}}
QLineEdit#NiraTopInput:focus {{
    color:{p['text_strong']};
    border-color:{p['ai']};
    background:{panel_rgba};
}}
QPushButton#NiraTopSend {{
    min-width:30px; max-width:30px; min-height:30px; max-height:30px;
    padding:0; border-radius:8px;
    color:{p['accent2']};
    background:{p['ai_soft']};
    border:1px solid rgba(138,112,255,90);
    font-size:13pt; font-weight:700;
}}
QPushButton#NiraTopSend:hover {{
    color:{p['frost']};
    border-color:{p['accent2']};
    background:{p['accent_soft']};
}}

QLabel#EngineRunning {{ color:{p['success']}; font-family:"Bahnschrift","Consolas"; font-size:15pt; font-weight:650; }}
QLabel#EnginePaused {{ color:{p['warning']}; font-family:"Bahnschrift","Consolas"; font-size:15pt; font-weight:650; }}
QLabel#EngineOffline {{ color:{p['muted']}; font-family:"Bahnschrift","Consolas"; font-size:15pt; font-weight:650; }}
QLabel#InterlockPass {{ color:{p['success']}; font-weight:650; }}
QLabel#InterlockFail {{ color:{p['danger']}; font-weight:650; }}
"""


APP_STYLE = get_app_style("eclipse")
