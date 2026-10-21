"""The corridorkey.com design system for Qt: its colour tokens, type and controls.

Token names and values follow the site's CSS custom properties (--canvas,
--surface, --line, --brand, ...). Patterns such as uppercase panel titles,
collapsible inspector sections, underlined tabs and the cream primary button
follow the CorridorKey standalone app shown on the site.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QLabel

# --- tokens (corridorkey.com :root) ---------------------------------------
CANVAS = "#0b0b0b"
CANVAS_2 = "#0f0f0d"
SURFACE = "#131312"
SURFACE_2 = "#17170f"
SURFACE_BRAND = "#1c1c11"
LINE = "#2c2c27"
LINE_STRONG = "#3a3a33"
LINE_BRAND = "#79781d"
TEXT = "#f4f1e8"
MUTED = "#b3b1a6"
SUBTLE = "#87877e"
BRAND = "#fff203"
BRAND_INK = "#1a1a05"
DANGER = "#ff7a7a"
CHROMA_KEY = "#28e07c"
# Derived: hover lines, disabled text, and the danger/chroma lines flattened onto the canvas.
LINE_HOVER = "#4d4d44"
DISABLED = "#5f5e57"
DANGER_LINE = "#5e3a37"
CHROMA_LINE = "#1d5a38"
VIEWER_BG = "#000000"

# Queue status pills: (text colour, outline colour).
STATUS_PILLS = {
    "Queued": (SUBTLE, LINE_STRONG),
    "Done": (CHROMA_KEY, CHROMA_LINE),
    "Failed": (DANGER, DANGER_LINE),
    "Stopped": (MUTED, LINE_STRONG),
}

FONT_FAMILIES = ["Inter", "Segoe UI Variable Text", "Segoe UI"]       # --font-display
MONO_FAMILIES = ["Cascadia Mono", "Cascadia Code", "Consolas"]         # --font-mono
ICONS = Path(__file__).resolve().parent / "icons"


def icon_path(name: str) -> str:
    return str(ICONS / f"{name}.svg")


def app_icon() -> QIcon:
    """The Corridor mark, every size from 16 to 256 px (built by `python -m gui.build_icon`)."""
    return QIcon(str(ICONS / "corridorkey.ico"))


def font(px: int = 13, weight: QFont.Weight = QFont.Weight.Normal,
         spacing: float = 0.0, mono: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(MONO_FAMILIES if mono else FONT_FAMILIES)
    f.setPixelSize(px)
    f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return f


def panel_title(text: str) -> QLabel:
    """An uppercase, letter-spaced panel header: QUEUE, INSPECTOR."""
    label = QLabel(text.upper())
    label.setObjectName("panelTitle")
    label.setFont(font(11, QFont.Weight.Bold, 1.6))
    return label


_ROLES = QPalette.ColorRole
_PALETTE = {
    _ROLES.Window: SURFACE,
    _ROLES.WindowText: TEXT,
    _ROLES.Base: CANVAS_2,
    _ROLES.AlternateBase: "#121211",
    _ROLES.ToolTipBase: SURFACE_2,
    _ROLES.ToolTipText: TEXT,
    _ROLES.PlaceholderText: SUBTLE,
    _ROLES.Text: TEXT,
    _ROLES.Button: "#171716",
    _ROLES.ButtonText: TEXT,
    _ROLES.BrightText: DANGER,
    _ROLES.Highlight: SURFACE_BRAND,
    _ROLES.HighlightedText: TEXT,
    _ROLES.Link: BRAND,
    _ROLES.Light: LINE_STRONG,
    _ROLES.Midlight: LINE,
    _ROLES.Mid: "#20201c",
    _ROLES.Dark: CANVAS,
    _ROLES.Shadow: "#000000",
}


def _style() -> str:
    icons = ICONS.as_posix()
    return f"""
* {{ outline: none; }}
QToolTip {{
    background: {SURFACE_2}; color: {TEXT}; border: 1px solid {LINE_STRONG}; padding: 6px 8px;
}}

/* frames */
QFrame#appHeader {{ background: {CANVAS}; border-bottom: 1px solid {LINE}; }}
QFrame#runBar {{ background: {CANVAS}; border-top: 1px solid {LINE}; }}
QWidget#sidePanel {{ background: {SURFACE}; }}
QWidget#viewerPanel {{ background: {CANVAS}; }}
QSplitter::handle {{ background: {LINE}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

/* type */
QLabel#panelTitle {{ color: {SUBTLE}; }}
QLabel#wordmark {{ color: {BRAND}; }}
QLabel#badge {{ background: {BRAND}; color: {BRAND_INK}; border-radius: 2px; padding: 1px 5px; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#subtle {{ color: {SUBTLE}; }}
QWidget#sectionBody QLabel {{ color: {MUTED}; }}
QWidget#sectionBody QLabel#clipName {{ color: {TEXT}; }}
QWidget#sectionBody QLabel#subtle {{ color: {SUBTLE}; }}
QLabel:disabled, QWidget#sectionBody QLabel:disabled {{ color: {DISABLED}; }}

/* inspector sections */
QPushButton#sectionHeader {{
    background: transparent; border: none; border-radius: 0;
    color: {TEXT}; text-align: left; padding: 10px 0 6px 0;
}}
QPushButton#sectionHeader:hover {{ color: #ffffff; background: transparent; }}
QFrame#sectionRule {{ background: {LINE}; border: none; min-height: 1px; max-height: 1px; }}

/* buttons */
QPushButton, QToolButton {{
    background: #171716; color: {TEXT};
    border: 1px solid {LINE_STRONG}; border-radius: 2px; padding: 5px 12px;
}}
QToolButton {{ padding: 4px 10px; }}
QPushButton:hover, QToolButton:hover {{ background: #1c1c1a; border-color: {LINE_HOVER}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {CANVAS_2}; }}
QPushButton:disabled, QToolButton:disabled {{
    color: {DISABLED}; background: {SURFACE}; border-color: {LINE};
}}
QToolButton:checked {{ border-color: {LINE_BRAND}; background: {SURFACE_BRAND}; }}
QPushButton#primary {{
    background: {TEXT}; color: {CANVAS}; border-color: {TEXT};
    font-weight: 700; padding: 6px 20px;
}}
QPushButton#primary:hover {{ background: #ffffff; border-color: #ffffff; }}
QPushButton#primary:disabled {{ background: {LINE}; border-color: {LINE}; color: {SUBTLE}; }}
QPushButton#danger:enabled {{
    color: {DANGER}; border-color: {DANGER_LINE}; background: rgba(255, 77, 77, 20);
}}
QPushButton#danger:enabled:hover {{ border-color: {DANGER}; }}
QToolButton#iconButton {{ background: transparent; border: 1px solid transparent; padding: 3px; }}
QToolButton#iconButton:hover {{ background: #1c1c1a; border-color: {LINE_STRONG}; }}

/* viewer tabs: Single | Wipe | Side by side in the app */
QFrame#tabStrip {{ background: {CANVAS_2}; border: 1px solid {LINE}; border-radius: 2px; }}
QToolButton#viewTab {{
    background: transparent; border: none; border-radius: 0;
    border-bottom: 2px solid transparent; color: {MUTED}; padding: 5px 12px 3px 12px;
}}
QToolButton#viewTab:hover {{ color: {TEXT}; background: transparent; }}
QToolButton#viewTab:checked {{
    color: {TEXT}; border-bottom-color: {BRAND}; background: transparent;
}}
QToolButton#viewTab:disabled {{ color: #4f4e48; }}

/* inputs */
QLineEdit, QComboBox, QAbstractSpinBox {{
    background: {CANVAS_2}; color: {TEXT};
    border: 1px solid {LINE_STRONG}; border-radius: 2px; padding: 4px 8px;
    selection-background-color: {LINE_BRAND}; selection-color: {TEXT};
}}
QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover {{ border-color: {LINE_HOVER}; }}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{ border-color: {LINE_BRAND}; }}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    color: {DISABLED}; background: #121211; border-color: {LINE};
}}
/* Spin boxes show no arrows, as in the app's inspector (wheel, keys and typing
   still work). The arrows are styled away rather than set to NoButtons, which
   collapses the value field to 1 px under a stylesheet in Qt 6.12. The value
   field then spans the whole box, so it carries the padding itself. */
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0; border: none; }}
QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{ image: none; width: 0; }}
QAbstractSpinBox > QLineEdit, QAbstractSpinBox > QLineEdit:hover,
QAbstractSpinBox > QLineEdit:focus, QAbstractSpinBox > QLineEdit:disabled {{
    background: transparent; border: none; padding: 0 8px;
}}
QAbstractSpinBox#valueSpin > QLineEdit {{ padding: 0 4px; }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none;
}}
QComboBox::down-arrow {{ image: url("{icons}/chevron-down.svg"); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {LINE_STRONG}; padding: 2px;
    selection-background-color: {SURFACE_BRAND}; selection-color: {BRAND};
}}
QAbstractSpinBox#valueSpin {{
    background: transparent; border-color: transparent; padding: 2px 4px; font-weight: 600;
}}
QAbstractSpinBox#valueSpin:hover {{ border-color: {LINE_STRONG}; }}
QAbstractSpinBox#valueSpin:focus {{ border-color: {LINE_BRAND}; background: {CANVAS_2}; }}
QAbstractSpinBox#valueSpin:disabled {{ background: transparent; border-color: transparent; }}

QCheckBox {{ spacing: 8px; color: {TEXT}; }}
QCheckBox:disabled {{ color: {DISABLED}; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; background: {CANVAS_2};
    border: 1px solid {LINE_STRONG}; border-radius: 2px;
}}
QCheckBox::indicator:hover {{ border-color: {LINE_HOVER}; }}
QCheckBox::indicator:checked {{
    background: {TEXT}; border-color: {TEXT}; image: url("{icons}/check.svg");
}}
QCheckBox::indicator:disabled {{ background: {SURFACE}; border-color: {LINE}; }}
QCheckBox::indicator:checked:disabled {{ background: {DISABLED}; border-color: {DISABLED}; }}

QSlider::groove:horizontal {{ height: 2px; background: {LINE_STRONG}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {TEXT}; border-radius: 1px; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 10px; height: 10px; margin: -4px 0; border-radius: 5px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::sub-page:horizontal:disabled,
QSlider::handle:horizontal:disabled {{ background: {DISABLED}; }}
/* frame scrubber: the app's green timeline line with a cream playhead */
QSlider#scrubber::groove:horizontal {{ height: 3px; background: {LINE}; }}
QSlider#scrubber::sub-page:horizontal {{ background: {CHROMA_KEY}; }}
QSlider#scrubber::handle:horizontal {{
    background: {TEXT}; width: 3px; height: 16px; margin: -7px 0; border-radius: 1px;
}}

/* lists */
QTreeView {{
    background: {CANVAS_2}; alternate-background-color: #121211;
    border: 1px solid {LINE}; border-radius: 2px;
}}
QTreeView::item {{ border-top: 1px solid transparent; border-bottom: 1px solid transparent; }}
QTreeView::item:hover {{ background: #161614; }}
QTreeView::item:selected {{
    background: {SURFACE_BRAND}; color: {TEXT};
    border-top: 1px solid {LINE_BRAND}; border-bottom: 1px solid {LINE_BRAND};
}}
QHeaderView::section {{
    background: {SURFACE}; color: {SUBTLE};
    border: none; border-bottom: 1px solid {LINE}; padding: 5px 6px;
}}

QProgressBar {{
    background: {LINE}; border: none; border-radius: 1px; min-height: 4px; max-height: 4px;
}}
QProgressBar::chunk {{ background: {BRAND}; border-radius: 1px; }}

QPlainTextEdit#log {{
    background: {CANVAS}; color: {MUTED}; border: none; padding: 6px 10px;
    selection-background-color: {LINE_BRAND};
}}

QMenu {{ background: {SURFACE}; border: 1px solid {LINE_STRONG}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 2px; color: {TEXT}; }}
QMenu::item:selected {{ background: {SURFACE_BRAND}; color: {BRAND}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 6px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {LINE_STRONG}; border-radius: 2px; margin: 2px; }}
QScrollBar::handle:vertical {{ min-height: 28px; }}
QScrollBar::handle:horizontal {{ min-width: 28px; }}
QScrollBar::handle:hover {{ background: {LINE_HOVER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    # Dark window frames on Windows 11 (Qt 6.8+).
    if hasattr(app.styleHints(), "setColorScheme"):
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    palette = QPalette()
    for role, colour in _PALETTE.items():
        palette.setColor(role, QColor(colour))
    for role in (_ROLES.WindowText, _ROLES.Text, _ROLES.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(DISABLED))
    app.setPalette(palette)
    app.setFont(font(13))
    app.setStyleSheet(_style())
