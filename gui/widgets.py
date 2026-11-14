"""Small composite inputs shared by the side panels."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .theme import font, icon_path


def combo(items: list[tuple[str, str]]) -> QComboBox:
    box = QComboBox()
    for label, data in items:
        box.addItem(label, data)
    return box


def select(box: QComboBox, data: str) -> None:
    box.setCurrentIndex(max(0, box.findData(data)))


class Section(QWidget):
    """A collapsible inspector section: chevron and bold title over its rows,
    a rule underneath. Remembers whether it was left open."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._key = f"sections/{title}"
        saved = QSettings().value(self._key)
        if saved is not None:
            expanded = saved in (True, "true")
        self.header = QPushButton(title)
        self.header.setObjectName("sectionHeader")
        self.header.setCheckable(True)
        self.header.setFont(font(14, QFont.Weight.Bold, -0.2))
        self.header.setIconSize(QSize(12, 12))
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        self.body = QWidget()
        self.body.setObjectName("sectionBody")
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(0, 2, 0, 12)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(8)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        rule = QFrame()
        rule.setObjectName("sectionRule")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        layout.addWidget(rule)
        self.header.setChecked(expanded)
        self._show(expanded)
        self.header.toggled.connect(self._toggled)

    def _toggled(self, expanded: bool) -> None:
        self._show(expanded)
        QSettings().setValue(self._key, expanded)

    def _show(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.header.setIcon(QIcon(icon_path("chevron-down" if expanded else "chevron-right")))


class ElidedLabel(QLabel):
    """One line elided in the middle to fit, like the app's path fields; the
    full text is the tooltip."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        super().setText(self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideMiddle, max(0, self.width())))


class SliderSpin(QWidget):
    """A 0–1 value as a slider with its value beside it, editable in place."""

    valueChanged = Signal(float)

    def __init__(self, value: float = 0.5, parent=None):
        super().__init__(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.spin = QDoubleSpinBox()
        self.spin.setObjectName("valueSpin")
        self.spin.setRange(0.0, 1.0)
        self.spin.setSingleStep(0.05)
        self.spin.setDecimals(2)
        self.spin.setFixedWidth(52)
        self.spin.setFont(font(13, QFont.Weight.DemiBold))
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v / 100))
        self.spin.valueChanged.connect(self._on_spin)
        self.setValue(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)
        self._sync_slider(value)

    def _on_spin(self, value: float) -> None:
        self._sync_slider(value)
        self.valueChanged.emit(value)

    def _sync_slider(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round(value * 100))
        self.slider.blockSignals(False)


class PathEdit(QWidget):
    """A path field with a Choose… button: folders when pick_dir, files otherwise."""

    changed = Signal(str)

    def __init__(self, title: str, pick_dir: bool = True, file_filter: str = "",
                 placeholder: str = "", parent=None):
        super().__init__(parent)
        self.title, self.pick_dir, self.file_filter = title, pick_dir, file_filter
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setClearButtonEnabled(True)
        button = QPushButton("Choose…")
        button.setToolTip(f"Browse for {title.lower()}")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)
        self.edit.textChanged.connect(lambda: self.changed.emit(self.text()))
        button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text)

    def _browse(self) -> None:
        start = self.text()
        if self.pick_dir:
            path = QFileDialog.getExistingDirectory(self, self.title, start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, self.title, start, self.file_filter)
        if path:
            self.setText(str(Path(path)))
