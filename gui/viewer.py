"""Frame viewer: the selected clip's input or any of its outputs, with a scrubber."""
from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPalette, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .clips import Clip
from .frames import FrameLoader
from .theme import TEXT, VIEWER_BG, font, icon_path

VIEWS = (("input", "Input"), ("alpha", "Alpha"), ("fg", "Foreground"),
         ("cutout", "Cutout"), ("checker", "Checker"))
PLACEHOLDER = ("Drop clips or frame folders here. Select a clip, then click Run queue "
               "below to create output previews.")


def _positive_fps(value) -> float | None:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if math.isfinite(fps) and fps > 0 else None


class ImageView(QWidget):
    """One frame fitted to the widget. The wheel zooms at the cursor, dragging
    pans, double-click fits again."""

    zoom_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.placeholder = ""
        self._image = QImage()
        self._zoom = 1.0          # relative to fit
        self._pan = QPointF()
        self._drag_from: tuple[QPointF, QPointF] | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def has_image(self) -> bool:
        return not self._image.isNull()

    def set_image(self, image: QImage) -> None:
        if image.isNull() or image.size() != self._image.size():
            self._zoom, self._pan = 1.0, QPointF()
        self._image = image
        self.update()
        self.zoom_changed.emit()

    def fit(self) -> None:
        self._zoom, self._pan = 1.0, QPointF()
        self.update()
        self.zoom_changed.emit()

    def actual_size(self) -> None:
        if self.has_image():
            self._zoom, self._pan = 1.0 / self._fit_scale(), QPointF()
            self.update()
            self.zoom_changed.emit()

    def scale(self) -> float:
        return self._fit_scale() * self._zoom

    def _fit_scale(self) -> float:
        if not self.has_image():
            return 1.0
        return min(self.width() / self._image.width(), self.height() / self._image.height())

    def _image_rect(self) -> QRectF:
        s = self.scale()
        w, h = self._image.width() * s, self._image.height() * s
        return QRectF(self.width() / 2 + self._pan.x() - w / 2,
                      self.height() / 2 + self._pan.y() - h / 2, w, h)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(VIEWER_BG))
        if not self.has_image():
            painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            painter.drawText(self.rect().adjusted(24, 24, -24, -24),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.placeholder)
            return
        # Smooth when shrinking; hard pixels when zoomed in, for checking edges.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.scale() < 2.0)
        painter.drawImage(self._image_rect(), self._image)

    def wheelEvent(self, event) -> None:
        if not self.has_image():
            return
        pos, before, old = event.position(), self._image_rect(), self.scale()
        # Keep the image pixel under the cursor where it is.
        u, v = (pos.x() - before.x()) / old, (pos.y() - before.y()) / old
        self._zoom = min(64.0, max(0.1, self._zoom * 1.25 ** (event.angleDelta().y() / 120)))
        s = self.scale()
        self._pan = QPointF(pos.x() - u * s + self._image.width() * s / 2 - self.width() / 2,
                            pos.y() - v * s + self._image.height() * s / 2 - self.height() / 2)
        self.update()
        self.zoom_changed.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag_from = (event.position(), QPointF(self._pan))
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_from is not None:
            start, pan = self._drag_from
            self._pan = pan + (event.position() - start)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_from = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()


class ViewerPanel(QWidget):
    open_folder = Signal(object)  # Clip

    def __init__(self, loader: FrameLoader, parent=None):
        super().__init__(parent)
        self.loader = loader
        self._clip: Clip | None = None
        self._paths: dict[str, Path] = {}
        self._view = "input"
        self._index = 0
        self._count = 0
        self._fps = 24.0
        self._sequence_fps = 24.0
        self._cutout_linear = False
        self._input_range = "auto"
        self._input_profile = "legacy"
        self._output_profiles: dict[str, str] = {}
        self._output_fps: dict[str, float | None] = {}
        self._manifest_native_fps: float | None = None
        self._input_fps: float | None = None
        self._display_transform = True
        self._waiting = False
        self._transport: list[QToolButton] = []

        self.setObjectName("viewerPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.image = ImageView()
        self.tabs = QButtonGroup(self)
        strip = QFrame()
        strip.setObjectName("tabStrip")
        strip_row = QHBoxLayout(strip)
        strip_row.setContentsMargins(2, 0, 2, 0)
        strip_row.setSpacing(0)
        for key, label in VIEWS:
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setObjectName("viewTab")
            button.setFont(font(13, QFont.Weight.DemiBold))
            button.setProperty("view", key)
            self.tabs.addButton(button)
            strip_row.addWidget(button)
        self.tabs.buttonClicked.connect(lambda button: self.show_view(button.property("view")))
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(strip)
        top.addStretch(1)
        self.view_transform = QCheckBox("View transform")
        self.view_transform.setToolTip("Decode the selected camera log/gamut to display sRGB. "
                                       "This affects only the Input preview.")
        self.view_transform.setChecked(True)
        self.view_transform.setVisible(False)
        self.view_transform.toggled.connect(self._set_view_transform)
        top.addWidget(self.view_transform)
        self.zoom_label = QLabel()
        self.zoom_label.setObjectName("subtle")
        top.addWidget(self.zoom_label)
        for text, tip, slot in (("Fit", "Fit to the window (F, or double-click)", self.image.fit),
                                ("1:1", "Actual pixels", self.image.actual_size)):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            top.addWidget(button)
        self.folder_button = QToolButton()
        self.folder_button.setText("Open folder")
        self.folder_button.setToolTip("Open this clip's output folder")
        self.folder_button.clicked.connect(lambda: self._clip and self.open_folder.emit(self._clip))
        top.addWidget(self.folder_button)

        self.play = self._icon_button("play", "Play / pause (Space)", self.toggle_play)
        transport = [self._icon_button("step-back", "Previous frame (←)", lambda: self.step(-1)),
                     self.play,
                     self._icon_button("step-forward", "Next frame (→)", lambda: self.step(1))]
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("scrubber")
        self.slider.valueChanged.connect(self._seek)
        self.frame_label = QLabel("–")
        self.frame_label.setObjectName("subtle")
        self.frame_label.setMinimumWidth(90)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.metadata_label = QLabel()
        self.metadata_label.setObjectName("subtle")
        self.guidance_label = QLabel()
        self.guidance_label.setObjectName("subtle")
        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        for button in transport:
            bottom.addWidget(button)
        bottom.addSpacing(8)
        bottom.addWidget(self.slider, 1)
        bottom.addSpacing(8)
        bottom.addWidget(self.frame_label)

        column = QVBoxLayout(self)
        column.setContentsMargins(10, 10, 10, 8)
        column.setSpacing(8)
        column.addLayout(top)
        column.addWidget(self.guidance_label)
        column.addWidget(self.metadata_label)
        column.addWidget(self.image, 1)
        column.addLayout(bottom)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        loader.loaded.connect(self._on_loaded)
        loader.failed.connect(self._on_failed)
        self.image.zoom_changed.connect(self._show_zoom)
        for key, slot in (("Space", self.toggle_play), ("Left", lambda: self.step(-1)),
                          ("Right", lambda: self.step(1)), ("F", self.image.fit)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        self.stop()
        self.set_clip(None, {})

    def set_clip(self, clip: Clip | None, outputs: dict[str, Path]) -> None:
        """Show a clip. Calling it again for the same clip refreshes its outputs
        and keeps the frame and view."""
        if clip is not self._clip:
            self.stop()
            self._index = 0
            self._input_fps = None
        self._clip = clip
        self._paths = {"input": clip.source, **outputs} if clip else {}
        self._input_profile = clip.input_color_space if clip else "legacy"
        self._output_profiles = {}
        self._output_fps = {}
        self._manifest_native_fps = None
        if clip and outputs:
            sidecar = next(iter(outputs.values())).parent / f"{clip.name}_color.json"
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                self._output_profiles = {
                    key: value["profile"] for key, value in metadata.get("outputs", {}).items()
                    if key in outputs and isinstance(value, dict) and value.get("profile")
                    and value.get("path", outputs[key].name) == outputs[key].name
                }
                self._output_fps = {
                    key: _positive_fps(value.get("native_fps"))
                    for key, value in metadata.get("outputs", {}).items()
                    if key in self._output_profiles and isinstance(value, dict)
                }
                self._manifest_native_fps = _positive_fps(metadata.get("native_fps"))
            except (OSError, ValueError, TypeError, AttributeError):
                # Older outputs have no color manifest; retain their historical
                # EXR-linear / video-sRGB interpretation.
                self._output_profiles = {}
        for button in self.tabs.buttons():
            key = button.property("view")
            button.setEnabled(key in self._paths)
            button.setToolTip("" if key in self._paths or key == "input"
                              else "Click Run queue below to create this output preview")
        if self._view not in self._paths:
            self._view = "input"
        self._check_tab()
        self.view_transform.setVisible(self._input_profile != "legacy" and self._view == "input")
        for widget in (self.folder_button, self.slider, *self._transport):
            widget.setEnabled(clip is not None)
        self.image.placeholder = PLACEHOLDER
        self.metadata_label.clear()
        self.guidance_label.setText(
            "Input preview · click Run queue below to generate output previews."
            if clip is not None and not outputs else "")
        if clip is None:
            self._count = 0
            self.frame_label.setText("–")
            self.image.set_image(QImage())
            return
        self._request()

    def set_sequence_fps(self, fps: float) -> None:
        self._sequence_fps = fps

    def set_cutout_linear(self, linear: bool) -> None:
        self._cutout_linear = linear

    def set_input_range(self, value: str) -> None:
        self._input_range = value

    def _set_view_transform(self, enabled: bool) -> None:
        self._display_transform = enabled
        if self._view == "input":
            self._request()

    def has_view(self, key: str) -> bool:
        return key in self._paths

    def show_view(self, key: str) -> None:
        if key in self._paths:
            self._view = key
            self._check_tab()
            self.view_transform.setVisible(self._input_profile != "legacy" and key == "input")
            self._request()

    def step(self, delta: int) -> None:
        if self._clip is None or not self._count:
            return
        self.stop()
        self._index = max(0, min(self._count - 1, self._index + delta))
        self._request()

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.stop()
        elif self._clip is not None:
            self._timer.start(max(1, round(1000 / (self._fps or 24.0))))
            self.play.setIcon(QIcon(icon_path("pause")))

    def stop(self) -> None:
        self._timer.stop()
        self.play.setIcon(QIcon(icon_path("play")))

    def _icon_button(self, icon: str, tip: str, slot) -> QToolButton:
        button = QToolButton()
        button.setObjectName("iconButton")
        button.setIcon(QIcon(icon_path(icon)))
        button.setIconSize(QSize(16, 16))
        button.setToolTip(tip)
        button.clicked.connect(slot)
        self._transport.append(button)
        return button

    def _check_tab(self) -> None:
        for button in self.tabs.buttons():
            button.setChecked(button.property("view") == self._view)

    def _request(self) -> None:
        self._waiting = True
        input_view = self._view == "input"
        profile_id = self._input_profile if input_view else self._output_profiles.get(self._view, "legacy")
        cutout_linear = self._cutout_linear
        if self._view == "cutout" and self._view in self._output_profiles:
            cutout_linear = profile_id == "linear_rec709"
        sequence_fps = self._sequence_fps
        if not input_view and self._paths[self._view].is_dir():
            sequence_fps = (self._output_fps.get(self._view) or
                            self._manifest_native_fps or self._input_fps or
                            self._sequence_fps)
        self.loader.request(self._paths[self._view], self._index,
                            linear_float=self._view != "alpha",
                            cutout=self._view == "cutout",
                            sequence_fps=sequence_fps,
                            cutout_linear=cutout_linear,
                            profile_id=profile_id,
                            display_transform=self._display_transform if input_view else True,
                            input_range=self._input_range if input_view else "auto")

    def _tick(self) -> None:
        if not self._waiting and self._count:
            self._index = (self._index + 1) % self._count
            self._request()

    def _seek(self, value: int) -> None:
        self._index = value
        self._request()

    def _current(self, path: str) -> bool:
        current = self._paths.get(self._view)
        return current is not None and path == str(current)

    def _on_loaded(self, path: str, index: int, count: int, fps: float, image: QImage) -> None:
        if not self._current(path):
            return
        self._waiting = False
        self._index, self._count, self._fps = index, count, fps
        if self._clip is not None and path == str(self._clip.source) and not self._clip.is_frames:
            self._input_fps = _positive_fps(fps)
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, count - 1))
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.frame_label.setText(f"<span style='color:{TEXT}; font-weight:600'>{index + 1}</span>"
                                 f" / {count}")
        duration = count / fps if fps > 0 else 0.0
        self.metadata_label.setText(f"{image.width()} × {image.height()} · {fps:g} fps · "
                                    f"{count} frames · {duration:.2f} s")
        self.image.set_image(image)

    def _on_failed(self, path: str, message: str) -> None:
        if not self._current(path):
            return
        self._waiting = False
        self.stop()
        self.image.placeholder = message
        self.image.set_image(QImage())

    def _show_zoom(self) -> None:
        zoom = f"{self.image.scale() * 100:.0f}%" if self.image.has_image() else ""
        self.zoom_label.setText(zoom)
