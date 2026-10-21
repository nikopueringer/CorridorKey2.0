"""Job settings shared by every clip in a run, remembered between sessions."""
from __future__ import annotations

import json

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .command import ALL_OUTPUTS, DEFAULT_TEMP_DIR, JobSettings
from .widgets import PathEdit, Section, SliderSpin, combo, select

OUTPUT_LABELS = {"alpha": "Alpha matte", "fg": "Foreground",
                 "cutout": "Cutout (RGBA)", "checker": "Checker preview"}
OUTPUT_TIPS = {
    "alpha": "The predicted matte.",
    "fg": "The predicted foreground colour, before despill.",
    "cutout": "Foreground with alpha: ProRes 4444 .mov, or RGBA EXR. Despilled.",
    "checker": "Foreground over a checkerboard with despill, for review.",
}
WINDOWS = [("80 / 40 · max continuity", "80/40"), ("48 / 24", "48/24"), ("24 / 12", "24/12"),
           ("8 / 4", "8/4"), ("1 / 1 · single frames", "1/1"), ("Custom", "custom")]
SETTINGS_KEY = "job"


class SettingsPanel(QScrollArea):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(0)
        for section in (self._outputs_group(), self._hints_group(), self._memory_group(),
                        self._frames_group(), self._range_group(), self._folder_group()):
            column.addWidget(section)
        column.addStretch(1)
        self.setWidget(body)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        try:
            saved = json.loads(QSettings().value(SETTINGS_KEY, "{}"))
        except (TypeError, ValueError):
            saved = {}
        self.load(JobSettings.from_dict(saved if isinstance(saved, dict) else {}))

    def _outputs_group(self) -> Section:
        box = Section("Outputs")
        form = box.form
        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        self.outputs: dict[str, QCheckBox] = {}
        for i, key in enumerate(ALL_OUTPUTS):
            check = QCheckBox(OUTPUT_LABELS[key])
            check.setToolTip(OUTPUT_TIPS[key])
            check.toggled.connect(self._on_output_toggled)
            self.outputs[key] = check
            grid.addWidget(check, i // 2, i % 2)
        form.addRow(grid)
        self.format = combo([("Preview video", "video"), ("EXR sequence · HDR", "exr")])
        self.format.setToolTip("Preview video is display-referred and clipped to 0–1.\n"
                               "EXR keeps unclamped scene-linear values.")
        self.exr_dtype = combo([("Half float", "half"), ("Full float", "float")])
        self.cutout_colour = combo([("Straight sRGB", "srgb"), ("Straight linear", "linear")])
        self.cutout_colour.setToolTip("Colour space of the cutout video. EXR is always linear.")
        self.despill = SliderSpin()
        self.despill.setToolTip("0 = no despill, 1 = full. Applies to cutout and checker.")
        form.addRow("Format", self.format)
        form.addRow("EXR type", self.exr_dtype)
        form.addRow("Cutout colour", self.cutout_colour)
        form.addRow("Despill", self.despill)
        for widget in (self.format, self.exr_dtype, self.cutout_colour):
            widget.currentIndexChanged.connect(self._on_change)
        self.despill.valueChanged.connect(self._on_change)
        return box

    def _hints_group(self) -> Section:
        box = Section("Hint trust")
        form = box.form
        self.hint_quality = SliderSpin()
        self.hint_quality.setToolTip("How far the model trusts a hint you provide:\n"
                                     "0.95 clean artist matte · 0.75 rough automated hint ·\n"
                                     "0.5 or less loose scribble.")
        self.carry = QCheckBox("Carry the model's own matte between windows")
        self.carry.setToolTip("Feeds each window's prediction forward as a hint (Hann carry).")
        self.carry_quality = SliderSpin()
        self.carry_quality.setToolTip("How far the model trusts its carried predictions.")
        form.addRow("Provided hint", self.hint_quality)
        form.addRow(self.carry)
        form.addRow("Carried hint", self.carry_quality)
        self.hint_quality.valueChanged.connect(self._on_change)
        self.carry.toggled.connect(self._on_change)
        self.carry_quality.valueChanged.connect(self._on_change)
        return box

    def _memory_group(self) -> Section:
        box = Section("GPU memory")
        form = box.form
        self.vram_mode = combo([("Auto · fit to free memory", "auto"), ("Manual", "manual")])
        self.vram_mode.setToolTip("Auto picks the largest window that fits each clip.")
        self.low_vram = QCheckBox("Low-VRAM mode (staged, slower)")
        self.window = combo(WINDOWS)
        self.window.setToolTip("Frames per stage-1 window / frames between windows.\n"
                               "Bigger windows: steadier over time, more VRAM.")
        self.chunk, self.stride = QSpinBox(), QSpinBox()
        for spin in (self.chunk, self.stride):
            spin.setRange(1, 400)
            spin.valueChanged.connect(self._on_window_spin)
        pair = QHBoxLayout()
        pair.addWidget(self.chunk)
        pair.addWidget(QLabel("/"))
        pair.addWidget(self.stride)
        self.temp_dir = PathEdit("Scratch folder", placeholder=DEFAULT_TEMP_DIR)
        self.temp_dir.setToolTip("Where feature caches go while a clip runs.")
        form.addRow("Mode", self.vram_mode)
        form.addRow(self.low_vram)
        form.addRow("Window", self.window)
        form.addRow("Chunk / stride", pair)
        form.addRow("Scratch", self.temp_dir)
        self.vram_mode.currentIndexChanged.connect(self._on_change)
        self.low_vram.toggled.connect(self._on_change)
        self.window.currentIndexChanged.connect(self._on_window_preset)
        self.temp_dir.changed.connect(self._on_change)
        return box

    def _frames_group(self) -> Section:
        box = Section("Frame sequences", expanded=False)
        form = box.form
        self.frame_colour = combo([("Auto · EXR linear, PNG/JPG sRGB", "auto"),
                                   ("Scene-linear", "linear"), ("sRGB", "srgb")])
        self.fps = QDoubleSpinBox()
        self.fps.setRange(1.0, 240.0)
        self.fps.setDecimals(3)
        self.fps.setToolTip("Frame rate written for frame-folder inputs.")
        form.addRow("Colour", self.frame_colour)
        form.addRow("Frame rate", self.fps)
        self.frame_colour.currentIndexChanged.connect(self._on_change)
        self.fps.valueChanged.connect(self._on_change)
        return box

    def _range_group(self) -> Section:
        box = Section("Frame range", expanded=False)
        form = box.form
        self.num_frames = QSpinBox()
        self.num_frames.setRange(0, 1_000_000)
        self.num_frames.setSpecialValueText("All frames")
        self.num_frames.setSuffix(" frames")
        self.num_frames.setToolTip("Process only part of each clip, for quick tests.")
        self.start_mode = combo([("From the start", "begin"), ("Centred segment", "middle"),
                                 ("Random segment", "random_middle")])
        self.start_mode.setToolTip("Which part of a video to take. Frame folders start at frame 1.")
        form.addRow("Length", self.num_frames)
        form.addRow("Take", self.start_mode)
        self.num_frames.valueChanged.connect(self._on_change)
        self.start_mode.currentIndexChanged.connect(self._on_change)
        return box

    def _folder_group(self) -> Section:
        box = Section("Output folder")
        form = box.form
        self.out_mode = combo([("Next to each clip · out_<name>", "beside"),
                               ("All in one folder", "custom")])
        self.out_parent = PathEdit("Output folder", placeholder="Folder for every out_<name>")
        form.addRow(self.out_mode)
        form.addRow(self.out_parent)
        self.out_mode.currentIndexChanged.connect(self._on_change)
        self.out_parent.changed.connect(self._on_change)
        return box

    def settings(self) -> JobSettings:
        custom_folder = self.out_mode.currentData() == "custom"
        return JobSettings(
            outputs=[key for key in ALL_OUTPUTS if self.outputs[key].isChecked()],
            output_format=self.format.currentData(),
            exr_dtype=self.exr_dtype.currentData(),
            cutout_linear=self.cutout_colour.currentData() == "linear",
            despill_strength=round(self.despill.value(), 3),
            hint_quality=round(self.hint_quality.value(), 3),
            carry_hint=self.carry.isChecked(),
            carry_hint_quality=round(self.carry_quality.value(), 3),
            frame_colour=self.frame_colour.currentData(),
            frame_dir_fps=self.fps.value(),
            auto_vram=self.vram_mode.currentData() == "auto",
            low_vram=self.low_vram.isChecked(),
            hann_chunk=self.chunk.value(),
            hann_stride=self.stride.value(),
            temp_dir=self.temp_dir.text() or DEFAULT_TEMP_DIR,
            num_frames=self.num_frames.value(),
            start_mode=self.start_mode.currentData(),
            output_parent=self.out_parent.text() if custom_folder else "",
        )

    def load(self, s: JobSettings) -> None:
        self._loading = True
        for key, check in self.outputs.items():
            check.setChecked(key in s.outputs)
        if not any(check.isChecked() for check in self.outputs.values()):
            self.outputs["alpha"].setChecked(True)
        select(self.format, s.output_format)
        select(self.exr_dtype, s.exr_dtype)
        select(self.cutout_colour, "linear" if s.cutout_linear else "srgb")
        self.despill.setValue(s.despill_strength)
        self.hint_quality.setValue(s.hint_quality)
        self.carry.setChecked(s.carry_hint)
        self.carry_quality.setValue(s.carry_hint_quality)
        select(self.frame_colour, s.frame_colour)
        self.fps.setValue(s.frame_dir_fps)
        select(self.vram_mode, "auto" if s.auto_vram else "manual")
        self.low_vram.setChecked(s.low_vram)
        self.chunk.setValue(s.hann_chunk)
        self.stride.setValue(s.hann_stride)
        select(self.window, f"{s.hann_chunk}/{s.hann_stride}")
        if self.window.currentData() != f"{s.hann_chunk}/{s.hann_stride}":
            select(self.window, "custom")
        self.temp_dir.setText("" if s.temp_dir == DEFAULT_TEMP_DIR else s.temp_dir)
        self.num_frames.setValue(s.num_frames)
        select(self.start_mode, s.start_mode)
        select(self.out_mode, "custom" if s.output_parent else "beside")
        self.out_parent.setText(s.output_parent)
        self._loading = False
        self._sync_enabled()

    def _on_change(self) -> None:
        if self._loading:
            return
        self._sync_enabled()
        QSettings().setValue(SETTINGS_KEY, json.dumps(self.settings().to_dict()))
        self.changed.emit()

    def _on_output_toggled(self, checked: bool) -> None:
        # At least one output stays on; infer.py has nothing to write otherwise.
        if not checked and not any(check.isChecked() for check in self.outputs.values()):
            check = self.sender()
            check.blockSignals(True)
            check.setChecked(True)
            check.blockSignals(False)
            return
        self._on_change()

    def _on_window_preset(self) -> None:
        preset = self.window.currentData()
        if preset != "custom" and not self._loading:
            chunk, stride = (int(v) for v in preset.split("/"))
            for spin, value in ((self.chunk, chunk), (self.stride, stride)):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)
        self._on_change()

    def _on_window_spin(self) -> None:
        if self._loading:
            return
        self.window.blockSignals(True)
        select(self.window, f"{self.chunk.value()}/{self.stride.value()}")
        if self.window.currentData() != f"{self.chunk.value()}/{self.stride.value()}":
            select(self.window, "custom")
        self.window.blockSignals(False)
        self._on_change()

    def _sync_enabled(self) -> None:
        outputs = {key for key, check in self.outputs.items() if check.isChecked()}
        exr = self.format.currentData() == "exr"
        self.exr_dtype.setEnabled(exr)
        self.cutout_colour.setEnabled("cutout" in outputs and not exr)
        self.despill.setEnabled(bool(outputs & {"cutout", "checker"}))
        self.carry_quality.setEnabled(self.carry.isChecked())
        manual = self.vram_mode.currentData() == "manual"
        self.low_vram.setEnabled(manual)
        self.window.setEnabled(manual)
        self.chunk.setEnabled(manual)
        self.stride.setEnabled(manual)
        self.start_mode.setEnabled(self.num_frames.value() > 0)
        self.out_parent.setEnabled(self.out_mode.currentData() == "custom")
