"""The selected clip's hint: found automatically from companion files, editable here."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel

from .clips import HINT_FIRST_FRAME, HINT_NONE, HINT_VIDEO, RUNNING, Clip
from .theme import font
from .widgets import ElidedLabel, PathEdit, Section, combo, select

HINT_FILTER = ("Hints (*.png *.jpg *.jpeg *.exr *.mp4 *.mov *.avi *.mkv *.webm);;"
               "All files (*)")


class ClipBox(Section):
    edited = Signal(object)  # Clip

    def __init__(self, parent=None):
        super().__init__("Clip", parent=parent)
        self._clip: Clip | None = None
        self._clips: list[Clip] = []
        self._loading = False
        self.name = QLabel()
        self.name.setObjectName("clipName")
        self.name.setFont(font(15, QFont.Weight.Bold, -0.3))
        self.where = ElidedLabel()
        self.where.setObjectName("subtle")
        self.kind = combo([("No hint", HINT_NONE),
                           ("First-frame matte", HINT_FIRST_FRAME),
                           ("Full hint video or folder", HINT_VIDEO)])
        self.kind.setToolTip(
            "No hint: the model picks the subject.\n"
            "First-frame matte: white = subject on frame 1, then the model tracks it.\n"
            "Full hint video: a matte or roto for every frame.\n"
            "Companion files (<clip>_AlphaHint.*, <clip>FrameHint.png) are picked up on add.")
        self.path = PathEdit("Hint", pick_dir=False, file_filter=HINT_FILTER,
                             placeholder="None attached")
        from corridorkey.camera_color import PROFILES
        self.input_color = combo([(profile.label, profile_id)
                                  for profile_id, profile in PROFILES.items()])
        self.input_color.setToolTip("Select the camera log/gamut recorded by this clip. "
                                    "Choose Legacy for standard display video or linear frame sequences.")
        self.form.addRow(self.name)
        self.form.addRow(self.where)
        self.form.addRow("Camera input", self.input_color)
        self.form.addRow("Alpha hint", self.kind)
        self.form.addRow(self.path)
        self.kind.currentIndexChanged.connect(self._on_edit)
        self.path.changed.connect(self._on_edit)
        self.input_color.currentIndexChanged.connect(self._on_color_edit)
        self.set_clips([])

    def set_clips(self, clips: list[Clip]) -> None:
        self._clips = [clip for clip in clips if clip.status != RUNNING]
        self._clip = self._clips[0] if len(self._clips) == 1 else None
        for widget in (self.kind, self.path, self.input_color):
            widget.blockSignals(True)
        self._loading = True
        if self._clip is None:
            self.name.setText(f"{len(clips)} clips selected" if clips else "No clip selected")
            self.where.setText("Select a single clip to set its hint." if clips
                               else "Drop videos or frame folders anywhere in the window.")
            select(self.kind, HINT_NONE)
            self.path.setText("")
            selected_profiles = {clip.input_color_space for clip in self._clips}
            if len(selected_profiles) == 1:
                select(self.input_color, next(iter(selected_profiles)))
            else:
                self.input_color.setCurrentIndex(-1)
        else:
            clip = self._clip
            self.name.setText(clip.name)
            self.where.setText(("Frame sequence · " if clip.is_frames else "Video · ")
                               + str(clip.source))
            select(self.kind, clip.hint_kind)
            self.path.setText(str(clip.hint_path) if clip.hint_path else "")
            select(self.input_color, clip.input_color_space)
        self._loading = False
        for widget in (self.kind, self.path, self.input_color):
            widget.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        editable = self._clip is not None and self._clip.status != RUNNING
        self.kind.setEnabled(editable)
        self.path.setEnabled(editable and self.kind.currentData() != HINT_NONE)
        self.input_color.setEnabled(bool(self._clips))

    def _on_edit(self) -> None:
        if self._clip is None:
            return
        self._clip.hint_kind = self.kind.currentData()
        text = self.path.text()
        self._clip.hint_path = Path(text) if text and self._clip.hint_kind != HINT_NONE else None
        self.refresh()
        self.edited.emit(self._clip)

    def _on_color_edit(self) -> None:
        if self._loading:
            return
        profile_id = self.input_color.currentData()
        if profile_id is None:
            return
        for clip in self._clips:
            clip.input_color_space = profile_id
        if self._clips:
            self.edited.emit(self._clip or self._clips[0])
