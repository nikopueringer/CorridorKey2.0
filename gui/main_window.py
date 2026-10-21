"""Main window: queue on the left, viewer in the middle, settings on the right."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .clip_box import ClipBox
from .clip_list import ClipList
from .clips import (
    DONE,
    FAILED,
    RUNNING,
    STOPPED,
    VIDEO_EXTS,
    Clip,
    discover,
    find_outputs,
    output_dir_for,
)
from .frames import FrameLoader
from .gpu import GpuMonitor
from .runner import QueueRunner
from .settings_panel import SettingsPanel
from .theme import font, icon_path, panel_title
from .viewer import ViewerPanel

CLIP_FILTER = (f"Clips ({' '.join('*' + ext for ext in sorted(VIDEO_EXTS))});;"
               "All files (*)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CorridorKey 2.0")
        self.resize(1480, 900)
        self.setAcceptDrops(True)
        self._counts: Counter[str] = Counter()

        self.loader = FrameLoader(self)
        self.runner = QueueRunner(self)
        self.gpu = GpuMonitor(self)
        self.queue = ClipList()
        self.viewer = ViewerPanel(self.loader)
        self.clip_box = ClipBox()
        self.settings = SettingsPanel()
        self.log = QPlainTextEdit()
        self.log.setObjectName("log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("infer.py output appears here.")
        self.log.setFont(font(12, mono=True))

        self.setCentralWidget(self._build())
        self._connect()
        self._restore()
        self._show_selection()

    def _header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("appHeader")
        header.setFixedHeight(44)
        row = QHBoxLayout(header)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(10)
        glyph = QLabel()
        glyph.setPixmap(QIcon(icon_path("glyph")).pixmap(QSize(22, 22)))
        wordmark = QLabel("CORRIDORKEY")
        wordmark.setObjectName("wordmark")
        wordmark.setFont(font(13, QFont.Weight.Black, 2.6))
        badge = QLabel("2.0")
        badge.setObjectName("badge")
        badge.setFont(font(11, QFont.Weight.Bold))
        row.addWidget(glyph, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(wordmark, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(badge, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addStretch(1)
        return header

    def _build(self) -> QWidget:
        left = QWidget()
        left.setObjectName("sidePanel")
        left.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        column = QVBoxLayout(left)
        column.setContentsMargins(12, 12, 10, 10)
        column.setSpacing(8)
        column.addWidget(panel_title("Queue"))
        column.addWidget(self.queue, 1)
        buttons = QHBoxLayout()
        for text, slot in (("Add clips…", self.browse_clips), ("Add folder…", self.browse_folder),
                           ("Remove", self.queue.remove_selected)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        column.addLayout(buttons)

        right = QWidget()
        right.setObjectName("sidePanel")
        right.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        column = QVBoxLayout(right)
        column.setContentsMargins(14, 12, 8, 0)
        column.setSpacing(0)
        column.addWidget(panel_title("Inspector"))
        column.addWidget(self.clip_box)
        column.addWidget(self.settings, 1)

        self.split = QSplitter(Qt.Orientation.Horizontal)
        for widget in (left, self.viewer, right):
            self.split.addWidget(widget)
        self.split.setStretchFactor(1, 1)
        self.split.setChildrenCollapsible(False)
        self.split.setHandleWidth(1)
        self.split.setSizes([340, 760, 380])

        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.addWidget(self.split)
        self.vsplit.addWidget(self.log)
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setHandleWidth(1)
        self.vsplit.setSizes([760, 140])

        bar = QFrame()
        bar.setObjectName("runBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 8, 14, 8)
        row.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(200)
        self.status = QLabel("Ready · drop clips to begin.")
        self.status.setObjectName("muted")
        self.gpu_label = QLabel()
        self.gpu_label.setObjectName("subtle")
        self.log_toggle = QToolButton()
        self.log_toggle.setText("Log")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(True)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.run_button = QPushButton("Run queue")
        self.run_button.setObjectName("primary")
        row.addWidget(self.progress)
        row.addWidget(self.status, 1)
        row.addWidget(self.gpu_label)
        row.addSpacing(6)
        for widget in (self.log_toggle, self.stop_button, self.run_button):
            row.addWidget(widget)

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self._header())
        column.addWidget(self.vsplit, 1)
        column.addWidget(bar)
        return central

    def _connect(self) -> None:
        self.queue.itemSelectionChanged.connect(self._show_selection)
        self.queue.open_folder.connect(self.open_output_folder)
        self.viewer.open_folder.connect(self.open_output_folder)
        self.clip_box.edited.connect(self.queue.refresh)
        self.settings.changed.connect(self._show_selection)
        self.runner.clip_started.connect(self._on_clip_started)
        self.runner.clip_changed.connect(self._on_clip_changed)
        self.runner.log.connect(self.log.appendPlainText)
        self.runner.finished.connect(self._on_run_finished)
        self.gpu.changed.connect(self.gpu_label.setText)
        self.log_toggle.toggled.connect(self.log.setVisible)
        self.run_button.clicked.connect(self.run_queue)
        self.stop_button.clicked.connect(self.runner.stop)
        add = QShortcut(QKeySequence(QKeySequence.StandardKey.Open), self)
        add.activated.connect(self.browse_clips)
        remove = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.queue)
        remove.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        remove.activated.connect(self.queue.remove_selected)

    def add_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        found = [clip for path in paths for clip in discover(path)]
        added = self.queue.add(found)
        if added:
            self.status.setText(f"Added {len(added)} clip{'s' if len(added) != 1 else ''}.")
            if not self.queue.selected_clips():
                self.queue.select(added[0])
        elif found:
            self.status.setText("Those clips are already in the queue.")
        else:
            self.status.setText(f"No clips found in {paths[0].name}.")

    def browse_clips(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add clips", self._last_dir(), CLIP_FILTER)
        if files:
            self._remember_dir(Path(files[0]).parent)
            self.add_paths([Path(f) for f in files])

    def browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Add a frame folder or a folder of clips", self._last_dir())
        if folder:
            self._remember_dir(Path(folder).parent)
            self.add_paths([Path(folder)])

    def open_output_folder(self, clip: Clip) -> None:
        folder = self._output_dir(clip)
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.status.setText(f"{clip.name} has no output folder yet.")

    def _output_dir(self, clip: Clip) -> Path:
        return clip.output_dir or output_dir_for(clip, self.settings.settings().output_parent)

    def _outputs(self, clip: Clip | None) -> dict[str, Path]:
        # A running clip's old outputs are being overwritten, so only its input is shown.
        if clip is None or clip.status == RUNNING:
            return {}
        return find_outputs(clip, self._output_dir(clip))

    def _show_selection(self) -> None:
        clips = self.queue.selected_clips()
        self.clip_box.set_clips(clips)
        clip = clips[0] if clips else None
        self.viewer.set_clip(clip, self._outputs(clip))

    def run_queue(self) -> None:
        if self.runner.is_running():
            return
        if self.queue.next_queued() is None:
            selected = self.queue.selected_clips()
            if not selected:
                self.status.setText("Nothing queued. Add clips, or select finished ones "
                                    "to run again.")
                return
            for clip in selected:
                self.queue.requeue(clip)
        self._counts.clear()
        self.loader.release()
        self.settings.setEnabled(False)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.runner.start(self.queue.next_queued, self.settings.settings())

    def _on_clip_started(self, clip: Clip) -> None:
        if clip in self.queue.selected_clips():
            self._show_selection()

    def _on_clip_changed(self, clip: Clip) -> None:
        self.queue.refresh(clip)
        selected = clip in self.queue.selected_clips()
        if clip.status == RUNNING:
            self.progress.setValue(int(clip.progress * 1000))
            self.status.setText(f"{clip.name} · {clip.detail}")
            return
        self._counts[clip.status] += 1
        if clip.status == FAILED:
            self.status.setText(f"{clip.name} failed: {clip.detail}")
            self.log_toggle.setChecked(True)
        if selected:
            self._show_selection()
            if clip.status == DONE:
                for view in ("checker", "alpha", "fg"):
                    if self.viewer.has_view(view):
                        self.viewer.show_view(view)
                        break

    def _on_run_finished(self) -> None:
        self.settings.setEnabled(True)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setValue(0)
        parts = [f"{self._counts[s]} {s.lower()}"
                 for s in (DONE, FAILED, STOPPED) if self._counts[s]]
        self.status.setText("Run finished: " + ", ".join(parts) + "." if parts else "Run finished.")
        self.clip_box.refresh()

    def report_error(self, text: str) -> None:
        self.log.appendPlainText(text)
        self.log_toggle.setChecked(True)
        self.status.setText("Something went wrong in the UI; the log has the details.")

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.add_paths(paths)
        event.acceptProposedAction()

    def _last_dir(self) -> str:
        return QSettings().value("window/lastDir", "")

    def _remember_dir(self, folder: Path) -> None:
        QSettings().setValue("window/lastDir", str(folder))

    def _restore(self) -> None:
        store = QSettings()
        if store.contains("window/geometry"):
            self.restoreGeometry(store.value("window/geometry"))
        for key, splitter in (("window/split", self.split), ("window/vsplit", self.vsplit)):
            if store.contains(key):
                splitter.restoreState(store.value(key))
        self.log_toggle.setChecked(store.value("window/log", "true") in (True, "true"))

    def closeEvent(self, event) -> None:
        if self.runner.is_running():
            answer = QMessageBox.question(
                self, "Stop keying?", "A clip is still being keyed. Stop it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        store = QSettings()
        store.setValue("window/geometry", self.saveGeometry())
        store.setValue("window/split", self.split.saveState())
        store.setValue("window/vsplit", self.vsplit.saveState())
        store.setValue("window/log", self.log_toggle.isChecked())
        self.loader.shutdown()
        event.accept()
