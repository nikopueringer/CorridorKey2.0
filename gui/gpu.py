"""Polls nvidia-smi so the run bar can show how much GPU memory is free."""
from __future__ import annotations

import shutil

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


class GpuMonitor(QObject):
    changed = Signal(str)

    def __init__(self, parent=None, interval_ms: int = 3000):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        if shutil.which("nvidia-smi"):
            self._timer.start(interval_ms)
            QTimer.singleShot(0, self._poll)

    def _poll(self) -> None:
        if self._proc is not None:
            return
        self._proc = QProcess(self)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)
        self._proc.start("nvidia-smi", ["--query-gpu=name,memory.used,memory.total",
                                        "--format=csv,noheader,nounits"])

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._timer.stop()
            self._proc.deleteLater()
            self._proc = None

    def _on_finished(self, *_) -> None:
        proc, self._proc = self._proc, None
        lines = proc.readAllStandardOutput().data().decode(errors="replace").strip().splitlines()
        proc.deleteLater()
        try:
            name, used, total = (part.strip() for part in lines[0].split(","))
            free_gb, total_gb = (int(total) - int(used)) / 1024, int(total) / 1024
        except (IndexError, ValueError):
            return
        name = name.replace("NVIDIA ", "").replace("GeForce ", "")
        self.changed.emit(f"{name} · {free_gb:.1f} of {total_gb:.0f} GB free")
