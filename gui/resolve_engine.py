"""Own the optional Linux OFX worker without loading CUDA into the Qt UI."""
from __future__ import annotations

import os
import signal
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .command import ROOT, python_exe


class ResolveEngine(QObject):
    log = Signal(str)
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc: QProcess | None = None
        self._buffer = ""

    def start(self):
        if self.proc is not None or not sys.platform.startswith("linux"):
            return
        proc = self.proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._finished)
        proc.errorOccurred.connect(self._error)
        self._buffer = ""
        self.changed.emit("Loading Resolve engine…")
        proc.start(python_exe(), ["-m", "ofx.worker"])

    def _read(self):
        if self.proc is None:
            return
        self._buffer += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self.log.emit(line)
            if line.startswith("CK_ENGINE_READY "):
                self.changed.emit("Resolve engine ready")

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log.emit(self.proc.errorString())
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, code, _status):
        self._read()
        if self._buffer:
            self.log.emit(self._buffer)
        proc, self.proc = self.proc, None
        if proc:
            proc.deleteLater()
        self.changed.emit("Resolve engine stopped" if code == 0 else "Resolve engine exited; see log")

    def stop(self):
        proc = self.proc
        if proc is None:
            return
        proc.terminate()
        if not proc.waitForFinished(1500):
            pid = proc.processId()
            try:
                if pid and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    proc.kill()
            except ProcessLookupError:
                pass
            proc.waitForFinished(3000)
