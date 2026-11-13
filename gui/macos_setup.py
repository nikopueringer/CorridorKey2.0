"""First-run model download for the bundled Mac app; never writes into .app."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import sys

from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtWidgets import QDialog, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout

from .command import ROOT


READY_MARKER = ".community-models-complete-v1"


def _files_present(root: Path) -> bool:
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in (
        "weights/corridorkey_v2.safetensors",
        "weights/foundation_stats.safetensors",
        "vfms/moge/model.pt",
        "vfms/rvm/pretrain_rvm_small16_256_204031069.npz",
        "vfms/cradio/C-RADIOv4-SO400M/config.json",
        "vfms/cradio/C-RADIOv4-SO400M/model.safetensors",
        "vfms/cradio/C-RADIOv4-SO400M/hf_model.py",
    ))


def assets_present(root: Path) -> bool:
    # The downloader fetches an entire C-RADIO code snapshot. The marker proves
    # it completed, even when the large weights finished before smaller files.
    return (root / READY_MARKER).is_file() and _files_present(root)


class DownloadDialog(QDialog):
    def __init__(self, root: Path):
        super().__init__()
        self.setWindowTitle("CorridorKey · Download models")
        self.resize(720, 420)
        self.root = root
        self.proc = QProcess(self)
        self.proc.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        # Inference is offline; initial model installation must be online.
        env.remove("HF_HUB_OFFLINE")
        env.remove("TRANSFORMERS_OFFLINE")
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        layout = QVBoxLayout(self)
        message = QLabel(f"Download about 5 GB of model files from their original publishers.\n"
                         f"Files will be kept in:\n{root}\n\n"
                         "Allow at least 12 GB of free disk space for download and temporary files.")
        message.setWordWrap(True)
        layout.addWidget(message)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)
        layout.addWidget(self.log)
        self.button = QPushButton("Download models")
        self.button.clicked.connect(self.start)
        layout.addWidget(self.button)
        self.proc.readyReadStandardOutput.connect(self.read)
        self.proc.finished.connect(self.finished_download)
        self.proc.errorOccurred.connect(self.failed)

    def start(self):
        self.button.setEnabled(False)
        self.proc.start(sys.executable, [str(ROOT / "scripts/download_weights.py"),
                                         "--package_root", str(self.root)])

    def read(self):
        self.log.appendPlainText(bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace"))

    def failed(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(self.proc.errorString())
            self.button.setEnabled(True)

    def finished_download(self, code, status):
        self.read()
        if code == 0 and status == QProcess.ExitStatus.NormalExit and _files_present(self.root):
            try:
                (self.root / READY_MARKER).write_text("Model download completed.\n")
                self.accept()
                return
            except OSError as exc:
                self.log.appendPlainText(str(exc))
        self.log.appendPlainText("Download did not finish. Check the log and retry; completed files are reused.")
        self.button.setText("Retry download")
        self.button.setEnabled(True)

    def reject(self):
        if self.proc.state() != QProcess.ProcessState.NotRunning:
            pid = self.proc.processId()
            try:
                if pid and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    self.proc.kill()
            except ProcessLookupError:
                pass
            self.proc.waitForFinished(5000)
        super().reject()


def prepare() -> bool:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        QMessageBox.information(None, "FFmpeg is required",
                                "Install FFmpeg in Terminal with:\n\nbrew install ffmpeg\n\n"
                                "Then open CorridorKey again. Homebrew is available at https://brew.sh.")
        return False
    root = Path(os.environ["CORRIDORKEY_ASSET_ROOT"]).expanduser()
    if assets_present(root):
        return True
    return DownloadDialog(root).exec() == QDialog.DialogCode.Accepted
