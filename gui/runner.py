"""Runs the queue through infer.py, one process per clip, as the wizard does."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from .clips import DONE, FAILED, RUNNING, STOPPED, Clip, output_dir_for
from .command import ROOT, JobSettings, command_line, infer_args, python_exe

# Milestones infer.py prints, in order (see corridorkey/runtime.py).
_LOADING = re.compile(r"^Loading CorridorKey v2 runtime weights")
_PROBED = re.compile(r"^\s*probed \d+ frames")
_AUTO_VRAM = re.compile(r"Auto VRAM: .*low_vram=(True|False)")
_RVM = re.compile(r"RVM continuous prepass")
_VFM = re.compile(r"frozen VFM prepass")
_STAGE1 = re.compile(r"Hann stage1: ")
_WINDOW = re.compile(r"^\s*window (\d+)/(\d+)")

# Stage 1 windows span this part of the bar; low-VRAM runs still have a
# second pass to write the outputs after the last window.
KEY_START, KEY_END, KEY_END_LOW_VRAM = 0.15, 0.97, 0.65

# Scratch files run_inference deletes in its finally block; a killed run leaves them.
_SCRATCH = ("_rvm_feat.f16", "_baton.f16", "_foundation_cradio.f32", "_foundation_moge.f32")


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class QueueRunner(QObject):
    clip_started = Signal(object)
    clip_changed = Signal(object)   # a clip's status, progress or detail moved
    log = Signal(str)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._clip: Clip | None = None
        self._next_clip: Callable[[], Clip | None] = lambda: None
        self._settings = JobSettings()
        self._stopping = False
        self._buffer = ""
        self._last_line = ""
        self._low_vram = False
        self._keying_since = 0.0

    def is_running(self) -> bool:
        return self._proc is not None

    def start(self, next_clip: Callable[[], Clip | None], settings: JobSettings) -> None:
        """Run clips until next_clip() returns None. Settings stay fixed for the run."""
        if self._proc is not None:
            return
        self._next_clip = next_clip
        self._settings = settings
        self._stopping = False
        self._start_next()

    def stop(self) -> None:
        """Kill the running clip, ffmpeg children included, and end the run."""
        if self._proc is None:
            return
        self._stopping = True
        pid = self._proc.processId()
        if sys.platform == "win32" and pid:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self._proc.kill()
        self._proc.waitForFinished(5000)

    def _start_next(self) -> None:
        clip = self._next_clip()
        if clip is None:
            self.finished.emit()
            return
        self._clip = clip
        clip.output_dir = output_dir_for(clip, self._settings.output_parent)
        args = infer_args(clip, self._settings, clip.output_dir)
        self._buffer, self._last_line = "", ""
        self._low_vram = self._settings.low_vram and not self._settings.auto_vram
        self._keying_since = time.monotonic()
        self._set(clip, RUNNING, 0.0, "Starting Python")
        self.clip_started.emit(clip)

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        for key, value in (("PYTHONUNBUFFERED", "1"), ("PYTHONIOENCODING", "utf-8"),
                           ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1")):
            env.insert(key, value)
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        exe = python_exe()
        self.log.emit(f"\n=== {clip.name} ===\n{command_line([exe, *args])}")
        proc.start(exe, args)

    def _read(self) -> None:
        if self._proc is None:
            return
        text = self._proc.readAllStandardOutput().data().decode("utf-8", "replace")
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            self._handle(line)

    def _drain(self) -> None:
        self._read()
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._handle(line)

    def _handle(self, line: str) -> None:
        self.log.emit(line)
        if line.strip():
            self._last_line = line.strip()
        clip = self._clip
        window = _WINDOW.search(line)
        auto_vram = _AUTO_VRAM.search(line)
        if window:
            done, total = int(window[1]), int(window[2])
            end = KEY_END_LOW_VRAM if self._low_vram else KEY_END
            detail = f"Keying {done}/{total}"
            if done < total:
                left = (time.monotonic() - self._keying_since) / done * (total - done)
                detail += f" · {_clock(left)} left"
            elif self._low_vram:
                detail = "Writing outputs"
            self._update(clip, KEY_START + (end - KEY_START) * done / total, detail)
        elif _STAGE1.search(line):
            self._keying_since = time.monotonic()
            self._update(clip, KEY_START, "Keying")
        elif auto_vram:
            self._low_vram = auto_vram[1] == "True"
        elif _VFM.search(line):
            self._update(clip, 0.12, "Feature prepass")
        elif _RVM.search(line):
            self._update(clip, 0.08, "Motion prepass")
        elif _PROBED.search(line):
            self._update(clip, 0.06, "Reading clip")
        elif _LOADING.search(line):
            self._update(clip, 0.02, "Loading model")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._drain()
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.deleteLater()
        clip = self._clip
        if self._stopping:
            self._remove_scratch(clip)
            self._set(clip, STOPPED, 0.0, "Stopped")
            self.finished.emit()
            return
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self._set(clip, DONE, 1.0, "Done")
        else:
            self._remove_scratch(clip)
            self._set(clip, FAILED, clip.progress,
                      self._last_line or f"infer.py exited with code {exit_code}")
        self._start_next()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return  # finished() follows every other error
        proc, self._proc = self._proc, None
        message = proc.errorString() if proc is not None else ""
        if proc is not None:
            proc.deleteLater()
        self._set(self._clip, FAILED, 0.0, f"Could not start {python_exe()}: {message}")
        self._start_next()

    def _remove_scratch(self, clip: Clip) -> None:
        temp = Path(self._settings.temp_dir)
        for suffix in _SCRATCH:
            try:
                (temp / f"{clip.name}{suffix}").unlink(missing_ok=True)
            except OSError:
                pass

    def _update(self, clip: Clip, progress: float, detail: str) -> None:
        self._set(clip, RUNNING, progress, detail)

    def _set(self, clip: Clip, status: str, progress: float, detail: str) -> None:
        clip.status, clip.progress, clip.detail = status, progress, detail
        self.clip_changed.emit(clip)
