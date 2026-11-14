"""Runs the queue through infer.py, one process per clip, as the wizard does."""
from __future__ import annotations

import re
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

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
_STAGE2 = re.compile(r"^\s*Stage2 rendering: frames=(\d+)")
_STAGE2_PROGRESS = re.compile(r"^\s*output frames (\d+)/(\d+)")
_KEYING_COMPLETE = re.compile(r"^\s*Keying complete; closing outputs")

# Stage 1 windows span this part of the bar; low-VRAM runs still have a
# second pass to write the outputs after the last window.
KEY_START, KEY_END, KEY_END_LOW_VRAM = 0.15, 0.97, 0.65

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
        self._scratch: tempfile.TemporaryDirectory | None = None
        self._clock = time.monotonic
        self._run_started_at: float | None = None
        self._clip_started_at: float | None = None
        self._keying_started_at: float | None = None
        self._keying_ended_at: float | None = None
        self._window_done = 0
        self._window_total = 0
        self._eta_deadline: float | None = None
        self._stage2_total = 0
        self._stage2_done = 0
        self._stage2_sample: tuple[int, float] | None = None
        self._phase = "Starting Python"
        self.last_run_timing = ""
        self._timing_timer = QTimer(self)
        self._timing_timer.setInterval(1000)
        self._timing_timer.timeout.connect(self._tick_timing)

    def is_running(self) -> bool:
        return self._proc is not None

    def start(self, next_clip: Callable[[], Clip | None], settings: JobSettings) -> None:
        """Run clips until next_clip() returns None. Settings stay fixed for the run."""
        if self._proc is not None:
            return
        self._next_clip = next_clip
        self._settings = settings
        self._stopping = False
        self._run_started_at = self._clock()
        self.last_run_timing = ""
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
            try:
                if sys.platform.startswith(("linux", "darwin")) and pid and os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    self._proc.kill()
            except ProcessLookupError:
                pass
        self._proc.waitForFinished(5000)

    def _start_next(self) -> None:
        clip = self._next_clip()
        if clip is None:
            self._finish_run()
            self.finished.emit()
            return
        self._clip = clip
        self._clip_started_at = self._clock()
        clip.timing_summary = ""
        self._keying_started_at = None
        self._keying_ended_at = None
        self._window_done = self._window_total = 0
        self._eta_deadline = None
        self._stage2_total = self._stage2_done = 0
        self._stage2_sample = None
        self._phase = "Starting Python"
        self._timing_timer.start()
        clip.output_dir = output_dir_for(clip, self._settings.output_parent)
        # Each job owns its scratch files; stopping it must not delete files
        # belonging to another UI or CLI run with the same clip name.
        try:
            Path(self._settings.temp_dir).mkdir(parents=True, exist_ok=True)
            self._scratch = tempfile.TemporaryDirectory(prefix="gui-", dir=self._settings.temp_dir)
        except OSError as exc:
            self._finish_clip_timing(clip, FAILED)
            self._set(clip, FAILED, 0.0, f"Cannot create temporary directory: {exc}")
            self._timing_timer.stop()
            self._finish_run()
            self.finished.emit()
            return
        settings = replace(self._settings, temp_dir=self._scratch.name)
        args = infer_args(clip, settings, clip.output_dir)
        self._buffer, self._last_line = "", ""
        self._low_vram = self._settings.low_vram and not self._settings.auto_vram
        self._keying_since = self._clock()
        self._set(clip, RUNNING, 0.0, "Starting Python")
        self.clip_started.emit(clip)

        proc = QProcess(self)
        if sys.platform.startswith(("linux", "darwin")):
            proc.setUnixProcessParameters(QProcess.UnixProcessFlag.CreateNewSession)
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
        stage2 = _STAGE2.search(line)
        stage2_progress = _STAGE2_PROGRESS.search(line)
        if window:
            done, total = int(window[1]), int(window[2])
            now = self._clock()
            self._window_done, self._window_total = done, total
            self._phase = ("Stage 1 complete" if done >= total and self._low_vram
                           else "Writing outputs" if done >= total else "Keying")
            if done >= total:
                self._eta_deadline = None
            if done >= total and not self._low_vram:
                self._keying_ended_at = now
            elif done >= 2 and done < total:
                key_start = self._keying_started_at if self._keying_started_at is not None else now
                per_window = max(0.0, (now - key_start) / done)
                self._eta_deadline = now + per_window * (total - done)
            end = KEY_END_LOW_VRAM if self._low_vram else KEY_END
            self._update(clip, KEY_START + (end - KEY_START) * done / total,
                         self._timing_detail(now))
        elif stage2:
            self._stage2_total = int(stage2[1])
            self._stage2_done = 0
            self._stage2_sample = None
            self._eta_deadline = None
            self._phase = "Stage 2 rendering"
            self._update(clip, clip.progress, self._timing_detail(self._clock()))
        elif stage2_progress:
            done, total = int(stage2_progress[1]), int(stage2_progress[2])
            now = self._clock()
            self._stage2_total = total
            self._stage2_done = done
            if self._stage2_sample is not None:
                previous_done, previous_time = self._stage2_sample
                delta_frames = done - previous_done
                delta_time = now - previous_time
                if delta_frames > 0 and delta_time > 0 and done < total:
                    self._eta_deadline = now + delta_time / delta_frames * (total - done)
            self._stage2_sample = (done, now)
            self._phase = "Stage 2 rendering"
            progress = KEY_END_LOW_VRAM + (1.0 - KEY_END_LOW_VRAM) * done / max(total, 1)
            self._update(clip, progress, self._timing_detail(now))
        elif _KEYING_COMPLETE.search(line):
            now = self._clock()
            self._keying_ended_at = now
            self._eta_deadline = None
            self._phase = "Writing outputs"
            self._update(clip, clip.progress, self._timing_detail(now))
        elif _STAGE1.search(line):
            self._keying_since = self._clock()
            self._keying_started_at = self._keying_since
            self._phase = "Keying"
            self._update(clip, KEY_START, self._timing_detail(self._keying_since))
        elif auto_vram:
            self._low_vram = auto_vram[1] == "True"
        elif _VFM.search(line):
            self._phase = "Feature prepass"
            self._update(clip, 0.12, "Feature prepass")
        elif _RVM.search(line):
            self._phase = "Motion prepass"
            self._update(clip, 0.08, "Motion prepass")
        elif _PROBED.search(line):
            self._phase = "Reading clip"
            self._update(clip, 0.06, "Reading clip")
        elif _LOADING.search(line):
            self._phase = "Loading model"
            self._update(clip, 0.02, "Loading model")

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._drain()
        proc, self._proc = self._proc, None
        self._remove_scratch(self._clip)
        if proc is not None:
            proc.deleteLater()
        clip = self._clip
        if self._stopping:
            self._remove_scratch(clip)
            self._finish_clip_timing(clip, STOPPED)
            self._set(clip, STOPPED, 0.0, "Stopped")
            self._timing_timer.stop()
            self._finish_run()
            self.finished.emit()
            return
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            self._finish_clip_timing(clip, DONE)
            self._set(clip, DONE, 1.0, "Done")
        else:
            self._remove_scratch(clip)
            self._finish_clip_timing(clip, FAILED)
            self._set(clip, FAILED, clip.progress,
                      self._last_line or f"infer.py exited with code {exit_code}")
        self._timing_timer.stop()
        self._start_next()

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error != QProcess.ProcessError.FailedToStart:
            return  # finished() follows every other error
        proc, self._proc = self._proc, None
        self._remove_scratch(self._clip)
        message = proc.errorString() if proc is not None else ""
        if proc is not None:
            proc.deleteLater()
        self._finish_clip_timing(self._clip, FAILED)
        self._set(self._clip, FAILED, 0.0, f"Could not start {python_exe()}: {message}")
        self._timing_timer.stop()
        self._start_next()

    def _remove_scratch(self, clip: Clip) -> None:
        if self._scratch is not None:
            try:
                self._scratch.cleanup()
            except OSError:
                pass
            self._scratch = None

    def _update(self, clip: Clip, progress: float, detail: str) -> None:
        self._set(clip, RUNNING, progress, detail)

    def _set(self, clip: Clip, status: str, progress: float, detail: str) -> None:
        clip.status, clip.progress, clip.detail = status, progress, detail
        self.clip_changed.emit(clip)

    def _eta_text(self, now: float) -> str:
        if self._eta_deadline is None:
            return ""
        remaining = self._eta_deadline - now
        if remaining < 0:
            return " · ETA updating"
        return f" · ~{_clock(remaining)} left"

    def _timing_detail(self, now: float) -> str:
        if self._keying_started_at is None:
            clip_start = self._clip_started_at if self._clip_started_at is not None else now
            return f"{self._phase} · {_clock(max(0.0, now - clip_start))} elapsed"
        keying_end = self._keying_ended_at
        keying_elapsed = max(0.0, ((keying_end if keying_end is not None else now)
                                   - self._keying_started_at))
        clip_start = self._clip_started_at if self._clip_started_at is not None else now
        total_elapsed = max(0.0, now - clip_start)
        if self._phase == "Writing outputs" or keying_end is not None:
            return (f"Writing outputs · {_clock(keying_elapsed)} keying · "
                    f"{_clock(total_elapsed)} elapsed")
        if self._stage2_total:
            detail = (f"Stage 2 rendering {self._stage2_done}/{self._stage2_total} · "
                      f"{_clock(keying_elapsed)} keying · {_clock(total_elapsed)} elapsed")
        elif self._window_total:
            phase = "Stage 1 keying" if self._low_vram else "Keying"
            detail = (f"{phase} {self._window_done}/{self._window_total} · "
                      f"{_clock(keying_elapsed)} elapsed")
        else:
            detail = f"Keying · {_clock(keying_elapsed)} elapsed"
        return detail + self._eta_text(now)

    def _tick_timing(self) -> None:
        clip = self._clip
        if clip is None or self._proc is None:
            return
        now = self._clock()
        clip_start = self._clip_started_at if self._clip_started_at is not None else now
        elapsed = max(0.0, now - clip_start)
        if self._keying_started_at is not None:
            detail = self._timing_detail(now)
        else:
            detail = f"{self._phase} · {_clock(elapsed)} elapsed"
        self._update(clip, clip.progress, detail)

    def _finish_clip_timing(self, clip: Clip | None, outcome: str) -> None:
        if clip is None:
            return
        now = self._clock()
        started = self._clip_started_at if self._clip_started_at is not None else now
        keying_end = self._keying_ended_at if self._keying_ended_at is not None else now
        total = max(0.0, now - started)
        keying = (max(0.0, keying_end - self._keying_started_at)
                  if self._keying_started_at is not None else 0.0)
        clip.timing_summary = (f"{outcome} · total {_clock(total)} · "
                               f"keying {_clock(keying)}")
        self.log.emit(f"Timing {clip.name}: {clip.timing_summary}")

    def _finish_run(self) -> None:
        self._timing_timer.stop()
        if self._run_started_at is not None:
            self.last_run_timing = f"Queue total {_clock(max(0.0, self._clock() - self._run_started_at))}"
        else:
            self.last_run_timing = "Queue total 0:00"
        self.log.emit(self.last_run_timing)
        self._run_started_at = None
