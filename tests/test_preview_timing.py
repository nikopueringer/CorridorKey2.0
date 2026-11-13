from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from gui import frames
from gui.clips import Clip
from gui.frames import FrameLoader
from gui.viewer import ViewerPanel


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait_until(app: QApplication, predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def make_edit_list_video(tmp_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for the edit-list preview regression")
    source, edited = tmp_path / "original.mp4", tmp_path / "edited.mp4"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x48:rate=50:duration=0.24", "-c:v", "mpeg4",
                    "-bf", "2", "-g", "12", str(source)], check=True)
    subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", "0.04", "-i", str(source),
                    "-c", "copy", str(edited)], check=True)
    return edited


def test_edit_list_preview_corrects_count_without_blocking_first_frame(app, tmp_path):
    source = make_edit_list_video(tmp_path)
    loader = FrameLoader()
    viewer = ViewerPanel(loader)
    failures: list[str] = []
    loader.failed.connect(lambda _path, message: failures.append(message))
    try:
        viewer.set_clip(Clip(source, source, False), {})

        # The first image can load using container metadata while the exact
        # count runs independently in the background.
        assert wait_until(app, lambda: viewer.image.has_image())
        assert wait_until(app, lambda: viewer._count == 10)
        assert viewer.metadata_label.text() == "64 × 48 · 50 fps · 10 frames · 0.20 s"
        assert viewer.slider.maximum() == 9

        viewer._seek(viewer.slider.maximum())
        assert wait_until(app, lambda: not viewer._waiting and viewer._index == 9)
        assert viewer.image.has_image()
        assert failures == []
    finally:
        viewer.close()
        loader.shutdown()
        app.processEvents()


def test_genuine_decode_error_is_not_hidden_by_exact_frame_count(app, monkeypatch, tmp_path):
    class BadSource:
        def __init__(self, path, *_args, **_kwargs):
            self.path = str(path)
            self.count = 10
            self.exact_count = 10
            self.fps = 25.0
            self.files = None
            self._interrupted = __import__("threading").Event()

        def set_count_callback(self, callback):
            callback(self.exact_count)

        def read(self, _index):
            raise OSError("synthetic corrupt frame")

        def close(self):
            pass

        def interrupt(self):
            self._interrupted.set()

    monkeypatch.setattr(frames, "FrameSource", BadSource)
    loader = FrameLoader()
    viewer = ViewerPanel(loader)
    failures: list[str] = []
    loader.failed.connect(lambda _path, message: failures.append(message))
    source = tmp_path / "corrupt.mov"
    source.touch()
    try:
        viewer.set_clip(Clip(source, source, False), {})
        assert wait_until(app, lambda: bool(failures))
        assert failures == ["synthetic corrupt frame"]
    finally:
        viewer.close()
        loader.shutdown()
        app.processEvents()


def test_failed_count_scan_releases_pending_real_decode_error(app, monkeypatch, tmp_path):
    class DeferredBadSource:
        instance = None

        def __init__(self, path, *_args, **_kwargs):
            type(self).instance = self
            self.count = 10
            self.exact_count = None
            self.count_scan_finished = False
            self.fps = 25.0
            self.files = None
            self._interrupted = threading.Event()
            self.linear_float = True
            self.cutout = False
            self.cutout_linear = False
            self.profile_id = "legacy"
            self.display_transform = True
            self.input_range = "auto"
            self.callback = None
            self.failed_read = threading.Event()

        def set_count_callback(self, callback):
            self.callback = callback

        def read(self, _index):
            self.failed_read.set()
            raise OSError("synthetic corrupt frame")

        def close(self):
            pass

        def interrupt(self):
            self._interrupted.set()

    monkeypatch.setattr(frames, "FrameSource", DeferredBadSource)
    loader = FrameLoader()
    viewer = ViewerPanel(loader)
    failures: list[str] = []
    loader.failed.connect(lambda _path, message: failures.append(message))
    source = tmp_path / "scan-failure.mov"
    source.touch()
    try:
        viewer.set_clip(Clip(source, source, False), {})
        assert wait_until(app, lambda: DeferredBadSource.instance.failed_read.is_set())
        assert failures == []  # waits briefly for exact count before classifying EOF
        DeferredBadSource.instance.count_scan_finished = True
        DeferredBadSource.instance.callback(None)  # FFmpeg scan failed or was unavailable
        assert wait_until(app, lambda: bool(failures))
        assert failures == ["synthetic corrupt frame"]
    finally:
        viewer.close()
        loader.shutdown()
        app.processEvents()


def test_cached_source_rebinds_count_completion_to_latest_request(app, monkeypatch, tmp_path):
    class DeferredSource:
        instance = None

        def __init__(self, path, *_args, **_kwargs):
            type(self).instance = self
            self.count = 12
            self.exact_count = None
            self.count_scan_finished = False
            self.fps = 25.0
            self.files = None
            self._interrupted = threading.Event()
            self.linear_float = True
            self.cutout = False
            self.cutout_linear = False
            self.profile_id = "legacy"
            self.display_transform = True
            self.input_range = "auto"
            self.callback = None
            self.failed_read = threading.Event()

        def set_count_callback(self, callback):
            self.callback = callback

        def read(self, index):
            if index == 11:
                self.failed_read.set()
                raise OSError("out of range from stale container count")
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def close(self):
            pass

        def interrupt(self):
            self._interrupted.set()

    monkeypatch.setattr(frames, "FrameSource", DeferredSource)
    loader = FrameLoader()
    viewer = ViewerPanel(loader)
    loaded: list[int] = []
    failures: list[str] = []
    loader.loaded.connect(lambda _path, index, _count, _fps, _image: loaded.append(index))
    loader.failed.connect(lambda _path, message: failures.append(message))
    path = tmp_path / "cached-source.mov"
    path.touch()
    try:
        viewer.set_clip(Clip(path, path, False), {})
        assert wait_until(app, lambda: loaded == [0]), (loaded, failures, viewer._waiting,
                                                         loader._request, loader._sources,
                                                         loader._active_source, loader._active_generation,
                                                         viewer._view, viewer._paths, viewer._index)
        loader.request(path, 11)
        source = DeferredSource.instance
        assert wait_until(app, lambda: source.failed_read.is_set())
        source.count = source.exact_count = 10
        source.count_scan_finished = True
        source.callback(10)
        assert wait_until(app, lambda: loaded == [0, 9])
        assert failures == []
    finally:
        viewer.close()
        loader.shutdown()
        app.processEvents()


def test_frame_source_interrupt_never_joins_count_callback_under_loader_lock(app, tmp_path):
    loader = FrameLoader()
    source = object.__new__(frames.FrameSource)
    source._interrupted = threading.Event()
    source._count_cancel = threading.Event()
    source._count_lock = threading.Lock()
    source._count_proc = None
    source._ffmpeg = None
    source._camera_reader = None
    callback_completed = threading.Event()

    def callback_needs_loader_lock():
        with loader._cond:
            callback_completed.set()

    callback_thread = threading.Thread(target=callback_needs_loader_lock)
    callback_thread.start()
    loader._active_source = source
    loader._active_generation = 0
    try:
        started = time.monotonic()
        loader.request(tmp_path / "cancel.mov", 0)
        elapsed = time.monotonic() - started
        assert elapsed < 0.25
        assert callback_completed.wait(1.0)
    finally:
        callback_thread.join(timeout=1.0)
        loader._active_source = None
        loader.shutdown()


def test_frame_source_close_attempts_every_resource_after_cleanup_failure():
    source = object.__new__(frames.FrameSource)
    closed = []

    def broken_reader():
        closed.append("reader")
        raise RuntimeError("reader cleanup failed")

    source._cancel_count = lambda: closed.append("count")
    source._stop_ffmpeg = lambda: closed.append("ffmpeg")
    source._camera_reader = SimpleNamespace(close=broken_reader)
    source.cap = SimpleNamespace(release=lambda: closed.append("capture"))

    with pytest.raises(RuntimeError, match="reader cleanup failed"):
        source.close()

    assert closed == ["count", "ffmpeg", "reader", "capture"]
    assert source._camera_reader is None
    assert source.cap is None


def test_queued_old_load_and_failure_are_discarded_after_new_request(app, monkeypatch, tmp_path):
    class ChangingSource:
        reads = 0
        first_read = threading.Event()
        second_read = threading.Event()

        def __init__(self, path, *_args, **_kwargs):
            self.count = self.exact_count = 1
            self.count_scan_finished = True
            self.fps = 24.0
            self.files = None
            self._interrupted = threading.Event()
            self.linear_float = True
            self.cutout = False
            self.cutout_linear = False
            self.profile_id = "legacy"
            self.display_transform = True
            self.input_range = "auto"

        def set_count_callback(self, callback):
            callback(1)

        def read(self, _index):
            type(self).reads += 1
            if type(self).reads == 1:
                type(self).first_read.set()
                raise OSError("stale decode error")
            type(self).second_read.set()
            return np.full((1, 1, 3), 2, dtype=np.uint8)

        def close(self):
            pass

        def interrupt(self):
            self._interrupted.set()

    monkeypatch.setattr(frames, "FrameSource", ChangingSource)
    loader = FrameLoader()
    path = tmp_path / "queued-signal.mov"
    path.touch()
    loaded: list[int] = []
    failures: list[str] = []
    loader.loaded.connect(lambda _path, _index, _count, _fps, image: loaded.append(
        image.pixelColor(0, 0).red()))
    loader.failed.connect(lambda _path, message: failures.append(message))
    try:
        loader.request(path, 0)
        assert ChangingSource.first_read.wait(1.0)
        deadline = time.monotonic() + 1.0
        while loader._active_generation is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        # The old error is now queued to the UI thread but not yet delivered.
        loader.request(path, 0)
        assert ChangingSource.second_read.wait(1.0)
        deadline = time.monotonic() + 1.0
        while loader._active_generation is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        app.processEvents()
        assert failures == []
        assert loaded == [2]
    finally:
        loader.shutdown()
        app.processEvents()


def test_same_generation_loaded_snapshot_cannot_restore_stale_frame_count(app, tmp_path):
    path = tmp_path / "same-generation.mov"
    path.touch()
    loader = FrameLoader()
    loader.request = lambda *_args, **_kwargs: None  # install UI state without a decoder request
    viewer = ViewerPanel(loader)
    image = QImage(64, 48, QImage.Format.Format_RGB888)
    try:
        viewer.set_clip(Clip(path, path, False), {})
        generation = 7
        with loader._cond:
            loader._generation = generation
            loader._latest_request = (str(path), 0, True, False, 24.0, False,
                                     "legacy", True, "auto", generation)
        # Replay loaded snapshots queued within the same generation. Both an
        # undercount and an overcount must defer to the authoritative scan.
        for stale_count in (192, 196):
            loader._count_for_generation.emit(generation, str(path), 194)
            loader._loaded_for_generation.emit(generation, str(path), 0, stale_count,
                                               25.0, image)
            app.processEvents()
            assert viewer._count == 194
            assert viewer.slider.maximum() == 193
            assert "194 frames" in viewer.metadata_label.text()
    finally:
        viewer.close()
        loader.shutdown()
        app.processEvents()


def test_interrupted_source_reopen_keeps_known_exact_frame_count(app, monkeypatch, tmp_path):
    path = make_edit_list_video(tmp_path)
    original = frames.FrameSource(path)
    reopened = None
    release_scan = threading.Event()
    try:
        assert original._count_scan_finished.wait(5.0)
        assert original.exact_count == 10
        original.interrupt()
        original.close()

        # Hold the new scan so this assertion can only pass from the identity-
        # keyed cache seeded by the first successful scan.
        entered_scan = threading.Event()

        def held_scan(_source):
            entered_scan.set()
            release_scan.wait(3.0)

        monkeypatch.setattr(frames.FrameSource, "_count_video_frames", held_scan)
        reopened = frames.FrameSource(path)
        assert not entered_scan.is_set()  # exact count is seeded synchronously
        assert reopened.count == 10
        assert reopened.exact_count == 10
    finally:
        release_scan.set()
        if reopened is not None:
            reopened.close()
        original.close()


def test_exact_count_cache_invalidates_when_file_identity_changes(monkeypatch, tmp_path):
    path = make_edit_list_video(tmp_path)
    capture = frames.cv2.VideoCapture(str(path))
    metadata_count = int(capture.get(frames.cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    first = frames.FrameSource(path)
    try:
        assert first._count_scan_finished.wait(5.0)
        assert first.exact_count == 10
        old_identity = frames._video_identity(path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        new_identity = frames._video_identity(path)
        assert new_identity != old_identity

        entered_scan = threading.Event()
        release_scan = threading.Event()

        def held_scan(_source):
            entered_scan.set()
            release_scan.wait(3.0)

        monkeypatch.setattr(frames.FrameSource, "_count_video_frames", held_scan)
        changed = frames.FrameSource(path)
        try:
            assert entered_scan.wait(1.0)
            assert changed.count == metadata_count
            assert changed.exact_count is None
            assert frames._cached_exact_count(new_identity) is None
        finally:
            release_scan.set()
            changed.close()
    finally:
        first.close()


def test_count_scan_result_is_discarded_if_file_changes_mid_scan(monkeypatch, tmp_path):
    path = make_edit_list_video(tmp_path)
    old_identity = frames._video_identity(path)
    scan_started = threading.Event()
    finish_scan = threading.Event()

    class FakePopen:
        def __init__(self, *_args, **_kwargs):
            self.returncode = None

        def poll(self):
            return self.returncode

        def communicate(self, timeout=None):
            scan_started.set()
            if not finish_scan.wait(3.0):
                raise subprocess.TimeoutExpired("fake ffmpeg", timeout)
            self.returncode = 0
            return "frame=10\nprogress=end\n", ""

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.returncode = -15
            finish_scan.set()

        def kill(self):
            self.returncode = -9
            finish_scan.set()

    monkeypatch.setattr(frames.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(frames, "validate_video", lambda _path: None)
    source = frames.FrameSource(path)
    metadata_count = source.count
    try:
        assert scan_started.wait(1.0)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
        new_identity = frames._video_identity(path)
        assert new_identity != old_identity
        finish_scan.set()
        assert source._count_scan_finished.wait(1.0)
        assert source.exact_count is None
        assert source.count == metadata_count
        assert frames._cached_exact_count(new_identity) is None
    finally:
        finish_scan.set()
        source.close()


def test_exact_count_cache_overrides_underreported_container_metadata(monkeypatch, tmp_path):
    path = tmp_path / "cached-count.mov"
    path.write_bytes(b"synthetic media identity")
    frames._remember_exact_count(frames._video_identity(path), 10)

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            values = {
                frames.cv2.CAP_PROP_FRAME_COUNT: 8,
                frames.cv2.CAP_PROP_FPS: 25,
                frames.cv2.CAP_PROP_FRAME_WIDTH: 64,
                frames.cv2.CAP_PROP_FRAME_HEIGHT: 48,
            }
            return values[prop]

        def release(self):
            pass

    monkeypatch.setattr(frames, "validate_video", lambda _path: None)
    monkeypatch.setattr(frames.cv2, "VideoCapture", lambda _path: FakeCapture())
    monkeypatch.setattr(frames.shutil, "which", lambda _name: None)
    source = frames.FrameSource(path)
    try:
        assert source.count == 10
        assert source.exact_count == 10
    finally:
        source.close()


def test_playback_timer_uses_first_loaded_video_fps(app, tmp_path):
    class StubLoader(QObject):
        loaded = Signal(str, int, int, float, object)
        failed = Signal(str, str)

        def request(self, *_args, **_kwargs):
            pass

        def shutdown(self):
            pass

    source = tmp_path / "synthetic.mov"
    source.touch()
    loader = StubLoader()
    viewer = ViewerPanel(loader)
    try:
        viewer.set_clip(Clip(source, source, False), {})
        viewer.toggle_play()
        assert viewer._timer.isActive()
        assert viewer._timer.interval() == 42  # provisional 24fps before metadata

        viewer._on_loaded(str(source), 0, 10, 50.0,
                          QImage(2, 2, QImage.Format.Format_RGB888))
        assert viewer.metadata_label.text() == "2 × 2 · 50 fps · 10 frames · 0.20 s"
        assert viewer._timer.isActive()
        assert viewer._timer.interval() == 20
    finally:
        viewer.close()
        app.processEvents()
