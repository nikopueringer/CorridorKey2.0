"""Linux lifecycle tests for the Qt GUI process runners."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from gui import runner
from gui import resolve_engine
from gui import gpu
from gui import frames
from gui.viewer import ViewerPanel
from gui.clips import Clip, DONE, FAILED, STOPPED
from gui.command import JobSettings


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait_until(app, predicate, timeout=8.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    app.processEvents()
    assert predicate(), "Qt operation did not finish before timeout"


def linux_only():
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux QProcess lifecycle behavior")


def unix_only():
    if not sys.platform.startswith(("linux", "darwin")):
        pytest.skip("Unix QProcess process-group behavior")


def setup_stub(monkeypatch, tmp_path, body):
    script = tmp_path / "stub_infer.py"
    script.write_text(body)
    monkeypatch.setattr(runner, "python_exe", lambda: sys.executable)
    monkeypatch.setattr(
        runner,
        "infer_args",
        lambda clip, settings, output: [str(script), settings.temp_dir, str(tmp_path / f"{clip.name}.json")],
    )
    return script


def test_queue_jobs_get_private_scratch_cleaned_on_success_and_failure(
    app, monkeypatch, tmp_path
):
    setup_stub(
        monkeypatch,
        tmp_path,
        """import json, pathlib, sys
scratch, record = map(pathlib.Path, sys.argv[1:])
(scratch / 'owned.bin').write_text('owned')
record.write_text(json.dumps({'scratch': str(scratch)}))
if record.stem == 'bad':
    print('intentional failure', flush=True)
    raise SystemExit(3)
""",
    )
    temp_root = tmp_path / "shared-temp"
    temp_root.mkdir()
    unrelated = temp_root / "other-run.bin"
    unrelated.write_text("keep")
    clips = [Clip(tmp_path / "good.mp4", tmp_path / "good.mp4", False),
             Clip(tmp_path / "bad.mp4", tmp_path / "bad.mp4", False)]
    pending = iter(clips)
    runner_obj = runner.QueueRunner()
    ended = []
    runner_obj.finished.connect(lambda: ended.append(True))
    runner_obj.start(lambda: next(pending, None), JobSettings(temp_dir=str(temp_root)))
    wait_until(app, lambda: bool(ended))

    import json

    scratch_paths = [Path(json.loads((tmp_path / f"{clip.name}.json").read_text())["scratch"])
                     for clip in clips]
    assert scratch_paths[0] != scratch_paths[1]
    assert all(not path.exists() for path in scratch_paths)
    assert unrelated.read_text() == "keep"
    assert clips[0].status == DONE
    assert clips[1].status == FAILED
    assert clips[1].detail == "intentional failure"


@pytest.mark.parametrize("platform_name", ["linux", "darwin"])
def test_stop_kills_process_group_child_and_cleans_only_job_scratch(
    app, monkeypatch, tmp_path, platform_name
):
    unix_only()
    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform=platform_name))
    setup_stub(
        monkeypatch,
        tmp_path,
        """import pathlib, subprocess, sys, time
scratch, record = map(pathlib.Path, sys.argv[1:])
(scratch / 'owned.bin').write_text('owned')
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
record.write_text(f'{scratch}\\n{child.pid}\\n')
while True: time.sleep(1)
""",
    )
    temp_root = tmp_path / "shared-temp"
    temp_root.mkdir()
    unrelated = temp_root / "unrelated.bin"
    unrelated.write_text("keep")
    clip = Clip(tmp_path / "stuck.mp4", tmp_path / "stuck.mp4", False)
    runner_obj = runner.QueueRunner()
    ended = []
    runner_obj.finished.connect(lambda: ended.append(True))
    runner_obj.start(lambda: clip, JobSettings(temp_dir=str(temp_root)))
    record = tmp_path / "stuck.json"
    wait_until(app, record.exists)
    scratch_path, child_pid_text = record.read_text().splitlines()
    child_pid = int(child_pid_text)

    runner_obj.stop()
    wait_until(app, lambda: bool(ended))

    assert clip.status == STOPPED
    assert not Path(scratch_path).exists()
    assert unrelated.read_text() == "keep"
    # A killed process can briefly remain as a zombie while its parent reaps it.
    status = Path(f"/proc/{child_pid}/status")
    if status.exists():
        assert "State:\tZ" in status.read_text()


def test_resolve_engine_clears_failed_to_start_process(app, monkeypatch):
    linux_only()
    monkeypatch.setattr(resolve_engine, "python_exe", lambda: "/missing/python-for-test")
    engine = resolve_engine.ResolveEngine()
    states = []
    engine.changed.connect(states.append)

    engine.start()
    wait_until(app, lambda: engine.proc is None)

    assert states[0] == "Loading Resolve engine…"
    assert states[-1] == "Resolve engine exited; see log"


def test_apple_silicon_gpu_label_has_no_fabricated_memory_telemetry(monkeypatch):
    monkeypatch.setattr(gpu.sys, "platform", "darwin")
    monkeypatch.setattr(gpu.platform, "machine", lambda: "arm64")
    assert gpu.GpuMonitor.status_text() == "Apple Silicon · MPS"


def test_cutout_preview_composites_straight_alpha_in_linear_light():
    import numpy as np

    from gui.frames import to_cutout_checker8

    samples = np.array([[[32768, 32768, 32768, 0],
                         [32768, 32768, 32768, 65535]]], dtype=np.uint16)
    displayed = to_cutout_checker8(samples, checker_size=1)
    assert tuple(displayed[0, 0]) == (184, 184, 184)  # transparent checker
    assert tuple(displayed[0, 1]) == (128, 128, 128)  # opaque sRGB midpoint

    linear_exr = np.array([[[0.5, 0.5, 0.5, 1.0]]], dtype=np.float32)
    exr_display = to_cutout_checker8(linear_exr)
    assert tuple(exr_display[0, 0]) == (188, 188, 188)

    linear_video = np.array([[[128, 128, 128, 255]]], dtype=np.uint8)
    assert tuple(to_cutout_checker8(linear_video, linear=True)[0, 0]) == (188, 188, 188)


def test_cutout_tab_available_when_it_is_the_only_output(app, tmp_path):
    from PySide6.QtGui import QImage

    clip = Clip(tmp_path / "clip.mov", tmp_path / "clip.mov", False)
    viewer = ViewerPanel(frames.FrameLoader())
    try:
        viewer.set_clip(clip, {})
        assert "Run queue" in viewer.guidance_label.text()
        image = QImage(1920, 1080, QImage.Format.Format_RGB888)
        viewer._on_loaded(str(clip.source), 0, 38, 25.0, image)
        assert viewer.metadata_label.text() == "1920 × 1080 · 25 fps · 38 frames · 1.52 s"

        viewer.set_clip(clip, {"cutout": tmp_path / "clip_cutout.mov"})
        cutout_button = next(button for button in viewer.tabs.buttons()
                             if button.property("view") == "cutout")
        assert cutout_button.isEnabled()
        viewer.show_view("cutout")
        assert viewer._view == "cutout"
    finally:
        viewer.loader.shutdown()


def test_video_fps_stays_metadata_driven_and_frame_folder_uses_sequence_fps(
    monkeypatch, tmp_path
):
    import numpy as np

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            return {
                frames.cv2.CAP_PROP_FRAME_COUNT: 12,
                frames.cv2.CAP_PROP_FPS: 25.0,
                frames.cv2.CAP_PROP_FRAME_WIDTH: 2,
                frames.cv2.CAP_PROP_FRAME_HEIGHT: 1,
            }[prop]

        def read(self):
            return True, np.zeros((1, 2, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(frames.cv2, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(frames, "validate_video", lambda path: None)
    video = frames.FrameSource(tmp_path / "video.mov", sequence_fps=30.0)
    assert video.fps == 25.0
    assert video.read(0).shape == (1, 2, 3)
    video.close()

    sequence = tmp_path / "sequence"
    sequence.mkdir()
    (sequence / "frame_0001.png").write_bytes(b"fake")
    monkeypatch.setattr(frames.cv2, "imdecode", lambda data, flags: np.zeros((1, 2, 3), np.uint8))
    folder = frames.FrameSource(sequence, sequence_fps=29.97)
    assert folder.fps == 29.97
    assert folder.read(0).shape == (1, 2, 3)
    folder.close()


def test_frame_loader_cancels_old_generation_without_poisoning_new_request(
    app, monkeypatch, tmp_path
):
    import threading
    import numpy as np

    first_started = threading.Event()

    class BlockingSource:
        def __init__(self, path, linear_float=True, cutout=False,
                     sequence_fps=24.0, cutout_linear=False, profile_id="legacy",
                     display_transform=True, input_range="auto"):
            self.linear_float = linear_float
            self.cutout = cutout
            self.cutout_linear = cutout_linear
            self.profile_id = profile_id
            self.display_transform = display_transform
            self.input_range = input_range
            self._interrupted = threading.Event()
            self.files = None
            self.count, self.fps = 2, 25.0
            self.width, self.height = 1, 1

        def read(self, index):
            if index == 0:
                first_started.set()
                while not self._interrupted.wait(0.005):
                    pass
                raise OSError("cancelled old request")
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def interrupt(self):
            self._interrupted.set()

        def close(self):
            self.interrupt()

    monkeypatch.setattr(frames, "FrameSource", BlockingSource)
    loader = frames.FrameLoader()
    loaded = []
    loader.loaded.connect(lambda path, index, count, fps, image: loaded.append(index))
    try:
        loader.request(tmp_path / "clip.mov", 0)
        assert first_started.wait(1.0)
        loader.request(tmp_path / "clip.mov", 1)
        wait_until(app, lambda: loaded == [1])
        assert loader._thread.is_alive()  # the worker remains available
    finally:
        loader.shutdown()


def test_frame_loader_shutdown_interrupts_active_read_quickly(app, monkeypatch, tmp_path):
    import threading

    started = threading.Event()

    class BlockingSource:
        def __init__(self, path, linear_float=True, cutout=False,
                     sequence_fps=24.0, cutout_linear=False, profile_id="legacy",
                     display_transform=True, input_range="auto"):
            self.linear_float, self.cutout = linear_float, cutout
            self.cutout_linear = cutout_linear
            self.profile_id = profile_id
            self.display_transform = display_transform
            self.input_range = input_range
            self._interrupted = threading.Event()
            self.files = None
            self.count, self.fps = 1, 25.0

        def read(self, index):
            started.set()
            while not self._interrupted.wait(0.005):
                pass
            raise OSError("stopped")

        def interrupt(self):
            self._interrupted.set()

        def close(self):
            self.interrupt()

    monkeypatch.setattr(frames, "FrameSource", BlockingSource)
    loader = frames.FrameLoader()
    loader.request(tmp_path / "slow.mov", 0)
    assert started.wait(1.0)
    before = time.monotonic()
    loader.shutdown()
    assert time.monotonic() - before < 2.0
    assert not loader._thread.is_alive()


def test_sequence_fps_control_is_disabled_for_selected_video(app, tmp_path):
    from gui.settings_panel import SettingsPanel

    panel = SettingsPanel()
    try:
        panel.set_sequence_source(False)
        assert not panel.fps.isEnabled()
        panel.set_sequence_source(True)
        assert panel.fps.isEnabled()
    finally:
        panel.close()


def test_cutout_video_uses_one_persistent_alpha_preserving_decoder(monkeypatch, tmp_path):
    import io
    import numpy as np

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, prop):
            return {
                frames.cv2.CAP_PROP_FRAME_COUNT: 2,
                frames.cv2.CAP_PROP_FPS: 25.0,
                frames.cv2.CAP_PROP_FRAME_WIDTH: 1,
                frames.cv2.CAP_PROP_FRAME_HEIGHT: 1,
            }[prop]

        def release(self):
            pass

    class FakeProcess:
        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(bytes((0, 0, 255, 255, 255, 0, 0, 0)))

        def poll(self):
            return 0

    calls = []
    monkeypatch.setattr(frames.cv2, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(frames, "validate_video", lambda path: None)
    monkeypatch.setattr(frames.shutil, "which", lambda name: "/fake/ffmpeg")
    monkeypatch.setattr(frames.subprocess, "Popen", lambda *a, **k: (calls.append((a, k)) or FakeProcess()))
    source = frames.FrameSource(tmp_path / "cutout.mov", cutout=True)
    try:
        opaque_red = source.read(0)
        transparent = source.read(1)
        assert len(calls) == 1
        assert "bgra" in calls[0][0][0]
        assert tuple(opaque_red[0, 0]) == (255, 0, 0)
        assert tuple(transparent[0, 0]) == (184, 184, 184)
    finally:
        source.close()
