"""Linux lifecycle tests for the Qt GUI process runners."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication

from gui import runner
from gui import resolve_engine
from gui import gpu
from gui.clips import Clip, DONE, FAILED, STOPPED
from gui.command import JobSettings


@pytest.fixture(scope="module")
def app():
    instance = QCoreApplication.instance() or QCoreApplication([])
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
