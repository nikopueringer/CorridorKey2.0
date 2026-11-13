from __future__ import annotations

import json
import subprocess

import pytest

from corridorkey import media_probe


def test_probe_reports_supported_dnxhd_and_caches_result(monkeypatch, tmp_path):
    path = tmp_path / "normal.mov"
    path.touch()
    completed = subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"streams": [{"codec_name": "dnxhd", "width": 1920,
                                        "height": 1080, "avg_frame_rate": "25/1",
                                        "nb_frames": "38"}]}),
        stderr="",
    )
    calls = []
    monkeypatch.setattr(media_probe.shutil, "which", lambda _: "/bin/ffprobe")
    monkeypatch.setattr(media_probe.subprocess, "run", lambda *a, **k: (calls.append(1) or completed))
    media_probe._probe_cached.cache_clear()

    info = media_probe.probe_video(path)
    again = media_probe.probe_video(path)

    assert info == again
    assert (info.codec, info.width, info.height, info.fps, info.frame_count) == (
        "dnxhd", 1920, 1080, 25.0, 38)
    assert len(calls) == 1


def test_probe_rejects_variable_act_with_actionable_transcode_guidance(monkeypatch, tmp_path):
    path = tmp_path / "unsupported.mov"
    path.touch()
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='{"streams": []}',
        stderr="Unsupported: variable ACT flag",
    )
    monkeypatch.setattr(media_probe.shutil, "which", lambda _: "/bin/ffprobe")
    monkeypatch.setattr(media_probe.subprocess, "run", lambda *a, **k: completed)
    media_probe._probe_cached.cache_clear()

    with pytest.raises(media_probe.UnsupportedVideoError) as raised:
        media_probe.validate_video(path)
    message = str(raised.value)
    assert "variable ACT flag" in message
    assert "ProRes 4444" in message
    assert "EXR frame sequence" in message
