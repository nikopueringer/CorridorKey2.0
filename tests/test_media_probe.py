from __future__ import annotations

import json
import subprocess
import shutil

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


def test_decoded_count_cache_tracks_source_replacement(monkeypatch, tmp_path):
    source = tmp_path / "clip.mov"
    source.touch()
    executable = tmp_path / "ffmpeg"
    executable.touch()
    calls = []
    monkeypatch.setattr(media_probe.shutil, "which", lambda _: str(executable))
    def count(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess([], 0, "frame=194\nprogress=end\n", "")
    monkeypatch.setattr(media_probe.subprocess, "run", count)
    assert media_probe.count_decoded_video_frames(source) == 194
    assert media_probe.count_decoded_video_frames(source) == 194
    assert len(calls) == 1
    source.write_bytes(b"replaced")
    assert media_probe.count_decoded_video_frames(source) == 194
    assert len(calls) == 2


@pytest.mark.parametrize("returncode,stdout,stderr", [
    (1, "frame=194\nprogress=end\n", "corrupt input packet"),
    (0, "frame=194\nprogress=end\n", "error while decoding"),
    (0, "frame=194\nprogress=continue\n", ""),
    (0, "frame=0\nprogress=end\n", ""),
])
def test_decoded_count_rejects_failed_or_incomplete_decode(monkeypatch, tmp_path,
                                                          returncode, stdout, stderr):
    source = tmp_path / "bad.mov"
    source.touch()
    executable = tmp_path / "ffmpeg"
    executable.touch()
    monkeypatch.setattr(media_probe.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(media_probe.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess([], returncode, stdout, stderr))
    with pytest.raises(ValueError):
        media_probe.count_decoded_video_frames(source)


def test_mp4_edit_list_uses_decoded_frames_for_full_clip_and_middle_range(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for the MP4 edit-list regression")
    source, edited = tmp_path / "original.mp4", tmp_path / "edited.mp4"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x48:rate=50:duration=0.24", "-c:v", "mpeg4",
                    "-bf", "2", "-g", "12", str(source)], check=True)
    subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", "0.04", "-i", str(source),
                    "-c", "copy", str(edited)], check=True)
    import cv2
    from corridorkey.video_io import probe_video_meta, iter_video_frame_chunks
    cap = cv2.VideoCapture(str(edited))
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    actual = media_probe.count_decoded_video_frames(edited, ffmpeg)
    assert declared == 12 and actual == 10
    options = dict(shape_quantum=16, native_resolution=True, start_mode="middle", seed=0,
                   model_scale=1.0, linear_brightness=1.0, linear_contrast=1.0,
                   linear_contrast_pivot=.18, decoded_frame_count=actual)
    middle = probe_video_meta(str(edited), num_frames=4, **options)
    assert middle["num_frames"] == 4 and middle["start_frame"] == 3
    full = probe_video_meta(str(edited), num_frames=-1, **options)
    assert full["total_frames"] == full["num_frames"] == 10
    full.update(input_color_space="sony_slog3_sgamut3_cine", ffmpeg_bin=ffmpeg)
    chunks = list(iter_video_frame_chunks(str(edited), full, chunk_size=4, need_model=False))
    assert sum(chunk.shape[0] for _, _, chunk in chunks) == 10
