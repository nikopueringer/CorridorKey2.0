from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from corridorkey.video_io import ExrSequenceWriter, ThreadedVideoWriter, WriterConfig


def _float_frames(count=1):
    return np.zeros((count, 2, 2, 3), dtype=np.float32)


def _u16_rgba_frames(count=1):
    frames = np.full((count, 16, 16, 4), 32768, dtype=np.uint16)
    frames[..., 3] = 65535
    return frames


def test_exr_writer_reports_successfully_written_frames_and_repeated_close(tmp_path):
    writer = ExrSequenceWriter(str(tmp_path / "exr-success"), 25, (2, 2),
                               dtype="float", maxsize=1)
    writer.write_many(_float_frames(3))
    writer.release()
    assert writer.written_frames == 3
    writer.release()
    assert writer.written_frames == 3


def test_exr_disk_failure_wakes_blocked_producer_and_release_is_idempotent(
    monkeypatch, tmp_path
):
    entered, fail = threading.Event(), threading.Event()

    def fail_imwrite(*_args, **_kwargs):
        entered.set()
        assert fail.wait(3), "test did not release the synthetic disk failure"
        raise OSError("synthetic disk full")

    monkeypatch.setattr("corridorkey.video_io.cv2.imwrite", fail_imwrite)
    writer = ExrSequenceWriter(str(tmp_path / "exr"), 25, (2, 2),
                               dtype="float", maxsize=1)
    writer.write_many(_float_frames())
    assert entered.wait(2)
    writer.write_many(_float_frames())  # fill the one-slot queue
    blocked_result = []

    def submit_third_batch():
        try:
            writer.write_many(_float_frames())
        except Exception as exc:  # capture the worker's original failure
            blocked_result.append(exc)

    producer = threading.Thread(target=submit_third_batch)
    producer.start()
    time.sleep(0.1)
    assert producer.is_alive(), "third write should be waiting on the full queue"
    fail.set()
    producer.join(3)
    assert not producer.is_alive(), "producer stayed blocked after writer failure"
    assert blocked_result and isinstance(blocked_result[0], OSError)
    assert "synthetic disk full" in str(blocked_result[0])

    with pytest.raises(OSError, match="synthetic disk full") as first:
        writer.release()
    with pytest.raises(OSError, match="synthetic disk full") as second:
        writer.release()
    assert first.value is second.value is writer.err
    assert not writer.thread.is_alive()
    assert writer.q.empty()


def test_ffmpeg_early_exit_is_reported_and_child_is_reaped(tmp_path):
    if os.name == "nt":
        pytest.skip("test uses a POSIX fake encoder executable")
    fake_ffmpeg = tmp_path / "ffmpeg-exits"
    fake_ffmpeg.write_text("#!/bin/sh\nexit 23\n")
    fake_ffmpeg.chmod(0o755)
    writer = ThreadedVideoWriter(
        str(tmp_path / "failed.mov"), 25, (2, 2),
        WriterConfig(ffmpeg_bin=str(fake_ffmpeg), threads=1), maxsize=1,
    )
    # Ensure the process has exited before enqueueing so the worker encounters
    # the encoder failure while a normal producer/release path is active.
    deadline = time.monotonic() + 3
    while writer.proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.proc.poll() == 23
    try:
        writer.write_many(np.zeros((1, 2, 2, 3), dtype=np.uint16))
    except Exception:
        pass
    with pytest.raises(Exception):
        writer.release()
    with pytest.raises(Exception):
        writer.release()
    assert not writer.thread.is_alive()
    assert writer.proc.poll() is not None


def test_abort_fences_full_queue_and_kills_encoder_ignoring_sigterm(tmp_path):
    if os.name == "nt":
        pytest.skip("test uses POSIX process signals")
    marker = tmp_path / "encoder-started"
    fake_ffmpeg = tmp_path / "ffmpeg-ignores-term"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(marker)!r}, 'w').close()\n"
        "while True: time.sleep(0.1)\n"
    )
    fake_ffmpeg.chmod(0o755)
    writer = ThreadedVideoWriter(
        str(tmp_path / "stalled.mov"), 25, (1024, 1024),
        WriterConfig(ffmpeg_bin=str(fake_ffmpeg), threads=1), maxsize=1,
    )
    deadline = time.monotonic() + 3
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()

    frame = np.zeros((1, 1024, 1024, 3), dtype=np.uint16)
    writer.write_many(frame)
    deadline = time.monotonic() + 3
    while not writer.q.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert writer.q.empty(), "worker did not take the first frame"
    writer.write_many(frame)  # occupy the queue while the worker blocks on pipe IO
    producer_errors = []

    def blocked_submit():
        try:
            writer.write_many(frame)
        except Exception as exc:
            producer_errors.append(exc)

    producer = threading.Thread(target=blocked_submit)
    producer.start()
    time.sleep(0.1)
    assert producer.is_alive(), "third frame should block behind the full queue"
    release_errors = []

    def abort_writer():
        try:
            writer.release(abort=True)
        except Exception as exc:
            release_errors.append(exc)

    releaser = threading.Thread(target=abort_writer)
    releaser.start()
    producer.join(1)
    assert not producer.is_alive(), "abort did not fence the blocked producer"
    releaser.join(5)
    assert not releaser.is_alive(), "abort did not kill/reap the stalled encoder"
    assert producer_errors and isinstance(producer_errors[0], RuntimeError)
    assert release_errors  # cancellation may retain the expected broken-pipe error
    assert not writer.thread.is_alive()
    assert writer.proc.poll() is not None


def test_prores_stream_metadata_uses_frame_level_transfer_and_keeps_camera_tags_unset(
    tmp_path,
):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg and FFprobe are required")

    def probe(path):
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_primaries,color_transfer,color_space", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        return json.loads(result.stdout)["streams"][0]

    tagged_path = tmp_path / "tagged.mov"
    tagged = ThreadedVideoWriter(
        str(tagged_path), 25, (16, 16), WriterConfig(ffmpeg_bin=ffmpeg, threads=1),
        channels=4, color_trc="linear",
    )
    tagged.write_many(_u16_rgba_frames())
    tagged.release()
    assert tagged.written_frames == 1
    tagged_metadata = probe(tagged_path)
    assert tagged_metadata["color_primaries"] == "bt709"
    assert tagged_metadata["color_transfer"] == "linear"

    camera_path = tmp_path / "camera.mov"
    camera = ThreadedVideoWriter(
        str(camera_path), 25, (16, 16), WriterConfig(ffmpeg_bin=ffmpeg, threads=1),
        channels=4, color_trc="linear", tag_color=False,
    )
    camera.write_many(_u16_rgba_frames())
    camera.release()
    assert camera.written_frames == 1
    camera_metadata = probe(camera_path)
    assert camera_metadata.get("color_primaries", "unknown") == "unknown"
    assert camera_metadata.get("color_transfer", "unknown") == "unknown"
