"""Unix socket lifecycle tests for the persistent OFX worker."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import shutil
import socket
import struct
import tempfile
import time

import numpy as np
import pytest

from ofx.protocol import (LINEAR, REQUEST, REQUEST_MAGIC, RESPONSE, RESPONSE_MAGIC,
                          receive)
from ofx.worker import serve


class FakeEngine:
    """Tiny deterministic engine; marker 1 blocks until its release file exists."""

    def __init__(self, root: Path, calls: str, release: str, started: str):
        self.calls = Path(calls)
        self.release = Path(release)
        self.started = Path(started)

    def key(self, pixels, linear: bool, despill: float, cancelled=lambda: False):
        marker = int(pixels[0, 0, 0])
        with self.calls.open("a") as stream:
            stream.write(f"{marker},{int(linear)},{despill}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if marker == 1:
            self.started.touch()
            deadline = time.monotonic() + 8
            while not self.release.exists() and time.monotonic() < deadline:
                if cancelled():
                    raise RuntimeError("unexpected cancellation")
                time.sleep(0.01)
        return np.array(pixels, copy=True)


class FakeFactory:
    def __init__(self, calls: str, release: str, started: str):
        self.args = calls, release, started

    def __call__(self, root):
        return FakeEngine(root, *self.args)


@pytest.fixture
def worker(tmp_path):
    # Keep AF_UNIX names below Darwin's 104-byte and Linux's 108-byte limits.
    run_dir = Path(tempfile.mkdtemp(prefix="ckw-", dir="/tmp"))
    sock_path = run_dir / "e.sock"
    calls = run_dir / "calls"
    release = run_dir / "release"
    started = run_dir / "started"
    proc = multiprocessing.get_context("spawn").Process(
        target=serve,
        args=(sock_path,),
        kwargs={"engine_factory": FakeFactory(str(calls), str(release), str(started))},
    )
    proc.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not sock_path.exists() and proc.is_alive():
        time.sleep(0.01)
    assert sock_path.exists(), "worker failed to create its Unix socket"
    try:
        yield sock_path, calls, release, started, proc
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(3)
        if proc.is_alive():
            proc.kill()
            proc.join(3)
        shutil.rmtree(run_dir, ignore_errors=True)


def connect(path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(3)
    sock.connect(str(path))
    return sock


def send_request(sock, marker=2, *, header=None, linear=True):
    pixels = np.full((1, 1, 4), marker, dtype="<f4")
    if header is None:
        header = REQUEST.pack(REQUEST_MAGIC, 1, 1, 4, LINEAR if linear else 0,
                              0.25, pixels.nbytes)
    sock.sendall(header + pixels.tobytes())
    return pixels


def get_response(sock):
    magic, status, size = RESPONSE.unpack(receive(sock, RESPONSE.size))
    payload = receive(sock, size)
    assert magic == RESPONSE_MAGIC
    if status:
        return status, payload.decode("utf-8", "replace")
    return status, np.frombuffer(payload, dtype="<f4").reshape(1, 1, 4)


def wait_for(path, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    assert path.exists(), f"timed out waiting for {path.name}"


def test_complete_queued_request_is_discarded_after_client_closes(worker):
    path, calls, release, started, proc = worker
    first = connect(path)
    first_pixels = send_request(first, 1)
    wait_for(started)

    abandoned = connect(path)
    send_request(abandoned, 2)
    abandoned.close()
    release.touch()

    status, result = get_response(first)
    first.close()
    assert status == 0
    np.testing.assert_array_equal(result, first_pixels)
    # A later sentinel proves the server accepted/handled the abandoned request.
    sentinel = connect(path)
    sentinel_pixels = send_request(sentinel, 3)
    status, result = get_response(sentinel)
    sentinel.close()
    assert status == 0
    np.testing.assert_array_equal(result, sentinel_pixels)
    assert calls.read_text().splitlines() == ["1,1,0.25", "3,1,0.25"]
    assert proc.is_alive()


def test_bad_request_does_not_stop_server_and_valid_request_succeeds(worker):
    path, calls, release, started, proc = worker
    bad = connect(path)
    invalid = REQUEST.pack(REQUEST_MAGIC, 1, 1, 4, 2, 0.25, 16)
    bad.sendall(invalid)
    status, message = get_response(bad)
    bad.close()
    assert status == 1
    assert "Unsupported" in message

    valid = connect(path)
    pixels = send_request(valid, 7, linear=False)
    status, result = get_response(valid)
    valid.close()
    assert status == 0
    np.testing.assert_array_equal(result, pixels)
    assert calls.read_text().splitlines() == ["7,0,0.25"]
    assert proc.is_alive()


def test_second_worker_cannot_unlink_or_disrupt_first(worker):
    path, calls, release, started, proc = worker
    second = multiprocessing.get_context("spawn").Process(
        target=serve,
        args=(path,),
        kwargs={"engine_factory": FakeFactory(str(calls), str(release), str(started))},
    )
    second.start()
    second.join(3)
    if second.is_alive():
        second.terminate()
        second.join(2)
        pytest.fail("second worker did not reject the held socket lock")
    assert second.exitcode != 0
    assert path.exists()

    valid = connect(path)
    pixels = send_request(valid, 9)
    status, result = get_response(valid)
    valid.close()
    assert status == 0
    np.testing.assert_array_equal(result, pixels)
    assert calls.read_text().splitlines() == ["9,1,0.25"]
    assert proc.is_alive()
