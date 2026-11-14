import os
from pathlib import Path
import socket
import struct

import pytest

from ofx.protocol import (REQUEST, REQUEST_MAGIC, default_socket, parse_request,
                          private_directory, receive, peer_uid, disconnected)


def test_peer_credentials_and_non_consuming_disconnect_detection():
    a, b = socket.socketpair()
    with a, b:
        assert peer_uid(a) == peer_uid(b) == os.getuid()
        assert not disconnected(b)
        a.sendall(b"frame")
        assert not disconnected(b)
        assert receive(b, 5) == b"frame"
        assert not disconnected(b)
        a.close()
        assert disconnected(b)


def packet(w=16, h=8, channels=4, flags=0, despill=0.5, size=None):
    return REQUEST.pack(REQUEST_MAGIC, w, h, channels, flags, despill,
                        w * h * 16 if size is None else size)


def test_header():
    assert REQUEST.size == 32
    assert parse_request(packet()) == (16, 8, 0, 0.5, 2048)
    assert default_socket() == Path(f"/tmp/corridorkey-{os.getuid()}/engine.sock")


@pytest.mark.parametrize("kwargs", [{"w": 0}, {"h": 8193}, {"w": 8192, "h": 8192},
                                   {"channels": 3}, {"flags": 2}, {"despill": float("nan")},
                                   {"despill": -1}, {"despill": 1.1}, {"size": 1}])
def test_bad_headers_rejected_before_allocating(kwargs):
    with pytest.raises(ValueError):
        parse_request(packet(**kwargs))


def test_truncated_packet():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(b"short")
        a.shutdown(socket.SHUT_WR)
        with pytest.raises(ConnectionError):
            receive(b, 32)


def test_private_directory(tmp_path):
    p = tmp_path / "engine"
    private_directory(p)
    assert p.stat().st_mode & 0o777 == 0o700
    p.chmod(0o755)
    with pytest.raises(PermissionError):
        private_directory(p)
    link = tmp_path / "link"
    link.symlink_to(p, target_is_directory=True)
    with pytest.raises(PermissionError):
        private_directory(link)


def test_gui_sequence_order_matches_runtime(tmp_path):
    from gui.clips import frame_files
    from corridorkey.video_io import frame_paths
    for n in ("f_1.PNG", "f_10.png", "f_2.png", "notes.txt"):
        (tmp_path / n).touch()
    assert [str(p) for p in frame_files(tmp_path)] == frame_paths(tmp_path)
    assert [p.name for p in frame_files(tmp_path)] == ["f_1.PNG", "f_2.png", "f_10.png"]
