"""Local, bounded float32 RGBA transport. No pickle, commands, or file paths.

All values are little-endian. Rows run top to bottom and RGB is straight.
Request: magic[8], width/u32, height/u32, channels/u32, flags/u32,
despill/f32, payload_bytes/u32, then RGBA float32 pixels.
Response: magic[8], status/u32, payload_bytes/u32, then pixels or UTF-8 error.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import socket
import select
import stat
import struct
import sys
from functools import lru_cache

REQUEST = struct.Struct("<8sIIIIfI")
RESPONSE = struct.Struct("<8sII")
REQUEST_MAGIC = b"CK2REQ01"
RESPONSE_MAGIC = b"CK2RES01"
MAX_PIXELS = 4096 * 4096
MAX_ERROR = 16384
LINEAR = 1
MAX_SOCKET_PATH_BYTES = 103 if sys.platform == "darwin" else 107


@lru_cache(maxsize=1)
def _getpeereid():
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.getpeereid
    function.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint))
    function.restype = ctypes.c_int
    return function


def peer_uid(sock: socket.socket) -> int:
    """Authenticate the connected local peer using OS credentials, fail closed."""
    if sys.platform == "darwin":
        import ctypes
        uid, gid = ctypes.c_uint(), ctypes.c_uint()
        if _getpeereid()(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return uid.value
    if sys.platform.startswith("linux"):
        _, uid, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        return uid
    raise RuntimeError("The OFX worker supports Linux and macOS only")


def default_socket() -> Path:
    # Same default as the C++ client, independent of environment inheritance.
    return Path(f"/tmp/corridorkey-{os.getuid()}/engine.sock")


def private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"Socket directory must be owned by you with mode 0700: {path}")


def receive(sock: socket.socket, count: int) -> bytes:
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        n = sock.recv_into(view[offset:])
        if not n:
            raise ConnectionError("Peer disconnected before the frame was complete")
        offset += n
    return data


def parse_request(header: bytes) -> tuple[int, int, int, float, int]:
    magic, width, height, channels, flags, despill, size = REQUEST.unpack(header)
    if magic != REQUEST_MAGIC or channels != 4 or flags & ~LINEAR:
        raise ValueError("Unsupported CorridorKey protocol or pixel format")
    if not (1 <= width <= 8192 and 1 <= height <= 8192 and width * height <= MAX_PIXELS):
        raise ValueError("Frame exceeds the supported limit of 16 megapixels / 8192 per side")
    if size != width * height * 16:
        raise ValueError("Invalid RGBA payload length")
    if not math.isfinite(despill) or not 0 <= despill <= 1:
        raise ValueError("Despill must be between zero and one")
    return width, height, flags, despill, size


def send_error(sock: socket.socket, error: Exception) -> None:
    payload = str(error).encode("utf-8", "replace")[:MAX_ERROR]
    sock.sendall(RESPONSE.pack(RESPONSE_MAGIC, 1, len(payload)) + payload)


def disconnected(sock: socket.socket) -> bool:
    """Detect OFX cancellation without consuming request data or waiting."""
    try:
        readable, _, errors = select.select([sock], [], [sock], 0)
        if errors:
            return True
        if readable:
            # EOF is readable on both Darwin and Linux. Never consume bytes;
            # only this worker thread reads the connection.
            return sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except BlockingIOError:
        return False
    except OSError:
        return True
    return False
