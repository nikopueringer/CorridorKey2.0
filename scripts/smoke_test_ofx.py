#!/usr/bin/env python3
"""Send a real image to an already running OFX engine; save float RGBA output.

python scripts/smoke_test_ofx.py /path/to/frame.jpg /path/to/result.npy
The result uses straight display RGB by default, or linear RGB with --linear.
"""
import argparse
import os
from pathlib import Path
import socket
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from ofx.protocol import REQUEST, REQUEST_MAGIC, RESPONSE, RESPONSE_MAGIC, default_socket, receive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--socket", type=Path, default=default_socket())
    parser.add_argument("--linear", action="store_true")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB or RGBA image")
    if image.dtype == np.uint8:
        image = image.astype("<f4") / 255
    elif image.dtype == np.uint16:
        image = image.astype("<f4") / 65535
    image = image[:, :, [2, 1, 0, 3] if image.shape[2] == 4 else [2, 1, 0]]
    if image.shape[2] == 3:
        image = np.concatenate([image, np.ones((*image.shape[:2], 1), dtype="<f4")], axis=2)
    h, w = image.shape[:2]
    payload = np.ascontiguousarray(image, dtype="<f4").tobytes()
    started = time.monotonic()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(600)
        connection.connect(str(args.socket))
        connection.sendall(REQUEST.pack(REQUEST_MAGIC, w, h, 4, int(args.linear), 0.5, len(payload)) + payload)
        magic, status, size = RESPONSE.unpack(receive(connection, RESPONSE.size))
        if magic != RESPONSE_MAGIC or size > max(len(payload), 16384):
            raise ValueError("Invalid engine response")
        data = receive(connection, size)
    if status:
        raise RuntimeError(data.decode("utf-8", "replace"))
    result = np.frombuffer(data, dtype="<f4").reshape(h, w, 4)
    assert np.isfinite(result).all()
    assert np.all((result[:, :, 3] >= 0) & (result[:, :, 3] <= 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result)
    print(f"Keyed {w}x{h} in {time.monotonic()-started:.2f}s; alpha {result[:, :, 3].min():.5f}..{result[:, :, 3].max():.5f}")
    if args.compare:
        expected = np.load(args.compare, allow_pickle=False)
        assert expected.shape == result.shape
        print(f"Bit-identical: {np.array_equal(expected, result)}; max error: {np.abs(expected-result).max():.9g}")
        assert np.array_equal(expected, result)


if __name__ == "__main__":
    main()
