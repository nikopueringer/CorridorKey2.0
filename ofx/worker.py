"""Persistent CUDA worker for Linux OpenFX. Start with python -m ofx.worker.

One model, one inference at a time. OFX requests use independent single-frame
windows, so seeking is deterministic. For temporal keying and hints use the
desktop queue and import its EXR or ProRes 4444 output into Resolve.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import tempfile
import traceback

from .protocol import (LINEAR, REQUEST, RESPONSE, RESPONSE_MAGIC, default_socket,
                       disconnected, parse_request, private_directory, receive, send_error)


class Cancelled(Exception):
    """The host abandoned this frame; drop it and move to the next client."""


class Engine:
    def __init__(self, root: Path):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from corridorkey.runtime import RuntimePaths, load_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch installation.")
        self.model = load_model(RuntimePaths.from_package(root), device="cuda")
        self.model.requires_grad_(False)

    def key(self, pixels, linear: bool, despill: float, cancelled=lambda: False):
        import cv2
        import numpy as np
        import torch
        from corridorkey import color_utils
        from corridorkey.runtime import InferenceConfig, WriterConfig, run_inference

        if not np.isfinite(pixels).all():
            raise ValueError("Input contains NaN or infinite pixels")
        # Unique scratch directories prevent collisions with queue jobs or
        # another application. Never read/write a pathname received from a host.
        with tempfile.TemporaryDirectory(prefix="corridorkey-ofx-") as scratch:
            root = Path(scratch)
            source = root / "input"
            source.mkdir()
            if not cv2.imwrite(str(source / "frame.exr"), pixels[:, :, 2::-1]):
                raise RuntimeError("Could not write the input EXR")
            cfg = InferenceConfig(
                hann_chunk=1, hann_stride=1, low_vram=True, auto_vram=False,
                carry_hint=False, outputs={"cutout"}, output_format="exr", exr_dtype="float",
                frame_dir_linear=linear, despill_strength=despill, temp_dir=root / "scratch",
            )
            hooks = []
            def check_cancel(*_):
                if cancelled():
                    raise Cancelled()

            try:
                check_cancel()
                # A preceding frame can leave stage 2 on CUDA. Move both stages
                # out before foundations are uploaded by the low-VRAM runtime.
                self.model.stage1.cpu()
                self.model.stage2.cpu()
                torch.cuda.empty_cache()
                # Abort at model phase/window boundaries. A running GPU kernel
                # itself cannot be preempted safely by the host's cancellation.
                for module in (self.model.foundations, self.model.foundations.rvm,
                               self.model.stage1, self.model.stage2):
                    if module is not None:
                        hooks.append(module.register_forward_pre_hook(check_cancel))
                run_inference(self.model, str(source), str(root / "output"), cfg, WriterConfig(),
                              initial_hint_path=None, hint_video_path=None)
                check_cancel()
                files = list((root / "output").glob("*_cutout_exr/cutout_*.exr"))
                if len(files) != 1:
                    raise RuntimeError("Expected exactly one keyed EXR from the engine")
                result = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
                if result is None or result.shape != pixels.shape:
                    raise RuntimeError("Engine returned an invalid frame")
                result = np.ascontiguousarray(result[:, :, [2, 1, 0, 3]], dtype="<f4")
                if not linear:
                    result[:, :, :3] = color_utils.linear_to_srgb(result[:, :, :3])
                result[:, :, 3] = np.clip(result[:, :, 3], 0, 1) * np.clip(pixels[:, :, 3], 0, 1)
                if not np.isfinite(result).all():
                    raise RuntimeError("Engine returned non-finite pixels")
                return result
            finally:
                for hook in hooks:
                    hook.remove()
                # The persistent worker must not monopolize unused cached VRAM.
                torch.cuda.empty_cache()


def serve(path: Path, engine_factory=Engine) -> None:
    import numpy as np

    private_directory(path.parent)
    if len(os.fsencode(path)) >= 108:
        raise ValueError("Unix socket path is too long (maximum 107 bytes)")
    # Lock lives for the entire server lifetime. A second start cannot unlink
    # the first server's socket. O_NOFOLLOW rejects a substituted lock symlink.
    lock_fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A CorridorKey engine already owns this socket") from None
        engine = engine_factory(Path(__file__).resolve().parents[1])
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise PermissionError(f"Refusing to replace a non-owned socket: {path}")
            path.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(path))
            os.chmod(path, 0o600)
            server.listen(4)
            server.settimeout(1)
            running = True

            def stop(*_):
                nonlocal running
                running = False

            previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
            print(f"CK_ENGINE_READY {path}", flush=True)
            try:
                while running:
                    try:
                        conn, _ = server.accept()
                    except socket.timeout:
                        continue
                    with conn:
                        conn.settimeout(30)
                        try:
                            _, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                            if uid != os.getuid():
                                raise PermissionError("Only the current user may connect")
                            w, h, flags, despill, size = parse_request(receive(conn, REQUEST.size))
                            pixels = np.frombuffer(receive(conn, size), dtype="<f4").reshape(h, w, 4)
                            if disconnected(conn):
                                continue
                            result = engine.key(pixels, bool(flags & LINEAR), despill,
                                                cancelled=lambda: not running or disconnected(conn))
                            payload = np.ascontiguousarray(result, dtype="<f4").tobytes()
                            if len(payload) != size:
                                raise RuntimeError("Invalid engine response size")
                            conn.sendall(RESPONSE.pack(RESPONSE_MAGIC, 0, size) + payload)
                        except Cancelled:
                            print("Cancelled obsolete OFX frame", flush=True)
                        except Exception as exc:
                            traceback.print_exc()
                            try:
                                send_error(conn, exc)
                            except OSError:
                                pass
            finally:
                path.unlink(missing_ok=True)
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_socket())
    args = parser.parse_args()
    serve(args.socket.absolute())


if __name__ == "__main__":
    main()
