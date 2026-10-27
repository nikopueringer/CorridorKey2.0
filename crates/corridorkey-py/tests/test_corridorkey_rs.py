"""`corridorkey_rs` against the product's fp32 reference (`scripts/parity/pipeline_reference.py`).

    uv run python scripts/parity/pipeline_reference.py /tmp/corridorkey-e2e
    cd crates/corridorkey-py && maturin develop --release
    CORRIDORKEY_RUNTIME=cuda pytest crates/corridorkey-py/tests -s

One keyer per process at a time (an 8 GB GPU does not hold two), so the exit test, which starts its
own process, runs before this module loads its keyer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import corridorkey_rs as ck

ROOT = Path(__file__).resolve().parents[3]
REF = Path(os.environ.get("CORRIDORKEY_E2E", "/tmp/corridorkey-e2e"))
NAMES = {"alpha": "alpha", "fg": "fg", "cutout": "cutout", "checker": "checker_comp_despill050"}

needs_reference = pytest.mark.skipif(not (REF / "frames_rgb8.u8").exists(), reason=f"run pipeline_reference.py {REF}")


def reference_frames() -> np.ndarray:
    meta = json.loads((REF / "meta.json").read_text())
    t, h, w, _ = meta["shapes"]["alpha"]
    return np.fromfile(REF / "frames_rgb8.u8", dtype=np.uint8).reshape(t, h, w, 3)


def test_runtimes():
    names = ck.available_runtimes()
    assert names, "no runtime compiled in"
    assert ck.current_runtime() in names
    with pytest.raises(ValueError):
        ck.set_runtime("no-such-runtime")
    ck.set_runtime(names[0])
    assert ck.current_runtime() == names[0]
    ck.set_runtime(None)


@needs_reference
def test_process_exits_promptly():
    """The device shutdown at exit must not wait out CubeCL's 30 s runner join."""
    script = (
        "import sys, numpy as np, corridorkey_rs as ck\n"
        f"k = ck.Keyer(root={str(ROOT)!r})\n"
        f"f = np.fromfile({str(REF / 'frames_rgb8.u8')!r}, dtype=np.uint8).reshape(12, 420, 664, 3)\n"
        "k.key_frames(f[:4], outputs=['alpha'])\n"
        "print('done', flush=True)\n"
    )
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "done"
    t = time.monotonic()
    assert p.wait(timeout=60) == 0
    assert time.monotonic() - t < 5, f"exit took {time.monotonic() - t:.1f} s"


@pytest.fixture(scope="module")
def keyer() -> ck.Keyer:
    return ck.Keyer(root=str(ROOT), precision="f32")


@needs_reference
def test_key_frames_matches_the_product(keyer):
    frames = reference_frames()
    out = keyer.key_frames(frames, outputs=list(NAMES), hann_chunk=8, hann_stride=4)
    for name, ref_name in NAMES.items():
        got = out[name]
        want = np.fromfile(REF / f"{ref_name}.f32", dtype=np.float32).reshape(got.shape[:3] + (-1,))
        want = want[..., 0] if got.ndim == 3 else want
        if got.ndim == 4:  # the reference is as written (B G R (A)); the bindings return R G B (A)
            want = want[..., [2, 1, 0, 3][: want.shape[-1]]]
        err = np.abs(got.astype(np.float64) - want)
        rel = (err / np.maximum(np.abs(want), 1.0)).max()
        print(f"{keyer.runtime} {name:8}: max {rel:.3e} (relative where |v| > 1), mean {err.mean():.3e}")
        assert rel < 1e-2 and err.mean() < 1e-4, name


HINTS_REF = Path(os.environ.get("CORRIDORKEY_E2E_HINTS", "/tmp/corridorkey-e2e-hints"))


@pytest.mark.skipif(not (HINTS_REF / "hints.f32").exists(), reason=f"run pipeline_reference.py {HINTS_REF} --hints")
def test_key_frames_with_hints_matches_the_product(keyer):
    frames = np.fromfile(HINTS_REF / "frames_rgb8.u8", dtype=np.uint8).reshape(12, 420, 664, 3)
    first = np.fromfile(HINTS_REF / "hint_first.f32", dtype=np.float32).reshape(420, 664)
    hints = np.fromfile(HINTS_REF / "hints.f32", dtype=np.float32).reshape(12, 420, 664)
    out = keyer.key_frames(frames, outputs=["alpha"], hann_chunk=8, hann_stride=4, hint_first_frame=first, hints=hints)
    want = np.fromfile(HINTS_REF / "alpha.f32", dtype=np.float32).reshape(12, 420, 664)
    err = np.abs(out["alpha"].astype(np.float64) - want)
    print(f"{keyer.runtime} alpha with hints: max {err.max():.3e}, mean {err.mean():.3e}")
    assert err.max() < 1e-2 and err.mean() < 1e-4


@needs_reference
def test_key_clip_writes_the_product_files(keyer, tmp_path):
    report = keyer.key_clip(str(REF / "frames"), str(tmp_path), outputs=["alpha", "fg"], hann_chunk=8, hann_stride=4)
    assert report["frames"] == 12 and (report["width"], report["height"]) == (664, 420)
    clip = report["clip"]
    assert [Path(p).name for p in report["written"]] == [f"{clip}_alpha.mp4", f"{clip}_fg.mp4"]
    for p in report["written"]:
        assert Path(p).stat().st_size > 0
    assert (tmp_path / f"{clip}_settings.txt").exists()


@needs_reference
def test_keying_releases_the_gil(keyer):
    frames = reference_frames()[:4]
    ticks = 0
    done = threading.Event()

    def count():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)

    t = threading.Thread(target=count)
    t.start()
    start = time.monotonic()
    keyer.key_frames(frames, outputs=["alpha"])
    elapsed = time.monotonic() - start
    done.set()
    t.join()
    # A held GIL would leave the counter near zero for the whole call.
    assert ticks > 0.25 * elapsed * 1000 * 0.5, f"{ticks} ticks in {elapsed:.2f} s"


def test_rejects_bad_input(keyer):
    with pytest.raises(ValueError):
        keyer.key_frames(np.zeros((2, 64, 64, 4), np.uint8))
    with pytest.raises(ValueError):
        keyer.key_frames(np.zeros((2, 64, 64, 3), np.uint8), outputs=["nope"])
