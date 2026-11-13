from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch
from torch import nn

import corridorkey.runtime as runtime
from corridorkey.outputs import OutputSnapshot, read_outputs
from corridorkey.video_io import WriterConfig


class _Stage2(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, rgb, _baton):
        b, t, _, h, w = rgb.shape
        return torch.zeros((b, t, 4, h, w), device=rgb.device) + self.anchor


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self._runtime_device = torch.device("cpu")
        self.foundations = nn.Module()
        self.foundations.rvm = None
        self.stage1 = nn.Module()
        self.stage2 = _Stage2()


def _make_source(path: Path, count=2):
    path.mkdir(parents=True)
    for index in range(count):
        rgb = np.full((16, 16, 3), 64 + 64 * index, np.uint8)
        assert cv2.imwrite(str(path / f"frame_{index:04d}.png"), rgb)
    return path


def _fake_hann(_model, _source, meta, *, on_baton_frames, **_kwargs):
    if on_baton_frames is None:
        return
    h, w = meta["shape_hw"]
    d16 = h // 16
    items = [
        (index, np.zeros((1, d16, w // 16), np.float16),
         torch.zeros((3, h, w), dtype=torch.bfloat16))
        for index in range(meta["num_frames"])
    ]
    on_baton_frames(items)


def _fake_render(meta, outputs, _checker, logits):
    t = logits.shape[0]
    h, w = meta["output_hw"]
    result = {}
    for name in sorted(outputs):
        if name == "alpha":
            result[name] = np.full((t, h, w, 1), 0.625, np.float32)
        elif name == "fg":
            result[name] = np.full((t, h, w, 3), 0.25, np.float32)
        elif name == "cutout":
            result[name] = np.full((t, h, w, 4), 0.5, np.float32)
        elif name == "checker":
            result[name] = np.full((t, h, w, 3), 0.75, np.float32)
    return result


def _config(tmp_path, outputs=("alpha", "fg")):
    return runtime.InferenceConfig(
        hann_chunk=1,
        hann_stride=1,
        shape_quantum=1,
        outputs=set(outputs),
        output_format="exr",
        exr_dtype="float",
        frame_dir_fps=25,
        temp_dir=tmp_path / "temp",
    )


def _install_fake_pipeline(monkeypatch):
    monkeypatch.setattr(runtime, "hann_stage1_stream", _fake_hann)
    monkeypatch.setattr(runtime, "_render_predictions_exr_torch", _fake_render)


def _run(model, source, output, cfg):
    return runtime.run_inference(
        model, str(source), str(output), cfg, WriterConfig(),
        initial_hint_path=None, hint_video_path=None,
    )


def test_run_inference_commits_real_exr_outputs_and_returns_generation_paths(
    monkeypatch, tmp_path
):
    _install_fake_pipeline(monkeypatch)
    source = _make_source(tmp_path / "source")
    output = tmp_path / "output"
    snapshot = _run(_Model(), source, output, _config(tmp_path))

    assert isinstance(snapshot, OutputSnapshot)
    assert set(snapshot) == {"alpha", "fg"}
    assert snapshot.metadata["schema_version"] == 2
    assert snapshot.metadata["outputs"]["alpha"]["frame_count"] == 2
    for name, directory in snapshot.items():
        assert directory.is_dir()
        frames = sorted(directory.glob("*.exr"))
        assert len(frames) == 2
        pixels = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
        assert pixels is not None
        assert pixels.dtype == np.float32
        expected_shape = (16, 16) if name == "alpha" else (16, 16, 3)
        assert tuple(pixels.shape) == expected_shape
        expected_value = 0.625 if name == "alpha" else 0.25
        np.testing.assert_allclose(pixels, expected_value, atol=1e-6)
    assert snapshot["alpha"].parent.parent.name == "renders"
    assert read_outputs(output, source, "source") == snapshot


def test_partial_write_and_release_failures_do_not_publish_or_mask_primary_error(
    monkeypatch, tmp_path
):
    _install_fake_pipeline(monkeypatch)
    source = _make_source(tmp_path / "source")
    output = tmp_path / "output"
    before = _run(_Model(), source, output, _config(tmp_path))
    manifest_path = output / "source_color.json"
    previous_manifest = manifest_path.read_bytes()

    actual_writer = runtime.ExrSequenceWriter
    release_attempts = []

    def failing_writer(*args, **kwargs):
        writer = actual_writer(*args, **kwargs)
        writer_name = Path(args[0]).name
        actual_write_many = writer.write_many
        actual_release = writer.release

        def write_many(frames):
            if writer_name == "source_fg_exr":
                raise OSError("synthetic inference write failure")
            return actual_write_many(frames)

        def release(abort=False):
            release_attempts.append((writer_name, abort))
            try:
                return actual_release(abort=abort)
            finally:
                raise RuntimeError(f"synthetic cleanup failure: {writer_name}")

        writer.write_many = write_many
        writer.release = release
        return writer

    monkeypatch.setattr(runtime, "ExrSequenceWriter", failing_writer)
    with pytest.raises(OSError, match="synthetic inference write failure"):
        _run(_Model(), source, output, _config(tmp_path))

    assert {name for name, _abort in release_attempts} == {"source_alpha_exr", "source_fg_exr"}
    assert all(abort for _name, abort in release_attempts)
    assert manifest_path.read_bytes() == previous_manifest
    after = read_outputs(output, source, "source")
    assert after == before
    assert not list((output / ".staging").glob("*"))


def test_close_readers_attempts_all_and_preserves_active_error(capsys):
    calls = []

    class BadReader:
        def __init__(self, name):
            self.name = name

        def close(self):
            calls.append(self.name)
            raise RuntimeError(f"close failed: {self.name}")

    first, second = BadReader("first"), BadReader("second")
    primary = ValueError("inference failed")
    try:
        raise primary
    except ValueError:
        runtime._close_readers(first, None, second)
    assert calls == ["first", "second"]
    captured = capsys.readouterr()
    assert "close failed: first" in captured.out
    assert "close failed: second" in captured.out


def test_ofx_engine_consumes_committed_path_returned_by_runtime(monkeypatch):
    from ofx.worker import Engine
    from corridorkey import color_utils

    engine = object.__new__(Engine)
    engine.device = torch.device("cpu")
    engine.model = _Model()
    pixels = np.zeros((2, 2, 4), dtype=np.float32)
    pixels[..., :3] = (0.2, 0.4, 0.6)
    pixels[..., 3] = 0.8
    observed = {}

    def fake_run_inference(_model, _input, output_dir, *_args, **_kwargs):
        generation = Path(output_dir) / "renders" / "committed-generation" / "cutout_exr"
        generation.mkdir(parents=True)
        # cv2 stores the EXR as BGRA; Engine.key converts the returned frame to RGBA.
        rendered = np.zeros((2, 2, 4), np.float32)
        rendered[..., :3] = (0.75, 0.5, 0.25)
        rendered[..., 3] = 0.5
        assert cv2.imwrite(str(generation / "cutout_000000.exr"), rendered)
        observed["returned"] = generation
        return OutputSnapshot({"cutout": generation}, {"schema_version": 2})

    monkeypatch.setattr(runtime, "run_inference", fake_run_inference)
    result = engine.key(pixels, linear=False, despill=0.0)
    assert observed["returned"].name == "cutout_exr"
    np.testing.assert_allclose(
        result[..., :3], color_utils.linear_to_srgb(
            np.broadcast_to(np.array([0.25, 0.5, 0.75], np.float32), (2, 2, 3)).copy()),
        atol=1e-6,
    )
    np.testing.assert_allclose(result[..., 3], 0.4, atol=1e-6)


def test_cli_rejects_invalid_stride_before_loading_model(monkeypatch, tmp_path):
    import infer

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    output_dir = tmp_path / "outputs"
    loaded = []
    monkeypatch.setattr(sys, "argv", [
        "infer.py", "--input", str(input_dir), "--output_dir", str(output_dir),
        "--hann_chunk", "2", "--hann_stride", "3",
    ])
    monkeypatch.setattr(infer, "load_model", lambda *a, **k: loaded.append(True))
    with pytest.raises(SystemExit) as exc:
        infer.main()
    assert exc.value.code == 2
    assert loaded == []

