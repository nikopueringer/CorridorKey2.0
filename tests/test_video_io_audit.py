from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from corridorkey.video_io import (
    SequentialHintFrameDirWindowReader,
    window_starts,
)


def test_window_starts_reject_invalid_chunk_and_stride():
    for chunk, stride in ((0, 1), (-1, 1), (10, 0), (10, 11)):
        with pytest.raises(ValueError):
            window_starts(20, chunk, stride)
    with pytest.raises(ValueError):
        window_starts(-1, 10, 5)


@pytest.mark.parametrize(
    ("frames", "chunk", "stride", "expected"),
    [
        (1, 4, 2, [0]),
        (4, 4, 4, [0]),
        (10, 4, 4, [0, 4, 6]),
        (10, 4, 2, [0, 2, 4, 6]),
        (10, 1, 1, list(range(10))),
    ],
)
def test_window_starts_cover_short_and_tail_windows(frames, chunk, stride, expected):
    starts = window_starts(frames, chunk, stride)
    assert starts == expected
    covered = {i for start in starts for i in range(start, min(start + chunk, frames))}
    assert covered == set(range(frames))


def _hint_meta(start_frame=2, num_frames=2):
    return {
        "start_frame": start_frame,
        "num_frames": num_frames,
        "shape_hw": (2, 2),
        "model_native_hw": (2, 2),
    }


def test_hint_frame_directory_applies_source_start_offset(tmp_path):
    for index, value in enumerate((0, 64, 128, 255)):
        frame = np.full((2, 2), value, np.uint8)
        assert cv2.imwrite(str(tmp_path / f"hint_{index}.png"), frame)
    reader = SequentialHintFrameDirWindowReader(str(tmp_path), _hint_meta())
    window = reader.window(0, 2)
    torch.testing.assert_close(
        window[:, 0, 0, 0], torch.tensor([128 / 255, 1.0], dtype=torch.bfloat16))


def test_hint_frame_directory_validates_absolute_required_length(tmp_path):
    for index, value in enumerate((0, 64, 128)):
        assert cv2.imwrite(str(tmp_path / f"hint_{index}.png"), np.full((2, 2), value, np.uint8))
    with pytest.raises(RuntimeError, match=r"need 4 for start=2, num_frames=2"):
        SequentialHintFrameDirWindowReader(str(tmp_path), _hint_meta())


def test_hint_frame_directory_validates_requested_window_bounds(tmp_path):
    for index, value in enumerate((0, 64, 128, 255)):
        assert cv2.imwrite(str(tmp_path / f"hint_{index}.png"), np.full((2, 2), value, np.uint8))
    reader = SequentialHintFrameDirWindowReader(str(tmp_path), _hint_meta())
    with pytest.raises(ValueError, match="outside"):
        reader.window(-1, 1)
    with pytest.raises(ValueError, match="outside"):
        reader.window(0, 3)
