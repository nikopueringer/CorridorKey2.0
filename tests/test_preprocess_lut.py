"""Regression checks for the uint8 preprocessing LUT against the old float path."""

import os
import sys
import types

import cv2
import numpy as np
import pytest
import torch


def _load_video_io():
    # Avoid corridorkey.__init__ importing the full model and its optional deps.
    root = os.path.join(os.path.dirname(__file__), "..", "corridorkey")
    package = types.ModuleType("corridorkey")
    package.__path__ = [root]
    sys.modules.setdefault("corridorkey", package)
    from corridorkey import video_io
    return video_io


def _original_video_views(vio, frame_bgr, meta, *, need_model):
    """Reference copied from origin/master's pre-LUT implementation."""
    h_out, w_out = meta["shape_hw"]
    h_model, w_model = meta["model_native_hw"]
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frame = vio.resize_hwc_lanczos(frame, (h_model, w_model))
    frame, _ = vio.center_pad_to_shape(frame, h_out, w_out)
    review = torch.from_numpy(frame.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    if not need_model and not vio.has_linear_adjust(meta):
        return None, review, review
    linear = vio.cu.srgb_to_linear(frame)
    linear_model = vio.apply_linear_adjust(linear, meta)
    if vio.has_linear_adjust(meta):
        foundation_np = vio.cu.linear_to_vfm_sdr(linear_model)
        foundation = torch.from_numpy(
            foundation_np.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    else:
        foundation = review
    if not need_model:
        return None, foundation, review
    model = vio.cu.linear_to_asinh(linear_model)
    model_t = torch.from_numpy(model.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    return model_t, foundation, review


def _original_frame_dir_views(vio, path, meta, *, need_model):
    """Reference copied from origin/master's pre-LUT implementation."""
    h_out, w_out = meta["shape_hw"]
    h_model, w_model = meta["model_native_hw"]
    display, linear = vio._read_frame_dir_views(
        path, exr_linear=bool(meta.get("exr_linear", False)))
    display = vio.resize_hwc_lanczos(display, (h_model, w_model))
    linear = vio.resize_hwc_lanczos(linear, (h_model, w_model))
    display, _ = vio.center_pad_to_shape(
        np.clip(display, 0.0, 1.0), h_out, w_out)
    display_t = torch.from_numpy(
        display.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    if not need_model:
        return None, display_t, display_t
    linear, _ = vio.center_pad_to_shape(linear, h_out, w_out)
    model = vio.cu.linear_to_asinh(linear)
    model_t = torch.from_numpy(model.transpose(2, 0, 1)).to(dtype=torch.bfloat16)
    return model_t, display_t, display_t


def _assert_tensor_tuple_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            assert got.dtype == torch.bfloat16
            assert torch.equal(got, want)


def _padded_meta(h, w, *, model_hw=None, gain=1.37, contrast=0.73, pivot=0.21):
    model_hw = model_hw or (h, w)
    out_hw = (model_hw[0] + 3, model_hw[1] + 5)
    return {
        "model_native_hw": model_hw,
        "shape_hw": out_hw,
        "linear_brightness": gain,
        "linear_contrast": contrast,
        "linear_contrast_pivot": pivot,
        "exr_linear": False,
    }


def _all_codes_odd_bgr():
    # Each RGB channel sees every byte value; all three have different orders.
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = np.stack((codes, codes[::-1], np.roll(codes, 5, axis=1)), axis=-1)
    return np.ascontiguousarray(rgb[..., ::-1])  # OpenCV/video input is BGR.


@pytest.mark.parametrize("need_model", [True, False])
def test_video_uint8_lut_matches_original_for_all_codes_and_odd_padding(need_model):
    vio = _load_video_io()
    frame_bgr = _all_codes_odd_bgr()
    meta = _padded_meta(*frame_bgr.shape[:2])

    actual = vio.preprocess_video_frame_views(frame_bgr, meta, need_model=need_model)
    expected = _original_video_views(vio, frame_bgr, meta, need_model=need_model)

    _assert_tensor_tuple_equal(actual, expected)


def test_video_uint8_lut_matches_original_without_adjustment_when_model_not_needed():
    vio = _load_video_io()
    frame_bgr = _all_codes_odd_bgr()
    meta = _padded_meta(*frame_bgr.shape[:2], gain=1.0, contrast=1.0)

    actual = vio.preprocess_video_frame_views(frame_bgr, meta, need_model=False)
    expected = _original_video_views(vio, frame_bgr, meta, need_model=False)

    _assert_tensor_tuple_equal(actual, expected)
    assert actual[1] is actual[2]


def test_scaled_video_input_uses_float_fallback_and_matches_original():
    vio = _load_video_io()
    frame_bgr = _all_codes_odd_bgr()
    # A 15x13 model input forces resize from 16x16 before odd-size padding.
    meta = _padded_meta(*frame_bgr.shape[:2], model_hw=(15, 13))

    actual = vio.preprocess_video_frame_views(frame_bgr, meta, need_model=True)
    expected = _original_video_views(vio, frame_bgr, meta, need_model=True)

    _assert_tensor_tuple_equal(actual, expected)


@pytest.mark.parametrize("gain,contrast", [(1.0, 1.0), (1.37, 0.73)])
def test_png16_uses_original_imread_color_quantization(tmp_path, gain, contrast):
    vio = _load_video_io()
    codes = np.arange(256, dtype=np.uint16).reshape(16, 16) * 257
    bgr16 = np.stack((codes, np.flipud(codes), np.fliplr(codes)), axis=-1)
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), bgr16)
    # Include actual frame-dir defaults and adjusted metadata to preserve the
    # original float path behavior for direct callers with custom metadata.
    meta = _padded_meta(16, 16, gain=gain, contrast=contrast)

    # origin/master's read_image_rgb uses IMREAD_COLOR for non-EXR, so 16-bit
    # PNG is reduced by OpenCV to 8-bit before float normalization. Lock that
    # existing behavior down while also comparing every preprocessed output.
    expected = _original_frame_dir_views(vio, str(path), meta, need_model=True)
    got_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert got_bgr is not None and got_bgr.dtype == np.uint8
    actual = vio.preprocess_frame_dir_path(str(path), meta, need_model=True)

    _assert_tensor_tuple_equal(actual, expected)
