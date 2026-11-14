"""Independent camera-colour regression checks from manufacturer references.

These deliberately include reference points and primary chromaticities: an
incorrect forward/inverse pair can pass a self-round-trip test.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import pytest
import torch

from corridorkey import color_utils as cu
from corridorkey.camera_color import decode_to_working, encode_from_working
from corridorkey.runtime import (
    _encode_rendered_camera_source,
    _render_predictions_exr_torch,
    _render_predictions_torch,
    _replace_rendered_rgb_with_source,
)
from corridorkey.video_io import ExrSequenceWriter


# ARRI's exposure-domain EI800 table and LogC4 specification; Sony's S-Log3
# technical summary; Apple's September 2025 Log 2 white paper, table 1.
@pytest.mark.parametrize("profile,linear,encoded", [
    ("arri_logc3_awg3_ei800", 0.0, 0.092809),
    ("arri_logc3_awg3_ei800", 0.18, 0.39100683203408376),
    ("arri_logc4_awg4", 0.0, 95 / 1023),
    ("arri_logc4_awg4", 0.18, 0.2783958365482653),
    ("sony_slog3_sgamut3", 0.0, 95 / 1023),
    ("sony_slog3_sgamut3", 0.18, 420 / 1023),
    ("apple_log2_awg2", 0.18, 0.488272),
])
def test_manufacturer_neutral_reference_points(profile, linear, encoded):
    actual = encode_from_working(np.full((1, 3), linear, np.float32), profile)
    np.testing.assert_allclose(actual, encoded, atol=1e-6, rtol=0)
    recovered = decode_to_working(np.full((1, 3), encoded, np.float32), profile)
    np.testing.assert_allclose(recovered, linear, atol=3e-6, rtol=0)


def test_logc4_negative_code_uses_published_linear_extension():
    # ARRI LogC4 specification §4.1.2: E' < 0 decodes as E' * s + t.
    # Independent evaluated reference for E'=-0.1 (not a generated round trip).
    result = decode_to_working(np.full((1, 3), -0.1, np.float32), "arri_logc4_awg4")
    np.testing.assert_allclose(result, -0.029416716980970217, atol=2e-8, rtol=0)


def test_fujifilm_published_reflection_code_table():
    # https://dl.fujifilm-x.com/technical-data/F-Log2C_DataSheet_E_Ver.1.0.pdf
    reflection = np.repeat(np.array([0.0, 0.18, 0.9])[:, None], 3, axis=1)
    encoded = encode_from_working(reflection, "fuji_flog2c_f_gamut_c")
    np.testing.assert_array_equal(np.rint(encoded * 1023), [[95]*3, [400]*3, [570]*3])


_REC709_TO_XYZ = np.array([
    [0.4123907992659595, 0.35758433938387796, 0.1804807884018343],
    [0.21263900587151036, 0.7151686787677559, 0.07219231536073371],
    [0.01933081871559185, 0.11919477979462599, 0.9505321522496607],
])


@pytest.mark.parametrize("profile,primaries,black,one", [
    # Apple Log 2 white paper, September 2025, §3.
    ("apple_log2_awg2", [[.725, .301], [.221, .814], [.068, -.076]],
     47.28711236 * .05641088**2,
     .08550479 * np.log2(1 + .00964052) + .69336945),
    # Official Fujifilm F-Log2 C datasheet, §3; transfer constants in §2-3.
    ("fuji_flog2c_f_gamut_c", [[.73470, .26530], [.02630, .97370], [.11730, -.02240]],
     .092864, .245281 * np.log10(5.555556 + .064829) + .384316),
])
def test_camera_primaries_match_manufacturer_xy(profile, primaries, black, one):
    camera_codes = np.where(np.eye(3), one, black).astype(np.float32)
    working = decode_to_working(camera_codes, profile)
    xyz = working @ _REC709_TO_XYZ.T
    xy = xyz[:, :2] / xyz.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(xy, primaries, atol=2e-6, rtol=0)
    assert np.any(working < 0), "Wide-gamut primaries must survive as signed working RGB"


def _render_fixture(despill=0.0):
    alpha = np.array([[0, .125, .5], [.75, .9, 1]], np.float32)
    linear = np.repeat(np.array([[-.03, 0, .18], [1, 4, 32]], np.float32)[..., None], 3, -1)
    # Include a green-heavy pixel so changing despill genuinely changes the
    # model foreground before the original-source replacement is applied.
    linear[1, 1] = [.1, 4, .2]
    model_rgb = cu.linear_to_asinh(torch.from_numpy(linear.transpose(2, 0, 1)))
    logits = torch.cat((torch.from_numpy(alpha)[None], model_rgb), 0)[None]
    meta = {"model_native_hw": (2, 3), "output_hw": (2, 3),
            "pad": (0, 0, 0, 0), "despill_strength": despill}
    checker = torch.full((3, 2, 3), .5)
    return meta, checker, logits, linear, alpha


def test_processed_source_uses_unclamped_signed_hdr_foreground():
    meta, checker, logits, linear, alpha = _render_fixture()
    rendered = _render_predictions_exr_torch(meta, {"fg", "cutout", "alpha"}, checker, logits)
    converted = _encode_rendered_camera_source(rendered, "arri_logc4_awg4", video=False)
    expected = encode_from_working(linear, "arri_logc4_awg4")
    np.testing.assert_allclose(converted["fg"][0, ..., ::-1], expected, atol=2e-6, rtol=2e-6)
    assert converted["fg"][0, 0, 0].min() < 0, "Negative camera codes must not be display-clipped"
    np.testing.assert_array_equal(converted["cutout"][0, ..., 3], alpha)
    np.testing.assert_array_equal(converted["alpha"][0, ..., 0], alpha)


def test_camera_video_policy_keeps_the_same_linear_alpha_as_legacy():
    meta, checker, logits, _, _ = _render_fixture()
    outputs = {"alpha", "cutout"}
    legacy = _render_predictions_torch(meta, outputs, checker, logits)
    linear = _render_predictions_exr_torch(meta, outputs, checker, logits)
    camera = _encode_rendered_camera_source(linear, "arri_logc4_awg4", video=True)
    np.testing.assert_array_equal(camera["alpha"], legacy["alpha"])
    np.testing.assert_array_equal(camera["cutout"][..., 3], legacy["cutout"][..., 3])


def test_source_rgb_adjacent_16bit_codes_survive_float_exr_and_despill(tmp_path):
    codes = np.array([
        [[32000, 32001, 32002], [65533, 65534, 65535], [1, 2, 3]],
        [[4094, 4095, 4096], [50000, 50001, 50002], [0, 32767, 32768]],
    ], np.uint16)
    source = codes.astype(np.float32) / 65535
    expected_bgr = source[..., ::-1]
    for despill in (0.0, 1.0):
        meta, checker, logits, _, alpha = _render_fixture(despill)
        rendered = _render_predictions_exr_torch(meta, {"fg", "cutout"}, checker, logits)
        result = _replace_rendered_rgb_with_source(rendered, [source])
        np.testing.assert_array_equal(result["fg"][0], expected_bgr)
        np.testing.assert_array_equal(result["cutout"][0, ..., :3], expected_bgr)
        np.testing.assert_array_equal(result["cutout"][0, ..., 3], alpha)
        directory = tmp_path / f"despill-{despill}"
        writer = ExrSequenceWriter(str(directory), 25, (3, 2), channels=4,
                                   dtype="float", frame_name="cutout")
        try:
            writer.write_many(result["cutout"])
        finally:
            writer.release()
        decoded = cv2.imread(str(directory / "cutout_000000.exr"), cv2.IMREAD_UNCHANGED)
        assert decoded is not None
        np.testing.assert_array_equal(decoded, result["cutout"][0])
        np.testing.assert_array_equal(np.rint(decoded[..., 2::-1] * 65535).astype(np.uint16), codes)
