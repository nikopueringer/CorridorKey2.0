import shutil
import subprocess

import numpy as np
import pytest
import torch

from corridorkey.camera_video import CameraVideoReader, read_frame_dir_codes
from corridorkey.video_io import preprocess_camera_code_views
from corridorkey.camera_color import decode_to_working, encode_from_working
from corridorkey.runtime import (
    _encode_rendered_camera_source,
    _replace_rendered_rgb_with_source,
    _write_color_manifest,
)
from corridorkey import color_utils as cu


def test_ffmpeg_camera_reader_preserves_16bit_rgb_and_absolute_frame_index(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for camera decode tests")
    h, w = 2, 3
    frames = np.array([
        [[[0, 95, 1023], [1024, 2345, 4096], [8192, 12000, 16384]],
         [[32768, 40000, 49152], [55000, 60000, 64000], [65000, 65534, 65535]]],
        [[[65535, 50000, 40000], [30000, 20000, 10000], [8000, 4000, 1000]],
         [[512, 1024, 2048], [3072, 4096, 5120], [6144, 7168, 8192]]],
    ], dtype=np.uint16)
    raw_path = tmp_path / "rgb48.raw"
    video_path = tmp_path / "camera.mkv"
    raw_path.write_bytes(frames.astype("<u2").tobytes())
    subprocess.run([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{w}x{h}",
        "-framerate", "24", "-i", str(raw_path), "-c:v", "ffv1",
        "-level", "3", "-pix_fmt", "gbrp16le", str(video_path),
    ], check=True)

    reader = CameraVideoReader(str(video_path), width=w, height=h,
                               ffmpeg_bin=ffmpeg, total_frames=2)
    try:
        got_second = reader.read_code_frame(1)
    finally:
        reader.close()
    expected = frames[1].astype(np.float32) / 65535.0
    np.testing.assert_allclose(got_second, expected, atol=1 / 65535, rtol=0)
    assert np.unique(got_second).size > 10
    assert not np.array_equal(got_second, np.round(got_second * 255) / 255)


def test_frame_directory_reader_preserves_uint16_codes(tmp_path):
    import cv2

    rgb = np.array([[[17, 1023, 40000], [65534, 2011, 52000]]], dtype=np.uint16)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    decoded = read_frame_dir_codes(str(path))
    np.testing.assert_array_equal(decoded, rgb.astype(np.float32) / 65535.0)


def test_camera_preprocess_decodes_hdr_before_resize_and_padding():
    # Distinct 16-bit codes at an odd native size exercise both high-bit input
    # preservation and the required decode-linear -> Lanczos -> reflect-pad order.
    codes = np.linspace(8000, 52000, 5 * 7 * 3, dtype=np.uint16).reshape(5, 7, 3)
    rgb = codes.astype(np.float32) / 65535.0
    meta = {
        "input_color_space": "red_log3g10_redwidegamut",
        "shape_hw": (8, 10),
        "model_native_hw": (6, 8),
        "linear_brightness": 1.15,
        "linear_contrast": 1.1,
        "linear_contrast_pivot": .18,
    }
    model, foundation, review = preprocess_camera_code_views(rgb, meta, need_model=True)

    # Independent float oracle from the public transform API, followed by the
    # documented preprocessing math and the same tensor precision boundary.
    linear = decode_to_working(rgb, meta["input_color_space"])
    linear = meta["linear_brightness"] * (
        meta["linear_contrast_pivot"] + meta["linear_contrast"] *
        (linear - meta["linear_contrast_pivot"]))
    import cv2
    linear = cv2.resize(linear, (8, 6), interpolation=cv2.INTER_LANCZOS4)
    linear = cv2.copyMakeBorder(linear, 1, 1, 1, 1, cv2.BORDER_REFLECT_101)
    expected_model = torch.from_numpy(
        cu.linear_to_asinh(linear).transpose(2, 0, 1)).to(torch.bfloat16)
    assert torch.equal(model, expected_model)
    assert model.shape == (3, 8, 10)
    assert foundation.shape == review.shape == model.shape

    # Code precision survives the transform path: this is not the 8-bit LUT path.
    rounded = np.round(rgb * 255).astype(np.uint8).astype(np.float32) / 255
    rounded_model, _, _ = preprocess_camera_code_views(rounded, meta, need_model=True)
    assert not torch.equal(model, rounded_model)


@pytest.mark.parametrize("profile", [
    "red_log3g10_redwidegamut", "canon_log3_cinema_gamut",
    "panasonic_vlog_v_gamut", "samsung_log_bt2020", "apple_log2_awg2",
])
def test_camera_profile_encode_decode_roundtrip_including_overrange(profile):
    source = np.array([[.09, .18, .3], [.45, .8, 1.2]], dtype=np.float32)
    working = decode_to_working(source, profile)
    recovered = encode_from_working(working, profile)
    np.testing.assert_allclose(recovered, source, atol=2e-5, rtol=2e-5)


def test_source_rgb_policy_keeps_original_channels_and_predicted_alpha():
    source = [np.array([[[.1, .3, .8], [.95, .2, .05]]], dtype=np.float32)]
    alpha = np.array([[[[.25], [.75]]]], dtype=np.float32)
    rendered = {
        "fg": np.zeros((1, 1, 2, 3), dtype=np.float32),
        "cutout": np.concatenate((np.zeros((1, 1, 2, 3), dtype=np.float32), alpha), -1),
    }
    result = _replace_rendered_rgb_with_source(rendered, source)
    np.testing.assert_array_equal(result["fg"], source[0][..., ::-1][None])
    np.testing.assert_array_equal(result["cutout"][..., :3], source[0][..., ::-1][None])
    np.testing.assert_array_equal(result["cutout"][..., 3:4], alpha)


def test_processed_source_exr_preserves_overrange_rgb_and_alpha():
    rendered = {
        "fg": np.full((1, 1, 1, 3), 1000.0, dtype=np.float32),
        "cutout": np.array([[[[1000.0, 1000.0, 1000.0, .5]]]], dtype=np.float32),
    }
    result = _encode_rendered_camera_source(
        rendered, "red_log3g10_redwidegamut", video=False)
    assert result["fg"].max() > 1.0
    assert result["cutout"][0, 0, 0, :3].max() > 1.0
    assert result["cutout"][0, 0, 0, 3] == .5


def test_processed_source_video_keeps_alpha_as_linear_matte_data():
    alpha = np.array([[[[.5]]]], dtype=np.float32)
    rendered = {
        "fg": np.full((1, 1, 1, 3), 1000.0, dtype=np.float32),
        "cutout": np.concatenate((np.full((1, 1, 1, 3), .5), alpha), axis=-1),
        "alpha": alpha.copy(),
    }
    result = _encode_rendered_camera_source(
        rendered, "red_log3g10_redwidegamut", video=True)
    assert result["fg"].dtype == np.uint16
    assert result["fg"].max() == 65535
    assert result["cutout"][0, 0, 0, 3] == int(.5 * 65535)
    assert np.all(result["alpha"] == int(.5 * 65535))


def test_legacy_source_policy_uses_srgb_or_linear_codes_as_declared():
    linear = np.array([[[[.18, .18, .18]]]], dtype=np.float32)
    normal = _encode_rendered_camera_source(
        {"fg": linear.copy()}, "legacy", video=False)
    frame_dir_linear = _encode_rendered_camera_source(
        {"fg": linear.copy()}, "legacy", video=False, legacy_linear=True)
    assert normal["fg"].mean() > frame_dir_linear["fg"].mean()
    np.testing.assert_array_equal(frame_dir_linear["fg"], linear)


def test_manifest_subset_rerender_keeps_profiles_for_existing_outputs(tmp_path):
    path = tmp_path / "clip_color.json"
    _write_color_manifest(path, {
        "schema_version": 1,
        "input_color_space": "red_log3g10_redwidegamut",
        "color_output": "processed_source",
        "input_range": "auto",
        "outputs": {"cutout": {"profile": "red_log3g10_redwidegamut", "alpha": "straight"}},
    })
    _write_color_manifest(path, {
        "schema_version": 1,
        "input_color_space": "legacy",
        "color_output": "processed_working",
        "input_range": "auto",
        "outputs": {"alpha": {"profile": "matte", "data": "matte"}},
    })
    import json
    manifest = json.loads(path.read_text())
    assert manifest["outputs"]["cutout"]["profile"] == "red_log3g10_redwidegamut"
    assert manifest["outputs"]["alpha"]["profile"] == "matte"
    assert list(tmp_path.glob("*.tmp-*")) == []
