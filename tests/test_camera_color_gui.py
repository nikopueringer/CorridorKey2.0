"""Focused regression checks for camera profile math and GUI command wiring."""
from __future__ import annotations

import numpy as np
import cv2
import json
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from corridorkey.camera_color import PROFILES, decode_to_working, encode_from_working
from gui.clips import Clip
from gui.command import JobSettings, infer_args
from gui.frames import FrameSource, _camera_display8, _linear_display8, to_cutout_checker8
from gui.viewer import ViewerPanel


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_published_grey_reference_values():
    middle_grey = np.full((1, 1, 3), .18, dtype=np.float32)
    vectors = {
        "apple_log_rec2020": .48827246,
        "apple_log2_awg2": .48827246,
        "samsung_log_bt2020": .527859,
        "sony_slog3_sgamut3": .41055718,
        "arri_logc3_awg3_ei800": .39100683,
        "arri_logc4_awg4": .27839583,
        "dji_dlog_dgamut": .39876456,
    }
    for profile_id, expected in vectors.items():
        encoded = encode_from_working(middle_grey, profile_id)
        np.testing.assert_allclose(encoded, expected, atol=3e-5, rtol=0)
        decoded = decode_to_working(encoded, profile_id)
        np.testing.assert_allclose(decoded, middle_grey, atol=2e-5, rtol=0)


def test_supported_profiles_roundtrip_signed_and_hdr_values():
    working = np.array([
        [-.03, .18, 4.0],
        [0.0, .01, .9],
    ], dtype=np.float32)
    assert len(PROFILES) >= 16
    for profile_id in PROFILES.keys() - {"legacy"}:
        encoded = encode_from_working(working, profile_id)
        decoded = decode_to_working(encoded, profile_id)
        assert encoded.dtype == decoded.dtype == np.float32
        assert np.isfinite(encoded).all(), profile_id
        np.testing.assert_allclose(decoded, working, atol=3e-5, rtol=2e-5, err_msg=profile_id)


def test_gui_args_include_nondefault_profile_options_only(tmp_path, monkeypatch):
    import gui.command as command

    monkeypatch.setattr(command, "INFER", tmp_path / "infer.py")
    video = Clip(tmp_path / "camera.mov", tmp_path / "camera.mov", False)
    defaults = infer_args(video, JobSettings(), tmp_path / "out")
    assert "--input_color_space" not in defaults
    assert "--color_output" not in defaults
    assert "--input_range" not in defaults

    video.input_color_space = "apple_log2_awg2"
    args = infer_args(video, JobSettings(color_output="processed_source", input_range="legal"),
                      tmp_path / "out")
    assert args[args.index("--input_color_space") + 1] == "apple_log2_awg2"
    assert args[args.index("--color_output") + 1] == "processed_source"
    assert args[args.index("--input_range") + 1] == "legal"

    legacy_original = infer_args(video, JobSettings(color_output="source_rgb",
                                                     output_format="video",
                                                     exr_dtype="half"), tmp_path / "out")
    assert legacy_original[legacy_original.index("--color_output") + 1] == "source_rgb"
    assert legacy_original[legacy_original.index("--output_format") + 1] == "exr"
    assert legacy_original[legacy_original.index("--exr_dtype") + 1] == "float"


def test_camera_profile_frame_sequence_is_not_double_decoded(tmp_path, monkeypatch):
    import gui.command as command

    monkeypatch.setattr(command, "INFER", tmp_path / "infer.py")
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "frame_0001.exr").write_bytes(b"placeholder")
    clip = Clip(folder, folder, True, input_color_space="arri_logc4_awg4")
    args = infer_args(clip, JobSettings(frame_colour="auto", input_range="legal"), tmp_path / "out")
    assert "--input_color_space" in args
    assert "--frame_dir_linear" not in args
    assert "--input_range" not in args  # code-range applies to video streams only


def test_input_preview_view_transform_decodes_camera_profile():
    codes = encode_from_working(
        np.full((1, 1, 3), .18, dtype=np.float32), "apple_log2_awg2")
    transformed = _camera_display8(codes, "apple_log2_awg2", True)
    raw = _camera_display8(codes, "apple_log2_awg2", False)
    np.testing.assert_allclose(transformed[0, 0], [118, 118, 118], atol=1)
    assert raw[0, 0, 0] > transformed[0, 0, 0]


def test_output_profiles_keep_standard_srgb_linear_and_matte_previews(tmp_path):
    folder = tmp_path / "sequence"
    folder.mkdir()
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[:, :, :3] = 128
    rgba[:, :, 3] = 255
    ok, encoded = cv2.imencode(".png", rgba)
    assert ok
    (folder / "frame_0001.png").write_bytes(encoded.tobytes())

    for profile in ("linear_rec709", "srgb", "matte"):
        source = FrameSource(folder, cutout=(profile != "matte"),
                             profile_id=profile, cutout_linear=profile == "linear_rec709")
        try:
            shown = source.read(0)
            assert shown.shape == (2, 2, 3)
            assert shown.dtype == np.uint8
        finally:
            source.close()


def test_explicit_output_transfer_is_respected_for_float_and_integer_media():
    srgb_cutout = np.full((1, 1, 4), .5, dtype=np.float32)
    srgb_cutout[:, :, 3] = 1
    shown_srgb = to_cutout_checker8(srgb_cutout, source_transfer="srgb")
    assert tuple(shown_srgb[0, 0]) == (128, 128, 128)

    linear_cutout = np.full((1, 1, 4), .18, dtype=np.float32)
    linear_cutout[:, :, 3] = 1
    assert tuple(to_cutout_checker8(linear_cutout, source_transfer="linear")[0, 0]) == (118, 118, 118)

    linear_video_pixel = np.full((1, 1, 3), 46, dtype=np.uint8)
    assert tuple(_linear_display8(linear_video_pixel)[0, 0]) == (118, 118, 118)


def test_video_native_fps_is_frozen_for_exr_output_sequence(qt_app, tmp_path):
    class Loader(QObject):
        loaded = Signal(str, int, int, float, object)
        failed = Signal(str, str)

        def __init__(self):
            super().__init__()
            self.requests = []

        def request(self, path, index, **kwargs):
            self.requests.append((path, index, kwargs))

    loader = Loader()
    viewer = ViewerPanel(loader)
    clip = Clip(tmp_path / "plate.mov", tmp_path / "plate.mov", False)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sequence = output_dir / "plate_fg_exr"
    sequence.mkdir()
    manifest = {
        "schema_version": 1,
        "native_fps": 25.0,
        "outputs": {"fg": {"profile": "linear_rec709"}},
    }
    (output_dir / "plate_color.json").write_text(json.dumps(manifest))

    viewer.set_sequence_fps(24.0)
    viewer.set_clip(clip, {"fg": sequence})
    viewer.set_sequence_fps(30.0)  # later UI settings do not reinterpret old output
    viewer.show_view("fg")
    assert loader.requests[-1][2]["sequence_fps"] == 25.0
    assert viewer._manifest_native_fps == 25.0


def test_old_exr_output_uses_known_source_fps_and_rejects_bad_manifest_fps(qt_app, tmp_path):
    class Loader(QObject):
        loaded = Signal(str, int, int, float, object)
        failed = Signal(str, str)

        def __init__(self):
            super().__init__()
            self.requests = []

        def request(self, path, index, **kwargs):
            self.requests.append(kwargs)

    loader = Loader()
    viewer = ViewerPanel(loader)
    clip = Clip(tmp_path / "plate.mov", tmp_path / "plate.mov", False)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    sequence = output_dir / "plate_fg_exr"
    sequence.mkdir()
    (output_dir / "plate_color.json").write_text(json.dumps({"native_fps": "nan", "outputs": {}}))
    viewer.set_sequence_fps(24.0)
    viewer.set_clip(clip, {"fg": sequence})
    viewer._input_fps = 25.0  # learned from the source video preview
    viewer.show_view("fg")
    assert viewer._manifest_native_fps is None
    assert loader.requests[-1]["sequence_fps"] == 25.0


def test_main_window_starts_and_applies_frame_folder_camera_profile(qt_app, tmp_path):
    from gui.main_window import MainWindow

    old_org = qt_app.organizationName()
    old_app = qt_app.applicationName()
    old_format = QSettings.defaultFormat()
    qt_app.setOrganizationName("CorridorKey-Gui-Smoke")
    qt_app.setApplicationName("Isolated-Test")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    QSettings().clear()

    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    (frame_dir / "frame_0001.png").write_bytes(encoded.tobytes())
    clip = Clip(frame_dir, frame_dir, True)
    window = None
    try:
        window = MainWindow()
        window.queue.add([clip])
        window.queue.select(clip)
        qt_app.processEvents()

        profile_index = window.clip_box.input_color.findData("apple_log2_awg2")
        assert profile_index >= 0
        window.clip_box.input_color.setCurrentIndex(profile_index)
        qt_app.processEvents()

        assert clip.input_color_space == "apple_log2_awg2"
        assert window.viewer._input_profile == "apple_log2_awg2"
        assert window.settings.color_output.isEnabled()
        assert not window.settings.input_range.isEnabled()  # frame folders have no video code range
        assert "EXR is scene-linear" in window.settings.format.toolTip()
    finally:
        if window is not None:
            window.close()
            qt_app.processEvents()
        QSettings().clear()
        qt_app.setOrganizationName(old_org)
        qt_app.setApplicationName(old_app)
        QSettings.setDefaultFormat(old_format)


def test_completed_output_path_and_fps_survive_other_format_and_subset_render(qt_app, tmp_path):
    from gui.clips import find_outputs
    from corridorkey.runtime import _write_color_manifest

    clip = Clip(tmp_path / "plate.mov", tmp_path / "plate.mov", False)
    sequence = tmp_path / "plate_cutout_exr"
    sequence.mkdir()
    alternate = tmp_path / "plate_cutout.mov"
    alternate.touch()
    os.utime(sequence, (1, 1))
    os.utime(alternate, (2, 2))
    path = tmp_path / "plate_color.json"
    _write_color_manifest(path, {
        "native_fps": 25.0, "outputs": {
            "cutout": {"path": sequence.name, "profile": "apple_log2_awg2", "native_fps": 25.0}}})
    _write_color_manifest(path, {
        "native_fps": 30.0, "outputs": {"alpha": {"profile": "matte", "native_fps": 30.0}}})
    outputs = find_outputs(clip, tmp_path)
    assert outputs["cutout"] == sequence
    class Loader(QObject):
        loaded = Signal(str, int, int, float, object)
        failed = Signal(str, str)

        def __init__(self):
            super().__init__()
            self.requests = []

        def request(self, path, index, **kwargs):
            self.requests.append(kwargs)

    loader = Loader()
    viewer = ViewerPanel(loader)
    viewer.set_clip(clip, outputs)
    viewer.show_view("cutout")
    assert loader.requests[-1]["sequence_fps"] == 25.0
    assert loader.requests[-1]["profile_id"] == "apple_log2_awg2"
