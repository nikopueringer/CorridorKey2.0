import concurrent.futures
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from corridorkey.outputs import (
    OWNER_FILE, OutputTransaction, output_metadata, read_outputs,
    source_identity, validate_destination,
)
from corridorkey.video_io import ExrSequenceWriter
from gui.clips import Clip, output_dir_for


def render(root, source, *, count=1, key="fg", value=.2, fps=25, profile="linear_rec709"):
    tx = OutputTransaction(root, source, "shot")
    folder = tx.stage / f"shot_{key}_exr"
    writer = ExrSequenceWriter(folder, fps, (2, 2), channels=3, dtype="float")
    writer.write_many(np.full((count, 2, 2, 3), value, np.float32))
    writer.release()
    tx.commit({"native_fps": fps, "outputs": {
        key: {"path": folder.name, "profile": profile, "native_fps": fps, "start_frame": 7}
    }}, count)
    return read_outputs(root, source, "shot")


def test_shorter_rerender_publishes_only_new_frames_and_preserves_previous_generation(tmp_path):
    source = tmp_path / "shot.mov"
    root = tmp_path / "out"
    old = render(root, source, count=3)
    new = render(root, source, count=1, value=.9)
    assert old["fg"] != new["fg"]
    assert len(list(old["fg"].glob("*.exr"))) == 3
    files = list(new["fg"].glob("*.exr"))
    assert len(files) == 1
    np.testing.assert_allclose(cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED), .9)
    assert new.entries["fg"]["frame_count"] == 1


def test_abort_and_partial_multi_output_do_not_replace_committed_outputs(tmp_path):
    source, root = tmp_path / "shot.mov", tmp_path / "out"
    before = render(root, source)
    manifest = (root / "shot_color.json").read_bytes()
    tx = OutputTransaction(root, source, "shot")
    (tx.stage / "shot_alpha_exr").mkdir()
    (tx.stage / "shot_alpha_exr/frame_000000.exr").write_bytes(b"new")
    with pytest.raises(RuntimeError, match="Missing or empty"):
        tx.commit({"outputs": {
            "alpha": {"path": "shot_alpha_exr"},
            "cutout": {"path": "shot_cutout_exr"},
        }}, 1)
    assert read_outputs(root, source, "shot") == before
    assert (root / "shot_color.json").read_bytes() == manifest
    tx.abort()
    assert not tx.stage.exists()


def test_subset_and_snapshot_preserve_previous_profile_fps_and_source_offset(tmp_path):
    source, root = tmp_path / "shot.mov", tmp_path / "out"
    first = render(root, source, key="cutout", fps=25, profile="apple_log2_awg2")
    render(root, source, key="alpha", fps=50, profile="matte")
    new = read_outputs(root, source, "shot")
    assert new["cutout"] == first["cutout"]
    assert new.entries["cutout"]["native_fps"] == 25
    assert new.entries["cutout"]["start_frame"] == 7
    render(root, source, key="cutout", fps=50, profile="linear_rec709")
    # A viewer may hold the old paths while another render is committed.
    _, entries = output_metadata("shot", first, root)
    assert entries["cutout"]["profile"] == "apple_log2_awg2"
    assert entries["cutout"]["native_fps"] == 25


def test_concurrent_subset_publication_keeps_every_output(tmp_path):
    source, root = tmp_path / "shot.mov", tmp_path / "out"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(render, root, source, key=key) for key in ("alpha", "fg")]
        for result in results:
            result.result(timeout=5)
    assert set(read_outputs(root, source, "shot")) == {"alpha", "fg"}


def test_failed_manifest_replace_leaves_previous_generation_visible(tmp_path, monkeypatch):
    import corridorkey.outputs as outputs
    source, root = tmp_path / "shot.mov", tmp_path / "out"
    old = render(root, source)
    original = outputs.atomic_json
    def fail(path, payload):
        if path.name == "shot_color.json":
            raise OSError("synthetic disk full")
        return original(path, payload)
    monkeypatch.setattr(outputs, "atomic_json", fail)
    with pytest.raises(OSError, match="disk full"):
        render(root, source, value=.9)
    assert read_outputs(root, source, "shot") == old
    assert len(list((root / "renders").iterdir())) == 2  # orphan never discovered


@pytest.mark.parametrize("bad_path", ["../outside.exr", "/tmp/outside.exr", ".staging/uncommitted"])
def test_unsafe_or_uncommitted_manifest_paths_are_not_discovered(tmp_path, bad_path):
    source, root = tmp_path / "shot.mov", tmp_path / "out"
    render(root, source)
    path = root / "shot_color.json"
    data = json.loads(path.read_text())
    data["outputs"]["fg"]["path"] = bad_path
    path.write_text(json.dumps(data))
    (root / "shot_fg.mp4").write_bytes(b"uncommitted newer file")
    assert read_outputs(root, source, "shot") == {}
    path.write_text("malformed json")
    assert read_outputs(root, source, "shot") == {}


def test_manifest_symlink_cannot_escape_output_root(tmp_path):
    source, root = tmp_path / "shot.mov", tmp_path / "out"
    render(root, source)
    outside = tmp_path / "outside.exr"
    outside.write_bytes(b"private")
    (root / "renders/escaped.exr").symlink_to(outside)
    path = root / "shot_color.json"
    data = json.loads(path.read_text())
    data["outputs"]["fg"]["path"] = "renders/escaped.exr"
    path.write_text(json.dumps(data))
    assert read_outputs(root, source, "shot") == {}


def test_source_identity_distinguishes_names_and_preserves_alias_identity(tmp_path):
    paths = [tmp_path / "a/shot.mov", tmp_path / "b/shot.mov", tmp_path / "a/shot.mp4"]
    clips = [Clip(p, p, False) for p in paths]
    destinations = [output_dir_for(c, str(tmp_path / "shared")) for c in clips]
    assert len(set(destinations)) == 3
    assert list(reversed(destinations)) == [output_dir_for(c, str(tmp_path / "shared")) for c in reversed(clips)]
    assert output_dir_for(clips[0]) != output_dir_for(clips[2])
    paths[0].parent.mkdir()
    paths[0].touch()
    alias = tmp_path / "alias.mov"
    alias.symlink_to(paths[0])
    assert source_identity(alias) == source_identity(paths[0])
    tx = OutputTransaction(destinations[0], paths[0], "shot")
    with pytest.raises(ValueError, match="another source"):
        OutputTransaction(destinations[0], paths[1], "shot")
    tx.abort()


def test_legacy_output_remains_readable_but_is_not_claimed_for_writing(tmp_path):
    source = tmp_path / "shot.mov"
    root = tmp_path / "legacy"
    root.mkdir()
    path = root / "shot_fg.mp4"
    path.write_bytes(b"old video")
    assert read_outputs(root, source, "shot")["fg"] == path
    with pytest.raises(ValueError, match="unowned results"):
        validate_destination(root, source)
    assert not (root / OWNER_FILE).exists()


def test_gui_window_controls_cannot_leave_gaps(monkeypatch, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    import gui.settings_panel as settings_module
    app = QApplication.instance() or QApplication([])
    store = QSettings(str(tmp_path / "isolated.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(settings_module, "QSettings", lambda: store)
    panel = settings_module.SettingsPanel()
    try:
        panel.chunk.setValue(2)
        panel.stride.setValue(4)
        assert panel.stride.value() == panel.stride.maximum() == 2
        panel.chunk.setValue(1)
        assert panel.stride.value() == panel.stride.maximum() == 1
    finally:
        panel.close()
        app.processEvents()
