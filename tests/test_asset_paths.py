from corridorkey.runtime import RuntimePaths
from gui.macos_setup import assets_present, READY_MARKER


def test_all_models_use_external_asset_root(tmp_path, monkeypatch):
    root = tmp_path / "Library" / "Application Support" / "CorridorKey2.0"
    monkeypatch.setenv("CORRIDORKEY_ASSET_ROOT", str(root))
    paths = RuntimePaths.from_package(tmp_path / "App.app" / "src")
    assert paths.package_root == tmp_path / "App.app" / "src"
    for path in (paths.weights, paths.foundation_stats, paths.rvm_ckpt,
                 paths.moge_checkpoint, paths.cradio_dir):
        assert path.is_relative_to(root)
    assert not assets_present(root)
    for path in (paths.weights, paths.foundation_stats, paths.rvm_ckpt,
                 paths.moge_checkpoint, paths.cradio_dir / "config.json",
                 paths.cradio_dir / "model.safetensors", paths.cradio_dir / "hf_model.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test placeholder")
    assert not assets_present(root)  # interrupted downloads never count as ready
    (root / READY_MARKER).write_text("complete")
    assert assets_present(root)
    paths.weights.write_bytes(b"")
    assert not assets_present(root)


def test_explicit_checkpoint_retains_its_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CORRIDORKEY_ASSET_ROOT", str(tmp_path / "models"))
    checkpoint = tmp_path / "custom.safetensors"
    paths = RuntimePaths.from_package(tmp_path / "src", checkpoint)
    assert paths.weights == checkpoint
    assert paths.foundation_stats.is_relative_to(tmp_path / "models")
