import importlib.util, os, sys, types
import pytest

def _load():
    # import video_io without the heavy package __init__
    root = os.path.join(os.path.dirname(__file__), "..", "corridorkey")
    pkg = types.ModuleType("corridorkey"); pkg.__path__ = [root]
    sys.modules.setdefault("corridorkey", pkg)
    from corridorkey import video_io
    return video_io

def test_natural_order_and_case(tmp_path):
    vio = _load()
    for n in ["f_10.png", "f_2.png", "f_1.png", "f_3.PNG", "notes.txt"]:
        (tmp_path / n).write_bytes(b"")
    (tmp_path / "sub.png").mkdir()
    got = [os.path.basename(p) for p in vio.frame_paths(tmp_path)]
    assert got == ["f_1.png", "f_2.png", "f_3.PNG", "f_10.png"]
