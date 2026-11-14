"""Exercise a relocated app's interpreter and GUI without downloading models."""
import os
from pathlib import Path
import platform
import sys


def main():
    import cv2
    import torch
    import torchvision
    import transformers
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from corridorkey.runtime import RuntimePaths, prepare_runtime_environment
    from .command import ROOT, python_exe
    from .main_window import MainWindow
    from .theme import apply_theme

    assert sys.platform == "darwin" and platform.machine() == "arm64"
    assert Path(sys.prefix).is_relative_to(ROOT.parent / "Python")
    assert Path(python_exe()).is_relative_to(ROOT.parent / "Python")
    assert os.environ.get("CORRIDORKEY_BUNDLED_APP") == "1"
    paths = RuntimePaths.from_package(ROOT)
    assert paths.package_root == ROOT
    assert paths.weights.is_relative_to(Path(os.environ["CORRIDORKEY_ASSET_ROOT"]))
    prepare_runtime_environment(paths)
    from moge.model.v2 import MoGeModel
    app = QApplication([sys.argv[0]])
    apply_theme(app)
    window = MainWindow()
    window.show()
    assert window.engine_button.isVisible()
    QTimer.singleShot(500, window.close)
    QTimer.singleShot(700, app.quit)
    code = app.exec()
    print(f"Bundle smoke test: {platform.machine()}, Python {platform.python_version()}, "
          f"PyTorch {torch.__version__}, OpenCV {cv2.__version__}; MPS available: {torch.backends.mps.is_available()}", flush=True)
    return code
