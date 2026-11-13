"""Desktop UI startup: the application, its icon and theme, and the main window."""
from __future__ import annotations

import ctypes
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication

from . import APP_USER_MODEL_ID
from .main_window import MainWindow
from .theme import app_icon, apply_theme


def main() -> int:
    # opencv-python exports its bundled Qt5 plugin path on Linux. PySide6 uses
    # Qt6 and cannot load those plugins; preserve any non-OpenCV user override.
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "/cv2/" in os.environ.get(key, ""):
            os.environ.pop(key)
    if sys.platform == "win32":
        # Own taskbar button and icon, rather than grouping under pythonw.exe.
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    app = QApplication(sys.argv)
    app.setOrganizationName("CorridorKey")
    app.setApplicationName("CorridorKey 2.0")
    app.setWindowIcon(app_icon())
    apply_theme(app)
    if os.environ.get("CORRIDORKEY_BUNDLED_APP") == "1":
        from .macos_setup import prepare
        if not prepare():
            return 0
    window = MainWindow()
    # pythonw.exe has no console, so stray exceptions go to the log panel instead.
    sys.excepthook = lambda *exc: window.report_error("".join(traceback.format_exception(*exc)))
    window.show()
    window.add_paths([Path(arg) for arg in sys.argv[1:]])
    return app.exec()
