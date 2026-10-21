"""Start the desktop UI from the repository root:

    python -m gui [clip_or_folder ...]

Clips, frame folders or folders of clips given on the command line are queued
on start. The UI needs PySide6 on top of requirements.txt:

    pip install PySide6-Essentials
"""
import sys

try:
    from .app import main
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("PySide6"):
        sys.exit("The desktop UI needs PySide6: pip install PySide6-Essentials")
    raise

sys.exit(main())
