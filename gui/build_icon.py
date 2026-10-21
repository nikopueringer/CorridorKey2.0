"""Render the desktop UI's app icon (gui/icons/corridorkey.ico):

    python -m gui.build_icon

The mark is the Corridor symbol in brand yellow on the canvas black. Sizes up
to 24 px use corridorkey-small.svg, a tighter crop: shrinking the whole symbol
that far merges the gaps between its arms. Larger sizes use corridorkey.svg.
Needs PySide6 (for SVG rendering) and Pillow (to write the multi-size .ico).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ICONS = Path(__file__).resolve().parent / "icons"
SMALL_SIZES = (16, 20, 24)
LARGE_SIZES = (32, 40, 48, 64, 128, 256)


def render_svg(path: Path, size: int) -> Image.Image:
    image = QImage(size, size, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(str(path)).render(painter)
    painter.end()
    data = bytes(image.constBits())
    return Image.frombuffer("RGBA", (size, size), data, "raw", "RGBA", image.bytesPerLine(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(ICONS / "corridorkey.ico"))
    args = ap.parse_args()

    app = QGuiApplication([])  # QPainter needs a GUI application
    images = [render_svg(ICONS / "corridorkey-small.svg", s) for s in SMALL_SIZES]
    images += [render_svg(ICONS / "corridorkey.svg", s) for s in LARGE_SIZES]
    largest = images[-1]
    largest.save(args.output, format="ICO", sizes=[im.size for im in images],
                 append_images=images[:-1])
    print(f"wrote {args.output} ({', '.join(str(s) for s in SMALL_SIZES + LARGE_SIZES)} px)")
    del app


if __name__ == "__main__":
    main()
