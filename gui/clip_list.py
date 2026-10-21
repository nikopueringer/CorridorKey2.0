"""The clip queue: one row per clip with its hint and run status."""
from __future__ import annotations

import os

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QMenu,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
)

from .clips import QUEUED, RUNNING, Clip
from .theme import BRAND, CANVAS, LINE_STRONG, MUTED, STATUS_PILLS, TEXT, font

CLIP_ROLE = Qt.ItemDataRole.UserRole
STATUS_COLUMN = 2


class _StatusDelegate(QStyledItemDelegate):
    """The status cell: a pill per state, like the app's READY badge, and the
    running phase over a thin brand-yellow bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pill_font = font(10, QFont.Weight.Bold, 1.2)
        self._detail_font = font(12)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), size.height() + 10)  # sets the row height for every column

    def paint(self, painter, option, index):
        clip = index.siblingAtColumn(0).data(CLIP_ROLE)
        if clip is None:
            return super().paint(painter, option, index)
        # Row background and selection as usual, without the cell's own text.
        cell = QStyleOptionViewItem(option)
        self.initStyleOption(cell, index)
        cell.text = ""
        style = cell.widget.style() if cell.widget else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, cell, painter, cell.widget)
        rect = QRectF(option.rect.adjusted(6, 0, -6, 0))
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if clip.status == RUNNING:
            track = QRectF(rect.left(), rect.bottom() - 5, rect.width(), 2)
            painter.fillRect(track, QColor(LINE_STRONG))
            painter.fillRect(QRectF(track.left(), track.top(), track.width() * clip.progress, 2),
                             QColor(BRAND))
            painter.setPen(QColor(TEXT))
            painter.setFont(self._detail_font)
            text = QFontMetricsF(self._detail_font).elidedText(
                clip.detail, Qt.TextElideMode.ElideRight, rect.width())
            painter.drawText(rect.adjusted(0, 0, 0, -5),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        else:
            colour, outline = STATUS_PILLS.get(clip.status, (MUTED, LINE_STRONG))
            label = clip.status.upper()
            metrics = QFontMetricsF(self._pill_font)
            height = metrics.height() + 4
            pill = QRectF(rect.left(), rect.center().y() - height / 2,
                          metrics.horizontalAdvance(label) + 16, height)
            painter.setPen(QPen(QColor(outline), 1))
            painter.setBrush(QColor(CANVAS))
            painter.drawRoundedRect(pill.adjusted(0.5, 0.5, -0.5, -0.5), height / 2, height / 2)
            painter.setPen(QColor(colour))
            painter.setFont(self._pill_font)
            painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)
        painter.restore()


class ClipList(QTreeWidget):
    open_folder = Signal(object)  # Clip

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[Clip, QTreeWidgetItem] = {}
        self.setHeaderLabels(["CLIP", "HINT", "STATUS"])
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        header = self.header()
        header.setFont(font(11, QFont.Weight.Bold, 1.2))
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(STATUS_COLUMN, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(1, 64)
        self.setColumnWidth(STATUS_COLUMN, 132)
        self.setItemDelegateForColumn(STATUS_COLUMN, _StatusDelegate(self))
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def clips(self) -> list[Clip]:
        return [self.topLevelItem(i).data(0, CLIP_ROLE) for i in range(self.topLevelItemCount())]

    def selected_clips(self) -> list[Clip]:
        items = sorted(self.selectedItems(), key=self.indexOfTopLevelItem)
        return [item.data(0, CLIP_ROLE) for item in items]

    def next_queued(self) -> Clip | None:
        return next((clip for clip in self.clips() if clip.status == QUEUED), None)

    def add(self, clips: list[Clip]) -> list[Clip]:
        """Append clips not already queued; returns the ones added."""
        known = {os.path.normcase(str(clip.source)) for clip in self._items}
        added = []
        for clip in clips:
            key = os.path.normcase(str(clip.source))
            if key in known:
                continue
            known.add(key)
            item = QTreeWidgetItem()
            item.setData(0, CLIP_ROLE, clip)
            self.addTopLevelItem(item)
            self._items[clip] = item
            self.refresh(clip)
            added.append(clip)
        return added

    def refresh(self, clip: Clip) -> None:
        item = self._items.get(clip)
        if item is None:
            return
        item.setText(0, clip.name)
        item.setToolTip(0, str(clip.path))
        item.setText(1, clip.hint_label)
        item.setToolTip(1, str(clip.hint_path) if clip.hint_path else "No hint")
        item.setText(STATUS_COLUMN, clip.status)
        item.setToolTip(STATUS_COLUMN, clip.detail or clip.status)

    def select(self, clip: Clip) -> None:
        if clip in self._items:
            self.setCurrentItem(self._items[clip])

    def requeue(self, clip: Clip) -> None:
        if clip.status != RUNNING:
            clip.status, clip.progress, clip.detail = QUEUED, 0.0, ""
            self.refresh(clip)

    def remove_selected(self) -> None:
        for clip in self.selected_clips():
            if clip.status == RUNNING:
                continue
            item = self._items.pop(clip)
            self.takeTopLevelItem(self.indexOfTopLevelItem(item))

    def _menu(self, pos) -> None:
        clips = self.selected_clips()
        if not clips:
            return
        menu = QMenu(self)
        menu.addAction("Run again").triggered.connect(lambda: [self.requeue(c) for c in clips])
        if len(clips) == 1:
            menu.addAction("Open output folder").triggered.connect(
                lambda: self.open_folder.emit(clips[0]))
        menu.addSeparator()
        menu.addAction("Remove").triggered.connect(self.remove_selected)
        menu.exec(self.viewport().mapToGlobal(pos))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.topLevelItemCount():
            return
        painter = QPainter(self.viewport())
        painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
        painter.drawText(self.viewport().rect().adjusted(16, 16, -16, -16),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                         "No clips yet.\nDrop videos or frame folders anywhere in this window.")
