"""Custom reusable Qt widgets for RemasterGO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


class DropTableWidget(QTableWidget):
    """Table widget supporting drag-and-drop of video media files."""

    sig_files_dropped = Signal(list)

    SUPPORTED_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".ts", ".m2ts", ".webm", ".wmv"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTableWidget.DragDropMode.DropOnly)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            valid = any(
                Path(u.toLocalFile()).suffix.lower() in self.SUPPORTED_EXTENSIONS
                for u in urls
            )
            if valid:
                event.acceptProposedAction()
                return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files: List[str] = []
        for url in event.mimeData().urls():
            fpath = url.toLocalFile()
            if os.path.isfile(fpath) and Path(fpath).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(fpath)

        if files:
            self.sig_files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            event.ignore()


class MetricCard(QFrame):
    """A card widget for displaying a live metric (e.g. FPS, ETA, Current Frame)."""

    def __init__(self, title: str, initial_value: str = "--", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #191922;
                border: 1px solid #2c2c3d;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600; border: none; background: transparent;")

        self.val_lbl = QLabel(initial_value)
        self.val_lbl.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: 700; border: none; background: transparent;")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.val_lbl)

    def set_value(self, val: str):
        self.val_lbl.setText(val)


class StatusBadge(QLabel):
    """A badge label showing a status pill with corresponding color."""

    COLORS = {
        "Pending": ("#374151", "#d1d5db"),
        "Processing": ("#0369a1", "#7dd3fc"),
        "Paused": ("#854d0e", "#fde047"),
        "Completed": ("#065f46", "#6ee7b7"),
        "Failed": ("#991b1b", "#fca5a5"),
        "Cancelled": ("#4b5563", "#9ca3af"),
    }

    def __init__(self, text: str = "Pending", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_status(text)

    def set_status(self, text: str):
        self.setText(text)
        bg, fg = self.COLORS.get(text, ("#374151", "#d1d5db"))
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border-radius: 10px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: 600;
            }}
        """)
