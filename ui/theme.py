"""Modern dark visual theme and QSS styles for RemasterGO."""

from __future__ import annotations

DARK_STYLE = """
QMainWindow, QDialog, QWidget {
    background-color: #121216;
    color: #f3f4f6;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

/* Group Boxes and Panels */
QGroupBox {
    background-color: #1c1c24;
    border: 1px solid #2e2e3a;
    border-radius: 8px;
    margin-top: 24px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    color: #e5e7eb;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    background-color: #272734;
    border: 1px solid #3b3b4d;
    border-radius: 4px;
    color: #38bdf8;
}

/* Buttons */
QPushButton {
    background-color: #262633;
    border: 1px solid #3b3b4f;
    border-radius: 6px;
    color: #f3f4f6;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #323244;
    border-color: #0ea5e9;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #1f1f2b;
}

QPushButton:disabled {
    background-color: #181820;
    border-color: #262630;
    color: #6b7280;
}

QPushButton#PrimaryButton {
    background-color: #0284c7;
    border: 1px solid #0369a1;
    color: #ffffff;
    font-weight: 600;
}

QPushButton#PrimaryButton:hover {
    background-color: #0ea5e9;
    border-color: #38bdf8;
}

QPushButton#DangerButton {
    background-color: #991b1b;
    border: 1px solid #7f1d1d;
    color: #ffffff;
}

QPushButton#DangerButton:hover {
    background-color: #dc2626;
}

/* Input Fields & Combos */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #181820;
    border: 1px solid #323242;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f9fafb;
    selection-background-color: #0284c7;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #0ea5e9;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #2e2e3d;
}

QComboBox QAbstractItemView {
    background-color: #1c1c26;
    border: 1px solid #38384a;
    color: #f3f4f6;
    selection-background-color: #0284c7;
    padding: 4px;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #272736;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #0284c7;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #f3f4f6;
    border: 2px solid #0284c7;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #38bdf8;
}

/* Table Widget */
QTableWidget {
    background-color: #181822;
    border: 1px solid #2d2d3c;
    border-radius: 8px;
    gridline-color: #252533;
    color: #f3f4f6;
    selection-background-color: #272738;
}

QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #20202d;
}

QTableWidget::item:selected {
    background-color: #28283a;
    color: #38bdf8;
}

QHeaderView::section {
    background-color: #1f1f2b;
    color: #9ca3af;
    padding: 8px;
    border: none;
    border-right: 1px solid #292938;
    border-bottom: 1px solid #2e2e40;
    font-weight: 600;
}

/* Progress Bars */
QProgressBar {
    background-color: #1a1a24;
    border: 1px solid #2c2c3b;
    border-radius: 6px;
    text-align: center;
    color: #f9fafb;
    font-weight: 600;
    height: 18px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #38bdf8);
    border-radius: 5px;
}

/* ScrollBars */
QScrollBar:vertical {
    border: none;
    background-color: #14141c;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #2e2e40;
    min-height: 25px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: #3e3e56;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* PlainTextEdit / Logs */
QPlainTextEdit {
    background-color: #0f0f14;
    border: 1px solid #292938;
    border-radius: 8px;
    color: #a7f3d0;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 12px;
    padding: 8px;
}

/* Labels */
QLabel {
    color: #d1d5db;
}

QLabel#HeaderTitle {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
}

QLabel#GpuBadge {
    background-color: #064e3b;
    color: #6ee7b7;
    border: 1px solid #047857;
    border-radius: 12px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 12px;
}
"""
