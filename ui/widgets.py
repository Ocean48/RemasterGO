"""Custom reusable Qt widgets for RemasterGO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
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


class EngineCacheInspectorDialog(QDialog):
    """Dialog for inspecting, verifying, and clearing cached TensorRT engines."""

    def __init__(self, cache_mgr, parent=None):
        super().__init__(parent)
        self.cache_mgr = cache_mgr
        self.setWindowTitle("TensorRT Engine Cache Inspector")
        self.resize(750, 420)
        self.setMinimumSize(600, 320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        hdr_lbl = QLabel("Cached TensorRT Engine Binaries")
        hdr_lbl.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(hdr_lbl)

        info_lbl = QLabel(f"Target GPU Architecture: {self.cache_mgr.gpu_name} | Location: {self.cache_mgr.cache_dir}")
        info_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(info_lbl)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Model", "Cache Key", "Resolution", "Precision", "Tiles", "File Size"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.populate_data)
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.accept)

        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_delete)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.populate_data()

    def populate_data(self):
        engines = self.cache_mgr.list_cached_engines()
        self.table.setRowCount(len(engines))
        for row, eng in enumerate(engines):
            self.table.setItem(row, 0, QTableWidgetItem(eng.model_name))
            self.table.setItem(row, 1, QTableWidgetItem(eng.cache_key))
            self.table.setItem(row, 2, QTableWidgetItem(f"{eng.resolution[0]}x{eng.resolution[1]}"))
            self.table.setItem(row, 3, QTableWidgetItem("FP16" if eng.fp16 else "FP32"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{eng.tiles}x{eng.tiles}" if eng.tiles > 1 else "1 (No Tiling)"))
            size_mb = eng.file_size_bytes / (1024 * 1024)
            self.table.setItem(row, 5, QTableWidgetItem(f"{size_mb:.1f} MB"))

    def _delete_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        cache_key = self.table.item(row, 1).text()
        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete cached engine {cache_key}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.cache_mgr.delete_cached_engine(cache_key)
            self.populate_data()


class DownloadEngineDialog(QDialog):
    """Dialog to download a precompiled TensorRT .engine binary or ONNX model directly from a URL."""

    PRESET_ENGINES = [
        ("Real-ESRGAN General x4 (ONNX Model)", "https://huggingface.co/JoPmt/Real_Esrgan_x2_Onnx_Tflite_Tfjs/resolve/main/ano_test/realesr-general-x4v3.onnx", "Real-ESRGAN_x4", "1920x1080"),
        ("Real-ESRGAN x4 Default (ONNX Model)", "https://huggingface.co/TheGuy444/Real-ESRGAN-ONNX/resolve/main/onnx/model.onnx", "Real-ESRGAN_x4", "1920x1080"),
        ("Real-ESRGAN AnimeVideo v3 (Weights)", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", "RealESRGAN_AnimeVideo_v3", "1920x1080"),
        ("Real-ESRGAN x4 Plus (Weights)", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "RealESRGAN_x4plus", "1920x1080"),
        ("Real-ESRGAN x4 Anime 6B (Weights)", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", "RealESRGAN_x4plus_anime_6B", "1920x1080"),
        ("Custom Engine / Model URL...", "", "Custom", "1920x1080")
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Precompiled TensorRT Engine")
        self.resize(560, 320)
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        hdr = QLabel("Download Precompiled TensorRT Engine (.engine)")
        hdr.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(hdr)

        self.lbl_desc = QLabel(
            "Download an ONNX model (to cache/models/ for compilation) or a compiled TensorRT hardware engine binary (.engine) directly into cache/engines/."
        )
        self.lbl_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_desc.setWordWrap(True)
        layout.addWidget(self.lbl_desc)

        # Preset selection
        row_preset = QHBoxLayout()
        lbl_preset = QLabel("Preset:")
        lbl_preset.setFixedWidth(120)
        self.combo_presets = QComboBox()
        for name, _, _, _ in self.PRESET_ENGINES:
            self.combo_presets.addItem(name)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        row_preset.addWidget(lbl_preset)
        row_preset.addWidget(self.combo_presets, 1)
        layout.addLayout(row_preset)

        # Direct URL
        row_url = QHBoxLayout()
        lbl_url = QLabel("Download URL:")
        lbl_url.setFixedWidth(120)
        self.txt_url = QLineEdit(self.PRESET_ENGINES[0][1])
        row_url.addWidget(lbl_url)
        row_url.addWidget(self.txt_url, 1)
        layout.addLayout(row_url)

        # Model Name
        row_model = QHBoxLayout()
        lbl_model = QLabel("Model Name:")
        lbl_model.setFixedWidth(120)
        self.txt_model = QLineEdit(self.PRESET_ENGINES[0][2])
        row_model.addWidget(lbl_model)
        row_model.addWidget(self.txt_model, 1)
        layout.addLayout(row_model)

        # Input Resolution
        row_res = QHBoxLayout()
        lbl_res = QLabel("Input Resolution:")
        lbl_res.setFixedWidth(120)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(32, 8192)
        self.spin_width.setValue(1920)
        self.spin_width.setSingleStep(2)
        lbl_x = QLabel("x")
        lbl_x.setStyleSheet("font-weight: bold; color: #94a3b8;")
        self.spin_height = QSpinBox()
        self.spin_height.setRange(32, 8192)
        self.spin_height.setValue(1080)
        self.spin_height.setSingleStep(2)

        self.combo_res_presets = QComboBox()
        self.combo_res_presets.addItems([
            "Quick Presets...",
            "1920x1080 (1080p FHD)",
            "1280x720 (720p HD)",
            "720x576 (PAL SD)",
            "720x480 (NTSC SD)",
            "640x480 (480p SD)",
            "640x360 (360p)",
            "384x288 (CIF / Low-Res)",
            "2560x1440 (1440p 2K)",
            "3840x2160 (4K UHD)"
        ])
        self.combo_res_presets.currentIndexChanged.connect(self._on_res_preset_selected)
        self.spin_width.valueChanged.connect(self._on_custom_res_typed)
        self.spin_height.valueChanged.connect(self._on_custom_res_typed)

        row_res.addWidget(lbl_res)
        row_res.addWidget(self.spin_width)
        row_res.addWidget(lbl_x)
        row_res.addWidget(self.spin_height)
        row_res.addWidget(self.combo_res_presets, 1)
        layout.addLayout(row_res)

        # Precision
        row_prec = QHBoxLayout()
        lbl_prec = QLabel("Precision:")
        lbl_prec.setFixedWidth(120)
        self.combo_prec = QComboBox()
        self.combo_prec.addItems(["FP16 (Half Precision)", "FP32 (Single Precision)"])
        row_prec.addWidget(lbl_prec)
        row_prec.addWidget(self.combo_prec, 1)
        layout.addLayout(row_prec)

        # Info note
        self.lbl_note = QLabel("Note: .onnx/.pth files are saved to cache/models/ and can be compiled into .engine via 'Build Engine'.")
        self.lbl_note.setStyleSheet("color: #38bdf8; font-size: 11px; font-style: italic;")
        self.lbl_note.setWordWrap(True)
        layout.addWidget(self.lbl_note)

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_download = QPushButton("Start Download")
        self.btn_download.setObjectName("PrimaryButton")
        self.btn_download.clicked.connect(self._validate_and_accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_download)
        layout.addLayout(btn_box)

    def _on_res_preset_selected(self, idx: int):
        if idx <= 0:
            return
        txt = self.combo_res_presets.currentText().split()[0]
        if "x" in txt:
            try:
                w_str, h_str = txt.split("x")
                self.spin_width.blockSignals(True)
                self.spin_height.blockSignals(True)
                self.spin_width.setValue(int(w_str))
                self.spin_height.setValue(int(h_str))
                self.spin_width.blockSignals(False)
                self.spin_height.blockSignals(False)
            except Exception:
                pass

    def _on_custom_res_typed(self):
        w = self.spin_width.value()
        h = self.spin_height.value()
        match_idx = 0
        for i in range(1, self.combo_res_presets.count()):
            txt = self.combo_res_presets.itemText(i).split()[0]
            if txt == f"{w}x{h}":
                match_idx = i
                break
        self.combo_res_presets.blockSignals(True)
        self.combo_res_presets.setCurrentIndex(match_idx)
        self.combo_res_presets.blockSignals(False)

    def _on_preset_changed(self, idx: int):
        if 0 <= idx < len(self.PRESET_ENGINES):
            _, url, model, res = self.PRESET_ENGINES[idx]
            self.txt_url.setText(url)
            self.txt_model.setText(model)
            if "x" in res:
                try:
                    w, h = map(int, res.split("x"))
                    self.spin_width.setValue(w)
                    self.spin_height.setValue(h)
                except Exception:
                    pass

            clean_url = url.split("?")[0].lower()
            if clean_url.endswith(".engine"):
                self.lbl_note.setText("Target: Direct TensorRT engine file (.engine) -> saved to cache/engines/ and registered.")
            elif clean_url.endswith(".onnx") or clean_url.endswith(".pth"):
                self.lbl_note.setText("Target: Model file -> saved to cache/models/. Click 'Build Engine' to compile into TensorRT .engine.")
            else:
                self.lbl_note.setText("Custom URL: .engine files will register to cache/engines/; .onnx/.pth files to cache/models/.")

    def _validate_and_accept(self):
        url = self.txt_url.text().strip()
        if not url.startswith("http://") and not url.startswith("https://"):
            QMessageBox.warning(self, "Invalid URL", "Please enter a valid HTTP/HTTPS URL to the .engine binary.")
            return
        self.accept()

    def get_data(self) -> dict:
        w = self.spin_width.value()
        h = self.spin_height.value()
        fp16 = "FP16" in self.combo_prec.currentText()
        return {
            "url": self.txt_url.text().strip(),
            "model_name": self.txt_model.text().strip() or "Custom_Engine",
            "resolution": (w, h),
            "fp16": fp16
        }


class CompileEngineDialog(QDialog):
    """Dialog to compile a TensorRT engine from an installed base model using trtexec."""

    def __init__(
        self,
        available_models: list[str],
        default_model: str = "",
        default_resolution: tuple[int, int] = (1920, 1080),
        trtexec_available: bool = True,
        trtexec_path: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Compile TensorRT Engine")
        self.resize(580, 380)
        self.trtexec_available = trtexec_available
        self.custom_trtexec_path = trtexec_path
        self.switch_to_download = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(18, 18, 18, 18)

        hdr = QLabel("Compile TensorRT Engine via trtexec")
        hdr.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(hdr)

        lbl_desc = QLabel("Compiles an ONNX model into an optimized TensorRT engine binary tailored for your GPU architecture.")
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Compiler Status Box
        self.box_status = QWidget()
        box_status_layout = QHBoxLayout(self.box_status)
        box_status_layout.setContentsMargins(10, 8, 10, 8)

        self.lbl_status = QLabel()
        self.btn_browse_trtexec = QPushButton("Browse trtexec.exe...")
        self.btn_browse_trtexec.clicked.connect(self._browse_trtexec)

        self.btn_switch_dl = QPushButton("Download Precompiled Instead")
        self.btn_switch_dl.setStyleSheet("background-color: #0284c7; color: white; font-weight: 600; padding: 4px 10px;")
        self.btn_switch_dl.clicked.connect(self._trigger_switch_to_download)

        box_status_layout.addWidget(self.lbl_status, 1)
        box_status_layout.addWidget(self.btn_browse_trtexec)
        box_status_layout.addWidget(self.btn_switch_dl)
        layout.addWidget(self.box_status)

        self._update_compiler_ui()

        row_model = QHBoxLayout()
        lbl_model = QLabel("Select Model File:")
        lbl_model.setFixedWidth(140)
        self.combo_models = QComboBox()
        self.combo_models.addItems(available_models if available_models else ["No models in cache/models/"])
        if default_model and default_model in available_models:
            self.combo_models.setCurrentText(default_model)
        row_model.addWidget(lbl_model)
        row_model.addWidget(self.combo_models, 1)
        layout.addLayout(row_model)

        row_res = QHBoxLayout()
        lbl_res = QLabel("Input Resolution:")
        lbl_res.setFixedWidth(140)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(32, 8192)
        self.spin_width.setValue(default_resolution[0] if default_resolution and len(default_resolution) >= 2 else 1920)
        self.spin_width.setSingleStep(2)
        lbl_x = QLabel("x")
        lbl_x.setStyleSheet("font-weight: bold; color: #94a3b8;")
        self.spin_height = QSpinBox()
        self.spin_height.setRange(32, 8192)
        self.spin_height.setValue(default_resolution[1] if default_resolution and len(default_resolution) >= 2 else 1080)
        self.spin_height.setSingleStep(2)

        self.combo_res_presets = QComboBox()
        self.combo_res_presets.addItems([
            "Quick Presets...",
            "1920x1080 (1080p FHD)",
            "1280x720 (720p HD)",
            "720x576 (PAL SD)",
            "720x480 (NTSC SD)",
            "640x480 (480p SD)",
            "640x360 (360p)",
            "384x288 (CIF / Low-Res)",
            "2560x1440 (1440p 2K)",
            "3840x2160 (4K UHD)"
        ])
        self.combo_res_presets.currentIndexChanged.connect(self._on_res_preset_selected)
        self.spin_width.valueChanged.connect(self._on_custom_res_typed)
        self.spin_height.valueChanged.connect(self._on_custom_res_typed)

        row_res.addWidget(lbl_res)
        row_res.addWidget(self.spin_width)
        row_res.addWidget(lbl_x)
        row_res.addWidget(self.spin_height)
        row_res.addWidget(self.combo_res_presets, 1)
        layout.addLayout(row_res)

        row_prec = QHBoxLayout()
        lbl_prec = QLabel("Precision Mode:")
        lbl_prec.setFixedWidth(140)
        self.combo_prec = QComboBox()
        self.combo_prec.addItems(["FP16 (Recommended for RTX GPUs)", "FP32 (Standard)"])
        row_prec.addWidget(lbl_prec)
        row_prec.addWidget(self.combo_prec, 1)
        layout.addLayout(row_prec)

        row_mem = QHBoxLayout()
        lbl_mem = QLabel("Max Workspace (MB):")
        lbl_mem.setFixedWidth(140)
        self.spin_mem = QSpinBox()
        self.spin_mem.setRange(512, 16384)
        self.spin_mem.setValue(2048)
        self.spin_mem.setSingleStep(512)
        row_mem.addWidget(lbl_mem)
        row_mem.addWidget(self.spin_mem, 1)
        layout.addLayout(row_mem)

        btn_box = QHBoxLayout()
        self.btn_compile = QPushButton("Compile Engine")
        self.btn_compile.setObjectName("PrimaryButton")
        self.btn_compile.clicked.connect(self._validate_and_accept)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_compile)
        layout.addLayout(btn_box)

    def _on_res_preset_selected(self, idx: int):
        if idx <= 0:
            return
        txt = self.combo_res_presets.currentText().split()[0]
        if "x" in txt:
            try:
                w_str, h_str = txt.split("x")
                self.spin_width.blockSignals(True)
                self.spin_height.blockSignals(True)
                self.spin_width.setValue(int(w_str))
                self.spin_height.setValue(int(h_str))
                self.spin_width.blockSignals(False)
                self.spin_height.blockSignals(False)
            except Exception:
                pass

    def _on_custom_res_typed(self):
        w = self.spin_width.value()
        h = self.spin_height.value()
        match_idx = 0
        for i in range(1, self.combo_res_presets.count()):
            txt = self.combo_res_presets.itemText(i).split()[0]
            if txt == f"{w}x{h}":
                match_idx = i
                break
        self.combo_res_presets.blockSignals(True)
        self.combo_res_presets.setCurrentIndex(match_idx)
        self.combo_res_presets.blockSignals(False)

    def _browse_trtexec(self):
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Locate trtexec.exe", "C:\\", "Executables (*trtexec.exe *trtexec);;All Files (*.*)"
        )
        if chosen and os.path.isfile(chosen):
            self.custom_trtexec_path = chosen
            self.trtexec_available = True
            self._update_compiler_ui()

    def _trigger_switch_to_download(self):
        self.switch_to_download = True
        self.reject()

    def _update_compiler_ui(self):
        if self.trtexec_available:
            path_desc = Path(self.custom_trtexec_path).name if self.custom_trtexec_path else "trtexec (System PATH)"
            self.lbl_status.setText(f"Compiler: Ready ({path_desc})")
            self.box_status.setStyleSheet("background-color: #064e3b; border: 1px solid #059669; border-radius: 6px;")
            self.lbl_status.setStyleSheet("color: #6ee7b7; font-weight: 600; font-size: 11px;")
            self.btn_switch_dl.setVisible(False)
        else:
            self.lbl_status.setText("Compiler: trtexec.exe not detected on system PATH.")
            self.box_status.setStyleSheet("background-color: #451a03; border: 1px solid #b45309; border-radius: 6px;")
            self.lbl_status.setStyleSheet("color: #fde047; font-weight: 600; font-size: 11px;")
            self.btn_switch_dl.setVisible(True)

    def _validate_and_accept(self):
        if not self.trtexec_available:
            msg = QMessageBox(self)
            msg.setWindowTitle("Compiler Not Found")
            msg.setText(
                "NVIDIA TensorRT compiler (trtexec.exe) was not found.\n\n"
                "To compile locally, please locate trtexec.exe or download NVIDIA TensorRT.\n"
                "Alternatively, you can download precompiled engines directly without compiling!"
            )
            btn_dl = msg.addButton("Download Precompiled Instead", QMessageBox.ButtonRole.ActionRole)
            btn_browse = msg.addButton("Browse for trtexec.exe...", QMessageBox.ButtonRole.ActionRole)
            btn_cancel = msg.addButton(QMessageBox.StandardButton.Cancel)

            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_dl:
                self._trigger_switch_to_download()
            elif clicked == btn_browse:
                self._browse_trtexec()
            return

        chosen_model = self.combo_models.currentText().strip()
        if chosen_model and not chosen_model.lower().endswith(".onnx"):
            QMessageBox.warning(
                self,
                "ONNX Model Required",
                f"'{chosen_model}' is a PyTorch checkpoint ({Path(chosen_model).suffix}).\n\n"
                f"NVIDIA trtexec only compiles ONNX models (.onnx) into TensorRT engines.\n\n"
                f"Please select an ONNX model (e.g. 'realesr-general-x4v3.onnx') or download one.\n"
                f"PyTorch (.pth) models run directly on your GPU using Real-ESRGAN Vulkan without needing compilation."
            )
            return

        self.accept()

    def get_data(self) -> dict:
        w = self.spin_width.value()
        h = self.spin_height.value()
        fp16 = "FP16" in self.combo_prec.currentText()
        return {
            "model_filename": self.combo_models.currentText(),
            "resolution": (w, h),
            "fp16": fp16,
            "workspace_mb": self.spin_mem.value()
        }

