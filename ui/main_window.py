"""PySide6 desktop main application window for RemasterGO."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QTime
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.cache_manager import EngineCacheManager
from core.job_queue import JobItem, JobQueueController
from core.probe import MediaMetadata, StrategyConfig, probe_media
from ui.theme import DARK_STYLE
from ui.widgets import DropTableWidget, MetricCard, StatusBadge


class MainWindow(QMainWindow):
    """Main desktop application window for RemasterGO."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RemasterGO - Hybrid Video Restoration & AI Upscaling Engine")
        self.resize(1280, 850)
        self.setMinimumSize(1024, 700)

        self.cache_mgr = EngineCacheManager()
        self.queue_controller = JobQueueController(self)

        self.custom_output_directory: Optional[str] = None

        self._setup_ui()
        self._connect_signals()

        self.log(f"RemasterGO initialized. Detected hardware: {self.cache_mgr.gpu_name}")
        self.log("Output destination: Same as input video location (default)")

    def _setup_ui(self):
        self.setStyleSheet(DARK_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        # Header bar
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_lbl = QLabel("RemasterGO")
        title_lbl.setObjectName("HeaderTitle")
        subtitle_lbl = QLabel("Hybrid Video Restoration & AI Super-Resolution Engine")
        subtitle_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        gpu_lbl = QLabel(f"GPU: {self.cache_mgr.gpu_name} (NVENC Accelerated)")
        gpu_lbl.setObjectName("GpuBadge")
        header_layout.addWidget(gpu_lbl)

        main_layout.addLayout(header_layout)

        # Main Splitter (Left: Queue, Right: Tuning / Strategy)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left: Queue Container
        queue_container = QWidget()
        queue_layout = QVBoxLayout(queue_container)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.setSpacing(10)

        queue_hdr = QHBoxLayout()
        queue_title = QLabel("Video Processing Queue")
        queue_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #f1f5f9;")
        queue_hdr.addWidget(queue_title)
        queue_hdr.addStretch()

        self.btn_add_files = QPushButton("Add Videos...")
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_clear_completed = QPushButton("Clear Completed")

        queue_hdr.addWidget(self.btn_add_files)
        queue_hdr.addWidget(self.btn_remove_selected)
        queue_hdr.addWidget(self.btn_clear_completed)
        queue_layout.addLayout(queue_hdr)

        # Queue Table
        self.table_queue = DropTableWidget()
        self.table_queue.setColumnCount(6)
        self.table_queue.setHorizontalHeaderLabels([
            "Source File", "Input Res", "Target Res", "Encoder", "Status", "Progress"
        ])
        self.table_queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        queue_layout.addWidget(self.table_queue)
        splitter.addWidget(queue_container)

        # Right: Strategy & Configuration Sidebar
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(12)

        # Strategy Box
        grp_strategy = QGroupBox("Restoration & AI Upscaling Strategy")
        strat_layout = QVBoxLayout(grp_strategy)
        strat_layout.setSpacing(10)

        # Target Resolution
        res_row = QHBoxLayout()
        res_lbl = QLabel("Target Resolution:")
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems([
            "1920x1080 (1080p FHD)",
            "2560x1440 (1440p QHD)",
            "3840x2160 (4K UHD)",
            "2x Native Upscale",
            "4x Native Upscale"
        ])
        res_row.addWidget(res_lbl)
        res_row.addWidget(self.combo_resolution)
        strat_layout.addLayout(res_row)

        # Model Selection
        model_row = QHBoxLayout()
        model_lbl = QLabel("AI Model Engine:")
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "Real-ESRGAN_x4 (Balanced)",
            "SPAN (Fast Film / Anime)",
            "Fast Super-Resolution (Direct TRT Sim)"
        ])
        model_row.addWidget(model_lbl)
        model_row.addWidget(self.combo_model)
        strat_layout.addLayout(model_row)

        # Deinterlacing
        deint_row = QHBoxLayout()
        self.chk_deinterlace = QCheckBox("Deinterlace (Auto-detect scan)")
        self.chk_deinterlace.setChecked(True)
        deint_row.addWidget(self.chk_deinterlace)
        strat_layout.addLayout(deint_row)

        # Denoising
        denoise_row = QHBoxLayout()
        self.chk_denoise = QCheckBox("Analog Tape Denoise")
        self.chk_denoise.setChecked(True)
        denoise_row.addWidget(self.chk_denoise)
        strat_layout.addLayout(denoise_row)

        # Texture Blend Ratio (80/20 default)
        blend_box = QVBoxLayout()
        self.lbl_blend = QLabel("Micro-Texture Blend: 80% AI / 20% Original")
        self.slider_blend = QSlider(Qt.Orientation.Horizontal)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(80)
        self.slider_blend.valueChanged.connect(self._on_blend_changed)
        blend_box.addWidget(self.lbl_blend)
        blend_box.addWidget(self.slider_blend)
        strat_layout.addLayout(blend_box)

        # Film Grain (Default 6)
        grain_box = QVBoxLayout()
        self.lbl_grain = QLabel("Dynamic Film Grain: 6 (Subtle)")
        self.slider_grain = QSlider(Qt.Orientation.Horizontal)
        self.slider_grain.setRange(0, 20)
        self.slider_grain.setValue(6)
        self.slider_grain.valueChanged.connect(self._on_grain_changed)
        grain_box.addWidget(self.lbl_grain)
        grain_box.addWidget(self.slider_grain)
        strat_layout.addLayout(grain_box)

        # Video Encoder
        enc_row = QHBoxLayout()
        enc_lbl = QLabel("Video Encoder:")
        self.combo_encoder = QComboBox()
        self.combo_encoder.addItems([
            "hevc_nvenc (NVIDIA HEVC High Quality)",
            "h264_nvenc (NVIDIA H.264 Fast)",
            "libx265 (CPU High Efficiency)",
            "libx264 (CPU Standard)"
        ])
        enc_row.addWidget(enc_lbl)
        enc_row.addWidget(self.combo_encoder)
        strat_layout.addLayout(enc_row)

        # Checkpoint Segment Duration
        seg_row = QHBoxLayout()
        seg_lbl = QLabel("Segment Chunking:")
        self.combo_segments = QComboBox()
        self.combo_segments.addItems([
            "5 Minutes (Recommended Checkpoint)",
            "10 Minutes",
            "1 Minute (Testing)",
            "Disabled (Single File)"
        ])
        seg_row.addWidget(seg_lbl)
        seg_row.addWidget(self.combo_segments)
        strat_layout.addLayout(seg_row)

        # Output Folder Selection
        out_box = QVBoxLayout()
        out_lbl = QLabel("Output Destination (Enforced .mkv):")
        out_btn_row = QHBoxLayout()
        self.lbl_out_dir = QLabel("Same as input video location (Default)")
        self.lbl_out_dir.setStyleSheet("color: #38bdf8; font-size: 11px;")
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._browse_output_dir)
        btn_reset_out = QPushButton("Reset")
        btn_reset_out.clicked.connect(self._reset_output_dir)
        out_btn_row.addWidget(self.lbl_out_dir, 1)
        out_btn_row.addWidget(btn_browse_out)
        out_btn_row.addWidget(btn_reset_out)
        out_box.addWidget(out_lbl)
        out_box.addLayout(out_btn_row)
        strat_layout.addLayout(out_box)

        sidebar_layout.addWidget(grp_strategy)
        sidebar_layout.addStretch()

        splitter.addWidget(sidebar_widget)
        splitter.setSizes([750, 450])
        main_layout.addWidget(splitter, 1)

        # Metrics Bar
        metrics_layout = QHBoxLayout()
        self.card_frame = MetricCard("Current Frame", "0 / 0")
        self.card_fps = MetricCard("Inference FPS", "--")
        self.card_stage = MetricCard("Pipeline Stage", "Idle")
        self.card_eta = MetricCard("Estimated Time", "--:--")

        metrics_layout.addWidget(self.card_frame)
        metrics_layout.addWidget(self.card_fps)
        metrics_layout.addWidget(self.card_stage)
        metrics_layout.addWidget(self.card_eta)
        main_layout.addLayout(metrics_layout)

        # Dual Progress Bars
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(4)

        current_job_hdr = QHBoxLayout()
        self.lbl_active_job = QLabel("Active Job: None")
        self.lbl_active_job.setStyleSheet("font-size: 12px; color: #cbd5e1; font-weight: 500;")
        current_job_hdr.addWidget(self.lbl_active_job)
        prog_layout.addLayout(current_job_hdr)

        self.bar_current = QProgressBar()
        self.bar_current.setRange(0, 100)
        self.bar_current.setValue(0)
        prog_layout.addWidget(self.bar_current)

        main_layout.addLayout(prog_layout)

        # Action Buttons
        actions_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Queue Processing")
        self.btn_start.setObjectName("PrimaryButton")
        self.btn_start.setFixedHeight(38)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setFixedHeight(38)
        self.btn_pause.setEnabled(False)

        self.btn_cancel = QPushButton("Cancel Active")
        self.btn_cancel.setObjectName("DangerButton")
        self.btn_cancel.setFixedHeight(38)
        self.btn_cancel.setEnabled(False)

        actions_layout.addWidget(self.btn_start, 2)
        actions_layout.addWidget(self.btn_pause, 1)
        actions_layout.addWidget(self.btn_cancel, 1)
        main_layout.addLayout(actions_layout)

        # Live Log Console
        log_hdr = QHBoxLayout()
        log_lbl = QLabel("Live Execution & Engine Logs")
        log_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #94a3b8;")
        self.btn_clear_logs = QPushButton("Clear Logs")
        self.btn_clear_logs.setFixedHeight(24)
        self.btn_clear_logs.clicked.connect(lambda: self.txt_logs.clear())
        log_hdr.addWidget(log_lbl)
        log_hdr.addStretch()
        log_hdr.addWidget(self.btn_clear_logs)
        main_layout.addLayout(log_hdr)

        self.txt_logs = QPlainTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setFixedHeight(120)
        main_layout.addWidget(self.txt_logs)

    def _connect_signals(self):
        self.btn_add_files.clicked.connect(self._open_file_dialog)
        self.btn_remove_selected.clicked.connect(self._remove_selected_job)
        self.btn_clear_completed.clicked.connect(self.queue_controller.clear_completed)

        self.table_queue.sig_files_dropped.connect(self._add_files_to_queue)

        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        self.queue_controller.sig_queue_updated.connect(self._refresh_table)
        self.queue_controller.sig_job_started.connect(self._on_job_started)
        self.queue_controller.sig_job_progress.connect(self._on_job_progress)
        self.queue_controller.sig_job_completed.connect(self._on_job_completed)
        self.queue_controller.sig_job_failed.connect(self._on_job_failed)
        self.queue_controller.sig_queue_finished.connect(self._on_queue_finished)
        self.queue_controller.sig_log.connect(self.log)

    def log(self, message: str):
        timestamp = QTime.currentTime().toString("hh:mm:ss")
        self.txt_logs.appendPlainText(f"[{timestamp}] {message}")
        self.txt_logs.ensureCursorVisible()

    def _on_blend_changed(self, val: int):
        self.lbl_blend.setText(f"Micro-Texture Blend: {val}% AI / {100 - val}% Original")

    def _on_grain_changed(self, val: int):
        self.lbl_grain.setText(f"Dynamic Film Grain: {val} ({'Off' if val == 0 else 'Subtle' if val < 10 else 'Medium'})")

    def _browse_output_dir(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select Output Directory", str(Path.cwd()))
        if chosen:
            self.custom_output_directory = chosen
            self.lbl_out_dir.setText(chosen)

    def _reset_output_dir(self):
        self.custom_output_directory = None
        self.lbl_out_dir.setText("Same as input video location (Default)")

    def _open_file_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            str(Path.cwd()),
            "Video Files (*.avi *.mp4 *.mkv *.mov *.ts *.m2ts *.webm);;All Files (*.*)"
        )
        if files:
            self._add_files_to_queue(files)

    def _add_files_to_queue(self, filepaths: List[str]):
        tw, th = self._get_target_resolution()
        model_name = self.combo_model.currentText().split()[0]
        enc_name = self.combo_encoder.currentText().split()[0]
        blend_ratio = self.slider_blend.value() / 100.0
        grain = self.slider_grain.value()
        seg_dur = self._get_segment_duration()

        for fpath in filepaths:
            try:
                media = probe_media(fpath)
                in_res = f"{media.width}x{media.height}"

                if "Native" in self.combo_resolution.currentText():
                    scale_mult = 2 if "2x" in self.combo_resolution.currentText() else 4
                    cur_tw = media.width * scale_mult
                    cur_th = media.height * scale_mult
                else:
                    cur_tw, cur_th = tw, th

                stem = Path(fpath).stem
                target_dir = Path(self.custom_output_directory) if self.custom_output_directory else Path(fpath).parent
                out_mkv = str(target_dir / f"{stem}_upscaled_{cur_th}p.mkv")

                strategy = StrategyConfig(
                    input_path=fpath,
                    output_path=out_mkv,
                    target_width=cur_tw,
                    target_height=cur_th,
                    model_name=model_name,
                    deinterlace=self.chk_deinterlace.isChecked() and media.video.is_interlaced if media.video else False,
                    denoise=self.chk_denoise.isChecked(),
                    blend_ai_ratio=blend_ratio,
                    film_grain_intensity=grain,
                    encoder=enc_name,
                    segment_duration_sec=seg_dur
                )

                self.queue_controller.add_job(fpath, strategy, media)

            except Exception as e:
                self.log(f"Failed to probe {fpath}: {e}")
                QMessageBox.warning(self, "Probe Error", f"Could not inspect {fpath}:\n{e}")

    def _get_target_resolution(self) -> tuple[int, int]:
        txt = self.combo_resolution.currentText()
        if "1080p" in txt:
            return 1920, 1080
        elif "1440p" in txt:
            return 2560, 1440
        elif "4K" in txt:
            return 3840, 2160
        return 1920, 1080

    def _get_segment_duration(self) -> float:
        txt = self.combo_segments.currentText()
        if "5 Minutes" in txt:
            return 300.0
        elif "10 Minutes" in txt:
            return 600.0
        elif "1 Minute" in txt:
            return 60.0
        return 0.0

    def _refresh_table(self):
        self.table_queue.setRowCount(len(self.queue_controller.jobs))
        for row, job in enumerate(self.queue_controller.jobs):
            # 0: Name
            self.table_queue.setItem(row, 0, QTableWidgetItem(Path(job.input_path).name))
            # 1: Input Res
            in_res = f"{job.media_info.width}x{job.media_info.height}" if job.media_info else "--"
            self.table_queue.setItem(row, 1, QTableWidgetItem(in_res))
            # 2: Target Res
            tgt_res = f"{job.strategy.target_width}x{job.strategy.target_height}"
            self.table_queue.setItem(row, 2, QTableWidgetItem(tgt_res))
            # 3: Encoder
            self.table_queue.setItem(row, 3, QTableWidgetItem(job.strategy.encoder))
            # 4: Status
            badge = StatusBadge(job.status)
            self.table_queue.setCellWidget(row, 4, badge)
            # 5: Progress
            pbar = QProgressBar()
            pbar.setValue(int(job.progress_percent))
            self.table_queue.setCellWidget(row, 5, pbar)

    def _remove_selected_job(self):
        selected_rows = self.table_queue.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.queue_controller.jobs):
            job = self.queue_controller.jobs[row]
            self.queue_controller.remove_job(job.id)

    def _on_start_clicked(self):
        if not self.queue_controller.jobs:
            QMessageBox.information(self, "Queue Empty", "Please add at least one video to the queue.")
            return

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.queue_controller.start_queue()

    def _on_pause_clicked(self):
        if self.btn_pause.text() == "Pause":
            self.queue_controller.pause_queue()
            self.btn_pause.setText("Resume")
        else:
            self.queue_controller.resume_queue()
            self.btn_pause.setText("Pause")

    def _on_cancel_clicked(self):
        self.queue_controller.cancel_active_job()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_pause.setText("Pause")

    def _on_job_started(self, job_id: str):
        job = next((j for j in self.queue_controller.jobs if j.id == job_id), None)
        if job:
            self.lbl_active_job.setText(f"Active Job: {Path(job.input_path).name} -> {Path(job.output_path).name}")
            self.card_stage.set_value("Processing")

    def _on_job_progress(self, job_id: str, cur_frame: int, total_frames: int, fps: float, stage: str):
        self.card_frame.set_value(f"{cur_frame:,} / {total_frames:,}")
        self.card_fps.set_value(f"{fps:.1f}" if fps > 0 else "--")
        self.card_stage.set_value(stage)

        pct = (cur_frame / total_frames * 100.0) if total_frames > 0 else 0.0
        self.bar_current.setValue(int(pct))

        if fps > 0 and total_frames > cur_frame:
            remaining_sec = int((total_frames - cur_frame) / fps)
            mins, secs = divmod(remaining_sec, 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                self.card_eta.set_value(f"{hours:02d}:{mins:02d}:{secs:02d}")
            else:
                self.card_eta.set_value(f"{mins:02d}:{secs:02d}")
        else:
            self.card_eta.set_value("--:--")

    def _on_job_completed(self, job_id: str, out_path: str):
        self.log(f"Job {job_id} successfully produced: {out_path}")
        self.bar_current.setValue(100)
        self.card_stage.set_value("Completed")

    def _on_job_failed(self, job_id: str, err: str):
        self.log(f"Job {job_id} failed: {err}")
        self.card_stage.set_value("Failed")

    def _on_queue_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.lbl_active_job.setText("Active Job: None (Queue Completed)")
        self.card_stage.set_value("Idle")
        self.card_fps.set_value("--")
        self.card_eta.set_value("--:--")
        QMessageBox.information(self, "Processing Finished", "All queued video upscaling jobs have completed!")
