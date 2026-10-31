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
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.job_queue import JobItem, JobQueueController
from core.model_downloader import (
    ModelDownloadWorker,
    ModelManager,
    PretrainedModelEntry,
)
from core.probe import MediaMetadata, StrategyConfig, detect_gpu_name, detect_scene_cuts, probe_media
from ui.theme import DARK_STYLE
from ui.widgets import (
    DropTableWidget,
    MetricCard,
    StatusBadge,
)


class MainWindow(QMainWindow):
    """Main desktop application window for RemasterGO with tabbed architecture."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RemasterGO - Hybrid Video Restoration & AI Upscaling Engine")
        self.resize(1260, 860)
        self.setMinimumSize(980, 680)

        self.gpu_name = detect_gpu_name()
        self.model_mgr = ModelManager()
        self.queue_controller = JobQueueController(self)
        self.active_download_worker: Optional[ModelDownloadWorker] = None

        self.custom_output_directory: Optional[str] = None

        self._setup_ui()
        self._connect_signals()

        self.log(f"RemasterGO initialized. Detected hardware: {self.gpu_name}")
        self.log("Output destination: Same as input video location (default)")

    def _setup_ui(self):
        self.setStyleSheet(DARK_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Top Header Bar
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

        gpu_lbl = QLabel(f"GPU: {self.gpu_name} (Vulkan AI + NVENC)")
        gpu_lbl.setObjectName("GpuBadge")
        header_layout.addWidget(gpu_lbl)

        main_layout.addLayout(header_layout)

        # Tabbed Central Widget to prevent overlap
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        tab_queue = self._create_queue_tab()
        tab_settings = self._create_settings_tab()
        tab_models = self._create_models_tab()
        tab_logs = self._create_logs_tab()

        self.tabs.addTab(tab_queue, "Queue & Processing")
        self.tabs.addTab(tab_settings, "Restoration Settings")
        self.tabs.addTab(tab_models, "AI Models")
        self.tabs.addTab(tab_logs, "Execution Logs")

        main_layout.addWidget(self.tabs, 1)

    def _create_queue_tab(self) -> QWidget:
        """Tab 1: Video Queue, Progress Metrics, and Process Controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Queue Toolbar
        queue_hdr = QHBoxLayout()
        queue_title = QLabel("Video Processing Queue")
        queue_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f1f5f9;")
        queue_hdr.addWidget(queue_title)
        queue_hdr.addStretch()

        self.btn_add_files = QPushButton("Add Videos...")
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_clear_completed = QPushButton("Clear Completed")

        queue_hdr.addWidget(self.btn_add_files)
        queue_hdr.addWidget(self.btn_remove_selected)
        queue_hdr.addWidget(self.btn_clear_completed)
        layout.addLayout(queue_hdr)

        # Queue Table
        self.table_queue = DropTableWidget()
        self.table_queue.setColumnCount(7)
        self.table_queue.setHorizontalHeaderLabels([
            "Source File", "AI Model", "Input Res", "Target Res", "Encoder", "Status", "Progress"
        ])
        self.table_queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table_queue, 1)

        # Real-time Metrics Row
        metrics_layout = QHBoxLayout()
        self.card_frame = MetricCard("Current Frame", "0 / 0")
        self.card_fps = MetricCard("Inference FPS", "--")
        self.card_engine = MetricCard("Running Engine", "Auto GPU")
        self.card_stage = MetricCard("Pipeline Stage", "Idle")
        self.card_eta = MetricCard("Estimated Time", "--:--")

        metrics_layout.addWidget(self.card_frame)
        metrics_layout.addWidget(self.card_fps)
        metrics_layout.addWidget(self.card_engine)
        metrics_layout.addWidget(self.card_stage)
        metrics_layout.addWidget(self.card_eta)
        layout.addLayout(metrics_layout)

        # Active Job and Progress Bars
        prog_layout = QVBoxLayout()
        prog_layout.setSpacing(6)

        current_job_hdr = QHBoxLayout()
        self.lbl_active_job = QLabel("Active Job: None")
        self.lbl_active_job.setStyleSheet("font-size: 12px; color: #cbd5e1; font-weight: 600;")
        self.lbl_selected_model = QLabel("Selected Model: Real-ESRGAN_x4")
        self.lbl_selected_model.setStyleSheet("font-size: 12px; color: #38bdf8; font-weight: 500;")
        current_job_hdr.addWidget(self.lbl_active_job)
        current_job_hdr.addStretch()
        current_job_hdr.addWidget(self.lbl_selected_model)
        prog_layout.addLayout(current_job_hdr)

        # Primary Active Job Progress Bar (Large, High Contrast)
        self.bar_current = QProgressBar()
        self.bar_current.setObjectName("ActiveJobProgressBar")
        self.bar_current.setRange(0, 100)
        self.bar_current.setValue(0)
        self.bar_current.setTextVisible(True)
        self.bar_current.setFormat("0.0% (0 / 0 frames)")
        prog_layout.addWidget(self.bar_current)

        # Secondary Overall Queue Progress Bar (Sleek Batch Tracker)
        queue_prog_hdr = QHBoxLayout()
        self.lbl_queue_prog = QLabel("Queue Progress: 0 of 0 jobs completed (0%)")
        self.lbl_queue_prog.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: 500;")
        queue_prog_hdr.addWidget(self.lbl_queue_prog)
        queue_prog_hdr.addStretch()
        prog_layout.addLayout(queue_prog_hdr)

        self.bar_queue = QProgressBar()
        self.bar_queue.setObjectName("QueueProgressBar")
        self.bar_queue.setRange(0, 100)
        self.bar_queue.setValue(0)
        self.bar_queue.setTextVisible(True)
        self.bar_queue.setFormat("0% (0 jobs)")
        prog_layout.addWidget(self.bar_queue)

        layout.addLayout(prog_layout)

        # Action Buttons (Start, Pause, Cancel)
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
        layout.addLayout(actions_layout)

        # Bottom Live Activity Status Ticker
        ticker_box = QHBoxLayout()
        self.lbl_ticker = QLabel("Status: Ready to process.")
        self.lbl_ticker.setStyleSheet("color: #94a3b8; font-size: 11px;")
        ticker_box.addWidget(self.lbl_ticker)
        layout.addLayout(ticker_box)

        return tab

    def _create_settings_tab(self) -> QWidget:
        """Tab 2: Restoration parameters, Tuning sliders, and Output configuration."""
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Group 1: Resolution & Model Strategy
        grp_res = QGroupBox("Target Resolution & AI Model Strategy")
        res_layout = QVBoxLayout(grp_res)
        res_layout.setSpacing(12)

        tip_res = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Target Resolution</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Sets the output video resolution and dimensions for AI upscaling.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Increase / Higher (e.g. 4K UHD, 1440p, 4x):</b><br>"
            "• Yields maximum sharpness, ultra-fine textures, and clarity on large screens.<br>"
            "• Increases processing time, GPU VRAM requirements, and output file size.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Decrease / Lower (e.g. 1080p FHD, 2x):</b><br>"
            "• Renders significantly faster with low GPU memory demand.<br>"
            "• Produces compact video file sizes with standard definition detail.</p>"
            "</div>"
        )
        tip_model = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>AI Model Engine</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Selects the neural network model used for super-resolution and artifact removal.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Higher / Specialized Models (e.g. Real-ESRGAN_x4plus, Anime_6B):</b><br>"
            "• Delivers higher quality restoration, crisp line art, and deep texture reconstruction.<br>"
            "• Requires more GPU compute power and longer processing time per frame.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Lower / Fast Models (e.g. realesr-general-x4v3, RealESRGAN_x2plus, SPAN):</b><br>"
            "• High-throughput processing with lower compute and VRAM requirements.<br>"
            "• Provides moderate enhancement suited for fast turnaround or cleaner source files.</p>"
            "</div>"
        )

        res_row = QHBoxLayout()
        res_lbl = QLabel("Target Resolution:")
        res_lbl.setFixedWidth(160)
        res_lbl.setToolTip(tip_res)
        self.combo_resolution = QComboBox()
        self.combo_resolution.setToolTip(tip_res)
        self.combo_resolution.addItems([
            "2x Native Scale (Aspect Ratio Preserved)",
            "4x Native Scale (Aspect Ratio Preserved)",
            "1x Native Restoration (Original Size)",
            "3x Native Scale (Aspect Ratio Preserved)",
            "1920x1080 (1080p FHD)",
            "2560x1440 (1440p QHD)",
            "3840x2160 (4K UHD)"
        ])
        res_row.addWidget(res_lbl)
        res_row.addWidget(self.combo_resolution, 1)
        res_layout.addLayout(res_row)

        model_row = QHBoxLayout()
        model_lbl = QLabel("AI Model Engine:")
        model_lbl.setFixedWidth(160)
        model_lbl.setToolTip(tip_model)
        self.combo_model = QComboBox()
        self.combo_model.setToolTip(tip_model)
        self.combo_model.addItems([
            "Nomos2_OTF_ESRGAN_x4 (Degraded / Noisy Old Video)",
            "uniscale_restore (Degraded / Old Video Restoration)",
            "ultramix-balanced-4x (Balanced Natural Detail & Low Noise)",
            "high-fidelity-4x (High-Fidelity Photorealistic Video)",
            "realesrgan-x4plus (General Real-World Video 4x)",
            "4xNomos8kSC (Compression-Resistant Film & Video 4x)",
            "remacri-4x (Fine Texture & Film Restoration 4x)",
            "ultrasharp-4x (Maximum Edge Sharpness 4x)",
            "RealESRGAN_General_WDN_x4_v3 (Fast Video with Denoise 4x)",
            "realesrgan-x4plus-anime (Anime & Sharp Line-Art 6B 4x)",
            "realesr-animevideov3 (Fast Animation & Temporal Stability)"
        ])
        model_row.addWidget(model_lbl)
        model_row.addWidget(self.combo_model, 1)
        res_layout.addLayout(model_row)

        # Selected Model Guide Card (Best For / Pros / Cons / Recommended Denoise)
        self.frame_model_guide = QWidget()
        self.frame_model_guide.setObjectName("ModelGuideCard")
        self.frame_model_guide.setStyleSheet(
            "QWidget#ModelGuideCard { background-color: #161b26; border: 1px solid #2d3748; border-radius: 8px; }"
        )
        guide_layout = QVBoxLayout(self.frame_model_guide)
        guide_layout.setContentsMargins(14, 10, 14, 10)
        guide_layout.setSpacing(6)

        guide_hdr_row = QHBoxLayout()
        self.lbl_guide_title = QLabel("Model Guide")
        self.lbl_guide_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38bdf8; border: none; background: transparent;")
        self.btn_apply_rec_denoise = QPushButton("Apply Recommended Denoise")
        self.btn_apply_rec_denoise.setStyleSheet(
            "background-color: #1e293b; color: #7dd3fc; border: 1px solid #0284c7; "
            "font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 5px;"
        )
        self.btn_apply_rec_denoise.clicked.connect(self._apply_recommended_denoise_for_current_model)
        guide_hdr_row.addWidget(self.lbl_guide_title, 1)
        guide_hdr_row.addWidget(self.btn_apply_rec_denoise)
        guide_layout.addLayout(guide_hdr_row)

        self.lbl_guide_body = QLabel()
        self.lbl_guide_body.setWordWrap(True)
        self.lbl_guide_body.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_guide_body.setStyleSheet("font-size: 12px; border: none; background: transparent;")
        guide_layout.addWidget(self.lbl_guide_body)

        res_layout.addWidget(self.frame_model_guide)

        layout.addWidget(grp_res)

        # Group 2: Artifact Cleaning & Scan
        grp_cleaning = QGroupBox("Artifact Cleaning & Scan Pre-processing")
        clean_layout = QVBoxLayout(grp_cleaning)
        clean_layout.setSpacing(10)

        tip_deinterlace = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Motion-Adaptive Deinterlacing</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Reconstructs interlaced video fields (e.g. 480i/1080i broadcast, VHS, DVD) into progressive frames using QTGMC / bwdif filters.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Enabled (Checked):</b><br>"
            "• Eliminates horizontal combing artifacts and jagged edges on moving objects.<br>"
            "• Prevents the AI model from amplifying interlacing artifacts into distorted patterns.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Disabled (Unchecked):</b><br>"
            "• Bypasses deinterlacing; recommended only for true progressive footage (e.g. 720p/1080p web or film).</p>"
            "</div>"
        )
        tip_denoise = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Analog Tape Denoise</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Applies multi-frame spatial-temporal noise filtration (KNLMeansCL / hqdn3d) before AI inference.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Enabled (Checked):</b><br>"
            "• Removes analog tape hiss, chroma noise, and sensor grain before upscaling.<br>"
            "• Helps the AI neural network focus on enhancing true subject details rather than noise.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Disabled (Unchecked):</b><br>"
            "• Retains original source grain; may cause the AI upscaler to accentuate background noise into blotchy textures.</p>"
            "</div>"
        )
        tip_scene_cuts = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Detect Scene Cut Boundaries</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Analyzes frame luminance and color histograms to detect camera shot changes.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Enabled (Checked):</b><br>"
            "• Snaps checkpoint segments cleanly to shot boundaries to prevent frame tearing and ghosting.<br>"
            "• Keeps temporal AI filters synchronized with shot changes for seamless transitions.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Disabled (Unchecked):</b><br>"
            "• Splits segments strictly based on elapsed time, which may slice across active scenes.</p>"
            "</div>"
        )

        self.chk_deinterlace = QCheckBox("Motion-Adaptive Deinterlace (QTGMC / bwdif auto-detect)")
        self.chk_deinterlace.setChecked(True)
        self.chk_deinterlace.setToolTip(tip_deinterlace)
        clean_layout.addWidget(self.chk_deinterlace)

        self.chk_denoise = QCheckBox("Analog Tape Denoise & Deblock (Deblock + TTempSmooth + Joint Bilateral)")
        self.chk_denoise.setChecked(True)
        self.chk_denoise.setToolTip(tip_denoise)
        clean_layout.addWidget(self.chk_denoise)

        denoise_str_box = QVBoxLayout()
        self.lbl_denoise_strength = QLabel("Denoise & Deblock Strength: 2.2 (Strong Old-Video Cleanup)")
        self.lbl_denoise_strength.setToolTip(tip_denoise)
        self.slider_denoise_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_denoise_strength.setToolTip(tip_denoise)
        self.slider_denoise_strength.setRange(5, 40)
        self.slider_denoise_strength.setValue(22)
        self.slider_denoise_strength.valueChanged.connect(self._on_denoise_strength_changed)
        denoise_str_box.addWidget(self.lbl_denoise_strength)
        denoise_str_box.addWidget(self.slider_denoise_strength)
        clean_layout.addLayout(denoise_str_box)

        self.chk_scene_cuts = QCheckBox("Detect Scene Cut Boundaries (Align Checkpoint Segments & Prevent Tearing)")
        self.chk_scene_cuts.setChecked(True)
        self.chk_scene_cuts.setToolTip(tip_scene_cuts)
        clean_layout.addWidget(self.chk_scene_cuts)

        self.chk_tta = QCheckBox("High-Precision TTA Mode (8x Geometric Ensemble - Maximum Final Video Quality, Slower)")
        self.chk_tta.setChecked(False)
        self.chk_tta.setToolTip(
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Test-Time Augmentation (TTA 8x Ensemble)</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Upscales each tile across 8 geometric rotations/reflections and averages the predictions.</p>"
            "<p style='color: #86efac; margin: 0;'>• Eliminates directional GAN artifacts and produces the highest possible final video clarity.<br>"
            "• Uses 200px VRAM-safe tiling (safe on 6GB/8GB GPUs), at ~4-6x longer inference time.</p></div>"
        )
        clean_layout.addWidget(self.chk_tta)

        layout.addWidget(grp_cleaning)

        # Group 3: Texture & Grain Tuning
        grp_tuning = QGroupBox("Micro-Texture & Dynamic Film Grain")
        tune_layout = QVBoxLayout(grp_tuning)
        tune_layout.setSpacing(12)

        tip_blend = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Micro-Texture Blend Ratio</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Controls the blending ratio between AI-upscaled detail and the original source image texture.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Increase (Towards 100% AI):</b><br>"
            "• Maximizes edge sharpness, clarity, and neural reconstruction.<br>"
            "• Higher values may appear overly smooth or slightly synthetic if source lacks organic texture.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Decrease (Towards 0% Original):</b><br>"
            "• Retains organic film grain, authentic optical softness, and original camera texture.<br>"
            "• Reduces AI super-resolution sharpness and detail enhancement.</p>"
            "</div>"
        )
        tip_grain = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Dynamic Film Grain Intensity</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Injects organic, frequency-matched procedural film grain over the rendered output.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Increase (Higher Value, e.g. 10 - 20):</b><br>"
            "• Adds a richer cinematic texture.<br>"
            "• Effectively masks AI plastic smoothness, color banding, and compression blocks in flat areas (e.g., skies/shadows).</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Decrease (Lower Value / 0 - Off):</b><br>"
            "• Yields a cleaner, noise-free image with smooth gradients.<br>"
            "• Low/zero values on heavy AI upscales may look unnaturally clean or sterile.</p>"
            "</div>"
        )

        blend_box = QVBoxLayout()
        self.lbl_blend = QLabel("Micro-Texture Blend: 100% AI / 0% Original")
        self.lbl_blend.setToolTip(tip_blend)
        self.slider_blend = QSlider(Qt.Orientation.Horizontal)
        self.slider_blend.setToolTip(tip_blend)
        self.slider_blend.setRange(0, 100)
        self.slider_blend.setValue(100)
        self.slider_blend.valueChanged.connect(self._on_blend_changed)
        blend_box.addWidget(self.lbl_blend)
        blend_box.addWidget(self.slider_blend)
        tune_layout.addLayout(blend_box)

        grain_box = QVBoxLayout()
        self.lbl_grain = QLabel("Dynamic Film Grain: 0 (Off)")
        self.lbl_grain.setToolTip(tip_grain)
        self.slider_grain = QSlider(Qt.Orientation.Horizontal)
        self.slider_grain.setToolTip(tip_grain)
        self.slider_grain.setRange(0, 20)
        self.slider_grain.setValue(0)
        self.slider_grain.valueChanged.connect(self._on_grain_changed)
        grain_box.addWidget(self.lbl_grain)
        grain_box.addWidget(self.slider_grain)
        tune_layout.addLayout(grain_box)

        layout.addWidget(grp_tuning)

        # Group 4: Encoding & Destination
        grp_out = QGroupBox("Video Encoding & Output Destination")
        out_layout = QVBoxLayout(grp_out)
        out_layout.setSpacing(12)

        tip_encoder = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Video Compression Encoder</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Selects the video codec and hardware accelerator used to render the final output MKV container.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Hardware NVENC (hevc_nvenc / h264_nvenc):</b><br>"
            "• Blazing fast GPU-accelerated encoding using dedicated NVIDIA silicon.<br>"
            "• Negligible CPU utilization, allowing maximum performance for AI processing.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Software CPU (libx265 / libx264):</b><br>"
            "• Slower encoding speed with heavy CPU utilization.<br>"
            "• Yields marginally higher compression efficiency at equivalent bitrates.</p>"
            "</div>"
        )
        tip_segments = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Segment Checkpoint Duration</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Sets the segment duration for chunked video processing. Chunks are rendered independently and concatenated losslessly.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Increase (e.g. 10 Minutes / Disabled):</b><br>"
            "• Produces fewer segment files and slightly reduces disk I/O on long, uninterrupted batches.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Decrease (e.g. 1 Minute / 5 Minutes):</b><br>"
            "• Creates frequent resume checkpoints, ensuring safety against unexpected interruptions or crashes.<br>"
            "• Keeps intermediate storage chunks smaller.</p>"
            "</div>"
        )
        tip_dest = (
            "<div style='font-size: 12px; line-height: 1.4;'>"
            "<p style='font-size: 13px; font-weight: bold; color: #38bdf8; margin: 0 0 4px 0;'>Output Destination Directory</p>"
            "<p style='color: #e2e8f0; margin: 0 0 6px 0;'>Specifies the directory where final processed MKV video files are saved.</p>"
            "<p style='color: #86efac; margin: 0 0 4px 0;'><b>Custom Directory (Browse):</b><br>"
            "• Consolidates all upscaled videos into a dedicated output folder of your choice.</p>"
            "<p style='color: #fca5a5; margin: 0;'><b>Default Location (Reset):</b><br>"
            "• Automatically exports each upscaled video to the same folder as its original source file.</p>"
            "</div>"
        )

        enc_row = QHBoxLayout()
        enc_lbl = QLabel("Video Encoder:")
        enc_lbl.setFixedWidth(160)
        enc_lbl.setToolTip(tip_encoder)
        self.combo_encoder = QComboBox()
        self.combo_encoder.setToolTip(tip_encoder)
        self.combo_encoder.addItems([
            "hevc_nvenc (NVIDIA HEVC High Quality)",
            "h264_nvenc (NVIDIA H.264 Fast)",
            "libx265 (CPU High Efficiency)",
            "libx264 (CPU Standard)"
        ])
        enc_row.addWidget(enc_lbl)
        enc_row.addWidget(self.combo_encoder, 1)
        out_layout.addLayout(enc_row)

        seg_row = QHBoxLayout()
        seg_lbl = QLabel("Segment Checkpoint Duration:")
        seg_lbl.setFixedWidth(160)
        seg_lbl.setToolTip(tip_segments)
        self.combo_segments = QComboBox()
        self.combo_segments.setToolTip(tip_segments)
        self.combo_segments.addItems([
            "5 Minutes (Recommended Checkpoint)",
            "10 Minutes",
            "1 Minute (Testing)",
            "Disabled (Single File)"
        ])
        seg_row.addWidget(seg_lbl)
        seg_row.addWidget(self.combo_segments, 1)
        out_layout.addLayout(seg_row)

        dest_box = QVBoxLayout()
        dest_lbl = QLabel("Output Destination (Strictly Enforced .mkv):")
        dest_lbl.setToolTip(tip_dest)
        dest_btn_row = QHBoxLayout()
        self.lbl_out_dir = QLabel("Same as input video location (Default)")
        self.lbl_out_dir.setStyleSheet("color: #38bdf8; font-size: 11px;")
        self.lbl_out_dir.setToolTip(tip_dest)
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.setToolTip(tip_dest)
        btn_browse_out.clicked.connect(self._browse_output_dir)
        btn_reset_out = QPushButton("Reset")
        btn_reset_out.setToolTip(tip_dest)
        btn_reset_out.clicked.connect(self._reset_output_dir)

        dest_btn_row.addWidget(self.lbl_out_dir, 1)
        dest_btn_row.addWidget(btn_browse_out)
        dest_btn_row.addWidget(btn_reset_out)
        dest_box.addWidget(dest_lbl)
        dest_box.addLayout(dest_btn_row)
        out_layout.addLayout(dest_box)

        layout.addWidget(grp_out)
        layout.addStretch()

        scroll.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll)
        return tab

    def _create_models_tab(self) -> QWidget:
        """Tab 3: Pretrained NCNN Vulkan AI Model Manager."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        models_hdr = QHBoxLayout()
        models_title = QLabel("Pretrained NCNN Vulkan AI Models (VRAM-Safe Tiled GPU Execution)")
        models_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        models_hdr.addWidget(models_title)
        models_hdr.addStretch()

        self.btn_refresh_models = QPushButton("Refresh Catalog")
        self.btn_refresh_models.clicked.connect(self._refresh_models_table)
        models_hdr.addWidget(self.btn_refresh_models)
        layout.addLayout(models_hdr)

        self.table_models = QTableWidget()
        self.table_models.setColumnCount(6)
        self.table_models.setHorizontalHeaderLabels([
            "Model Name", "Category", "Scale", "Size", "Status", "Action"
        ])
        self.table_models.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_models.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_models.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_models.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_models.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_models.setColumnWidth(5, 220)
        self.table_models.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        self.table_models.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_models.verticalHeader().setDefaultSectionSize(52)
        self.table_models.itemSelectionChanged.connect(self._on_catalog_table_selection_changed)
        layout.addWidget(self.table_models, 1)

        # Interactive Model Details & Pros/Cons Panel in AI Models Tab
        catalog_guide_box = QGroupBox("Selected Model Guide (GoodFor / Pros / Cons)")
        catalog_guide_layout = QVBoxLayout(catalog_guide_box)
        catalog_guide_layout.setContentsMargins(12, 10, 12, 10)
        catalog_guide_layout.setSpacing(6)

        catalog_guide_hdr = QHBoxLayout()
        self.lbl_catalog_guide_title = QLabel("Click any model above to inspect its Pros, Cons, and Best Use Case")
        self.lbl_catalog_guide_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #38bdf8;")
        self.btn_use_catalog_model = QPushButton("Use This Model in Restoration Settings")
        self.btn_use_catalog_model.setStyleSheet(
            "background-color: #0284c7; color: #ffffff; font-weight: 600; font-size: 11px; padding: 5px 12px; border-radius: 5px;"
        )
        self.btn_use_catalog_model.clicked.connect(self._use_selected_catalog_model)
        catalog_guide_hdr.addWidget(self.lbl_catalog_guide_title, 1)
        catalog_guide_hdr.addWidget(self.btn_use_catalog_model)
        catalog_guide_layout.addLayout(catalog_guide_hdr)

        self.lbl_catalog_guide_body = QLabel()
        self.lbl_catalog_guide_body.setWordWrap(True)
        self.lbl_catalog_guide_body.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_catalog_guide_body.setStyleSheet("font-size: 12px;")
        catalog_guide_layout.addWidget(self.lbl_catalog_guide_body)
        layout.addWidget(catalog_guide_box)

        download_box = QVBoxLayout()
        self.lbl_download_status = QLabel("All installed models run directly on your GPU via Real-ESRGAN Vulkan.")
        self.lbl_download_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        download_box.addWidget(self.lbl_download_status)

        self.model_download_bar = QProgressBar()
        self.model_download_bar.setRange(0, 100)
        self.model_download_bar.setValue(0)
        download_box.addWidget(self.model_download_bar)
        layout.addLayout(download_box)

        self._refresh_models_table()

        return tab

    def _create_logs_tab(self) -> QWidget:
        """Tab 4: Detailed Monospace Execution & Diagnostics Console."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        log_hdr = QHBoxLayout()
        log_title = QLabel("Application & Execution Console")
        log_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")
        log_hdr.addWidget(log_title)
        log_hdr.addStretch()

        self.btn_clear_logs = QPushButton("Clear Console")
        self.btn_clear_logs.clicked.connect(lambda: self.txt_logs.clear())
        self.btn_save_logs = QPushButton("Save Logs...")
        self.btn_save_logs.clicked.connect(self._save_logs_to_file)

        log_hdr.addWidget(self.btn_clear_logs)
        log_hdr.addWidget(self.btn_save_logs)
        layout.addLayout(log_hdr)

        self.txt_logs = QPlainTextEdit()
        self.txt_logs.setReadOnly(True)
        layout.addWidget(self.txt_logs, 1)

        return tab

    def _connect_signals(self):
        self.btn_add_files.clicked.connect(self._open_file_dialog)
        self.btn_remove_selected.clicked.connect(self._remove_selected_job)
        self.btn_clear_completed.clicked.connect(self.queue_controller.clear_completed)

        self.table_queue.sig_files_dropped.connect(self._add_files_to_queue)

        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)

        self.combo_model.currentTextChanged.connect(self._on_model_selection_changed)

        self.queue_controller.sig_queue_updated.connect(self._refresh_table)
        self.queue_controller.sig_job_started.connect(self._on_job_started)
        self.queue_controller.sig_job_progress.connect(self._on_job_progress)
        self.queue_controller.sig_job_completed.connect(self._on_job_completed)
        self.queue_controller.sig_job_failed.connect(self._on_job_failed)
        self.queue_controller.sig_queue_finished.connect(self._on_queue_finished)
        self.queue_controller.sig_log.connect(self.log)

        # Initial display setup for selected model and engine
        self._on_model_selection_changed(self.combo_model.currentText())

    def _get_engine_display_name(self, model_name: str, res: tuple[int, int] = (1920, 1080)) -> str:
        """Determine human-friendly display name of the GPU engine."""
        from core.ai_upscaler import is_ai_binary_available
        if is_ai_binary_available():
            return "Real-ESRGAN Vulkan"
        return "FFmpeg CAS"

    def _format_model_guide_html(self, entry: PretrainedModelEntry) -> str:
        return (
            f"<div style='line-height: 1.45;'>"
            f"<p style='margin: 0 0 4px 0; color: #f8fafc;'>"
            f"<b style='color: #38bdf8;'>Best For:</b> {entry.best_for}</p>"
            f"<p style='margin: 0 0 4px 0; color: #86efac;'>"
            f"<b>Good (Pros):</b> {entry.pros}</p>"
            f"<p style='margin: 0; color: #fca5a5;'>"
            f"<b>Trade-offs (Cons):</b> {entry.cons}</p>"
            f"</div>"
        )

    def _apply_recommended_denoise_for_current_model(self):
        model_name = self.combo_model.currentText().split()[0]
        entry = self.model_mgr.find_model_entry(model_name)
        if entry and hasattr(self, "slider_denoise_strength"):
            target_val = int(round(entry.recommended_denoise * 10))
            self.slider_denoise_strength.setValue(target_val)
            self.log(f"Applied recommended Denoise & Deblock Strength ({entry.recommended_denoise:.1f}) for {entry.name}")

    def _on_model_selection_changed(self, model_txt: str):
        model_name = model_txt.split()[0]
        if hasattr(self, "lbl_selected_model"):
            self.lbl_selected_model.setText(f"Selected Model: {model_name}")
        if hasattr(self, "card_engine") and (not hasattr(self, "queue_controller") or not self.queue_controller._is_running):
            self.card_engine.set_value(self._get_engine_display_name(model_name))

        entry = self.model_mgr.find_model_entry(model_name)
        if entry and hasattr(self, "lbl_guide_body"):
            self.lbl_guide_title.setText(
                f"{entry.name}  |  Speed: {entry.speed_tier}  |  Recommended Denoise: {entry.recommended_denoise:.1f}"
            )
            self.lbl_guide_body.setText(self._format_model_guide_html(entry))
            if hasattr(self, "btn_apply_rec_denoise"):
                self.btn_apply_rec_denoise.setText(f"Apply Recommended Denoise ({entry.recommended_denoise:.1f})")

    def _on_catalog_table_selection_changed(self):
        catalog = self.model_mgr.list_catalog()
        if not catalog or not hasattr(self, "lbl_catalog_guide_body"):
            return
        selected_rows = self.table_models.selectionModel().selectedRows() if self.table_models.selectionModel() else []
        row = selected_rows[0].row() if selected_rows else 0
        if 0 <= row < len(catalog):
            entry = catalog[row]
            self.lbl_catalog_guide_title.setText(
                f"{entry.name} ({entry.category})  —  Speed: {entry.speed_tier}  |  Recommended Denoise: {entry.recommended_denoise:.1f}"
            )
            self.lbl_catalog_guide_body.setText(self._format_model_guide_html(entry))

    def _use_selected_catalog_model(self):
        catalog = self.model_mgr.list_catalog()
        selected_rows = self.table_models.selectionModel().selectedRows() if self.table_models.selectionModel() else []
        row = selected_rows[0].row() if selected_rows else 0
        if 0 <= row < len(catalog):
            entry = catalog[row]
            stem = Path(entry.filename).stem.lower().replace("-x4", "")
            for idx in range(self.combo_model.count()):
                item_token = self.combo_model.itemText(idx).split()[0].lower()
                if item_token == Path(entry.filename).stem.lower() or stem in item_token:
                    self.combo_model.setCurrentIndex(idx)
                    break
            self._apply_recommended_denoise_for_current_model()
            self.tabs.setCurrentIndex(1)

    def log(self, message: str):
        timestamp = QTime.currentTime().toString("hh:mm:ss")
        entry = f"[{timestamp}] {message}"
        if hasattr(self, "txt_logs"):
            self.txt_logs.appendPlainText(entry)
            self.txt_logs.ensureCursorVisible()
        if hasattr(self, "lbl_ticker"):
            self.lbl_ticker.setText(f"Status: {message}")

    def _on_blend_changed(self, val: int):
        self.lbl_blend.setText(f"Micro-Texture Blend: {val}% AI / {100 - val}% Original")

    def _on_denoise_strength_changed(self, val: int):
        strength = val / 10.0
        level = "Light" if strength < 1.5 else ("Medium" if strength < 2.1 else ("Strong Old-Video Cleanup" if strength < 3.1 else "Heavy Degraded Restoration"))
        self.lbl_denoise_strength.setText(f"Denoise & Deblock Strength: {strength:.1f} ({level})")

    def _on_grain_changed(self, val: int):
        self.lbl_grain.setText(f"Dynamic Film Grain: {val} ({'Off' if val == 0 else 'Subtle' if val < 10 else 'Medium'})")

    def _save_logs_to_file(self):
        chosen, _ = QFileDialog.getSaveFileName(self, "Save Logs", "remastergo.log", "Log Files (*.log *.txt)")
        if chosen:
            with open(chosen, "w", encoding="utf-8") as f:
                f.write(self.txt_logs.toPlainText())
            self.log(f"Saved logs to: {chosen}")

    # -------------------------------------------------------------
    # AI Model Downloader Management
    # -------------------------------------------------------------
    def _refresh_models_table(self):
        """Populate the AI model catalog table with download and delete actions."""
        catalog = self.model_mgr.list_catalog()
        self.table_models.setRowCount(len(catalog))
        for row, entry in enumerate(catalog):
            self.table_models.setItem(row, 0, QTableWidgetItem(entry.name))
            self.table_models.setItem(row, 1, QTableWidgetItem(entry.category))
            self.table_models.setItem(row, 2, QTableWidgetItem(f"{entry.scale}x"))
            self.table_models.setItem(row, 3, QTableWidgetItem(f"{entry.filesize_mb:.1f} MB"))

            status_text = "Installed" if entry.installed else "Available"
            status_item = QTableWidgetItem(status_text)
            self.table_models.setItem(row, 4, status_item)

            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(6, 4, 6, 4)
            action_layout.setSpacing(8)

            if entry.installed:
                btn_info = QPushButton("Vulkan GPU Ready")
                btn_info.setStyleSheet(
                    "background-color: #1e293b; color: #38bdf8; border: 1px solid #0284c7; "
                    "font-weight: 600; font-size: 11px; padding: 6px 8px; border-radius: 5px; min-height: 28px;"
                )
                btn_info.setToolTip("Model is installed and ready for native Vulkan GPU upscaling.")
                action_layout.addWidget(btn_info)

                btn_del = QPushButton("Delete")
                btn_del.setStyleSheet(
                    "background-color: #b91c1c; color: #ffffff; font-weight: 600; font-size: 12px; "
                    "padding: 6px 12px; border-radius: 5px; min-height: 28px;"
                )
                btn_del.clicked.connect(lambda _, m=entry.id: self._delete_model(m))
                action_layout.addWidget(btn_del)
            else:
                btn_dl = QPushButton("Download Model")
                btn_dl.setStyleSheet(
                    "background-color: #0284c7; color: #ffffff; font-weight: 600; font-size: 12px; "
                    "padding: 6px 20px; border-radius: 5px; min-height: 28px;"
                )
                btn_dl.clicked.connect(lambda _, e=entry: self._download_model(e))
                action_layout.addWidget(btn_dl)

            self.table_models.setCellWidget(row, 5, action_widget)

            row_tip = self._format_model_guide_html(entry)
            for col in range(5):
                it = self.table_models.item(row, col)
                if it:
                    it.setToolTip(row_tip)

        if self.table_models.rowCount() > 0 and not self.table_models.selectionModel().selectedRows():
            self.table_models.selectRow(0)
        self._on_catalog_table_selection_changed()

    def _download_model(self, model_entry: PretrainedModelEntry):
        """Initiate asynchronous download of a pretrained model file."""
        if self.active_download_worker and self.active_download_worker.isRunning():
            QMessageBox.warning(self, "Download in Progress", "A model download is already active.")
            return

        self.model_download_bar.setValue(0)
        self.lbl_download_status.setText(f"Connecting to download {model_entry.name}...")
        self.log(f"Starting download of {model_entry.name} from {model_entry.download_url}")

        self.active_download_worker = ModelDownloadWorker(
            model_entry=model_entry,
            target_dir=str(self.model_mgr.models_dir),
            parent=self
        )
        self.active_download_worker.sig_progress.connect(self._on_model_download_progress)
        self.active_download_worker.sig_status.connect(self._on_model_download_status)
        self.active_download_worker.sig_finished.connect(self._on_model_download_finished)
        self.active_download_worker.sig_error.connect(self._on_model_download_error)
        self.active_download_worker.start()

    def _on_model_download_progress(self, model_id: str, downloaded: int, total: int, pct: float):
        self.model_download_bar.setValue(int(pct))
        dl_mb = downloaded / (1024 * 1024)
        tot_mb = total / (1024 * 1024)
        self.lbl_download_status.setText(f"Downloading {model_id}: {dl_mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)")

    def _on_model_download_status(self, model_id: str, status_msg: str):
        self.lbl_download_status.setText(status_msg)
        self.log(f"[Downloader] {model_id}: {status_msg}")

    def _on_model_download_finished(self, model_id: str, local_path: str):
        self.model_download_bar.setValue(100)
        self.lbl_download_status.setText(f"Installed {model_id} to {Path(local_path).name}")
        self.log(f"Model {model_id} successfully installed to {local_path}")
        self._refresh_models_table()
        QMessageBox.information(self, "Download Complete", f"AI Model '{model_id}' is now installed and ready.")

    def _on_model_download_error(self, model_id: str, err: str):
        self.lbl_download_status.setText(f"Download failed: {err}")
        self.log(f"Error downloading {model_id}: {err}")
        QMessageBox.critical(self, "Download Failed", f"Could not download model {model_id}:\n{err}")
        self._refresh_models_table()

    def _delete_model(self, model_id: str):
        confirm = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete model '{model_id}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.model_mgr.delete_model(model_id)
            self.log(f"Deleted model: {model_id}")
            self._refresh_models_table()

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
        denoise_str = self.slider_denoise_strength.value() / 10.0 if hasattr(self, "slider_denoise_strength") else 2.2
        seg_dur = self._get_segment_duration()

        for fpath in filepaths:
            try:
                media = probe_media(fpath)
                in_res = f"{media.display_width}x{media.display_height}"

                res_txt = self.combo_resolution.currentText()
                if "Native" in res_txt or "Scale" in res_txt:
                    if "1x" in res_txt:
                        scale_mult = 1
                    elif "3x" in res_txt:
                        scale_mult = 3
                    elif "4x" in res_txt:
                        scale_mult = 4
                    else:
                        scale_mult = 2
                    cur_tw = media.display_width * scale_mult
                    cur_th = media.display_height * scale_mult
                else:
                    cur_tw, cur_th = tw, th
                    # Automatically adapt target resolution orientation for vertical/portrait videos
                    if media.is_portrait and cur_tw > cur_th:
                        cur_tw, cur_th = cur_th, cur_tw

                # Ensure dimensions are even for video codec compatibility
                cur_tw = cur_tw + (cur_tw % 2)
                cur_th = cur_th + (cur_th % 2)

                stem = Path(fpath).stem
                target_dir = Path(self.custom_output_directory) if self.custom_output_directory else Path(fpath).parent
                out_mkv = str(target_dir / f"{model_name}_{stem}_upscaled_{cur_th}p.mkv")

                strategy = StrategyConfig(
                    input_path=fpath,
                    output_path=out_mkv,
                    target_width=cur_tw,
                    target_height=cur_th,
                    model_name=model_name,
                    tta_mode=self.chk_tta.isChecked() if hasattr(self, "chk_tta") else False,
                    deinterlace=self.chk_deinterlace.isChecked() and media.video.is_interlaced if media.video else False,
                    denoise=self.chk_denoise.isChecked(),
                    denoise_strength=denoise_str,
                    blend_ai_ratio=blend_ratio,
                    film_grain_intensity=grain,
                    encoder=enc_name,
                    segment_duration_sec=seg_dur,
                    scene_cuts=[]
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

    def _update_queue_progress(self):
        """Update the overall queue progress bar and badge based on batch status."""
        if not hasattr(self, "bar_queue"):
            return
        total_jobs = len(self.queue_controller.jobs)
        if total_jobs == 0:
            self.bar_queue.setValue(0)
            self.bar_queue.setFormat("0% (0 jobs)")
            if hasattr(self, "lbl_queue_prog"):
                self.lbl_queue_prog.setText("Queue Progress: 0 of 0 jobs completed (0%)")
            return

        completed_jobs = sum(1 for j in self.queue_controller.jobs if j.status == "Completed")
        active_pct = 0.0
        if self.queue_controller.active_job:
            active_pct = self.queue_controller.active_job.progress_percent / 100.0

        overall_pct = ((completed_jobs + active_pct) / total_jobs) * 100.0
        self.bar_queue.setValue(int(overall_pct))
        self.bar_queue.setFormat(f"{overall_pct:.1f}% ({completed_jobs}/{total_jobs} jobs done)")
        if hasattr(self, "lbl_queue_prog"):
            self.lbl_queue_prog.setText(f"Queue Progress: {completed_jobs} of {total_jobs} jobs completed ({overall_pct:.1f}%)")

    def _refresh_table(self):
        self.table_queue.setRowCount(len(self.queue_controller.jobs))
        for row, job in enumerate(self.queue_controller.jobs):
            # 0: Name
            self.table_queue.setItem(row, 0, QTableWidgetItem(Path(job.input_path).name))
            # 1: AI Model
            self.table_queue.setItem(row, 1, QTableWidgetItem(job.strategy.model_name))
            # 2: Input Res
            in_res = f"{job.media_info.width}x{job.media_info.height}" if job.media_info else "--"
            self.table_queue.setItem(row, 2, QTableWidgetItem(in_res))
            # 3: Target Res
            tgt_res = f"{job.strategy.target_width}x{job.strategy.target_height}"
            self.table_queue.setItem(row, 3, QTableWidgetItem(tgt_res))
            # 4: Encoder
            self.table_queue.setItem(row, 4, QTableWidgetItem(job.strategy.encoder))
            # 5: Status
            badge = StatusBadge(job.status)
            self.table_queue.setCellWidget(row, 5, badge)
            # 6: Progress
            pbar = QProgressBar()
            pbar.setObjectName("TableProgressBar")
            pbar.setRange(0, 100)
            pbar.setValue(int(job.progress_percent))
            pbar.setTextVisible(True)
            pbar.setFormat(f"{int(job.progress_percent)}%")
            if job.status == "Completed":
                pbar.setStyleSheet("QProgressBar::chunk { background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #10b981); }")
            elif job.status == "Failed":
                pbar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
            self.table_queue.setCellWidget(row, 6, pbar)

        self._update_queue_progress()

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
            res = (job.media_info.width, job.media_info.height) if job.media_info else (1920, 1080)
            engine_name = self._get_engine_display_name(job.strategy.model_name, res=res)
            self.lbl_active_job.setText(f"Active Job: {Path(job.input_path).name} -> {Path(job.output_path).name} (Model: {job.strategy.model_name})")
            self.card_engine.set_value(engine_name)
            self.card_stage.set_value("Processing")
            self.bar_current.setValue(0)
            self.bar_current.setFormat("0.0% (Starting...)")
            self._update_queue_progress()

    def _on_job_progress(self, job_id: str, cur_frame: int, total_frames: int, fps: float, stage: str):
        self.card_frame.set_value(f"{cur_frame:,} / {total_frames:,}")
        self.card_fps.set_value(f"{fps:.1f}" if fps > 0 else "--")
        if "AI Upscaling" in stage or "Real-ESRGAN" in stage:
            self.card_engine.set_value("Real-ESRGAN Vulkan")
        elif "Encoding" in stage:
            self.card_engine.set_value("Video Encoder")
        elif "FFmpeg" in stage:
            self.card_engine.set_value("FFmpeg CAS")
        self.card_stage.set_value(stage)

        pct = (cur_frame / total_frames * 100.0) if total_frames > 0 else 0.0
        new_bar_val = max(0, min(100, int(pct)))
        self.bar_current.setValue(new_bar_val)
        self.bar_current.setFormat(f"{pct:.1f}% ({cur_frame:,} / {total_frames:,} frames)")
        self._update_queue_progress()

        # Update table progress bar for the active job
        for row, job in enumerate(self.queue_controller.jobs):
            if job.id == job_id:
                pbar = self.table_queue.cellWidget(row, 6)
                if isinstance(pbar, QProgressBar):
                    pbar.setValue(new_bar_val)
                    pbar.setFormat(f"{pct:.1f}%")
                break

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
        self.bar_current.setFormat("100.0% (Completed)")
        self.card_stage.set_value("Completed")
        self._update_queue_progress()

    def _on_job_failed(self, job_id: str, err: str):
        self.log(f"Job {job_id} failed: {err}")
        self.card_stage.set_value("Failed")
        self.bar_current.setFormat("Failed")
        self._update_queue_progress()

    def _on_queue_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.lbl_active_job.setText("Active Job: None (Queue Completed)")
        self.card_engine.set_value(self._get_engine_display_name(self.combo_model.currentText().split()[0]))
        self.card_stage.set_value("Idle")
        self.card_fps.set_value("--")
        self.card_eta.set_value("--:--")
        self.bar_current.setValue(100)
        self.bar_current.setFormat("100.0% (Queue Completed)")
        self._update_queue_progress()
        QMessageBox.information(self, "Processing Finished", "All queued video upscaling jobs have completed!")
