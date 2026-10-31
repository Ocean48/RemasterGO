"""Unit tests for PySide6 UI components and MainWindow instantiation."""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.widgets import (
    MetricCard,
    StatusBadge,
)


def test_ui_widgets_instantiation():
    app = QApplication.instance() or QApplication(sys.argv)

    card = MetricCard("FPS", "60.0")
    assert card.val_lbl.text() == "60.0"
    card.set_value("120.0")
    assert card.val_lbl.text() == "120.0"

    badge = StatusBadge("Processing")
    assert badge.text() == "Processing"
    badge.set_status("Completed")
    assert badge.text() == "Completed"


def test_main_window_instantiation():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    assert window.windowTitle().startswith("RemasterGO")
    assert window.table_queue.columnCount() == 7
    assert window.card_engine is not None
    assert window.bar_current is not None
    assert window.bar_queue is not None
    assert any(
        k in window.lbl_selected_model.text()
        for k in ("uniscale", "ultramix", "high-fidelity", "Nomos", "RealPLKSR", "UltraSharp", "Real-ESRGAN", "realesr", "SPAN")
    )
    assert window.combo_resolution.count() > 0
    assert window.combo_encoder.count() > 0


def test_main_window_default_output_to_input_location():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._add_files_to_queue(["sample_640x360.avi"])
    assert len(window.queue_controller.jobs) == 1
    job = window.queue_controller.jobs[0]
    expected_dir = str(Path("sample_640x360.avi").parent.resolve())
    actual_dir = str(Path(job.output_path).parent.resolve())
    assert actual_dir == expected_dir
    selected_model = window.combo_model.currentText().split()[0]
    assert Path(job.output_path).name.startswith(f"{selected_model}_")


def test_tabbed_ui_navigation():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    assert window.tabs.count() == 4
    tab_names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert "Queue & Processing" in tab_names
    assert "Restoration Settings" in tab_names
    assert "AI Models" in tab_names
    assert "Execution Logs" in tab_names

    # Check model downloader table
    assert window.table_models.columnCount() == 6
    assert window.table_models.rowCount() >= 4


def test_cli_helper_commands(capsys):
    from main import list_models_cli
    code_models = list_models_cli()
    assert code_models == 0
    captured = capsys.readouterr()
    assert "Real-ESRGAN x4 Plus" in captured.out


def test_restoration_settings_tooltips():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    # Verify tooltips are present on controls and explain increase/decrease effects
    assert "Target Resolution" in window.combo_resolution.toolTip()
    assert "Increase / Higher" in window.combo_resolution.toolTip()
    assert "Decrease / Lower" in window.combo_resolution.toolTip()

    assert "AI Model Engine" in window.combo_model.toolTip()
    assert "Higher / Specialized Models" in window.combo_model.toolTip()
    assert "Lower / Fast Models" in window.combo_model.toolTip()

    assert "Motion-Adaptive Deinterlacing" in window.chk_deinterlace.toolTip()
    assert "Enabled" in window.chk_deinterlace.toolTip()
    assert "Disabled" in window.chk_deinterlace.toolTip()

    assert "Analog Tape Denoise" in window.chk_denoise.toolTip()
    assert "Enabled" in window.chk_denoise.toolTip()
    assert "Disabled" in window.chk_denoise.toolTip()

    assert "Detect Scene Cut Boundaries" in window.chk_scene_cuts.toolTip()
    assert "Enabled" in window.chk_scene_cuts.toolTip()
    assert "Disabled" in window.chk_scene_cuts.toolTip()

    assert "Micro-Texture Blend Ratio" in window.slider_blend.toolTip()
    assert "Increase" in window.slider_blend.toolTip()
    assert "Decrease" in window.slider_blend.toolTip()

    assert "Dynamic Film Grain Intensity" in window.slider_grain.toolTip()
    assert "Increase" in window.slider_grain.toolTip()
    assert "Decrease" in window.slider_grain.toolTip()

    assert "Video Compression Encoder" in window.combo_encoder.toolTip()
    assert "Hardware NVENC" in window.combo_encoder.toolTip()
    assert "Software CPU" in window.combo_encoder.toolTip()

    assert "Segment Checkpoint Duration" in window.combo_segments.toolTip()
    assert "Increase" in window.combo_segments.toolTip()
    assert "Decrease" in window.combo_segments.toolTip()

    assert "Output Destination Directory" in window.lbl_out_dir.toolTip()


def test_model_guide_pros_cons_display():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    # Verify initial model guide card in Restoration Settings shows Best For, Good (Pros), and Trade-offs (Cons)
    assert "Best For:" in window.lbl_guide_body.text()
    assert "Good (Pros):" in window.lbl_guide_body.text()
    assert "Trade-offs (Cons):" in window.lbl_guide_body.text()

    # Switch to a different model and verify guide updates dynamically
    prev_text = window.lbl_guide_body.text()
    window.combo_model.setCurrentIndex(7)  # ultrasharp-4x
    assert window.lbl_guide_body.text() != prev_text
    assert "Best For:" in window.lbl_guide_body.text()

    # Verify Apply Recommended Denoise button updates slider
    window.btn_apply_rec_denoise.click()
    assert window.slider_denoise_strength.value() == 15

    # Verify AI Models tab details panel and "Use This Model" button
    window.table_models.selectRow(1)
    assert "Best For:" in window.lbl_catalog_guide_body.text()
    window.btn_use_catalog_model.click()
    assert window.tabs.currentIndex() == 1






