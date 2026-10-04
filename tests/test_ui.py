"""Unit tests for PySide6 UI components and MainWindow instantiation."""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.widgets import (
    CompileEngineDialog,
    DownloadEngineDialog,
    EngineCacheInspectorDialog,
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
    assert "Real-ESRGAN" in window.lbl_selected_model.text() or "realesr" in window.lbl_selected_model.text() or "SPAN" in window.lbl_selected_model.text()
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


def test_engine_cache_inspector_dialog():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    dialog = EngineCacheInspectorDialog(window.cache_mgr)
    assert dialog.windowTitle() == "TensorRT Engine Cache Inspector"
    assert dialog.table.columnCount() == 6
    dialog.close()


def test_tabbed_ui_navigation():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    assert window.tabs.count() == 4
    tab_names = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert "Queue & Processing" in tab_names
    assert "Restoration Settings" in tab_names
    assert "AI Models & Engines" in tab_names
    assert "Execution Logs" in tab_names

    # Check model downloader and engines table
    assert window.table_models.columnCount() == 6
    assert window.table_models.rowCount() >= 4
    assert window.table_engines.columnCount() == 6


def test_download_engine_dialog():
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = DownloadEngineDialog()
    assert dlg.windowTitle() == "Download Precompiled TensorRT Engine"
    data = dlg.get_data()
    assert "url" in data
    assert "model_name" in data
    assert data["resolution"] == (1920, 1080)
    dlg.close()


def test_compile_engine_dialog(monkeypatch):
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = CompileEngineDialog(
        available_models=["SPAN_4x.onnx"],
        default_model="SPAN_4x.onnx",
        default_resolution=(384, 288)
    )
    assert dlg.windowTitle() == "Compile TensorRT Engine"
    data = dlg.get_data()
    assert data["model_filename"] == "SPAN_4x.onnx"
    assert data["resolution"] == (384, 288)
    assert data["fp16"] is True

    # Test typing custom resolution
    dlg.spin_width.setValue(640)
    dlg.spin_height.setValue(480)
    custom_data = dlg.get_data()
    assert custom_data["resolution"] == (640, 480)

    # Test preset selection
    dlg.combo_res_presets.setCurrentIndex(1)  # 1920x1080
    assert dlg.spin_width.value() == 1920
    assert dlg.spin_height.value() == 1080

    # Test browse trtexec without exception
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", ""))
    dlg._browse_trtexec()
    dlg.close()


def test_cli_helper_commands(capsys):
    from main import list_engines_cli, list_models_cli
    code_models = list_models_cli()
    assert code_models == 0
    captured = capsys.readouterr()
    assert "Real-ESRGAN x4 Plus" in captured.out

    code_engines = list_engines_cli()
    assert code_engines == 0





