"""Unit tests for PySide6 UI components and MainWindow instantiation."""

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.widgets import MetricCard, StatusBadge


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
    assert window.table_queue.columnCount() == 6
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

