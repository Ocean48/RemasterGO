"""Unit tests for segment checkpointing, state serialization, and recovery."""

import tempfile
from pathlib import Path

from core.checkpoint import CheckpointManager


def test_single_segment_planning():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            job_dir=tmpdir,
            input_path="test.avi",
            final_output_path="out.mkv",
            total_duration=120.0,
            segment_duration=300.0
        )
        assert len(mgr.state.segments) == 1
        seg = mgr.state.segments[0]
        assert seg.start_sec == 0.0
        assert seg.duration_sec == 120.0
        assert seg.output_filename == "segment_0000.mkv"


def test_multi_segment_planning_with_scene_cuts():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 750 seconds total, 300s chunking, scene cut at 295s and 598s
        mgr = CheckpointManager(
            job_dir=tmpdir,
            input_path="test.avi",
            final_output_path="out.mkv",
            total_duration=750.0,
            segment_duration=300.0,
            scene_cut_times=[100.0, 295.0, 598.0]
        )
        assert len(mgr.state.segments) == 3
        # First segment should snap to scene cut at 295.0
        assert mgr.state.segments[0].duration_sec == 295.0
        # Second segment starts at 295.0 and snaps to 598.0 (duration 303.0)
        assert mgr.state.segments[1].start_sec == 295.0
        assert mgr.state.segments[1].duration_sec == 303.0


def test_checkpoint_save_and_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            job_dir=tmpdir,
            input_path="test.avi",
            final_output_path="out.mkv",
            total_duration=600.0,
            segment_duration=300.0
        )
        mgr.mark_segment_processing(0)
        mgr.mark_segment_completed(0, rendered_frames=9000)

        # Reload from same job_dir
        mgr2 = CheckpointManager(
            job_dir=tmpdir,
            input_path="test.avi",
            final_output_path="out.mkv",
            total_duration=600.0,
            segment_duration=300.0
        )
        assert len(mgr2.state.segments) == 2
        # Without on-disk file, validate_existing_segments resets missing files to pending
        assert mgr2.state.total_duration == 600.0


def test_concat_file_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            job_dir=tmpdir,
            input_path="test.avi",
            final_output_path="out.mkv",
            total_duration=600.0,
            segment_duration=300.0
        )
        concat_txt = mgr.generate_concat_file()
        assert Path(concat_txt).is_file()
        content = Path(concat_txt).read_text(encoding="utf-8")
        assert "file '" in content
        assert "segment_0000.mkv" in content
        assert "segment_0001.mkv" in content


def test_segment_completed_rendered_frames_retained():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(
            job_dir=tmpdir,
            input_path="test.avi",
            final_output_path="out.mkv",
            total_duration=600.0,
            segment_duration=300.0
        )
        mgr.mark_segment_completed(0, rendered_frames=9000)
        assert mgr.state.segments[0].status == "completed"
        assert mgr.state.segments[0].rendered_frames == 9000
