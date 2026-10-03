"""PySide6 PipelineWorker execution engine for RemasterGO."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from core.cache_manager import EngineCacheManager
from core.checkpoint import CheckpointManager, SegmentInfo
from core.filtergraph import FiltergraphBuilder
from core.probe import MediaMetadata, StrategyConfig, probe_media


class PipelineWorker(QThread):
    """Worker thread orchestrating the video restoration and upscaling pipeline."""

    sig_progress = Signal(int, int, float, str)  # current_frame, total_frames, fps, stage_name
    sig_status_change = Signal(str)
    sig_log = Signal(str)
    sig_finished = Signal(str)
    sig_error = Signal(str)

    def __init__(self, config: StrategyConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._is_cancelled = False
        self._is_paused = False
        self.ffmpeg_proc: Optional[subprocess.Popen] = None
        self.vspipe_proc: Optional[subprocess.Popen] = None

        # Enforce strict .mkv output
        if not self.config.output_path.lower().endswith(".mkv"):
            base = os.path.splitext(self.config.output_path)[0]
            self.config.output_path = f"{base}.mkv"

        self.media_info: Optional[MediaMetadata] = None
        self.cache_mgr = EngineCacheManager()
        self.checkpoint_mgr: Optional[CheckpointManager] = None

    def run(self):
        """Main execution sequence."""
        try:
            self._is_cancelled = False
            self.sig_status_change.emit("Analyzing media and PTS timecodes...")
            self.sig_log.emit("Stage 1: Probing media metadata and PTS timestamps...")

            # Stage 1: Probe Media
            self.media_info = probe_media(self.config.input_path)
            total_frames = self.media_info.total_frames
            fps = self.media_info.fps or 30.0

            self.sig_log.emit(
                f"Source: {self.media_info.width}x{self.media_info.height} @ {fps:.2f} fps, "
                f"{total_frames} frames, duration {self.media_info.duration:.2f}s"
            )

            if self.media_info.video and self.media_info.video.is_interlaced:
                self.sig_log.emit(f"Detected interlaced scan ({self.media_info.video.field_order}). Deinterlacing will be applied.")
                self.config.deinterlace = True

            # Stage 1b: Engine Cache Check
            self.sig_log.emit("Stage 1b: Checking TensorRT engine cache...")
            engine_info = self.cache_mgr.get_engine_info(
                self.config.model_name,
                (self.media_info.width, self.media_info.height),
                fp16=True
            )
            self.sig_log.emit(
                f"Engine key: {engine_info.cache_key} (GPU: {engine_info.gpu_name}, Status: {'Cached' if engine_info.exists else 'Initialized'})"
            )

            # Stage 2-4: Segment Planning & Execution
            out_stem = Path(self.config.output_path).stem
            job_cache_dir = Path("cache") / out_stem
            scene_cut_times = [sc.pts_time for sc in self.config.scene_cuts]

            self.checkpoint_mgr = CheckpointManager(
                job_dir=str(job_cache_dir),
                input_path=self.config.input_path,
                final_output_path=self.config.output_path,
                total_duration=self.media_info.duration,
                segment_duration=self.config.segment_duration_sec,
                scene_cut_times=scene_cut_times
            )

            pending_segments = self.checkpoint_mgr.get_pending_segments()
            total_segments = len(self.checkpoint_mgr.state.segments)

            self.sig_log.emit(f"Segment checkpoint manager ready: {len(pending_segments)}/{total_segments} segments pending.")

            rendered_total_frames = 0
            # Calculate frames already rendered in completed segments
            for seg in self.checkpoint_mgr.state.segments:
                if seg.status == "completed":
                    rendered_total_frames += seg.rendered_frames

            # Execute pending segments
            for seg in pending_segments:
                if self._is_cancelled:
                    self.sig_log.emit("Pipeline cancelled by user.")
                    return

                while self._is_paused:
                    time.sleep(0.5)
                    if self._is_cancelled:
                        return

                self.sig_status_change.emit(f"Processing segment {seg.index + 1}/{total_segments}...")
                self.sig_log.emit(
                    f"Starting Segment {seg.index + 1}/{total_segments} "
                    f"[{seg.start_sec:.1f}s - {seg.start_sec + seg.duration_sec:.1f}s] -> {seg.output_filename}"
                )

                self.checkpoint_mgr.mark_segment_processing(seg.index)
                seg_frames = self._execute_segment(seg, total_frames, rendered_total_frames)

                if self._is_cancelled:
                    return

                rendered_total_frames += seg_frames
                self.checkpoint_mgr.mark_segment_completed(seg.index, rendered_frames=seg_frames)
                self.sig_log.emit(f"Segment {seg.index + 1} completed.")

            # Stage 4b: Lossless Concatenation to final MKV
            if not self._is_cancelled:
                self.sig_status_change.emit("Concatenating segments into final MKV...")
                self.sig_log.emit("Finalizing: Losslessly merging segments into final output...")
                final_mkv = self.checkpoint_mgr.concatenate_segments()
                self.sig_progress.emit(total_frames, total_frames, 0.0, "Completed")
                self.sig_status_change.emit("Job completed successfully.")
                self.sig_log.emit(f"Render complete! Output saved to: {final_mkv}")
                self.sig_finished.emit(final_mkv)

        except Exception as e:
            self.terminate_processes()
            self.sig_error.emit(str(e))
            self.sig_log.emit(f"Error during execution: {e}")

    def _execute_segment(
        self,
        seg: SegmentInfo,
        total_job_frames: int,
        previously_rendered_frames: int
    ) -> int:
        """Execute a single video segment through the FFmpeg filtergraph pipeline."""
        if not self.media_info:
            return 0

        cmd = FiltergraphBuilder.build_ffmpeg_cmd(
            strategy=self.config,
            media=self.media_info,
            output_file=seg.output_path,
            start_sec=seg.start_sec if self.config.segment_duration_sec > 0 and len(self.checkpoint_mgr.state.segments) > 1 else None,
            duration_sec=seg.duration_sec if self.config.segment_duration_sec > 0 and len(self.checkpoint_mgr.state.segments) > 1 else None,
            progress_pipe=True
        )

        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )

        frame_pattern = re.compile(r"frame=(\d+)")
        fps_pattern = re.compile(r"fps=([\d\.]+)")

        cur_segment_frame = 0
        cur_fps = 0.0

        if self.ffmpeg_proc.stdout:
            for line in self.ffmpeg_proc.stdout:
                if self._is_cancelled:
                    self.terminate_processes()
                    break

                line = line.strip()
                frame_match = frame_pattern.search(line)
                fps_match = fps_pattern.search(line)

                if frame_match:
                    cur_segment_frame = int(frame_match.group(1))
                if fps_match:
                    cur_fps = float(fps_match.group(1))

                if line.startswith("progress=continue") or line.startswith("progress=end"):
                    overall_frame = min(total_job_frames, previously_rendered_frames + cur_segment_frame)
                    self.sig_progress.emit(
                        overall_frame,
                        total_job_frames,
                        cur_fps,
                        f"Upscaling Segment {seg.index + 1}"
                    )

        ret = self.ffmpeg_proc.wait()
        if ret != 0 and not self._is_cancelled:
            stderr_out = ""
            if self.ffmpeg_proc.stderr:
                stderr_out = self.ffmpeg_proc.stderr.read()
            raise RuntimeError(f"FFmpeg failed with exit code {ret}: {stderr_out[-500:] if stderr_out else 'Unknown error'}")

        return cur_segment_frame

    def pause(self):
        """Pause pipeline worker."""
        self._is_paused = True
        self.sig_status_change.emit("Paused")
        self.sig_log.emit("Pipeline paused.")

    def resume(self):
        """Resume pipeline worker."""
        self._is_paused = False
        self.sig_status_change.emit("Resuming...")
        self.sig_log.emit("Pipeline resumed.")

    def terminate_processes(self):
        """Safely terminate child subprocesses."""
        if self.ffmpeg_proc and self.ffmpeg_proc.poll() is None:
            try:
                self.ffmpeg_proc.terminate()
            except Exception:
                pass
        if self.vspipe_proc and self.vspipe_proc.poll() is None:
            try:
                self.vspipe_proc.terminate()
            except Exception:
                pass

    def cancel(self):
        """Request non-blocking cancellation."""
        self._is_cancelled = True
        self.terminate_processes()
        self.sig_status_change.emit("Cancelled")
