"""PySide6 PipelineWorker execution engine for RemasterGO."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from core.ai_upscaler import is_ai_binary_available, run_ai_segment_upscale
from core.cache_manager import EngineCacheManager
from core.checkpoint import CheckpointManager, SegmentInfo
from core.filtergraph import FiltergraphBuilder
from core.probe import MediaMetadata, StrategyConfig, get_binary_path, probe_media
from core.vapoursynth_builder import VapourSynthScriptBuilder, is_vspipe_available


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
        self._max_emitted_frame = 0

    def _emit_monotonic_progress(self, current_frame: int, total_frames: int, fps: float, stage: str):
        """Emit progress signal ensuring current_frame is monotonic and never regresses backwards."""
        self._max_emitted_frame = max(self._max_emitted_frame, current_frame)
        self.sig_progress.emit(
            self._max_emitted_frame,
            total_frames,
            fps,
            stage
        )

    def run(self):
        """Main execution sequence."""
        try:
            self._is_cancelled = False
            self._max_emitted_frame = 0
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
            self.sig_log.emit("Stage 1b: Checking TensorRT engine cache and downloaded models...")
            engine_info = self.cache_mgr.get_engine_info(
                self.config.model_name,
                (self.media_info.width, self.media_info.height),
                fp16=True
            )
            status_desc = "Cached Engine" if engine_info.exists else ("Downloaded Model" if engine_info.source_model_path else "Uncached (Filtergraph Fallback)")
            self.sig_log.emit(
                f"Engine key: {engine_info.cache_key} (GPU: {engine_info.gpu_name}, Status: {status_desc})"
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
                    if seg.rendered_frames <= 0 and fps > 0:
                        seg.rendered_frames = int(round(seg.duration_sec * fps))
                    rendered_total_frames += seg.rendered_frames

            if rendered_total_frames > 0:
                self.sig_log.emit(f"Resuming pipeline from frame {rendered_total_frames}/{total_frames}...")
                self._emit_monotonic_progress(rendered_total_frames, total_frames, 0.0, "Resuming Checkpoint")

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

                if seg_frames <= 0 and seg.duration_sec > 0 and fps > 0:
                    seg_frames = int(round(seg.duration_sec * fps))

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
        """Execute a single segment via VapourSynth TensorRT, Native Real-ESRGAN GPU Engine, or FFmpeg fallback."""
        engine_info = self.cache_mgr.get_engine_info(
            model_name=self.config.model_name,
            input_resolution=(self.media_info.width, self.media_info.height),
            fp16=True
        )

        # Priority 1: VapourSynth + vs-mlrt TensorRT Zero-Copy VRAM Pipeline (if .engine exists and vspipe is available)
        if engine_info.exists:
            if is_vspipe_available():
                try:
                    self.sig_log.emit(f"Running VapourSynth TensorRT Zero-Copy GPU pipeline for segment {seg.index + 1}...")
                    return self._execute_segment_vspipe(seg, total_job_frames, previously_rendered_frames, tiles=1)
                except Exception as e:
                    err_str = str(e).lower()
                    if "out of memory" in err_str or "oom" in err_str or "cuda" in err_str:
                        self.sig_log.emit("VRAM OOM detected during inference! Retrying with 2x2 spatial tiling (tiles=4)...")
                        try:
                            return self._execute_segment_vspipe(seg, total_job_frames, previously_rendered_frames, tiles=4)
                        except Exception as retry_err:
                            self.sig_log.emit(f"Tiled VapourSynth inference failed: {retry_err}. Falling back to Native GPU AI engine.")
                    else:
                        self.sig_log.emit(f"VapourSynth engine pipeline failed ({e}). Falling back to Native GPU AI engine.")
            else:
                self.sig_log.emit(
                    f"[Engine Notice] Compiled TensorRT engine '{Path(engine_info.engine_path).name}' found, but VapourSynth (vspipe) is not installed on the system. "
                    f"Falling back to Real-ESRGAN Vulkan engine."
                )

        # Priority 2: Real-ESRGAN Native GPU AI Neural Network Engine (NVIDIA RTX 5060 Ti)
        if is_ai_binary_available():
            try:
                self.sig_log.emit(
                    f"Running Real-ESRGAN Native GPU AI Neural Network ({self.cache_mgr.gpu_name}) for segment {seg.index + 1}..."
                )

                def on_ai_progress(cur_f: int, cur_fps: float, stage_label: Optional[str] = None):
                    overall_f = min(total_job_frames, previously_rendered_frames + cur_f)
                    current_stage = stage_label if stage_label else f"AI Upscaling Seg {seg.index + 1}"
                    self._emit_monotonic_progress(
                        overall_f,
                        total_job_frames,
                        cur_fps,
                        current_stage
                    )

                return run_ai_segment_upscale(
                    strategy=self.config,
                    media=self.media_info,
                    seg=seg,
                    log_cb=self.sig_log.emit,
                    progress_cb=on_ai_progress,
                    cancel_check=lambda: self._is_cancelled
                )
            except Exception as ai_err:
                self.sig_log.emit(f"GPU AI neural network error: {ai_err}. Falling back to FFmpeg filtergraph.")

        # Priority 3: Fallback to direct FFmpeg filtergraph execution with Contrast Adaptive Sharpening
        self.sig_log.emit(f"Running fallback FFmpeg filtergraph pipeline for segment {seg.index + 1}...")
        return self._execute_segment_ffmpeg(seg, total_job_frames, previously_rendered_frames)

    def _execute_segment_vspipe(
        self,
        seg: SegmentInfo,
        total_job_frames: int,
        previously_rendered_frames: int,
        tiles: int = 1
    ) -> int:
        """Execute segment using vspipe stdout piped into FFmpeg stdin."""
        if not self.media_info:
            return 0

        fps = self.media_info.fps or 30.0
        vpy_dir = Path("cache") / "temp_scripts"
        vpy_dir.mkdir(parents=True, exist_ok=True)
        vpy_file = vpy_dir / f"segment_{seg.index:04d}_{tiles}tiles.vpy"

        start_frame = int(round(seg.start_sec * fps)) if seg.start_sec > 0 else 0
        num_frames = int(round(seg.duration_sec * fps)) if seg.duration_sec > 0 else None

        engine_info = self.cache_mgr.get_engine_info(
            model_name=self.config.model_name,
            input_resolution=(self.media_info.width, self.media_info.height),
            fp16=True,
            tiles=tiles
        )
        target_model_file = engine_info.engine_path if engine_info.exists else (engine_info.source_model_path or engine_info.engine_path)

        VapourSynthScriptBuilder.write_script_file(
            target_path=str(vpy_file),
            strategy=self.config,
            media=self.media_info,
            engine_path=target_model_file,
            start_frame=start_frame,
            num_frames=num_frames,
            tiles=tiles,
            fp16=True
        )

        vspipe_bin = get_binary_path("vspipe")
        vspipe_cmd = [vspipe_bin, str(vpy_file.resolve()), "-"]

        ffmpeg_cmd = FiltergraphBuilder.build_piped_ffmpeg_cmd(
            strategy=self.config,
            media=self.media_info,
            output_file=seg.output_path,
            fps=fps,
            start_sec=seg.start_sec if self.config.segment_duration_sec > 0 and len(self.checkpoint_mgr.state.segments) > 1 else None,
            duration_sec=seg.duration_sec if self.config.segment_duration_sec > 0 and len(self.checkpoint_mgr.state.segments) > 1 else None,
            progress_pipe=True
        )

        try:
            self.vspipe_proc = subprocess.Popen(
                vspipe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=self.vspipe_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace"
            )

            if self.vspipe_proc.stdout:
                self.vspipe_proc.stdout.close()
                self.vspipe_proc.stdout = None

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
                        self._emit_monotonic_progress(
                            overall_frame,
                            total_job_frames,
                            cur_fps,
                            f"Upscaling Segment {seg.index + 1} (vspipe)"
                        )

            ret_ffmpeg = self.ffmpeg_proc.wait()
            vspipe_err = b""
            if self.vspipe_proc:
                if self.vspipe_proc.poll() is None:
                    try:
                        _, vspipe_err = self.vspipe_proc.communicate()
                    except Exception:
                        pass
                elif self.vspipe_proc.stderr:
                    try:
                        vspipe_err = self.vspipe_proc.stderr.read()
                    except Exception:
                        pass

            if ret_ffmpeg != 0 and not self._is_cancelled:
                err_text = vspipe_err.decode("utf-8", errors="replace") if isinstance(vspipe_err, bytes) else str(vspipe_err)
                raise RuntimeError(f"VapourSynth vspipe failed: {err_text[-500:] if err_text else 'Unknown error'}")

            return cur_segment_frame if cur_segment_frame > 0 else (num_frames if num_frames else int(round(seg.duration_sec * fps)))

        finally:
            if vpy_file.is_file():
                try:
                    vpy_file.unlink(missing_ok=True)
                except OSError:
                    pass

    def _execute_segment_ffmpeg(
        self,
        seg: SegmentInfo,
        total_job_frames: int,
        previously_rendered_frames: int
    ) -> int:
        """Execute a single video segment through the direct FFmpeg filtergraph pipeline."""
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
                    self._emit_monotonic_progress(
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

        fallback_frames = int(round(seg.duration_sec * (self.media_info.fps or 30.0))) if seg.duration_sec > 0 else 0
        return cur_segment_frame if cur_segment_frame > 0 else fallback_frames

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
