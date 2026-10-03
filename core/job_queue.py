"""Job queue controller and queue lifecycle orchestration for RemasterGO."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from core.pipeline import PipelineWorker
from core.probe import MediaMetadata, StrategyConfig, probe_media


@dataclass
class JobItem:
    id: str
    input_path: str
    output_path: str
    strategy: StrategyConfig
    status: str = "Pending"  # "Pending", "Processing", "Paused", "Completed", "Failed", "Cancelled"
    progress_percent: float = 0.0
    current_fps: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    stage: str = "Ready"
    error_message: Optional[str] = None
    media_info: Optional[MediaMetadata] = None


class JobQueueController(QObject):
    """Manages the queue of video restoration and upscaling jobs."""

    sig_queue_updated = Signal()
    sig_job_started = Signal(str)  # job_id
    sig_job_progress = Signal(str, int, int, float, str)  # job_id, cur_frame, total_frames, fps, stage
    sig_job_completed = Signal(str, str)  # job_id, output_path
    sig_job_failed = Signal(str, str)  # job_id, error
    sig_job_status_changed = Signal(str, str)  # job_id, status
    sig_queue_finished = Signal()
    sig_log = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: List[JobItem] = []
        self.active_job: Optional[JobItem] = None
        self.current_worker: Optional[PipelineWorker] = None
        self._is_running = False

    def add_job(
        self,
        input_path: str,
        strategy: StrategyConfig,
        media: Optional[MediaMetadata] = None
    ) -> JobItem:
        """Add a video job to the queue."""
        # Ensure output ends in .mkv
        if not strategy.output_path.lower().endswith(".mkv"):
            base = os.path.splitext(strategy.output_path)[0]
            strategy.output_path = f"{base}.mkv"

        stem = Path(input_path).stem
        now = datetime.now(timezone.utc).strftime("%H%M%S_%f")[:10]
        job_id = f"job_{stem}_{now}"

        total_frames = media.total_frames if media else 0

        job = JobItem(
            id=job_id,
            input_path=input_path,
            output_path=strategy.output_path,
            strategy=strategy,
            total_frames=total_frames,
            media_info=media
        )
        self.jobs.append(job)
        self.sig_queue_updated.emit()
        self.sig_log.emit(f"Added job to queue: {Path(input_path).name} -> {Path(strategy.output_path).name}")
        return job

    def remove_job(self, job_id: str):
        """Remove a job from the queue if not currently active."""
        if self.active_job and self.active_job.id == job_id:
            self.cancel_active_job()

        self.jobs = [j for j in self.jobs if j.id != job_id]
        self.sig_queue_updated.emit()

    def clear_queue(self):
        """Clear all non-processing jobs."""
        if self._is_running:
            self.cancel_active_job()
        self.jobs = []
        self.sig_queue_updated.emit()

    def clear_completed(self):
        """Remove finished/failed/cancelled jobs."""
        self.jobs = [j for j in self.jobs if j.status in ("Pending", "Processing", "Paused")]
        self.sig_queue_updated.emit()

    def start_queue(self):
        """Start executing pending jobs in the queue."""
        if self._is_running:
            return
        self._is_running = True
        self._process_next_job()

    def _process_next_job(self):
        """Pick the next pending job and start the worker thread."""
        pending_jobs = [j for j in self.jobs if j.status == "Pending"]
        if not pending_jobs:
            self._is_running = False
            self.active_job = None
            self.sig_queue_finished.emit()
            self.sig_log.emit("All jobs in queue have finished.")
            return

        job = pending_jobs[0]
        self.active_job = job
        job.status = "Processing"
        job.stage = "Starting"
        self.sig_job_status_changed.emit(job.id, "Processing")
        self.sig_job_started.emit(job.id)
        self.sig_queue_updated.emit()

        self.current_worker = PipelineWorker(config=job.strategy)
        self.current_worker.sig_progress.connect(self._on_worker_progress)
        self.current_worker.sig_status_change.connect(self._on_worker_status)
        self.current_worker.sig_log.connect(self.sig_log.emit)
        self.current_worker.sig_finished.connect(self._on_worker_finished)
        self.current_worker.sig_error.connect(self._on_worker_error)
        self.current_worker.start()

    def _on_worker_progress(self, cur_frame: int, total_frames: int, fps: float, stage: str):
        if self.active_job:
            self.active_job.current_frame = cur_frame
            self.active_job.total_frames = total_frames
            self.active_job.current_fps = fps
            self.active_job.stage = stage
            pct = (cur_frame / total_frames * 100.0) if total_frames > 0 else 0.0
            self.active_job.progress_percent = min(100.0, pct)
            self.sig_job_progress.emit(self.active_job.id, cur_frame, total_frames, fps, stage)

    def _on_worker_status(self, status_msg: str):
        if self.active_job:
            self.active_job.stage = status_msg
            self.sig_job_status_changed.emit(self.active_job.id, status_msg)

    def _on_worker_finished(self, out_path: str):
        if self.active_job:
            self.active_job.status = "Completed"
            self.active_job.progress_percent = 100.0
            self.active_job.stage = "Finished"
            self.sig_job_completed.emit(self.active_job.id, out_path)
            self.sig_queue_updated.emit()

        self.current_worker = None
        self.active_job = None

        if self._is_running:
            self._process_next_job()

    def _on_worker_error(self, err_msg: str):
        if self.active_job:
            self.active_job.status = "Failed"
            self.active_job.error_message = err_msg
            self.active_job.stage = "Error"
            self.sig_job_failed.emit(self.active_job.id, err_msg)
            self.sig_queue_updated.emit()

        self.current_worker = None
        self.active_job = None

        if self._is_running:
            self._process_next_job()

    def pause_queue(self):
        """Pause the active job."""
        if self.current_worker and self.active_job:
            self.current_worker.pause()
            self.active_job.status = "Paused"
            self.sig_job_status_changed.emit(self.active_job.id, "Paused")
            self.sig_queue_updated.emit()

    def resume_queue(self):
        """Resume active or paused queue."""
        if self.current_worker and self.active_job and self.active_job.status == "Paused":
            self.current_worker.resume()
            self.active_job.status = "Processing"
            self.sig_job_status_changed.emit(self.active_job.id, "Processing")
            self.sig_queue_updated.emit()
        elif not self._is_running:
            self.start_queue()

    def cancel_active_job(self):
        """Cancel currently executing job."""
        if self.current_worker:
            self.current_worker.cancel()
        if self.active_job:
            self.active_job.status = "Cancelled"
            self.active_job.stage = "Cancelled"
            self.sig_job_status_changed.emit(self.active_job.id, "Cancelled")
            self.sig_queue_updated.emit()

        self.current_worker = None
        self.active_job = None

    def cancel_all(self):
        """Cancel active job and set all pending jobs to Cancelled."""
        self._is_running = False
        self.cancel_active_job()
        for j in self.jobs:
            if j.status == "Pending":
                j.status = "Cancelled"
        self.sig_queue_updated.emit()
