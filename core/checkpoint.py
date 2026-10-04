"""Segment-based checkpointing, resumption, and concatenation manager for RemasterGO."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from core.probe import get_binary_path


@dataclass
class SegmentInfo:
    index: int
    start_sec: float
    duration_sec: float
    output_filename: str
    output_path: str
    status: str = "pending"  # "pending", "processing", "completed", "failed"
    rendered_frames: int = 0
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentInfo":
        return cls(**data)


@dataclass
class CheckpointState:
    job_id: str
    input_path: str
    final_output_path: str
    total_duration: float
    segment_duration: float
    segments: List[SegmentInfo] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "input_path": self.input_path,
            "final_output_path": self.final_output_path,
            "total_duration": self.total_duration,
            "segment_duration": self.segment_duration,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointState":
        segments_data = data.pop("segments", [])
        segments = [SegmentInfo.from_dict(s) for s in segments_data]
        return cls(segments=segments, **data)


class CheckpointManager:
    """Orchestrates segment chunking, crash recovery tracking, and lossless MKV concat."""

    def __init__(
        self,
        job_dir: str,
        input_path: str,
        final_output_path: str,
        total_duration: float,
        segment_duration: float = 300.0,
        scene_cut_times: Optional[List[float]] = None,
        job_id: Optional[str] = None
    ):
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.job_dir / "job_checkpoint.json"

        # Strictly enforce .mkv output
        if not final_output_path.lower().endswith(".mkv"):
            base = os.path.splitext(final_output_path)[0]
            final_output_path = f"{base}.mkv"

        if job_id is None:
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stem = Path(input_path).stem
            job_id = f"job_{stem}_{now_str}"

        self.state = CheckpointState(
            job_id=job_id,
            input_path=str(input_path),
            final_output_path=str(final_output_path),
            total_duration=total_duration,
            segment_duration=segment_duration,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat()
        )

        # Plan or load existing segments
        if self.checkpoint_file.is_file():
            self.load()
        else:
            self._plan_segments(scene_cut_times or [])
            self.save()

    def _plan_segments(self, scene_cuts: List[float]):
        """Plan segment boundaries, aligning cuts with nearby scene transitions where possible."""
        dur = self.state.total_duration
        seg_dur = self.state.segment_duration

        if seg_dur <= 0 or dur <= seg_dur:
            # Single segment for short videos or when chunking is disabled
            fname = "segment_0000.mkv"
            self.state.segments = [
                SegmentInfo(
                    index=0,
                    start_sec=0.0,
                    duration_sec=dur,
                    output_filename=fname,
                    output_path=str(self.job_dir / fname)
                )
            ]
            return

        segments: List[SegmentInfo] = []
        cur_start = 0.0
        idx = 0

        while cur_start < dur:
            target_end = min(cur_start + seg_dur, dur)

            # Snap to closest scene transition within +/- 15 seconds of target_end if not last segment
            if target_end < dur and scene_cuts:
                candidates = [
                    sc for sc in scene_cuts
                    if abs(sc - target_end) <= 15.0 and sc > (cur_start + 5.0)
                ]
                if candidates:
                    target_end = min(candidates, key=lambda sc: abs(sc - target_end))

            chunk_len = target_end - cur_start
            fname = f"segment_{idx:04d}.mkv"
            segments.append(SegmentInfo(
                index=idx,
                start_sec=cur_start,
                duration_sec=chunk_len,
                output_filename=fname,
                output_path=str(self.job_dir / fname)
            ))
            cur_start = target_end
            idx += 1

        self.state.segments = segments

    def save(self):
        """Save current checkpoint state to job_checkpoint.json."""
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def load(self):
        """Load state from job_checkpoint.json and validate existing segment files."""
        with open(self.checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.state = CheckpointState.from_dict(data)
        self.validate_existing_segments()

    def validate_existing_segments(self):
        """Verify previously rendered segment files on disk and populate frame counts."""
        ffprobe_bin = get_binary_path("ffprobe")
        for seg in self.state.segments:
            seg_path = Path(seg.output_path)
            if seg_path.is_file() and seg_path.stat().st_size > 1024:
                # Probe segment to confirm valid MKV container and video stream
                try:
                    cmd = [
                        ffprobe_bin,
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=codec_name,nb_frames",
                        "-of", "csv=p=0",
                        str(seg_path)
                    ]
                    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    out_text = res.stdout.strip()
                    if out_text:
                        seg.status = "completed"
                        seg.size_bytes = seg_path.stat().st_size
                        parts = out_text.split(",")
                        if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) > 0:
                            seg.rendered_frames = int(parts[1])
                    else:
                        seg.status = "pending"
                except Exception:
                    seg.status = "pending"
            else:
                seg.status = "pending"
                seg.rendered_frames = 0
                seg.size_bytes = 0

    def get_pending_segments(self) -> List[SegmentInfo]:
        """Return all segments that still require processing."""
        return [s for s in self.state.segments if s.status != "completed"]

    def mark_segment_processing(self, index: int):
        """Mark a segment as active."""
        if 0 <= index < len(self.state.segments):
            self.state.segments[index].status = "processing"
            self.save()

    def mark_segment_completed(self, index: int, rendered_frames: int = 0):
        """Mark a segment as successfully rendered and verified."""
        if 0 <= index < len(self.state.segments):
            seg = self.state.segments[index]
            seg.status = "completed"
            seg.rendered_frames = rendered_frames
            seg_file = Path(seg.output_path)
            if seg_file.is_file():
                seg.size_bytes = seg_file.stat().st_size
            self.save()

    def mark_segment_failed(self, index: int):
        """Mark a segment as failed."""
        if 0 <= index < len(self.state.segments):
            self.state.segments[index].status = "failed"
            self.save()

    def generate_concat_file(self) -> str:
        """Write FFmpeg concat demuxer segment list."""
        concat_file = self.job_dir / "segment_list.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for seg in sorted(self.state.segments, key=lambda s: s.index):
                # Use absolute or relative path with forward slashes for FFmpeg concat safe format
                norm_path = Path(seg.output_path).resolve().as_posix()
                f.write(f"file '{norm_path}'\n")
        return str(concat_file)

    def concatenate_segments(self) -> str:
        """Losslessly merge all rendered segments into the target .mkv container."""
        ffmpeg_bin = get_binary_path("ffmpeg")
        concat_txt = self.generate_concat_file()
        output_mkv = self.state.final_output_path
        Path(output_mkv).parent.mkdir(parents=True, exist_ok=True)

        # If only 1 segment, copy directly to destination if paths differ
        if len(self.state.segments) == 1:
            single = self.state.segments[0].output_path
            if Path(single).resolve() != Path(output_mkv).resolve():
                shutil.copy2(single, output_mkv)
            return output_mkv

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            "-f", "matroska",
            output_mkv
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_mkv

    def cleanup_segments(self):
        """Delete intermediate segment MKV files upon successful job finalization."""
        for seg in self.state.segments:
            p = Path(seg.output_path)
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
        concat_txt = self.job_dir / "segment_list.txt"
        if concat_txt.is_file():
            try:
                concat_txt.unlink()
            except OSError:
                pass
