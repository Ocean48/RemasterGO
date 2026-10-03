"""Media analysis, PTS tracking, and scene-cut detection module for RemasterGO."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple


def get_binary_path(name: str) -> str:
    """Resolve executable path, preferring workspace root binaries over system PATH."""
    current_dir = Path(__file__).resolve().parent.parent
    local_binary = current_dir / f"{name}.exe" if os.name == "nt" else current_dir / name
    if local_binary.is_file():
        return str(local_binary)

    system_binary = shutil.which(name)
    if system_binary:
        return system_binary

    return name


@dataclass
class AudioStreamInfo:
    index: int
    codec_name: str
    sample_rate: int
    channels: int
    channel_layout: str
    bit_rate: Optional[int] = None
    duration: float = 0.0


@dataclass
class VideoStreamInfo:
    index: int
    codec_name: str
    width: int
    height: int
    pix_fmt: str
    r_frame_rate: str
    fps: float
    nb_frames: int
    duration: float
    sar: str = "1:1"
    dar: str = "16:9"
    field_order: str = "progressive"
    is_interlaced: bool = False
    bit_rate: Optional[int] = None


@dataclass
class MediaMetadata:
    filepath: str
    format_name: str
    duration: float
    size_bytes: int
    video: Optional[VideoStreamInfo] = None
    audio: List[AudioStreamInfo] = field(default_factory=list)

    @property
    def has_audio(self) -> bool:
        return len(self.audio) > 0

    @property
    def total_frames(self) -> int:
        if self.video:
            return self.video.nb_frames
        return 0

    @property
    def fps(self) -> float:
        if self.video:
            return self.video.fps
        return 0.0

    @property
    def width(self) -> int:
        return self.video.width if self.video else 0

    @property
    def height(self) -> int:
        return self.video.height if self.video else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneBoundary:
    frame_index: int
    pts_time: float
    score: float


@dataclass
class StrategyConfig:
    input_path: str
    output_path: str
    target_width: int = 1920
    target_height: int = 1080
    model_name: str = "Real-ESRGAN_x4"
    deinterlace: bool = False
    field_order: str = "progressive"  # "progressive", "tff", "bff"
    denoise: bool = True
    denoise_strength: float = 1.2
    blend_ai_ratio: float = 0.8
    film_grain_intensity: int = 6
    encoder: str = "hevc_nvenc"
    cq: int = 18
    preset: str = "p6"
    segment_duration_sec: float = 300.0
    scene_cuts: List[SceneBoundary] = field(default_factory=list)

    def __post_init__(self):
        # Enforce .mkv output container
        if not self.output_path.lower().endswith(".mkv"):
            base = os.path.splitext(self.output_path)[0]
            self.output_path = f"{base}.mkv"


def _parse_fraction(rate_str: str) -> float:
    """Parse fraction strings like '30000/1001' or '24' to float."""
    if not rate_str or rate_str == "0/0":
        return 0.0
    parts = rate_str.split("/")
    if len(parts) == 2:
        try:
            denom = float(parts[1])
            if denom == 0:
                return 0.0
            return float(parts[0]) / denom
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


def probe_media(filepath: str) -> MediaMetadata:
    """Probe media file using ffprobe and return detailed MediaMetadata."""
    ffprobe_bin = get_binary_path("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(filepath)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace"
    )

    data = json.loads(result.stdout)
    format_info = data.get("format", {})
    streams = data.get("streams", [])

    duration_str = format_info.get("duration", "0.0")
    total_duration = float(duration_str) if duration_str else 0.0
    size_bytes = int(format_info.get("size", 0))
    format_name = format_info.get("format_name", "unknown")

    video_info: Optional[VideoStreamInfo] = None
    audio_list: List[AudioStreamInfo] = []

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and video_info is None:
            r_rate = s.get("r_frame_rate", "0/0")
            fps = _parse_fraction(r_rate)
            if fps == 0.0:
                fps = _parse_fraction(s.get("avg_frame_rate", "0/0"))

            nb_frames_str = s.get("nb_frames")
            if nb_frames_str and nb_frames_str.isdigit():
                nb_frames = int(nb_frames_str)
            elif total_duration > 0 and fps > 0:
                nb_frames = int(round(total_duration * fps))
            else:
                nb_frames = 0

            field_order = s.get("field_order", "progressive")
            is_interlaced = field_order in ("tt", "bb", "tb", "bt") or s.get("is_avc") == "false" and "interlaced" in field_order.lower()

            bit_rate = int(s.get("bit_rate")) if s.get("bit_rate", "").isdigit() else None
            stream_dur = float(s.get("duration", total_duration))

            video_info = VideoStreamInfo(
                index=int(s.get("index", 0)),
                codec_name=s.get("codec_name", "unknown"),
                width=int(s.get("width", 0)),
                height=int(s.get("height", 0)),
                pix_fmt=s.get("pix_fmt", "yuv420p"),
                r_frame_rate=r_rate,
                fps=fps,
                nb_frames=nb_frames,
                duration=stream_dur,
                sar=s.get("sample_aspect_ratio", "1:1"),
                dar=s.get("display_aspect_ratio", "16:9"),
                field_order=field_order,
                is_interlaced=is_interlaced,
                bit_rate=bit_rate
            )
        elif codec_type == "audio":
            sample_rate = int(s.get("sample_rate", 44100))
            channels = int(s.get("channels", 2))
            audio_bitrate = int(s.get("bit_rate")) if s.get("bit_rate", "").isdigit() else None
            a_dur = float(s.get("duration", total_duration))

            audio_list.append(AudioStreamInfo(
                index=int(s.get("index", 1)),
                codec_name=s.get("codec_name", "unknown"),
                sample_rate=sample_rate,
                channels=channels,
                channel_layout=s.get("channel_layout", "stereo"),
                bit_rate=audio_bitrate,
                duration=a_dur
            ))

    return MediaMetadata(
        filepath=str(filepath),
        format_name=format_name,
        duration=total_duration,
        size_bytes=size_bytes,
        video=video_info,
        audio=audio_list
    )


def detect_scene_cuts(
    filepath: str,
    threshold: float = 0.3,
    max_duration: Optional[float] = None
) -> List[SceneBoundary]:
    """Detect scene cut boundaries using ffmpeg scene detection filter."""
    ffmpeg_bin = get_binary_path("ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-v", "info",
    ]
    if max_duration:
        cmd.extend(["-t", str(max_duration)])

    cmd.extend([
        "-i", str(filepath),
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-"
    ])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    boundaries: List[SceneBoundary] = []
    # Parse showinfo lines: n:  15 pts: 15015 pts_time:0.500500 ...
    pattern = re.compile(r"n:\s*(\d+)\s+pts:\s*\d+\s+pts_time:([\d\.]+)")

    for line in proc.stderr.splitlines():
        if "Parsed_showinfo_1" in line or "showinfo" in line:
            match = pattern.search(line)
            if match:
                frame_idx = int(match.group(1))
                pts_t = float(match.group(2))
                boundaries.append(SceneBoundary(
                    frame_index=frame_idx,
                    pts_time=pts_t,
                    score=threshold
                ))

    return boundaries


def create_strategy(
    media: MediaMetadata,
    output_dir: Optional[str] = None,
    target_width: int = 1920,
    target_height: int = 1080,
    model_name: str = "Real-ESRGAN_x4",
    encoder: str = "hevc_nvenc",
    blend_ai_ratio: float = 0.8,
    film_grain: int = 6,
    segment_duration: float = 300.0,
    detect_scenes: bool = False
) -> StrategyConfig:
    """Generate StrategyConfig based on probed media metadata and target preferences."""
    input_stem = Path(media.filepath).stem
    out_dir = Path(output_dir) if output_dir else Path(media.filepath).parent
    out_mkv = str(out_dir / f"{input_stem}_upscaled_{target_height}p.mkv")

    deinterlace = False
    field_order = "progressive"
    if media.video:
        deinterlace = media.video.is_interlaced
        field_order = media.video.field_order

    scene_cuts: List[SceneBoundary] = []
    if detect_scenes and media.duration < 600:
        scene_cuts = detect_scene_cuts(media.filepath, threshold=0.3)

    return StrategyConfig(
        input_path=media.filepath,
        output_path=out_mkv,
        target_width=target_width,
        target_height=target_height,
        model_name=model_name,
        deinterlace=deinterlace,
        field_order=field_order,
        denoise=True,
        denoise_strength=1.2,
        blend_ai_ratio=blend_ai_ratio,
        film_grain_intensity=film_grain,
        encoder=encoder,
        segment_duration_sec=segment_duration,
        scene_cuts=scene_cuts
    )
