"""Native GPU AI neural network upscaling engine using Real-ESRGAN Vulkan for RemasterGO."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from core.checkpoint import SegmentInfo
from core.probe import MediaMetadata, StrategyConfig, get_binary_path


def get_ai_binary_path() -> Optional[str]:
    """Resolve path to realesrgan-ncnn-vulkan binary."""
    proj_root = Path(__file__).resolve().parent.parent
    local_bin = proj_root / "bin" / "realesrgan" / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")
    if local_bin.is_file():
        return str(local_bin)

    sys_bin = shutil.which("realesrgan-ncnn-vulkan")
    if sys_bin:
        return sys_bin

    return None


def is_ai_binary_available() -> bool:
    """Check if the native GPU AI engine binary is installed and executable."""
    return get_ai_binary_path() is not None


def resolve_model_name(selected_name: str) -> str:
    """Map UI/CLI model selection to the closest embedded Real-ESRGAN neural network model."""
    clean = selected_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if "anime" in clean and ("6b" in clean or "x4plus" in clean):
        return "realesrgan-x4plus-anime"
    elif "anime" in clean or "span" in clean or "compact" in clean:
        return "realesr-animevideov3"
    elif "x4" in clean or "realesrgan" in clean or "balanced" in clean:
        return "realesrgan-x4plus"
    return "realesrgan-x4plus"


def calculate_scale_factor(in_w: int, in_h: int, target_w: int, target_h: int) -> int:
    """Determine the optimal AI super-resolution integer scale factor (2, 3, or 4)."""
    if in_h <= 0 or target_h <= 0:
        return 4
    ratio = target_h / in_h
    if ratio <= 2.2:
        return 2
    elif ratio <= 3.2:
        return 3
    return 4


def run_ai_segment_upscale(
    strategy: StrategyConfig,
    media: MediaMetadata,
    seg: SegmentInfo,
    log_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> int:
    """Perform real GPU AI super-resolution on a video segment:
    1. Extract video frames with FFmpeg.
    2. Upscale frames with Real-ESRGAN GPU neural network.
    3. Encode upscaled frames into Matroska (.mkv) container with audio and micro-texture tuning.
    4. Remove temporary image frames.
    """
    ai_bin = get_ai_binary_path()
    if not ai_bin:
        raise FileNotFoundError("Real-ESRGAN GPU AI binary not found.")

    ffmpeg_bin = get_binary_path("ffmpeg")
    fps = media.fps or 30.0
    scale = calculate_scale_factor(media.width, media.height, strategy.target_width, strategy.target_height)
    model_name = resolve_model_name(strategy.model_name)

    # Ensure model supports the chosen scale
    if model_name in ("realesrgan-x4plus", "realesrgan-x4plus-anime"):
        scale = 4

    job_cache = Path("cache") / Path(strategy.output_path).stem
    job_cache.mkdir(parents=True, exist_ok=True)
    frames_in = job_cache / f"seg_{seg.index:04d}_in"
    frames_out = job_cache / f"seg_{seg.index:04d}_out"
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Extract frames from video
        if log_cb:
            log_cb(f"[AI Engine] Extracting frames for Segment {seg.index + 1}...")

        extract_cmd = [
            ffmpeg_bin, "-y",
            "-v", "error"
        ]
        if seg.start_sec > 0:
            extract_cmd.extend(["-ss", f"{seg.start_sec:.4f}"])
        extract_cmd.extend(["-i", strategy.input_path])
        if seg.duration_sec > 0:
            extract_cmd.extend(["-t", f"{seg.duration_sec:.4f}"])

        extract_cmd.extend([
            "-q:v", "2",
            str(frames_in / "frame_%08d.png")
        ])

        subprocess.run(extract_cmd, check=True)
        extracted_frames = list(frames_in.glob("*.png"))
        total_frames_in_seg = len(extracted_frames)

        if total_frames_in_seg == 0:
            return 0

        if log_cb:
            log_cb(f"[AI Engine] Running Real-ESRGAN {model_name} (Scale {scale}x) on RTX GPU ({total_frames_in_seg} frames)...")

        # Step 2: GPU AI Upscaling
        ai_models_dir = (Path(ai_bin).parent / "models").resolve()
        ai_cmd = [
            ai_bin,
            "-i", str(frames_in.resolve()),
            "-o", str(frames_out.resolve()),
            "-n", model_name,
            "-s", str(scale),
            "-m", str(ai_models_dir),
            "-g", "0",
            "-f", "png"
        ]

        ai_proc = subprocess.Popen(
            ai_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )

        start_time = time.time()
        last_reported_count = -1
        cur_f = 0
        cur_fps = 0.0

        if ai_proc.stdout:
            while ai_proc.poll() is None:
                if cancel_check and cancel_check():
                    ai_proc.terminate()
                    return 0
                line = ai_proc.stdout.readline()
                cur_f = sum(1 for entry in os.scandir(frames_out) if entry.name.endswith(".png"))
                elapsed = time.time() - start_time
                cur_fps = (cur_f / elapsed) if elapsed > 0.5 else 0.0
                if cur_f != last_reported_count:
                    last_reported_count = cur_f
                    if progress_cb:
                        progress_cb(cur_f, cur_fps, f"AI Upscaling Seg {seg.index + 1}")

        ret_ai = ai_proc.wait()
        if ret_ai != 0:
            raise RuntimeError(f"Real-ESRGAN GPU inference failed with exit code {ret_ai}")

        cur_f = sum(1 for entry in os.scandir(frames_out) if entry.name.endswith(".png"))
        elapsed = time.time() - start_time
        cur_fps = (cur_f / elapsed) if elapsed > 0.5 else 0.0
        if progress_cb:
            progress_cb(cur_f, cur_fps, f"AI Upscaling Seg {seg.index + 1}")

        if cancel_check and cancel_check():
            return 0

        # Step 3: Post-processing and NVENC Encoding
        if log_cb:
            log_cb(f"[AI Engine] Encoding upscaled frames for Segment {seg.index + 1} with micro-texture blend & NVENC...")

        tw = strategy.target_width
        th = strategy.target_height
        grain = max(0, min(30, strategy.film_grain_intensity))
        ai_weight = max(0.0, min(1.0, strategy.blend_ai_ratio))
        orig_weight = max(0.0, min(1.0, 1.0 - ai_weight))

        # Build filtergraph: Blend AI video stream with spline-scaled original video for texture preservation
        # Ensure PTS alignment, framerate synchronization, and shortest=1 to prevent stream deadlocks
        filter_steps = [
            f"[0:v]setpts=PTS-STARTPTS,scale={tw}:{th}:flags=spline[ai_scaled]",
            f"[1:v]setpts=PTS-STARTPTS,fps={fps:.4f},scale={tw}:{th}:flags=spline[orig_scaled]",
            f"[ai_scaled][orig_scaled]blend=all_expr='A*{ai_weight:.2f}+B*{orig_weight:.2f}':shortest=1[blended]"
        ]
        if grain > 0:
            filter_steps.append(f"[blended]noise=alls={grain}:allf=t+u[final_v]")
            final_label = "[final_v]"
        else:
            final_label = "[blended]"

        encode_cmd = [
            ffmpeg_bin, "-y",
            "-progress", "pipe:1",
            "-nostats",
            "-framerate", f"{fps:.4f}",
            "-i", str(frames_out / "frame_%08d.png")
        ]

        # Audio and original video stream from source (Input 1)
        if seg.start_sec > 0:
            encode_cmd.extend(["-ss", f"{seg.start_sec:.4f}"])
        encode_cmd.extend(["-i", strategy.input_path])
        if seg.duration_sec > 0:
            encode_cmd.extend(["-t", f"{seg.duration_sec:.4f}"])

        encode_cmd.extend([
            "-filter_complex", "; ".join(filter_steps),
            "-map", final_label,
            "-map", "1:a?",
            "-fps_mode", "passthrough"
        ])

        encoder = strategy.encoder.lower()
        if "nvenc" in encoder:
            encode_cmd.extend([
                "-c:v", encoder,
                "-preset", strategy.preset,
                "-cq", str(strategy.cq),
                "-pix_fmt", "yuv420p"
            ])
        else:
            encode_cmd.extend([
                "-c:v", "libx264",
                "-crf", str(strategy.cq),
                "-pix_fmt", "yuv420p"
            ])

        if media.has_audio:
            encode_cmd.extend([
                "-af", "aresample=async=1000",
                "-c:a", "aac",
                "-b:a", "256k"
            ])
        else:
            encode_cmd.append("-an")

        encode_cmd.extend([
            "-f", "matroska",
            seg.output_path
        ])

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace"
        )

        frame_pattern = re.compile(r"frame=(\d+)")
        fps_pattern = re.compile(r"fps=([\d\.]+)")
        cur_enc_frame = 0
        cur_enc_fps = 0.0

        if encode_proc.stdout:
            for line in encode_proc.stdout:
                if cancel_check and cancel_check():
                    encode_proc.terminate()
                    return 0
                line = line.strip()
                f_m = frame_pattern.search(line)
                fps_m = fps_pattern.search(line)
                if f_m:
                    cur_enc_frame = int(f_m.group(1))
                if fps_m:
                    cur_enc_fps = float(fps_m.group(1))
                if line.startswith("progress=continue") or line.startswith("progress=end"):
                    if progress_cb:
                        progress_cb(cur_enc_frame, cur_enc_fps, f"Encoding Seg {seg.index + 1}")

        ret_enc = encode_proc.wait()
        if ret_enc != 0 and not (cancel_check and cancel_check()):
            err_msg = encode_proc.stderr.read() if encode_proc.stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg encoding failed with exit code {ret_enc}: {err_msg[-500:] if err_msg else ''}")

        return total_frames_in_seg

    finally:
        # Step 4: Cleanup temporary frame directories
        shutil.rmtree(frames_in, ignore_errors=True)
        shutil.rmtree(frames_out, ignore_errors=True)
