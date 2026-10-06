"""Dynamic filtergraph and FFmpeg encoding command builder for RemasterGO."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from core.probe import MediaMetadata, StrategyConfig, get_binary_path


@dataclass
class FiltergraphResult:
    filter_complex: str
    output_video_label: str
    output_audio_label: Optional[str] = None


class FiltergraphBuilder:
    """Builds complex video/audio filtergraphs and FFmpeg CLI execution arguments."""

    @staticmethod
    def build_filtergraph(
        strategy: StrategyConfig,
        media: MediaMetadata
    ) -> FiltergraphResult:
        """Construct the multi-stage filtergraph: deinterlace, denoise, upscale, blend, grain."""
        steps: List[str] = []
        cur_node = "0:v"

        # Stage 1: Deinterlace if required
        if strategy.deinterlace:
            parity = "auto"
            if strategy.field_order == "tff":
                parity = "tff"
            elif strategy.field_order == "bff":
                parity = "bff"
            steps.append(f"[{cur_node}]bwdif=mode=send_field:parity={parity}:deint=all[deint]")
            cur_node = "deint"

        # Stage 2: Spatial-temporal analog artifact denoising
        if strategy.denoise:
            # Subtle analog noise reduction without erasing fine textures
            luma_s = strategy.denoise_strength * 1.2
            chroma_s = strategy.denoise_strength * 1.0
            luma_t = strategy.denoise_strength * 1.8
            chroma_t = strategy.denoise_strength * 1.5
            steps.append(
                f"[{cur_node}]hqdn3d={luma_s:.1f}:{chroma_s:.1f}:{luma_t:.1f}:{chroma_t:.1f}[denoised]"
            )
            cur_node = "denoised"

        # Stage 3: Split into Original and AI/Super-Resolution pathways
        steps.append(f"[{cur_node}]split=2[orig_branch][ai_branch]")

        tw = strategy.target_width
        th = strategy.target_height

        # Spline-scaled original clean stream for texture retention
        steps.append(f"[orig_branch]scale={tw}:{th}:flags=spline[scaled_orig]")

        # Super-resolution scaling pathway with edge refinement
        steps.append(f"[ai_branch]scale={tw}:{th}:flags=lanczos+accurate_rnd[ai_scaled]")
        steps.append(f"[ai_scaled]cas=strength=0.6[ai_sharpened]")

        # Stage 4: Layer Blending (80% AI / 20% Original micro-texture by default)
        ai_weight = max(0.0, min(1.0, strategy.blend_ai_ratio))
        orig_weight = max(0.0, min(1.0, 1.0 - ai_weight))
        blend_expr = f"A*{ai_weight:.2f}+B*{orig_weight:.2f}"
        steps.append(f"[ai_sharpened][scaled_orig]blend=all_expr='{blend_expr}'[blended]")

        # Stage 5: Dynamic subtle film grain injection
        grain_strength = max(0, min(30, strategy.film_grain_intensity))
        if grain_strength > 0:
            steps.append(f"[blended]noise=alls={grain_strength}:allf=t+u[final_video]")
            final_v_label = "[final_video]"
        else:
            final_v_label = "[blended]"

        filter_str = "; ".join(steps)
        return FiltergraphResult(
            filter_complex=filter_str,
            output_video_label=final_v_label
        )

    @classmethod
    def build_ffmpeg_cmd(
        cls,
        strategy: StrategyConfig,
        media: MediaMetadata,
        output_file: Optional[str] = None,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        progress_pipe: bool = True
    ) -> List[str]:
        """Generate complete FFmpeg CLI command list."""
        ffmpeg_bin = get_binary_path("ffmpeg")
        target_out = output_file or strategy.output_path

        # Strictly enforce .mkv output
        if not target_out.lower().endswith(".mkv"):
            base = os.path.splitext(target_out)[0]
            target_out = f"{base}.mkv"

        cmd = [ffmpeg_bin, "-y"]

        if progress_pipe:
            cmd.extend(["-progress", "pipe:1", "-nostats"])

        # Input seeking for segment chunking
        if start_sec is not None and start_sec > 0:
            cmd.extend(["-ss", f"{start_sec:.4f}"])

        cmd.extend(["-i", strategy.input_path])

        if duration_sec is not None and duration_sec > 0:
            cmd.extend(["-t", f"{duration_sec:.4f}"])

        # Filtergraph
        fg = cls.build_filtergraph(strategy, media)
        cmd.extend(["-filter_complex", fg.filter_complex])
        cmd.extend(["-map", fg.output_video_label])

        # Audio stream handling & lip-sync preservation
        if media.has_audio:
            cmd.extend([
                "-map", "0:a",
                "-af", "aresample=async=1000",
                "-c:a", "aac",
                "-b:a", "256k"
            ])
        else:
            cmd.append("-an")

        # Video encoding parameters
        encoder = strategy.encoder.lower()
        if "nvenc" in encoder:
            cmd.extend([
                "-c:v", encoder,
                "-preset", strategy.preset,
                "-cq", str(strategy.cq),
                "-pix_fmt", "yuv420p"
            ])
        elif encoder in ("libx265", "hevc"):
            cmd.extend([
                "-c:v", "libx265",
                "-crf", str(strategy.cq),
                "-preset", "medium",
                "-pix_fmt", "yuv420p"
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(strategy.cq),
                "-preset", "medium",
                "-pix_fmt", "yuv420p"
            ])

        # PTS preservation and MKV container enforcement
        cmd.extend([
            "-metadata:s:v:0", "rotate=0",
            "-fps_mode", "passthrough",
            "-f", "matroska",
            target_out
        ])

        return cmd

    @classmethod
    def build_piped_ffmpeg_cmd(
        cls,
        strategy: StrategyConfig,
        media: MediaMetadata,
        output_file: Optional[str] = None,
        fps: Optional[float] = None,
        start_sec: Optional[float] = None,
        duration_sec: Optional[float] = None,
        progress_pipe: bool = True
    ) -> List[str]:
        """Generate FFmpeg CLI command list reading raw Y4M/YUV from stdin pipe (from vspipe)."""
        ffmpeg_bin = get_binary_path("ffmpeg")
        target_out = output_file or strategy.output_path

        if not target_out.lower().endswith(".mkv"):
            base = os.path.splitext(target_out)[0]
            target_out = f"{base}.mkv"

        effective_fps = fps or media.fps or 30.0
        tw = strategy.target_width
        th = strategy.target_height

        cmd = [ffmpeg_bin, "-y"]

        if progress_pipe:
            cmd.extend(["-progress", "pipe:1", "-nostats"])

        # Input 0: Raw Y4M frame stream from VapourSynth stdout pipe
        cmd.extend([
            "-f", "yuv4mpegpipe",
            "-i", "pipe:0"
        ])

        # Input 1: Original media for audio extraction & micro-texture blend
        if start_sec is not None and start_sec > 0:
            cmd.extend(["-ss", f"{start_sec:.4f}"])
        if duration_sec is not None and duration_sec > 0:
            cmd.extend(["-t", f"{duration_sec:.4f}"])
        cmd.extend(["-i", strategy.input_path])

        # Filtergraph: Blend AI pipe stream with spline-scaled original for texture preservation
        ai_weight = max(0.0, min(1.0, strategy.blend_ai_ratio))
        orig_weight = max(0.0, min(1.0, 1.0 - ai_weight))
        grain_strength = max(0, min(30, strategy.film_grain_intensity))

        if orig_weight > 0.01:
            filter_steps = [
                f"[0:v]setpts=PTS-STARTPTS[ai_pipe]",
                f"[1:v]setpts=PTS-STARTPTS,fps={effective_fps:.4f},scale={tw}:{th}:flags=spline[scaled_orig]",
                f"[ai_pipe][scaled_orig]blend=all_expr='A*{ai_weight:.2f}+B*{orig_weight:.2f}':shortest=1[blended]"
            ]
            current_v = "[blended]"
        else:
            filter_steps = [
                f"[0:v]setpts=PTS-STARTPTS[ai_pipe]"
            ]
            current_v = "[ai_pipe]"

        if grain_strength > 0:
            filter_steps.append(f"{current_v}noise=alls={grain_strength}:allf=t+u[final_video]")
            final_v = "[final_video]"
        else:
            final_v = current_v

        cmd.extend(["-filter_complex", "; ".join(filter_steps)])
        cmd.extend(["-map", final_v])

        # Audio handling from Input 1
        if media.has_audio:
            cmd.extend([
                "-map", "1:a?",
                "-af", "aresample=async=1000",
                "-c:a", "aac",
                "-b:a", "256k"
            ])
        else:
            cmd.append("-an")

        # Video encoder selection
        encoder = strategy.encoder.lower()
        if "nvenc" in encoder:
            cmd.extend([
                "-c:v", encoder,
                "-preset", strategy.preset,
                "-cq", str(strategy.cq),
                "-pix_fmt", "yuv420p"
            ])
        elif encoder in ("libx265", "hevc"):
            cmd.extend([
                "-c:v", "libx265",
                "-crf", str(strategy.cq),
                "-preset", "medium",
                "-pix_fmt", "yuv420p"
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(strategy.cq),
                "-preset", "medium",
                "-pix_fmt", "yuv420p"
            ])

        cmd.extend([
            "-metadata:s:v:0", "rotate=0",
            "-shortest",
            "-fps_mode", "passthrough",
            "-f", "matroska",
            target_out
        ])

        return cmd
