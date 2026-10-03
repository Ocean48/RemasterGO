"""VapourSynth pipeline script builder and runtime validator for RemasterGO."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from core.probe import MediaMetadata, StrategyConfig, get_binary_path


def is_vapoursynth_available() -> bool:
    """Check if the VapourSynth Python core package is importable."""
    try:
        import vapoursynth  # type: ignore # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def is_vspipe_available() -> bool:
    """Check if vspipe executable is available in PATH or project folder."""
    binary = get_binary_path("vspipe")
    if os.path.isabs(binary) and os.path.isfile(binary):
        return True
    return shutil.which("vspipe") is not None


def is_vsmlrt_available() -> bool:
    """Check if vs-mlrt (TensorRT VapourSynth runtime) is importable."""
    try:
        import vsmlrt  # type: ignore # noqa: F401
        return True
    except (ImportError, Exception):
        return False


class VapourSynthScriptBuilder:
    """Generates standalone Python VapourSynth (.vpy) script execution graphs."""

    @classmethod
    def build_script(
        cls,
        strategy: StrategyConfig,
        media: MediaMetadata,
        engine_path: str,
        start_frame: Optional[int] = None,
        num_frames: Optional[int] = None,
        tiles: int = 1,
        fp16: bool = True,
        scene_cut_frames: Optional[List[int]] = None
    ) -> str:
        """Generate Python code for the VapourSynth GPU processing pipeline."""
        input_file = Path(strategy.input_path).resolve().as_posix()
        engine_file = Path(engine_path).resolve().as_posix()

        tff_flag = "True" if strategy.field_order in ("tff", "progressive") else "False"
        denoise_strength = strategy.denoise_strength

        lines: List[str] = [
            "# RemasterGO Auto-Generated VapourSynth Pipeline Graph",
            "# Stage 2: Deinterlacing & Denoising | Stage 3: Zero-Copy TensorRT Inference",
            "import os",
            "import sys",
            "import glob",
            "from pathlib import Path",
            "import vapoursynth as vs",
            "",
            "core = vs.core",
            "",
            "# Auto-load local plugins if present",
            "workspace_dir = Path(__file__).resolve().parent.parent.parent",
            "plugin_dirs = [",
            "    workspace_dir / 'bin' / 'vapoursynth' / 'plugins',",
            "    workspace_dir / 'bin' / 'vapoursynth' / 'vs-coreplugins',",
            "]",
            "for pdir in plugin_dirs:",
            "    if pdir.is_dir():",
            "        for dll in pdir.glob('*.dll'):",
            "            try:",
            "                core.std.LoadPlugin(str(dll))",
            "            except Exception:",
            "                pass",
            "",
            "# Stage 1b: Video Source Decoder",
            "source_path = " + repr(input_file),
            "if hasattr(core, 'bs'):",
            "    clip = core.bs.VideoSource(source=source_path)",
            "elif hasattr(core, 'ffms2'):",
            "    clip = core.ffms2.Source(source=source_path)",
            "elif hasattr(core, 'lsmas'):",
            "    clip = core.lsmas.LWLibavSource(source=source_path)",
            "else:",
            "    raise RuntimeError('No compatible VapourSynth source plugin found (bs, ffms2, or lsmas)')",
            ""
        ]

        # Segment slicing at the clip level
        if start_frame is not None and start_frame > 0:
            if num_frames is not None and num_frames > 0:
                end_f = start_frame + num_frames
                lines.append(f"# Segment boundary slice [{start_frame}:{end_f}]")
                lines.append(f"clip = clip[{start_frame}:{end_f}]")
            else:
                lines.append(f"# Segment boundary slice [{start_frame}:]")
                lines.append(f"clip = clip[{start_frame}:]")
        elif num_frames is not None and num_frames > 0:
            lines.append(f"# Segment boundary slice [0:{num_frames}]")
            lines.append(f"clip = clip[0:{num_frames}]")

        # Stage 2: Deinterlacing (QTGMC)
        if strategy.deinterlace:
            lines.extend([
                "",
                "# Stage 2a: QTGMC Motion-Adaptive Deinterlacing",
                "try:",
                "    if hasattr(core, 'qtmgc'):",
                f"        clip = core.qtmgc.QTGMC(clip, Preset='Slower', TFF={tff_flag})",
                "    else:",
                "        import havsfunc",
                f"        clip = havsfunc.QTGMC(clip, Preset='Slower', TFF={tff_flag})",
                "except Exception as e:",
                "    # Fallback to internal bwdif filter if QTGMC helper is absent",
                "    if hasattr(core, 'bwdif'):",
                f"        clip = core.bwdif.Bwdif(clip, field=1 if {tff_flag} else 0)",
                "    else:",
                "        raise e",
            ])

        # Stage 2b: Analog tape denoising (KNLMeansCL / BM3D)
        if strategy.denoise:
            lines.extend([
                "",
                "# Stage 2b: Denoising (KNLMeansCL / BM3D)",
                "if hasattr(core, 'knlm'):",
                f"    clip = core.knlm.KNLMeansCL(clip, d=2, a=2, h={denoise_strength:.2f})",
                "elif hasattr(core, 'bm3d'):",
                f"    clip = core.bm3d.BM3D(clip, sigma=[{denoise_strength:.2f}, {denoise_strength:.2f}, {denoise_strength:.2f}])",
            ])

        # Stage 3: Zero-Copy TensorRT Super-Resolution (vs-mlrt)
        tile_pad = 10 if tiles > 1 else 0
        lines.extend([
            "",
            "# Stage 3: Direct VRAM TensorRT Inference via vs-mlrt",
            "import vsmlrt",
            f"model_engine = {repr(engine_file)}",
            f"clip = vsmlrt.TRT(clip, model_path=model_engine, tiles={tiles}, tile_pad={tile_pad}, fp16={fp16})",
            "",
            "# Output standard progressive YUV420P stream",
            "clip = core.resize.Bicubic(clip, format=vs.YUV420P8)",
            "clip.set_output()",
            ""
        ])

        return "\n".join(lines)

    @classmethod
    def write_script_file(
        cls,
        target_path: str,
        strategy: StrategyConfig,
        media: MediaMetadata,
        engine_path: str,
        start_frame: Optional[int] = None,
        num_frames: Optional[int] = None,
        tiles: int = 1,
        fp16: bool = True
    ) -> str:
        """Write generated VapourSynth pipeline code to a temporary or target .vpy file."""
        script_code = cls.build_script(
            strategy=strategy,
            media=media,
            engine_path=engine_path,
            start_frame=start_frame,
            num_frames=num_frames,
            tiles=tiles,
            fp16=fp16
        )
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(script_code)
        return str(target)
