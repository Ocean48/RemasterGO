"""RemasterGO - Hybrid Video Restoration & AI Upscaling Engine.

Main application entry point supporting both PySide6 Desktop GUI and CLI batch processing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication

from core.cache_manager import EngineCacheManager
from core.model_downloader import ModelManager
from core.pipeline import PipelineWorker
from core.probe import StrategyConfig, detect_scene_cuts, probe_media


def list_models_cli() -> int:
    """Print model catalog and installation status."""
    mgr = ModelManager()
    catalog = mgr.list_catalog()
    print("=" * 80)
    print(f"{'ID':<26} {'Name':<28} {'Scale':<6} {'Size':<10} {'Status':<10}")
    print("-" * 80)
    for m in catalog:
        status = "Installed" if m.installed else "Available"
        print(f"{m.id:<26} {m.name[:27]:<28} {m.scale}x{'':<4} {m.filesize_mb:5.1f} MB   {status:<10}")
    print("=" * 80)
    print(f"Models directory: {mgr.models_dir.resolve()}")
    return 0


def download_model_cli(model_id: str) -> int:
    """Download model(s) from catalog in CLI mode."""
    mgr = ModelManager()
    catalog = mgr.list_catalog()

    if model_id.lower() == "all":
        targets = [m for m in catalog if not m.installed]
        if not targets:
            print("All models in the catalog are already installed.")
            return 0
    else:
        entry = mgr.get_model(model_id)
        if not entry:
            print(f"Error: Model '{model_id}' not found in catalog.", file=sys.stderr)
            print("Run with --list-models to view valid model IDs.")
            return 1
        targets = [entry]

    for m in targets:
        print(f"\nDownloading '{m.name}' ({m.filesize_mb:.1f} MB)...")
        print(f"URL: {m.download_url}")

        def on_progress(downloaded: int, total: int, pct: float):
            dl_mb = downloaded / (1024 * 1024)
            tot_mb = total / (1024 * 1024) if total > 0 else 0
            bar_len = 30
            filled = int(bar_len * (pct / 100.0))
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r[{bar}] {pct:5.1f}% | {dl_mb:5.1f} MB / {tot_mb:5.1f} MB")
            sys.stdout.flush()

        try:
            saved_path = mgr.download_model_sync(m.id, progress_cb=on_progress)
            print(f"\nSuccess! Installed to: {saved_path}")
        except Exception as e:
            print(f"\nFailed to download {m.id}: {e}", file=sys.stderr)
            return 1

    return 0


def list_engines_cli() -> int:
    """List compiled TensorRT engines in the cache."""
    cache_mgr = EngineCacheManager()
    engines = cache_mgr.list_cached_engines()
    print("=" * 80)
    print(f"TensorRT Engine Cache (GPU: {cache_mgr.gpu_name})")
    print(f"Cache Directory: {cache_mgr.cache_dir.resolve()}")
    print("-" * 80)
    if not engines:
        print("No compiled TensorRT engines found in cache/engines/.")
    else:
        print(f"{'Model':<22} {'Cache Key':<34} {'Res':<10} {'Prec':<6} {'Size':<10}")
        print("-" * 80)
        for e in engines:
            sz_mb = e.file_size_bytes / (1024 * 1024)
            prec = "FP16" if e.fp16 else "FP32"
            res = f"{e.resolution[0]}x{e.resolution[1]}"
            print(f"{e.model_name[:21]:<22} {e.cache_key:<34} {res:<10} {prec:<6} {sz_mb:5.1f} MB")
    print("=" * 80)
    return 0


def build_engine_cli(model_input: str, resolution_str: str = "1920x1080", fp16: bool = True, workspace_mb: int = 2048) -> int:
    """Compile an ONNX model into a TensorRT .engine binary."""
    cache_mgr = EngineCacheManager()
    model_mgr = ModelManager()

    # Parse resolution
    parts = resolution_str.lower().split("x")
    tw, th = int(parts[0]), int(parts[1])

    # Find model file
    model_path = None
    if os.path.isfile(model_input):
        model_path = os.path.abspath(model_input)
    else:
        # Check in models dir
        candidate = model_mgr.models_dir / model_input
        if candidate.is_file():
            model_path = str(candidate)
        else:
            # Check catalog by id
            entry = model_mgr.get_model(model_input)
            if entry and entry.local_path:
                model_path = entry.local_path

    if not model_path:
        print(f"Error: Could not locate model '{model_input}'.", file=sys.stderr)
        print(f"Please check cache/models/ or download it using --download-model.")
        return 1

    model_name = Path(model_path).stem
    print("=" * 70)
    print("TensorRT Engine Compilation")
    print("=" * 70)
    print(f"Source Model: {model_path}")
    print(f"Resolution:   {tw}x{th}")
    print(f"Precision:    {'FP16' if fp16 else 'FP32'}")
    print(f"GPU:          {cache_mgr.gpu_name}")
    print(f"Workspace:    {workspace_mb} MB")
    print("=" * 70)

    try:
        info = cache_mgr.compile_engine(
            onnx_model_path=model_path,
            model_name=model_name,
            input_resolution=(tw, th),
            fp16=fp16,
            max_workspace_mb=workspace_mb
        )
        print(f"\nCompilation Successful!")
        print(f"Engine Path: {info.engine_path}")
        print(f"Cache Key:   {info.cache_key}")
        print(f"Size:        {info.file_size_bytes / (1024 * 1024):.1f} MB")
        return 0
    except Exception as e:
        print(f"\nCompilation Failed: {e}", file=sys.stderr)
        return 1


def run_cli(args: argparse.Namespace) -> int:
    """Run video upscaling in headless CLI mode."""
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    # Resolve output path: default to the input video location
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = Path(input_path).stem
        input_dir = Path(input_path).parent
        output_path = str(input_dir / f"{args.model}_{stem}_upscaled.mkv")

    # Enforce .mkv output
    if not output_path.lower().endswith(".mkv"):
        output_path = f"{os.path.splitext(output_path)[0]}.mkv"

    # Parse resolution
    tw, th = 1920, 1080
    if args.resolution:
        try:
            parts = args.resolution.lower().split("x")
            tw, th = int(parts[0]), int(parts[1])
        except Exception:
            print(f"Warning: Invalid resolution '{args.resolution}', defaulting to 1920x1080.")

    print("=" * 60)
    print("RemasterGO Headless Video Restoration & Upscaling Engine")
    print("=" * 60)
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Resolution:  {tw}x{th}")
    print(f"Model:       {args.model}")
    print(f"Encoder:     {args.encoder}")
    print(f"Blend:       {int(args.blend * 100)}% AI / {int((1.0 - args.blend) * 100)}% Original")
    print(f"Film Grain:  {args.grain}")
    print(f"Segment Dur: {args.chunk_duration}s")
    print("=" * 60)

    # Initialize minimal QCoreApplication for Qt signal/event loop if needed
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv)

    # Probe media
    media = probe_media(input_path)

    deinterlace = args.deinterlace or (media.video.is_interlaced if media.video else False)

    scene_cuts = []
    if args.scene_cuts:
        print("Detecting scene transitions for segment alignment...")
        scene_cuts = detect_scene_cuts(input_path, threshold=0.3)
        print(f"Detected {len(scene_cuts)} scene cuts.")

    strategy = StrategyConfig(
        input_path=input_path,
        output_path=output_path,
        target_width=tw,
        target_height=th,
        model_name=args.model,
        deinterlace=deinterlace,
        denoise=not args.no_denoise,
        blend_ai_ratio=args.blend,
        film_grain_intensity=args.grain,
        encoder=args.encoder,
        segment_duration_sec=args.chunk_duration,
        scene_cuts=scene_cuts
    )

    exit_code = 0
    worker = PipelineWorker(config=strategy)

    def on_progress(cur: int, total: int, fps: float, stage: str):
        pct = (cur / total * 100.0) if total > 0 else 0.0
        bar_len = 28
        filled = int(bar_len * (pct / 100.0))
        if filled >= bar_len:
            bar = "=" * bar_len
        elif filled > 0:
            bar = "=" * (filled - 1) + ">" + " " * (bar_len - filled)
        else:
            bar = " " * bar_len

        eta_str = "--:--"
        if fps > 0 and total > cur:
            rem_sec = int((total - cur) / fps)
            m, s = divmod(rem_sec, 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        sys.stdout.write(
            f"\r[{bar}] {pct:5.1f}% | Frame {cur:5d}/{total:5d} | {fps:5.1f} FPS | ETA {eta_str} | {stage[:22]:<22}"
        )
        sys.stdout.flush()

    def on_log(msg: str):
        print(f"\n[RemasterGO] {msg}")

    def on_error(err: str):
        nonlocal exit_code
        exit_code = 1
        print(f"\n[ERROR] {err}", file=sys.stderr)
        app.quit()

    def on_finished(out: str):
        print(f"\nSuccess! Rendered: {out}")
        app.quit()

    worker.sig_progress.connect(on_progress)
    worker.sig_log.connect(on_log)
    worker.sig_error.connect(on_error)
    worker.sig_finished.connect(on_finished)

    worker.start()
    app.exec()
    return exit_code


def run_gui() -> int:
    """Launch the PySide6 Desktop GUI."""
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("RemasterGO")
    app.setOrganizationName("RemasterGO")

    # Set explicit font to prevent QFont::setPointSize <= 0 warning on High-DPI screens
    from PySide6.QtGui import QFont
    default_font = QFont("Segoe UI", 10)
    app.setFont(default_font)

    from ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RemasterGO - Hybrid Video Restoration & AI Upscaling Engine"
    )
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI mode instead of GUI")
    parser.add_argument("-i", "--input", type=str, help="Input video file path")
    parser.add_argument("-o", "--output", type=str, help="Target output video file path (enforced .mkv, default: input video location)")
    parser.add_argument("-r", "--resolution", type=str, default="1920x1080", help="Target resolution (e.g. 1920x1080, 2560x1440, 3840x2160)")
    parser.add_argument("-m", "--model", type=str, default="Real-ESRGAN_x4", help="AI model name")
    parser.add_argument("-e", "--encoder", type=str, default="hevc_nvenc", help="Video encoder (hevc_nvenc, h264_nvenc, libx265, libx264)")
    parser.add_argument("-b", "--blend", type=float, default=0.8, help="AI to original texture blend ratio (0.0 to 1.0)")
    parser.add_argument("-g", "--grain", type=int, default=6, help="Dynamic film grain intensity (0 to 20)")
    parser.add_argument("-c", "--chunk-duration", type=float, default=300.0, help="Segment checkpoint duration in seconds (0 to disable)")
    parser.add_argument("--deinterlace", action="store_true", help="Force deinterlacing on")
    parser.add_argument("--no-denoise", action="store_true", help="Disable spatial/temporal denoising")
    parser.add_argument("--scene-cuts", action="store_true", help="Detect scene cut boundaries and snap checkpoints")

    # Model and engine management options
    parser.add_argument("--list-models", action="store_true", help="List available and installed pretrained AI models")
    parser.add_argument("--download-model", type=str, metavar="MODEL_ID", help="Download a model by ID or 'all'")
    parser.add_argument("--list-engines", action="store_true", help="List compiled TensorRT engines in cache/engines/")
    parser.add_argument("--build-engine", type=str, metavar="MODEL_FILE", help="Compile an ONNX model into a TensorRT .engine binary")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of default FP16 when building engine")
    parser.add_argument("--workspace-mb", type=int, default=2048, help="Max workspace memory in MB for TensorRT compilation (default: 2048)")

    args = parser.parse_args()

    if args.list_models:
        return list_models_cli()

    if args.download_model:
        return download_model_cli(args.download_model)

    if args.list_engines:
        return list_engines_cli()

    if args.build_engine:
        return build_engine_cli(
            model_input=args.build_engine,
            resolution_str=args.resolution,
            fp16=not args.fp32,
            workspace_mb=args.workspace_mb
        )

    if args.cli or args.input:
        if not args.input:
            parser.error("--input is required in CLI mode")
        return run_cli(args)
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())
