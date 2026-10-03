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

from core.pipeline import PipelineWorker
from core.probe import StrategyConfig, probe_media


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
        output_path = str(input_dir / f"{stem}_upscaled.mkv")

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
        segment_duration_sec=args.chunk_duration
    )

    exit_code = 0
    worker = PipelineWorker(config=strategy)

    def on_progress(cur: int, total: int, fps: float, stage: str):
        pct = (cur / total * 100.0) if total > 0 else 0.0
        bar_len = 30
        filled = int(bar_len * (pct / 100.0))
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}% | Frame {cur:5d}/{total:5d} | {fps:5.1f} FPS | {stage[:25]:<25}")
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

    args = parser.parse_args()

    if args.cli or args.input:
        if not args.input:
            parser.error("--input is required in CLI mode")
        return run_cli(args)
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())
