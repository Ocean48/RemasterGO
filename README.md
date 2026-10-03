# RemasterGO - Hybrid Video Restoration & AI Upscaling Engine

RemasterGO is a high-performance video restoration and AI upscaling desktop application built with **Python 3.12**, **PySide6**, **FFmpeg**, and **NVIDIA GPU acceleration** (NVENC HEVC/H.264). It implements the complete pipeline specified in `video_upscaling_technical_plan.md`, featuring segment-based checkpoint recovery, 80/20 micro-texture blending, dynamic film grain, audio presentation timestamp (PTS) preservation, and strict Matroska (`.mkv`) container enforcement.

---

## Folder Structure Explained

```
RemasterGO/
├── .venv/                  # Python virtual environment (PySide6, pytest, numpy)
├── cache/                  # Internal runtime cache & checkpoint storage
│   ├── engines/            # Cached TensorRT engines and sidecar JSON metadata
│   └── <job_stem>/         # Segment chunks (segment_XXXX.mkv) & job_checkpoint.json
├── core/                   # Core engine pipeline
│   ├── __init__.py
│   ├── cache_manager.py    # TensorRT & model cache management (MD5 hashing)
│   ├── checkpoint.py       # Segment chunking, recovery state & lossless concat
│   ├── filtergraph.py      # Deinterlace, denoise, blend, grain & encoding CLI builder
│   ├── job_queue.py        # Multi-job queue controller and state tracking
│   ├── pipeline.py         # PipelineWorker (QThread) subprocess orchestration
│   └── probe.py            # ffprobe media analysis, PTS tracking & scene detection
├── output/                 # The ONLY final destination directory for upscaled videos
├── tests/                  # Automated pytest unit & integration test suite
├── ui/                     # PySide6 desktop GUI
│   ├── __init__.py
│   ├── main_window.py      # Main window (Queue, Strategy Sidebar, Metrics, Logs)
│   ├── theme.py            # Dark mode QSS stylesheet
│   └── widgets.py          # Drag-and-drop table, status badges, metric cards
├── ffmpeg.exe              # Bundled FFmpeg binary with NVENC support
├── ffprobe.exe             # Bundled ffprobe binary
├── main.py                 # Application entrypoint (Desktop GUI & CLI runner)
├── requirements.txt        # Pinned Python package dependencies
└── video_upscaling_technical_plan.md
```

### Output Location Behavior

* **Default Behavior**: Rendered `.mkv` videos are saved directly in the **same directory as the input video file** (e.g., `<input_dir>/<stem>_upscaled_<res>.mkv`).
* **Custom Output Override**:
  - In the **GUI**: Click **Browse...** in the Output Destination card to choose a custom directory. Click **Reset** to return to the default (input video location).
  - In the **CLI**: Pass `-o <custom_path.mkv>` to override the output file location.
* **`cache/`**: Internal directory used for segment chunks (`segment_XXXX.mkv`) and recovery states (`job_checkpoint.json`). Once rendering finishes, segments are losslessly joined into the final `.mkv` at the input video's location.

---

## Key Features

1. **Hardware-Accelerated Encoding**:
   - Native support for NVIDIA NVENC (`hevc_nvenc`, `h264_nvenc`) utilizing the installed RTX 5060 Ti GPU.
   - Automatic fallback to CPU software encoders (`libx265`, `libx264`) when hardware encoding is unavailable.

2. **Strict Matroska (`.mkv`) Container Enforcement**:
   - All final files and intermediate chunks are written to `.mkv`. Unlike MP4 (which requires a trailing `moov` atom), MKV streams headers up-front, guaranteeing file readability even during crashes or power loss.

3. **Segment-Based Crash Recovery**:
   - Long video jobs can be divided into configurable chunks (e.g. 5 minutes).
   - If processing is interrupted, restarting the job reads `job_checkpoint.json`, verifies completed segments with `ffprobe`, skips already rendered chunks, and resumes where it left off.
   - Finished segments are losslessly joined using FFmpeg's concat demuxer (`-c copy`).

4. **Micro-Texture Preservation (80/20 Blending)**:
   - Blends the AI upscaled stream with a spline-scaled version of the original cleaned footage (default: 80% AI / 20% Original) to retain fine surface details and avoid plastic-looking over-smoothing.

5. **Dynamic Film Grain Injection**:
   - Adds subtle temporal film grain (`noise=alls=6:allf=t+u`) to mask artificial neural network smoothing artifacts.

6. **Audio & Lip-Sync Preservation**:
   - Preserves original presentation timestamps with `-fps_mode passthrough` and applies asynchronous resampling (`-af aresample=async=1000`) for permanent audio/video synchronization.

7. **Dual Modes: Desktop GUI & Headless CLI**:
   - Interactive modern dark-mode GUI with drag-and-drop video queue, live inference FPS/ETA stats, and real-time log viewer.
   - Scriptable CLI for automation, server environments, and batch scripts.

---

## Installation & Setup

1. **Prerequisites**:
   - Windows 10/11 64-bit
   - NVIDIA GeForce GPU with updated drivers (tested on RTX 5060 Ti)
   - Python 3.12 (in `.venv`)

2. **Install Dependencies**:
   ```powershell
   & ".\.venv\Scripts\pip.exe" install -r requirements.txt
   ```

---

## Usage

### 1. Launch Desktop GUI

```powershell
& ".\.venv\Scripts\python.exe" main.py
```

* Drag and drop video files (`.avi`, `.mp4`, `.mkv`, `.mov`) into the queue table.
* Choose target resolution (1080p, 1440p, 4K, 2x, 4x), AI model, and encoder.
* Adjust micro-texture blend ratio (slider) and film grain intensity.
* Click **Start Queue Processing**. Monitor live FPS, elapsed time, ETA, and logs.

### 2. Headless CLI Processing

```powershell
& ".\.venv\Scripts\python.exe" main.py --cli -i "sample_640x360.avi" -o "output\upscaled.mkv" -r "1920x1080" -e "hevc_nvenc" -b 0.8 -g 6 -c 300
```

#### CLI Options:
| Flag | Description | Default |
| --- | --- | --- |
| `--cli` | Run in headless CLI mode | False |
| `-i, --input` | Path to source video file (required) | None |
| `-o, --output` | Target output path (strictly `.mkv`) | `<input_dir>/<name>_upscaled.mkv` (Same as input) |
| `-r, --resolution` | Target resolution (`1920x1080`, `2560x1440`, `3840x2160`) | `1920x1080` |
| `-m, --model` | AI model engine name | `Real-ESRGAN_x4` |
| `-e, --encoder` | Video encoder (`hevc_nvenc`, `h264_nvenc`, `libx265`, `libx264`) | `hevc_nvenc` |
| `-b, --blend` | AI to original micro-texture blend ratio (0.0 to 1.0) | `0.8` (80% AI / 20% Original) |
| `-g, --grain` | Dynamic film grain intensity (0 to 20) | `6` |
| `-c, --chunk-duration` | Checkpoint segment duration in seconds (`0` for single pass) | `300.0` (5 minutes) |
| `--deinterlace` | Force deinterlacing on | Auto-detected |
| `--no-denoise` | Disable spatial/temporal analog tape denoising | False |

---

## Running Automated Tests

Run the complete test suite using pytest:

```powershell
& ".\.venv\Scripts\pytest.exe" -v
```
