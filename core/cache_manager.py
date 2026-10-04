"""TensorRT engine and model cache manager for RemasterGO."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class EngineCacheInfo:
    cache_key: str
    engine_path: str
    exists: bool
    model_name: str
    gpu_name: str
    resolution: Tuple[int, int]
    fp16: bool
    tiles: int = 1
    tile_pad: int = 10
    file_size_bytes: int = 0
    source_model_path: Optional[str] = None


class EngineCacheManager:
    """Manages TensorRT engine artifacts and OOM tiling strategies."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        custom_trtexec_path: Optional[str] = None
    ):
        base_dir = Path(__file__).resolve().parent.parent / "cache"
        self.cache_dir = Path(cache_dir) if cache_dir else (base_dir / "engines")
        self.models_dir = Path(models_dir) if models_dir else (base_dir / "models")
        self.custom_trtexec_path = custom_trtexec_path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_name = self._detect_gpu_name()

    def find_model_source(self, model_name: str) -> Optional[str]:
        """Look in cache/models/ for base model weights (.onnx, .pth, .engine) matching model_name."""
        clean_name = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
        for p in self.models_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in (".onnx", ".pth", ".bin", ".engine"):
                stem_clean = p.stem.lower().replace("-", "").replace("_", "").replace(" ", "")
                if clean_name in stem_clean or stem_clean in clean_name:
                    return str(p)
        return None

    def _detect_gpu_name(self) -> str:
        """Query GPU name via nvidia-smi with fallback to generic identifier."""
        try:
            cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            gpu = res.stdout.strip().splitlines()[0]
            if gpu:
                return gpu
        except Exception:
            pass
        return "NVIDIA_RTX_GPU"

    @property
    def gpu_name(self) -> str:
        return self._gpu_name

    def compute_cache_key(
        self,
        model_name: str,
        input_resolution: Tuple[int, int],
        fp16: bool = True
    ) -> str:
        """Generate MD5 hash based on (Model Weight + GPU Compute Capability + Input Resolution + Precision)."""
        precision_str = "FP16" if fp16 else "FP32"
        w, h = input_resolution
        key_raw = f"{model_name}_{self._gpu_name}_{w}x{h}_{precision_str}"
        return hashlib.md5(key_raw.encode("utf-8")).hexdigest()

    def get_engine_info(
        self,
        model_name: str,
        input_resolution: Tuple[int, int],
        fp16: bool = True,
        tiles: int = 1
    ) -> EngineCacheInfo:
        """Inspect cache and return engine information."""
        cache_key = self.compute_cache_key(model_name, input_resolution, fp16)
        engine_file = self.cache_dir / f"{cache_key}.engine"

        engine_size = engine_file.stat().st_size if engine_file.is_file() else 0
        source_model = self.find_model_source(model_name)

        return EngineCacheInfo(
            cache_key=cache_key,
            engine_path=str(engine_file),
            exists=engine_file.is_file(),
            model_name=model_name,
            gpu_name=self._gpu_name,
            resolution=input_resolution,
            fp16=fp16,
            tiles=tiles,
            tile_pad=10 if tiles > 1 else 0,
            file_size_bytes=engine_size,
            source_model_path=source_model
        )

    def record_engine_metadata(self, info: EngineCacheInfo) -> str:
        """Save JSON sidecar metadata for the cached engine."""
        meta_file = self.cache_dir / f"{info.cache_key}.json"
        engine_file = Path(info.engine_path)
        actual_size = engine_file.stat().st_size if engine_file.is_file() else info.file_size_bytes

        meta = {
            "cache_key": info.cache_key,
            "engine_path": str(engine_file.resolve()) if engine_file.is_file() else info.engine_path,
            "model_name": info.model_name,
            "gpu_name": info.gpu_name,
            "resolution": list(info.resolution),
            "fp16": info.fp16,
            "tiles": info.tiles,
            "tile_pad": info.tile_pad,
            "file_size_bytes": actual_size
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return str(meta_file)

    def register_engine(
        self,
        source_engine_file: str,
        model_name: str,
        input_resolution: Tuple[int, int] = (1920, 1080),
        fp16: bool = True,
        tiles: int = 1
    ) -> EngineCacheInfo:
        """Register or copy a .engine file into the cache with canonical key and metadata."""
        import shutil
        src_path = Path(source_engine_file).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"Engine file not found: {source_engine_file}")

        cache_key = self.compute_cache_key(model_name, input_resolution, fp16)
        dest_engine = self.cache_dir / f"{cache_key}.engine"

        # Move/copy to canonical cache key location if not already there
        if src_path.resolve() != dest_engine.resolve():
            if src_path.parent.resolve() == self.cache_dir.resolve():
                src_path.replace(dest_engine)
            else:
                shutil.copy2(src_path, dest_engine)

        info = EngineCacheInfo(
            cache_key=cache_key,
            engine_path=str(dest_engine),
            exists=True,
            model_name=model_name,
            gpu_name=self._gpu_name,
            resolution=input_resolution,
            fp16=fp16,
            tiles=tiles,
            tile_pad=10 if tiles > 1 else 0,
            file_size_bytes=dest_engine.stat().st_size
        )
        self.record_engine_metadata(info)
        return info

    def get_trtexec_binary(self) -> Optional[str]:
        """Resolve trtexec executable path across custom path, workspace root, TENSORRT_PATH, and PATH."""
        import shutil

        # 1. User-configured custom path
        if self.custom_trtexec_path and os.path.isfile(self.custom_trtexec_path):
            return self.custom_trtexec_path

        # 2. Local binary in project root or bin directories
        proj_root = Path(__file__).resolve().parent.parent
        candidate_local_bins = [
            proj_root / ("trtexec.exe" if os.name == "nt" else "trtexec"),
            proj_root / "bin" / ("trtexec.exe" if os.name == "nt" else "trtexec"),
            proj_root / "bin" / "tensorrt" / "bin" / ("trtexec.exe" if os.name == "nt" else "trtexec"),
            proj_root / "bin" / "trtexec" / ("trtexec.exe" if os.name == "nt" else "trtexec"),
        ]
        for c in candidate_local_bins:
            if c.is_file():
                return str(c)

        # 3. Environment variable TENSORRT_PATH
        trt_env = os.environ.get("TENSORRT_PATH")
        if trt_env:
            env_bin = Path(trt_env) / "bin" / ("trtexec.exe" if os.name == "nt" else "trtexec")
            if env_bin.is_file():
                return str(env_bin)

        # 4. CUDA path bin folder
        cuda_env = os.environ.get("CUDA_PATH")
        if cuda_env:
            cuda_bin = Path(cuda_env) / "bin" / ("trtexec.exe" if os.name == "nt" else "trtexec")
            if cuda_bin.is_file():
                return str(cuda_bin)

        # 5. System PATH lookup
        sys_bin = shutil.which("trtexec")
        if sys_bin:
            return sys_bin

        return None

    def is_trtexec_available(self) -> bool:
        """Check if NVIDIA TensorRT trtexec CLI is available."""
        return self.get_trtexec_binary() is not None

    def list_cached_engines(self) -> list[EngineCacheInfo]:
        """List all cached TensorRT engines found in the cache directory, pruning orphaned JSONs."""
        # Prune orphaned metadata sidecars
        for meta_p in self.cache_dir.glob("*.json"):
            engine_sibling = self.cache_dir / f"{meta_p.stem}.engine"
            if not engine_sibling.is_file():
                try:
                    meta_p.unlink(missing_ok=True)
                except OSError:
                    pass

        engines: list[EngineCacheInfo] = []
        for p in self.cache_dir.glob("*.engine"):
            cache_key = p.stem
            size_b = p.stat().st_size
            meta_path = self.cache_dir / f"{cache_key}.json"
            model_name = "Unknown"
            resolution = (1920, 1080)
            fp16 = True
            tiles = 1
            gpu_name = self._gpu_name

            if meta_path.is_file():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    model_name = meta.get("model_name", model_name)
                    resolution = tuple(meta.get("resolution", resolution))
                    fp16 = meta.get("fp16", fp16)
                    tiles = meta.get("tiles", tiles)
                    gpu_name = meta.get("gpu_name", gpu_name)
                except Exception:
                    pass
            else:
                # Auto-generate metadata for manually added engine file
                meta_info = EngineCacheInfo(
                    cache_key=cache_key,
                    engine_path=str(p),
                    exists=True,
                    model_name=cache_key,
                    gpu_name=gpu_name,
                    resolution=resolution,
                    fp16=fp16,
                    tiles=tiles,
                    tile_pad=0,
                    file_size_bytes=size_b
                )
                self.record_engine_metadata(meta_info)
                model_name = cache_key

            engines.append(EngineCacheInfo(
                cache_key=cache_key,
                engine_path=str(p),
                exists=True,
                model_name=model_name,
                gpu_name=gpu_name,
                resolution=resolution,
                fp16=fp16,
                tiles=tiles,
                tile_pad=10 if tiles > 1 else 0,
                file_size_bytes=size_b
            ))
        return engines

    def read_engine_metadata(self, cache_key: str) -> Optional[dict]:
        """Read sidecar JSON metadata for a specific cache key."""
        meta_file = self.cache_dir / f"{cache_key}.json"
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def delete_cached_engine(self, cache_key: str) -> bool:
        """Remove a cached engine binary and its metadata sidecar."""
        engine_file = self.cache_dir / f"{cache_key}.engine"
        meta_file = self.cache_dir / f"{cache_key}.json"
        deleted = False
        if engine_file.is_file():
            engine_file.unlink(missing_ok=True)
            deleted = True
        if meta_file.is_file():
            meta_file.unlink(missing_ok=True)
            deleted = True
        return deleted

    def _trtexec_supports_flag(self, trtexec_bin: str, flag: str) -> bool:
        """Check if the installed trtexec CLI binary accepts a specific flag (e.g. --fp16)."""
        try:
            res = subprocess.run([trtexec_bin, "--help"], capture_output=True, text=True, timeout=10)
            output = (res.stdout or "") + (res.stderr or "")
            return flag in output
        except Exception:
            return False

    def compile_engine(
        self,
        onnx_model_path: str,
        model_name: str,
        input_resolution: Tuple[int, int],
        fp16: bool = True,
        max_workspace_mb: int = 2048
    ) -> EngineCacheInfo:
        """Compile an ONNX model into a TensorRT .engine binary via trtexec if present."""
        if not onnx_model_path.lower().endswith(".onnx"):
            suffix = Path(onnx_model_path).suffix or "unknown"
            raise ValueError(
                f"Cannot compile '{Path(onnx_model_path).name}' into a TensorRT engine with trtexec.\n\n"
                f"trtexec requires an ONNX model file (.onnx), but the chosen file is a PyTorch checkpoint ({suffix}).\n\n"
                f"To compile a TensorRT engine, please use an ONNX model (e.g., 'realesr-general-x4v3.onnx').\n"
                f"PyTorch (.pth) models run directly on your GPU using the Real-ESRGAN Vulkan engine without compilation."
            )

        info = self.get_engine_info(model_name, input_resolution, fp16=fp16)
        if info.exists:
            return info

        trtexec_bin = self.get_trtexec_binary()
        if not trtexec_bin:
            raise FileNotFoundError(
                "NVIDIA TensorRT compiler (trtexec.exe) was not found on your system PATH or environment.\n\n"
                "How to proceed:\n"
                "1. Click 'Download Engine from URL...' to download precompiled engines directly without needing trtexec (Recommended).\n"
                "2. Or locate trtexec.exe manually using 'Browse for trtexec.exe...'\n"
                "3. Or download NVIDIA TensorRT from https://developer.nvidia.com/tensorrt and add its bin/ folder to PATH."
            )

        in_w, in_h = input_resolution
        shape_spec = f"input:1x3x{in_h}x{in_w}"
        cmd = [
            trtexec_bin,
            f"--onnx={onnx_model_path}",
            f"--minShapes={shape_spec}",
            f"--optShapes={shape_spec}",
            f"--maxShapes={shape_spec}",
            f"--shapes={shape_spec}",
            f"--saveEngine={info.engine_path}",
            f"--memPoolSize=workspace:{max_workspace_mb}"
        ]
        # Only pass --fp16 if the trtexec binary supports it (TensorRT 8.x/9.x; TRT 10+ is strongly typed)
        if fp16 and self._trtexec_supports_flag(trtexec_bin, "--fp16"):
            cmd.append("--fp16")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err_output = (proc.stderr or "").strip() or (proc.stdout or "").strip()
            err_lines = [
                line for line in err_output.splitlines()
                if any(k in line for k in ("[E]", "Error", "ERROR", "error", "Fatal", "FATAL"))
            ]
            err_summary = "\n".join(err_lines[-10:]) if err_lines else (err_output[-1000:] if err_output else "Unknown trtexec error")
            raise RuntimeError(f"trtexec compilation failed (exit code {proc.returncode}):\n{err_summary}")

        info.exists = os.path.isfile(info.engine_path)
        if info.exists:
            info.file_size_bytes = os.path.getsize(info.engine_path)
            self.record_engine_metadata(info)

        return info

    def get_oom_fallback_info(self, info: EngineCacheInfo) -> EngineCacheInfo:
        """Return a tiled configuration (2x2 grid, 4 tiles) to handle Out-Of-Memory (OOM) conditions."""
        return EngineCacheInfo(
            cache_key=info.cache_key,
            engine_path=info.engine_path,
            exists=info.exists,
            model_name=info.model_name,
            gpu_name=info.gpu_name,
            resolution=info.resolution,
            fp16=info.fp16,
            tiles=4,
            tile_pad=10,
            file_size_bytes=info.file_size_bytes
        )
