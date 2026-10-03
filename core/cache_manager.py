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


class EngineCacheManager:
    """Manages TensorRT engine artifacts and OOM tiling strategies."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent.parent / "cache" / "engines"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_name = self._detect_gpu_name()

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

        return EngineCacheInfo(
            cache_key=cache_key,
            engine_path=str(engine_file),
            exists=engine_file.is_file(),
            model_name=model_name,
            gpu_name=self._gpu_name,
            resolution=input_resolution,
            fp16=fp16,
            tiles=tiles,
            tile_pad=10 if tiles > 1 else 0
        )

    def record_engine_metadata(self, info: EngineCacheInfo) -> str:
        """Save JSON sidecar metadata for the cached engine."""
        meta_file = self.cache_dir / f"{info.cache_key}.json"
        meta = {
            "cache_key": info.cache_key,
            "engine_path": info.engine_path,
            "model_name": info.model_name,
            "gpu_name": info.gpu_name,
            "resolution": list(info.resolution),
            "fp16": info.fp16,
            "tiles": info.tiles,
            "tile_pad": info.tile_pad,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return str(meta_file)

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
            tile_pad=10
        )
