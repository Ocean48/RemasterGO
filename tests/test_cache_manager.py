"""Unit tests for TensorRT and model engine cache manager."""

import tempfile
from pathlib import Path

from core.cache_manager import EngineCacheManager


def test_engine_cache_key_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EngineCacheManager(cache_dir=tmpdir)
        k1 = mgr.compute_cache_key("Real-ESRGAN_x4", (640, 360), fp16=True)
        k2 = mgr.compute_cache_key("Real-ESRGAN_x4", (640, 360), fp16=True)
        k3 = mgr.compute_cache_key("Real-ESRGAN_x4", (1280, 720), fp16=True)
        k4 = mgr.compute_cache_key("Real-ESRGAN_x4", (640, 360), fp16=False)

        assert k1 == k2
        assert k1 != k3
        assert k1 != k4
        assert len(k1) == 32  # Valid MD5 hex length


def test_engine_info_and_metadata_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EngineCacheManager(cache_dir=tmpdir)
        info = mgr.get_engine_info("SPAN", (1920, 1080), fp16=True)

        assert info.model_name == "SPAN"
        assert info.resolution == (1920, 1080)
        assert info.fp16 is True
        assert info.tiles == 1
        assert Path(info.engine_path).parent == Path(tmpdir)

        # Save metadata
        meta_file = mgr.record_engine_metadata(info)
        assert Path(meta_file).is_file()


def test_oom_fallback_tiling():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EngineCacheManager(cache_dir=tmpdir)
        info = mgr.get_engine_info("Real-ESRGAN_x4", (1920, 1080), fp16=True)
        fallback = mgr.get_oom_fallback_info(info)

        assert fallback.tiles == 4
        assert fallback.tile_pad == 10
