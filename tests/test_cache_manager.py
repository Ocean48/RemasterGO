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


def test_list_and_delete_cached_engines():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EngineCacheManager(cache_dir=tmpdir)
        info = mgr.get_engine_info("Real-ESRGAN_x4", (640, 360), fp16=True)

        # Create dummy engine file and metadata
        Path(info.engine_path).write_bytes(b"\x00" * 2048)
        info.file_size_bytes = 2048
        info.exists = True
        mgr.record_engine_metadata(info)

        engines = mgr.list_cached_engines()
        assert len(engines) == 1
        assert engines[0].cache_key == info.cache_key
        assert engines[0].file_size_bytes == 2048

        meta = mgr.read_engine_metadata(info.cache_key)
        assert meta is not None
        assert meta["model_name"] == "Real-ESRGAN_x4"

        deleted = mgr.delete_cached_engine(info.cache_key)
        assert deleted is True
        assert len(mgr.list_cached_engines()) == 0


def test_register_engine_and_auto_prune():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = EngineCacheManager(cache_dir=tmpdir)

        # Create source engine file
        src_file = Path(tmpdir) / "source_test.engine"
        src_file.write_bytes(b"\x00" * 4096)

        info = mgr.register_engine(
            source_engine_file=str(src_file),
            model_name="CustomTestModel",
            input_resolution=(1920, 1080),
            fp16=True
        )

        assert info.exists is True
        assert info.file_size_bytes == 4096
        assert Path(info.engine_path).is_file()

        # Check listed engines
        cached = mgr.list_cached_engines()
        assert len(cached) == 1
        assert cached[0].model_name == "CustomTestModel"

        # Create orphaned json
        orphan_json = Path(tmpdir) / "non_existent_engine.json"
        orphan_json.write_text("{}", encoding="utf-8")

        # Listing engines prunes orphaned json
        cached_after = mgr.list_cached_engines()
        assert len(cached_after) == 1
        assert not orphan_json.exists()


def test_find_model_source_in_cache_models():
    with tempfile.TemporaryDirectory() as enginedir, tempfile.TemporaryDirectory() as modeldir:
        mgr = EngineCacheManager(cache_dir=enginedir, models_dir=modeldir)

        # Initially no model found
        assert mgr.find_model_source("SPAN_4x") is None

        # Simulate downloading a model into cache/models/
        model_file = Path(modeldir) / "SPAN_4x.onnx"
        model_file.write_bytes(b"\x00" * 1024)

        found = mgr.find_model_source("SPAN_4x")
        assert found == str(model_file)

        # Check that get_engine_info detects the downloaded model
        info = mgr.get_engine_info("SPAN_4x", (1920, 1080), fp16=True)
        assert info.exists is False
        assert info.source_model_path == str(model_file)

        # Test find_onnx_model_source matching and fallback
        assert mgr.find_onnx_model_source("SPAN_4x") == str(model_file)
        assert mgr.find_onnx_model_source("Unknown_Model") == str(model_file)


def test_compile_engine_validates_onnx_extension():
    import pytest
    with tempfile.TemporaryDirectory() as enginedir, tempfile.TemporaryDirectory() as modeldir:
        mgr = EngineCacheManager(cache_dir=enginedir, models_dir=modeldir)
        pth_file = Path(modeldir) / "RealESRGAN_x4plus.pth"
        pth_file.write_bytes(b"\x00" * 1024)

        with pytest.raises(ValueError, match="trtexec requires an ONNX model file"):
            mgr.compile_engine(
                onnx_model_path=str(pth_file),
                model_name="RealESRGAN_x4plus",
                input_resolution=(1920, 1080),
                fp16=True
            )


def test_trtexec_flag_detection():
    mgr = EngineCacheManager()
    trtexec_bin = mgr.get_trtexec_binary()
    if trtexec_bin:
        # Check whether flag detection runs safely without raising exceptions
        supports_onnx = mgr._trtexec_supports_flag(trtexec_bin, "--onnx")
        assert supports_onnx is True
        supports_fake = mgr._trtexec_supports_flag(trtexec_bin, "--non_existent_flag_xyz")
        assert supports_fake is False


