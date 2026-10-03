"""Unit tests for native Real-ESRGAN GPU AI neural network upscaler."""

from pathlib import Path
import pytest

from core.ai_upscaler import (
    calculate_scale_factor,
    get_ai_binary_path,
    is_ai_binary_available,
    resolve_model_name,
)


def test_ai_binary_availability():
    path = get_ai_binary_path()
    assert path is not None
    assert Path(path).is_file()
    assert is_ai_binary_available() is True


def test_resolve_model_name():
    assert resolve_model_name("Real-ESRGAN_x4 (Balanced)") == "realesrgan-x4plus"
    assert resolve_model_name("Real-ESRGAN x4 Plus") == "realesrgan-x4plus"
    assert resolve_model_name("Real-ESRGAN x4 Anime (6B)") == "realesrgan-x4plus-anime"
    assert resolve_model_name("RealESRGANv2 Anime 2x") == "realesr-animevideov3"
    assert resolve_model_name("SPAN (Fast Film / Anime)") == "realesr-animevideov3"
    assert resolve_model_name("Compact 2x Lightweight") == "realesr-animevideov3"


def test_calculate_scale_factor():
    assert calculate_scale_factor(640, 360, 1280, 720) == 2
    assert calculate_scale_factor(640, 360, 1920, 1080) == 3
    assert calculate_scale_factor(480, 270, 1920, 1080) == 4
    assert calculate_scale_factor(0, 0, 1920, 1080) == 4


def test_progress_callback_signature_compatibility():
    from core.probe import MediaMetadata, StrategyConfig
    from core.checkpoint import SegmentInfo
    from core.ai_upscaler import run_ai_segment_upscale
    import inspect

    sig = inspect.signature(run_ai_segment_upscale)
    assert "progress_cb" in sig.parameters
    assert "cancel_check" in sig.parameters
    assert "log_cb" in sig.parameters

