"""Unit tests for VapourSynth script builder and dual-pipeline execution components."""

from pathlib import Path
import pytest

from core.filtergraph import FiltergraphBuilder
from core.probe import MediaMetadata, StrategyConfig, VideoStreamInfo
from core.vapoursynth_builder import (
    VapourSynthScriptBuilder,
    is_vapoursynth_available,
    is_vspipe_available,
    is_vsmlrt_available,
)


@pytest.fixture
def sample_media():
    return MediaMetadata(
        filepath="sample.avi",
        format_name="avi",
        duration=10.0,
        size_bytes=1000000,
        video=VideoStreamInfo(
            index=0,
            codec_name="h264",
            width=640,
            height=360,
            pix_fmt="yuv420p",
            r_frame_rate="30/1",
            fps=30.0,
            nb_frames=300,
            duration=10.0,
            is_interlaced=True,
            field_order="tff"
        ),
        audio=[]
    )


@pytest.fixture
def sample_strategy():
    return StrategyConfig(
        input_path="sample.avi",
        output_path="sample_upscaled.mkv",
        target_width=1920,
        target_height=1080,
        model_name="Real-ESRGAN_x4",
        deinterlace=True,
        field_order="tff",
        denoise=True,
        denoise_strength=1.5,
        blend_ai_ratio=0.8,
        film_grain_intensity=6,
        encoder="hevc_nvenc"
    )


def test_availability_functions():
    # Should return booleans without throwing
    assert isinstance(is_vapoursynth_available(), bool)
    assert isinstance(is_vspipe_available(), bool)
    assert isinstance(is_vsmlrt_available(), bool)


def test_build_vapoursynth_script(sample_strategy, sample_media):
    engine_path = "cache/engines/test_engine.engine"
    script = VapourSynthScriptBuilder.build_script(
        strategy=sample_strategy,
        media=sample_media,
        engine_path=engine_path,
        start_frame=30,
        num_frames=60,
        tiles=1,
        fp16=True
    )

    assert "import vapoursynth as vs" in script
    assert "core.bs.VideoSource" in script or "core.ffms2.Source" in script or "core.lsmas" in script
    assert "clip = clip[30:90]" in script
    assert "QTGMC" in script
    assert "KNLMeansCL" in script or "BM3D" in script
    assert "core.trt.Model" in script
    assert "test_engine.engine" in script
    assert "clip.set_output()" in script


def test_build_vapoursynth_script_tiled_oom_fallback(sample_strategy, sample_media):
    engine_path = "cache/engines/test_engine.engine"
    script = VapourSynthScriptBuilder.build_script(
        strategy=sample_strategy,
        media=sample_media,
        engine_path=engine_path,
        start_frame=0,
        num_frames=100,
        tiles=4,
        fp16=True
    )

    assert "core.trt.Model" in script


def test_write_script_file(sample_strategy, sample_media, tmp_path):
    out_vpy = tmp_path / "test_pipeline.vpy"
    res_path = VapourSynthScriptBuilder.write_script_file(
        target_path=str(out_vpy),
        strategy=sample_strategy,
        media=sample_media,
        engine_path="cache/engines/fake.engine"
    )

    assert Path(res_path).is_file()
    content = Path(res_path).read_text(encoding="utf-8")
    assert "core.trt.Model" in content


def test_build_piped_ffmpeg_cmd(sample_strategy, sample_media):
    cmd = FiltergraphBuilder.build_piped_ffmpeg_cmd(
        strategy=sample_strategy,
        media=sample_media,
        output_file="out.mkv",
        fps=30.0,
        start_sec=1.0,
        duration_sec=5.0
    )

    cmd_str = " ".join(cmd)
    assert "-i pipe:0" in cmd_str
    assert "-f yuv4mpegpipe" in cmd_str
    assert "blend=" in cmd_str
    assert "noise=alls=6" in cmd_str
    assert "-f matroska" in cmd_str
    assert "out.mkv" in cmd_str
