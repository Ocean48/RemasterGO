"""Unit tests for media probe and metadata extraction."""

from pathlib import Path
import pytest

from core.probe import (
    MediaMetadata,
    StrategyConfig,
    _parse_fraction,
    create_strategy,
    get_binary_path,
    probe_media,
)


def test_binary_path_resolution():
    ffprobe = get_binary_path("ffprobe")
    ffmpeg = get_binary_path("ffmpeg")
    assert Path(ffprobe).exists()
    assert Path(ffmpeg).exists()


def test_parse_fraction():
    assert pytest.approx(_parse_fraction("30000/1001"), 0.001) == 29.970
    assert pytest.approx(_parse_fraction("24000/1001"), 0.001) == 23.976
    assert _parse_fraction("25") == 25.0
    assert _parse_fraction("0/0") == 0.0
    assert _parse_fraction("") == 0.0


def test_probe_sample_video():
    sample_file = "sample_640x360.avi"
    assert Path(sample_file).exists()

    meta = probe_media(sample_file)
    assert meta.width == 640
    assert meta.height == 360
    assert meta.total_frames == 400
    assert pytest.approx(meta.fps, 0.01) == 29.97
    assert meta.duration > 13.0
    assert meta.has_audio is False


def test_probe_sample_with_audio():
    sample_file = "sample_1280x720_surfing_with_audio.avi"
    assert Path(sample_file).exists()

    meta = probe_media(sample_file)
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.has_audio is True
    assert len(meta.audio) == 1
    assert meta.audio[0].channels == 2
    assert meta.audio[0].sample_rate == 48000


def test_strategy_mkv_enforcement():
    cfg = StrategyConfig(
        input_path="input.avi",
        output_path="test_output.mp4"
    )
    assert cfg.output_path.endswith(".mkv")


def test_create_strategy(tmp_path):
    meta = probe_media("sample_640x360.avi")
    strategy = create_strategy(
        meta,
        output_dir=str(tmp_path),
        target_width=1920,
        target_height=1080,
        encoder="hevc_nvenc"
    )
    assert strategy.output_path.endswith(".mkv")
    assert strategy.target_width == 1920


def test_create_strategy_default_input_location():
    meta = probe_media("sample_640x360.avi")
    strategy = create_strategy(meta)
    expected_parent = Path(meta.filepath).parent.resolve()
    actual_parent = Path(strategy.output_path).parent.resolve()
    assert expected_parent == actual_parent
    assert strategy.output_path.endswith(".mkv")
    assert strategy.target_height == 1080
    assert strategy.encoder == "hevc_nvenc"


def test_portrait_rotation_and_display_dimensions():
    from core.probe import VideoStreamInfo
    vinfo = VideoStreamInfo(
        index=0,
        codec_name="h264",
        width=1920,
        height=1080,
        pix_fmt="yuv420p",
        r_frame_rate="30/1",
        fps=30.0,
        nb_frames=300,
        duration=10.0,
        rotation=90,
        display_width=1080,
        display_height=1920,
        is_portrait=True
    )
    meta = MediaMetadata(
        filepath="phone_video.mp4",
        format_name="mov,mp4",
        duration=10.0,
        size_bytes=5000000,
        video=vinfo,
        audio=[]
    )
    assert meta.rotation == 90
    assert meta.display_width == 1080
    assert meta.display_height == 1920
    assert meta.is_portrait is True

    # When strategy is created with default 1920x1080, it should auto-adapt to 1080x1920
    strategy = create_strategy(meta, target_width=1920, target_height=1080)
    assert strategy.target_width == 1080
    assert strategy.target_height == 1920
    assert "upscaled_1920p.mkv" in strategy.output_path

    # When strategy is created with explicit scale_factor=2, it should produce 2160x3840
    strat_scaled = create_strategy(meta, scale_factor=2.0)
    assert strat_scaled.target_width == 2160
    assert strat_scaled.target_height == 3840
    assert "upscaled_2x_3840p.mkv" in strat_scaled.output_path

