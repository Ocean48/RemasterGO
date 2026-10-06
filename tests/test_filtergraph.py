"""Unit tests for filtergraph generation and FFmpeg command construction."""

from core.filtergraph import FiltergraphBuilder
from core.probe import MediaMetadata, StrategyConfig, VideoStreamInfo


def test_filtergraph_with_blend_and_grain():
    vinfo = VideoStreamInfo(
        index=0,
        codec_name="mpeg4",
        width=640,
        height=360,
        pix_fmt="yuv420p",
        r_frame_rate="30/1",
        fps=30.0,
        nb_frames=300,
        duration=10.0,
        is_interlaced=False
    )
    media = MediaMetadata(
        filepath="input.avi",
        format_name="avi",
        duration=10.0,
        size_bytes=1000000,
        video=vinfo,
        audio=[]
    )
    strategy = StrategyConfig(
        input_path="input.avi",
        output_path="output.mkv",
        target_width=1920,
        target_height=1080,
        deinterlace=False,
        denoise=True,
        blend_ai_ratio=0.8,
        film_grain_intensity=6,
        encoder="hevc_nvenc"
    )

    fg = FiltergraphBuilder.build_filtergraph(strategy, media)
    assert "hqdn3d" in fg.filter_complex
    assert "scale=1920:1080:flags=spline" in fg.filter_complex
    assert "blend=all_expr='A*0.80+B*0.20'" in fg.filter_complex
    assert "noise=alls=6:allf=t+u" in fg.filter_complex
    assert fg.output_video_label == "[final_video]"


def test_filtergraph_deinterlacing_option():
    vinfo = VideoStreamInfo(
        index=0,
        codec_name="mpeg2video",
        width=720,
        height=480,
        pix_fmt="yuv420p",
        r_frame_rate="30000/1001",
        fps=29.97,
        nb_frames=300,
        duration=10.0,
        is_interlaced=True,
        field_order="tff"
    )
    media = MediaMetadata(
        filepath="interlaced.avi",
        format_name="avi",
        duration=10.0,
        size_bytes=1000000,
        video=vinfo,
        audio=[]
    )
    strategy = StrategyConfig(
        input_path="interlaced.avi",
        output_path="output.mkv",
        target_width=1920,
        target_height=1080,
        deinterlace=True,
        field_order="tff"
    )

    fg = FiltergraphBuilder.build_filtergraph(strategy, media)
    assert "bwdif=mode=send_field:parity=tff" in fg.filter_complex


def test_ffmpeg_cmd_parameters():
    vinfo = VideoStreamInfo(
        index=0,
        codec_name="mpeg4",
        width=640,
        height=360,
        pix_fmt="yuv420p",
        r_frame_rate="30/1",
        fps=30.0,
        nb_frames=300,
        duration=10.0
    )
    media = MediaMetadata(
        filepath="input.avi",
        format_name="avi",
        duration=10.0,
        size_bytes=1000000,
        video=vinfo,
        audio=[]
    )
    strategy = StrategyConfig(
        input_path="input.avi",
        output_path="out.mkv",
        encoder="hevc_nvenc",
        cq=18,
        preset="p6"
    )

    cmd = FiltergraphBuilder.build_ffmpeg_cmd(
        strategy=strategy,
        media=media,
        start_sec=10.0,
        duration_sec=30.0
    )

    assert "-ss" in cmd
    assert "10.0000" in cmd
    assert "-t" in cmd
    assert "30.0000" in cmd
    assert "-c:v" in cmd
    assert "hevc_nvenc" in cmd
    assert "-cq" in cmd
    assert "18" in cmd
    assert "-metadata:s:v:0" in cmd
    assert "rotate=0" in cmd
    assert "-fps_mode" in cmd
    assert "passthrough" in cmd
    assert "-f" in cmd
    assert "matroska" in cmd
    assert cmd[-1].endswith(".mkv")
