from homelab_subs.core.srt import _format_timestamp, segments_to_srt, write_srt_file
from homelab_subs.core.transcription import Segment


def test_format_timestamp_basic():
    assert _format_timestamp(0.0) == "00:00:00,000"
    assert _format_timestamp(1.234) == "00:00:01,234"
    assert _format_timestamp(61.0) == "00:01:01,000"
    # negative gets clamped to zero
    assert _format_timestamp(-5.0) == "00:00:00,000"


def test_segments_to_srt_simple():
    segments = [
        Segment(index=1, start=0.0, end=1.5, text="Hello world"),
        Segment(index=2, start=2.0, end=3.0, text="Second line"),
    ]

    srt = segments_to_srt(segments)
    expected = (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "Hello world\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "Second line\n"
        "\n"
    )

    assert srt == expected


def test_write_srt_file(tmp_path):
    segments = [
        Segment(index=1, start=0.0, end=1.0, text="Test"),
    ]
    out = tmp_path / "test.srt"
    write_srt_file(segments, out)

    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "Test" in content
