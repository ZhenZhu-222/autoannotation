from __future__ import annotations

from app.core.utils import sanitize_filename


def test_sanitize_filename_keeps_extension_with_non_ascii_name() -> None:
    out = sanitize_filename("装配视频.mp4")
    assert out.endswith(".mp4")


def test_sanitize_filename_fallback_when_stem_is_empty() -> None:
    out = sanitize_filename("....", fallback="video.mp4")
    assert out.endswith(".mp4")
    assert out != ".mp4"
