from __future__ import annotations

import numpy as np

from app.services.qwen_annotation_service import (
    _extract_green_bboxes,
    _render_green_canvas_from_shapes,
    _sanitize_svg_overlay,
)


def test_svg_whitelist_rejects_illegal_tag() -> None:
    shapes, safe_svg, ok, reason = _sanitize_svg_overlay(
        svg_overlay=(
            '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
            "<script>alert(1)</script>"
            "</svg>"
        ),
        width=120,
        height=80,
        reject_on_invalid_svg=True,
    )
    assert ok is False
    assert not shapes
    assert "不允许的SVG标签" in reason
    assert "<script" not in safe_svg


def test_svg_render_and_green_extract_rect() -> None:
    shapes, _, ok, reason = _sanitize_svg_overlay(
        svg_overlay=(
            '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
            '<rect x="10" y="12" width="60" height="40" fill="none" stroke="#00FF00" stroke-width="3"/>'
            "</svg>"
        ),
        width=120,
        height=80,
        reject_on_invalid_svg=True,
    )
    assert ok, reason
    canvas = _render_green_canvas_from_shapes(shapes, 120, 80)
    boxes, mask, green_pixels, contour_count, quality, reject_reason = _extract_green_bboxes(
        green_canvas=canvas,
        width=120,
        height=80,
        hsv_profile="neon",
    )
    assert mask.shape == (80, 120)
    assert green_pixels > 0
    assert contour_count >= 1
    assert quality > 0
    assert reject_reason == ""
    assert len(boxes) >= 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 12 and y1 <= 14
    assert x2 >= 65 and y2 >= 45


def test_extract_rejects_non_green_canvas() -> None:
    canvas = np.zeros((80, 120, 3), dtype=np.uint8)
    canvas[:, :] = (0, 0, 255)  # red
    boxes, mask, green_pixels, contour_count, quality, reject_reason = _extract_green_bboxes(
        green_canvas=canvas,
        width=120,
        height=80,
        hsv_profile="neon",
    )
    assert mask.shape == (80, 120)
    assert green_pixels == 0
    assert contour_count == 0
    assert quality == 0.0
    assert not boxes
    assert "未提取到亮绿色区域" in reject_reason
