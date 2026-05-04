from __future__ import annotations

from app.services.annotation_service import xyxy_to_yolo
from app.services.label_format import get_label_task_spec, label_line_to_task, sort_label_lines


def test_xyxy_to_yolo_normalized() -> None:
    line = xyxy_to_yolo(3, 10, 20, 30, 60, 100, 200)
    parts = line.split()
    assert parts[0] == "3"
    assert abs(float(parts[1]) - 0.2) < 1e-6
    assert abs(float(parts[2]) - 0.2) < 1e-6
    assert abs(float(parts[3]) - 0.2) < 1e-6
    assert abs(float(parts[4]) - 0.2) < 1e-6


def test_sort_label_lines_groups_by_class_then_position() -> None:
    lines = [
        "2 0.800000 0.100000 0.100000 0.100000",
        "1 0.900000 0.200000 0.100000 0.100000",
        "2 0.100000 0.050000 0.100000 0.100000",
        "1 0.100000 0.100000 0.100000 0.100000",
    ]

    assert sort_label_lines(lines, "detect") == [
        "1 0.100000 0.100000 0.100000 0.100000",
        "1 0.900000 0.200000 0.100000 0.100000",
        "2 0.100000 0.050000 0.100000 0.100000",
        "2 0.800000 0.100000 0.100000 0.100000",
    ]


def test_label_task_specs_preserve_line_inference() -> None:
    assert get_label_task_spec("detect").line_value_count == 4
    assert get_label_task_spec("segment").min_points == 3
    assert get_label_task_spec("obb").fixed_points == 4
    assert label_line_to_task("0 0.5 0.5 0.1 0.1") == "detect"
    assert label_line_to_task("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2") == "obb"
    assert label_line_to_task("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2 0.15 0.25") == "segment"
