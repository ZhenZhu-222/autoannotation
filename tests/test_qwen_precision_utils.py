from __future__ import annotations

import asyncio
import json
import math
import re

import cv2
import numpy as np

from app.services.qwen_annotation_service import (
    _build_calibration_markers,
    _dedupe_yolo_lines_iou,
    _estimate_affine_from_markers,
    _merge_consensus_objects,
    _transform_bbox,
    _transform_points,
)
from app.services.qwen_service import QwenService


def test_affine_calibration_recovers_scaled_coordinates() -> None:
    markers = _build_calibration_markers(1200, 800)
    # Simulate model coordinates from an internal scaled/shifted space.
    # true mapping(model -> original): x = 2 * x' - 40, y = 2 * y' - 30
    model_points = []
    for m in markers:
        ox = float(m["x"])
        oy = float(m["y"])
        mx = (ox + 40.0) / 2.0
        my = (oy + 30.0) / 2.0
        model_points.append({"id": m["id"], "x": mx, "y": my})

    matrix, err_px, used = _estimate_affine_from_markers(markers, model_points)
    assert used == 5
    assert math.isfinite(err_px)
    assert err_px < 1.0

    corrected = _transform_bbox(matrix, [200.0, 100.0, 350.0, 260.0], 1200, 800)
    assert corrected is not None
    x1, y1, x2, y2 = corrected
    # Expected mapped bbox around [360,170,660,490]
    assert abs(x1 - 360.0) < 3.0
    assert abs(y1 - 170.0) < 3.0
    assert abs(x2 - 660.0) < 3.0
    assert abs(y2 - 490.0) < 3.0


def test_consensus_merge_keeps_stable_boxes_and_drops_outlier() -> None:
    samples = [
        [{"label": "zebra", "bbox": [100, 100, 200, 200], "confidence": 0.9}],
        [{"label": "zebra", "bbox": [102, 101, 201, 199], "confidence": 0.88}],
        [{"label": "zebra", "bbox": [500, 500, 540, 540], "confidence": 0.85}],
    ]
    merged, best_vote = _merge_consensus_objects(samples=samples, sample_count=3, min_votes=2, iou_threshold=0.45)
    assert best_vote >= (2 / 3)
    assert len(merged) == 1
    box = merged[0]["bbox"]
    assert box[0] < 110 and box[1] < 110
    assert box[2] > 190 and box[3] > 190


def test_affine_calibration_transforms_polygon_points_for_segment_and_obb() -> None:
    markers = _build_calibration_markers(1200, 800)
    model_markers = []
    for m in markers:
        ox = float(m["x"])
        oy = float(m["y"])
        model_markers.append({"id": m["id"], "x": (ox + 40.0) / 2.0, "y": (oy + 30.0) / 2.0})

    matrix, err_px, used = _estimate_affine_from_markers(markers, model_markers)
    assert used == 5
    assert err_px < 1.0

    corrected = _transform_points(
        matrix,
        [[200.0, 100.0], [350.0, 100.0], [330.0, 260.0], [210.0, 240.0]],
        1200,
        800,
    )
    assert len(corrected) == 4
    assert abs(corrected[0][0] - 360.0) < 3.0
    assert abs(corrected[0][1] - 170.0) < 3.0
    assert abs(corrected[2][0] - 620.0) < 3.0
    assert abs(corrected[2][1] - 490.0) < 3.0


def test_append_dedupe_by_iou_removes_near_duplicates() -> None:
    lines = [
        "1 0.500000 0.500000 0.300000 0.300000",
        "1 0.500500 0.500500 0.300000 0.300000",
        "2 0.500000 0.500000 0.300000 0.300000",
    ]
    deduped = _dedupe_yolo_lines_iou(lines, iou_threshold=0.98)
    # class 1 should keep one, class 2 is independent
    assert len(deduped) == 2
    assert any(line.startswith("1 ") for line in deduped)
    assert any(line.startswith("2 ") for line in deduped)


def test_qwen_service_keeps_calibration_points_in_same_raw_space(monkeypatch) -> None:
    original_points = [[20.0, 10.0], [84.0, 10.0], [82.0, 60.0], [18.0, 58.0]]

    async def _fake_chat_json(system_prompt: str, user_content: list[dict], model_name: str, key: str) -> str:
        _ = (system_prompt, user_content, model_name, key)
        prompt_text = "\n".join(str(item.get("text") or "") for item in user_content if item.get("type") == "text")
        size_match = re.search(r"width=(\d+), height=(\d+)", prompt_text)
        send_w = int(size_match.group(1)) if size_match else 120
        send_h = int(size_match.group(2)) if size_match else 80
        scale = send_w / 120.0
        resized_h = int(round(80.0 * scale))
        pad_y = (send_h - resized_h) // 2

        def original_to_send(point):
            x, y = point
            return [float(x) * scale, float(y) * scale + pad_y]

        def qwen_shift(point):
            x, y = point
            return [(float(x) + 40.0) / 2.0, (float(y) + 30.0) / 2.0]

        raw_points = [qwen_shift(original_to_send(point)) for point in original_points]
        raw_bbox = [
            min(p[0] for p in raw_points),
            min(p[1] for p in raw_points),
            max(p[0] for p in raw_points),
            max(p[1] for p in raw_points),
        ]
        marker_points = {
            "tl": [0.0, float(pad_y)],
            "tr": [float(send_w - 1), float(pad_y)],
            "bl": [0.0, float(pad_y + resized_h - 1)],
            "br": [float(send_w - 1), float(pad_y + resized_h - 1)],
            "center": [float(send_w - 1) / 2.0, float(pad_y) + (float(resized_h - 1) / 2.0)],
        }
        payload = {
            "image_size": [send_w, send_h],
            "calibration_points": [
                {"id": key, "x": qwen_shift(value)[0], "y": qwen_shift(value)[1]}
                for key, value in marker_points.items()
            ],
            "objects": [
                {
                    "label": "person",
                    "bbox": raw_bbox,
                    "points": raw_points,
                    "confidence": 0.93,
                }
            ],
            "note": "service-affine-raw-space",
        }
        return json.dumps(payload, ensure_ascii=False)

    monkeypatch.setattr("app.services.qwen_service.QwenService._chat_json", _fake_chat_json)

    img = np.zeros((80, 120, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok

    resp = asyncio.run(
        QwenService.detect_objects_on_image(
            image_name="demo.jpg",
            image_bytes=bytes(buf),
            image_width=120,
            image_height=80,
            description="把 person 按多边形标出来",
            reference_images=[],
            qwen_model="qwen-vl-max-latest",
            api_key="dummy",
            label_task="segment",
        )
    )

    send_w, send_h = resp["sent_image_size"]
    scale = send_w / 120.0
    resized_h = int(round(80.0 * scale))
    pad_y = (send_h - resized_h) // 2
    calibration_points = resp["calibration_points"]
    assert len(calibration_points) == 5
    assert abs(float(calibration_points[0]["x"]) - (20.0 / scale)) < 0.2
    assert abs(float(calibration_points[0]["y"]) - 0.0) < 1e-6
    assert abs(float(calibration_points[1]["x"]) - (((send_w - 1.0) + 40.0) / 2.0 / scale)) < 0.2

    object_points = resp["objects"][0]["points"]
    assert len(object_points) == 4
    assert abs(float(object_points[0][0]) - (((20.0 * scale) + 40.0) / 2.0 / scale)) < 0.2
    assert abs(float(object_points[0][1]) - 0.0) < 1e-6
