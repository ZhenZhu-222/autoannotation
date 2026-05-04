# ============================================================
# Qwen 大模型打标服务
# 职责：通过 Qwen-VL 视觉大模型对图片集进行自动标注
#       支持 SVG 输出解析 / 多轮采样融合 / 质量评分
# ============================================================
from __future__ import annotations

import asyncio
import csv
import json
import math
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import yaml

from app.core.config import OUTPUTS_DIR
from app.core.overlay_draw import draw_overlay
from app.core.state import AppState
from app.services.annotation_service import AnnotationService, rebuild_classes_txt, xyxy_to_yolo
from app.services.class_name_service import collect_used_class_stats, read_used_imageset_class_names
from app.services.label_format import normalize_label_task, parse_label_line, points_to_shape, shape_to_overlay_item, sort_label_lines
from app.services.qwen_service import QwenService
from app.services.task_manager import JobContext


# 将标签名标准化（去空格+小写），用于类别聚类匹配
def _normalize_label_key(label: str) -> str:
    text = re.sub(r"\s+", " ", (label or "").strip())
    return text.lower()


# 计算两个 xyxy 检测框的 IoU（交并比）
def _xyxy_iou(box1: list[float], box2: list[float]) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    inter_w = max(0.0, min(x12, x22) - max(x11, x21))
    inter_h = max(0.0, min(y12, y22) - max(y11, y21))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    den = a1 + a2 - inter
    if den <= 0:
        return 0.0
    return inter / den


# 对多轮采样结果按类别+IoU聚类，返回满足投票阈值的共识目标列表和最佳一致率
def _merge_consensus_objects(
    samples: list[list[dict]],
    sample_count: int,
    min_votes: int,
    iou_threshold: float = 0.45,
) -> tuple[list[dict], float]:
    buckets: dict[str, list[dict]] = {}
    for sample_idx, objects in enumerate(samples):
        for obj in objects:
            key = _normalize_label_key(str(obj.get("label") or ""))
            if not key:
                continue
            bbox = obj.get("bbox") or []
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            buckets.setdefault(key, []).append(
                {
                    "sample_idx": sample_idx,
                    "label": str(obj.get("label") or key),
                    "bbox": [float(v) for v in bbox],
                    "points": list(obj.get("points") or []),
                    "shape_type": str(obj.get("shape_type") or "detect"),
                    "confidence": float(obj.get("confidence", 0.0) or 0.0),
                }
            )

    merged: list[dict] = []
    best_vote = 0.0
    for _, entries in buckets.items():
        clusters: list[dict] = []
        for item in entries:
            assigned = False
            for cluster in clusters:
                iou = _xyxy_iou(item["bbox"], cluster["center_bbox"])
                if iou >= iou_threshold:
                    cluster["items"].append(item)
                    xs1 = [x["bbox"][0] for x in cluster["items"]]
                    ys1 = [x["bbox"][1] for x in cluster["items"]]
                    xs2 = [x["bbox"][2] for x in cluster["items"]]
                    ys2 = [x["bbox"][3] for x in cluster["items"]]
                    cluster["center_bbox"] = [
                        float(np.median(xs1)),
                        float(np.median(ys1)),
                        float(np.median(xs2)),
                        float(np.median(ys2)),
                    ]
                    assigned = True
                    break
            if not assigned:
                clusters.append({"center_bbox": item["bbox"][:], "items": [item]})

        for cluster in clusters:
            sample_ids = {int(x["sample_idx"]) for x in cluster["items"]}
            vote = len(sample_ids) / max(1, sample_count)
            best_vote = max(best_vote, vote)
            if len(sample_ids) < max(1, min_votes):
                continue
            xs1 = [x["bbox"][0] for x in cluster["items"]]
            ys1 = [x["bbox"][1] for x in cluster["items"]]
            xs2 = [x["bbox"][2] for x in cluster["items"]]
            ys2 = [x["bbox"][3] for x in cluster["items"]]
            scores = [x["confidence"] for x in cluster["items"]]
            best_item = sorted(cluster["items"], key=lambda x: (-float(x.get("confidence", 0.0)), -len(x.get("points") or [])))[0]
            merged.append(
                {
                    "label": cluster["items"][0]["label"],
                    "bbox": [
                        float(np.median(xs1)),
                        float(np.median(ys1)),
                        float(np.median(xs2)),
                        float(np.median(ys2)),
                    ],
                    "points": list(best_item.get("points") or []),
                    "shape_type": str(best_item.get("shape_type") or "detect"),
                    "confidence": round(float(np.median(scores)) if scores else 0.0, 4),
                    "vote_ratio": round(vote, 4),
                }
            )

    merged.sort(key=lambda x: (x.get("label", ""), -float(x.get("vote_ratio", 0.0))))
    return merged, best_vote


# 去除 YOLO 标签行中 IoU 高度重叠的重复框
def _dedupe_yolo_lines_iou(lines: list[str], label_task: str = "detect", iou_threshold: float = 0.98) -> list[str]:
    effective_task = normalize_label_task(label_task)
    parsed: list[tuple[str, int, float, float, float, float]] = []
    for line in lines:
        item = parse_label_line(line, effective_task)
        if not item:
            continue
        x1, y1, x2, y2 = item.bbox_norm()
        parsed.append((line, item.class_id, x1, y1, x2, y2))

    kept: list[tuple[str, int, float, float, float, float]] = []
    for row in parsed:
        line, cls_id, x1, y1, x2, y2 = row
        dup = False
        for kept_row in kept:
            _, k_cls, kx1, ky1, kx2, ky2 = kept_row
            if cls_id != k_cls:
                continue
            iou = _xyxy_iou([x1, y1, x2, y2], [kx1, ky1, kx2, ky2])
            if iou >= iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(row)
    return [x[0] for x in kept]


# 根据图片尺寸生成四角+中心 5 个校准标记点（像素坐标）
def _build_calibration_markers(width: int, height: int) -> list[dict]:
    safe_w = max(1.0, float(width))
    safe_h = max(1.0, float(height))
    return [
        {"id": "tl", "x": 0.0, "y": 0.0},
        {"id": "tr", "x": safe_w - 1.0, "y": 0.0},
        {"id": "bl", "x": 0.0, "y": safe_h - 1.0},
        {"id": "br", "x": safe_w - 1.0, "y": safe_h - 1.0},
        {"id": "center", "x": (safe_w - 1.0) / 2.0, "y": (safe_h - 1.0) / 2.0},
    ]


# 根据预期与检测到的校准锁点估计仿射变换矩阵，返回矩阵、平均误差和配对数
def _estimate_affine_from_markers(
    expected_markers: list[dict],
    detected_markers: list[dict],
) -> tuple[np.ndarray | None, float, int]:
    expected_by_id = {str(item.get("id")): item for item in expected_markers}
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for item in detected_markers:
        marker_id = str(item.get("id") or "")
        expected = expected_by_id.get(marker_id)
        if not expected:
            continue
        try:
            src = (float(item["x"]), float(item["y"]))
            dst = (float(expected["x"]), float(expected["y"]))
        except (KeyError, TypeError, ValueError):
            continue
        pairs.append((src, dst))
    if len(pairs) < 3:
        return None, float("inf"), len(pairs)

    lhs = []
    rhs = []
    for (sx, sy), (dx, dy) in pairs:
        lhs.append([sx, sy, 1.0, 0.0, 0.0, 0.0])
        lhs.append([0.0, 0.0, 0.0, sx, sy, 1.0])
        rhs.append(dx)
        rhs.append(dy)
    mat, *_ = np.linalg.lstsq(np.asarray(lhs, dtype=float), np.asarray(rhs, dtype=float), rcond=None)
    affine = np.asarray(
        [
            [mat[0], mat[1], mat[2]],
            [mat[3], mat[4], mat[5]],
        ],
        dtype=float,
    )

    errors = []
    for (sx, sy), (dx, dy) in pairs:
        pred_x = affine[0, 0] * sx + affine[0, 1] * sy + affine[0, 2]
        pred_y = affine[1, 0] * sx + affine[1, 1] * sy + affine[1, 2]
        errors.append(math.hypot(pred_x - dx, pred_y - dy))
    err_px = float(np.mean(errors)) if errors else 0.0
    return affine, err_px, len(pairs)


# 将 xyxy 检测框经仿射变换映射到原图坐标，并裁剪到图片范围内
def _transform_bbox(
    matrix: np.ndarray | None,
    bbox: list[float],
    width: int,
    height: int,
) -> list[float] | None:
    if matrix is None or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    points = np.asarray(
        [
            [x1, y1, 1.0],
            [x2, y1, 1.0],
            [x1, y2, 1.0],
            [x2, y2, 1.0],
        ],
        dtype=float,
    )
    transformed = (matrix @ points.T).T
    xs = transformed[:, 0].tolist()
    ys = transformed[:, 1].tolist()
    x1_t = min(max(0.0, min(xs)), float(max(1, width) - 1))
    y1_t = min(max(0.0, min(ys)), float(max(1, height) - 1))
    x2_t = min(max(0.0, max(xs)), float(max(1, width) - 1))
    y2_t = min(max(0.0, max(ys)), float(max(1, height) - 1))
    if x2_t <= x1_t or y2_t <= y1_t:
        return None
    return [x1_t, y1_t, x2_t, y2_t]


# ============================================================
# 仿射纠正（AI 坐标偏移修正）
# 说明：
# - AI 模型在部分图片上会基于内部缩放/偏移后的坐标系返回 bbox / polygon / obb 点位；
# - 这里通过 tl/tr/bl/br/center 五个 calibration_points 拟合仿射矩阵；
# - calibration_points 的前置缩放/letterbox 还原在 qwen_service 完成，这里只处理“残余偏移/仿射修正”；
# - 必须同时纠正 detect / segment / obb，不能只修 bbox，
#   否则 segment/obb 在训练和精修时会出现整体漂移或角点错位；
# - 后续若升级 prompt、切换模型、调整坐标缩放逻辑，请优先检查这一段是否仍然生效。
# ============================================================
# 将 xyxy 检测框转为顺时针四角点列表
def _rect_points_from_bbox(bbox: list[float]) -> list[list[float]]:
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# 从多边形点列表计算包围矩形（xyxy）
def _bbox_from_points(points: list[list[float]] | list[tuple[float, float]]) -> list[float] | None:
    if not points:
        return None
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


# 将多边形点列表经仿射变换映射，并裁剪到图片范围内
def _transform_points(
    matrix: np.ndarray | None,
    points: list[list[float]] | list[tuple[float, float]],
    width: int,
    height: int,
) -> list[list[float]]:
    if matrix is None or not points:
        return []
    try:
        pts = np.asarray([[float(p[0]), float(p[1]), 1.0] for p in points], dtype=float)
    except (TypeError, ValueError, IndexError):
        return []
    transformed = (matrix @ pts.T).T
    safe_w = float(max(1, width) - 1)
    safe_h = float(max(1, height) - 1)
    out: list[list[float]] = []
    for row in transformed:
        out.append([
            min(max(0.0, float(row[0])), safe_w),
            min(max(0.0, float(row[1])), safe_h),
        ])
    return out


# 将任意多边形点计算最小外接旋转矩形（OBB）四角点
def _obb_points_from_points(points: list[list[float]]) -> list[list[float]]:
    if len(points) == 4:
        return [[float(p[0]), float(p[1])] for p in points]
    if len(points) < 3:
        return []
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).tolist()
    return [[float(p[0]), float(p[1])] for p in box]


# 对检测对象列表整体施加仿射校正，返回校正后的对象列表
def _apply_affine_to_objects(
    objects: list[dict],
    matrix: np.ndarray | None,
    width: int,
    height: int,
    label_task: str,
) -> list[dict]:
    if matrix is None or not objects:
        return list(objects)

    effective_task = normalize_label_task(label_task)
    corrected: list[dict] = []
    for obj in objects:
        shape_type = normalize_label_task(obj.get("shape_type"), effective_task)
        bbox = obj.get("bbox") or []
        raw_points = obj.get("points") or []
        transformed_points = _transform_points(matrix, raw_points, width, height)
        transformed_bbox = _transform_bbox(matrix, bbox, width, height) if isinstance(bbox, list) and len(bbox) == 4 else None

        if shape_type == "detect":
            if transformed_bbox is None and transformed_points:
                transformed_bbox = _bbox_from_points(transformed_points)
            if transformed_bbox is None:
                continue
            transformed_points = _rect_points_from_bbox(transformed_bbox)
        elif shape_type == "obb":
            if not transformed_points:
                transformed_points = _transform_points(matrix, _rect_points_from_bbox(bbox), width, height)
            transformed_points = _obb_points_from_points(transformed_points)
            if not transformed_points:
                if transformed_bbox is None:
                    continue
                transformed_points = _rect_points_from_bbox(transformed_bbox)
            transformed_bbox = _bbox_from_points(transformed_points)
            if transformed_bbox is None:
                continue
        else:
            if not transformed_points:
                transformed_points = _transform_points(matrix, _rect_points_from_bbox(bbox), width, height)
            if len(transformed_points) < 3:
                if transformed_bbox is None:
                    continue
                transformed_points = _rect_points_from_bbox(transformed_bbox)
            transformed_bbox = _bbox_from_points(transformed_points)
            if transformed_bbox is None:
                continue

        corrected.append(
            {
                **obj,
                "shape_type": shape_type,
                "bbox": [float(v) for v in transformed_bbox],
                "points": [[float(p[0]), float(p[1])] for p in transformed_points],
            }
        )
    return corrected


# 将 CSS 颜色字符串（十六进制/rgb）解析为 BGR 整数元组
def _parse_color_to_bgr(color: str) -> tuple[int, int, int] | None:
    text = (color or "").strip().lower()
    if not text or text in {"none", "transparent"}:
        return None
    named = {
        "lime": (0, 255, 0),
        "green": (0, 255, 0),
        "#0f0": (0, 255, 0),
        "#00ff00": (0, 255, 0),
    }
    if text in named:
        return named[text]
    if re.fullmatch(r"#[0-9a-f]{6}", text):
        rgb = text[1:]
        return (int(rgb[4:6], 16), int(rgb[2:4], 16), int(rgb[0:2], 16))
    if re.fullmatch(r"#[0-9a-f]{3}", text):
        rgb = "".join(ch * 2 for ch in text[1:])
        return (int(rgb[4:6], 16), int(rgb[2:4], 16), int(rgb[0:2], 16))
    return None


# 安全解析 SVG 属性中的浮点数
def _svg_float(raw: str | None, default: float = 0.0) -> float:
    text = str(raw or "").strip().replace("px", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


# 解析 SVG 叠加层，提取合法形状列表；返回 (shapes, clean_svg, success, reason)
def _sanitize_svg_overlay(
    svg_overlay: str,
    width: int,
    height: int,
    reject_on_invalid_svg: bool = True,
) -> tuple[list[dict], str, bool, str]:
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    svg_text = (svg_overlay or "").strip()
    if not svg_text:
        return [], "", True, ""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        reason = "SVG解析失败"
        return [], "", (not reject_on_invalid_svg), reason

    allowed_tags = {"svg", "rect", "polygon", "polyline", "line", "circle", "ellipse"}
    shapes: list[dict] = []
    reason = ""

    def local_name(tag: str) -> str:
        return tag.split("}", 1)[-1].lower()

    for node in root.iter():
        tag = local_name(node.tag)
        if tag not in allowed_tags:
            reason = f"不允许的SVG标签: {tag}"
            return [], f'<svg xmlns="http://www.w3.org/2000/svg" width="{safe_width}" height="{safe_height}"></svg>', False, reason
        if tag == "svg":
            continue

        stroke = _parse_color_to_bgr(node.attrib.get("stroke", ""))
        fill = _parse_color_to_bgr(node.attrib.get("fill", ""))
        stroke_width = max(1, int(round(_svg_float(node.attrib.get("stroke-width"), 2.0))))
        shape: dict | None = None
        if tag == "rect":
            x = _svg_float(node.attrib.get("x"))
            y = _svg_float(node.attrib.get("y"))
            rect_w = _svg_float(node.attrib.get("width"))
            rect_h = _svg_float(node.attrib.get("height"))
            shape = {"tag": tag, "x": x, "y": y, "width": rect_w, "height": rect_h}
        elif tag in {"polygon", "polyline"}:
            raw_points = re.split(r"[\s,]+", str(node.attrib.get("points") or "").strip())
            coords = [float(x) for x in raw_points if x]
            if len(coords) >= 4 and len(coords) % 2 == 0:
                points = [(int(round(coords[i])), int(round(coords[i + 1]))) for i in range(0, len(coords), 2)]
                shape = {"tag": tag, "points": points}
        elif tag == "line":
            shape = {
                "tag": tag,
                "x1": _svg_float(node.attrib.get("x1")),
                "y1": _svg_float(node.attrib.get("y1")),
                "x2": _svg_float(node.attrib.get("x2")),
                "y2": _svg_float(node.attrib.get("y2")),
            }
        elif tag == "circle":
            shape = {
                "tag": tag,
                "cx": _svg_float(node.attrib.get("cx")),
                "cy": _svg_float(node.attrib.get("cy")),
                "r": _svg_float(node.attrib.get("r")),
            }
        elif tag == "ellipse":
            shape = {
                "tag": tag,
                "cx": _svg_float(node.attrib.get("cx")),
                "cy": _svg_float(node.attrib.get("cy")),
                "rx": _svg_float(node.attrib.get("rx")),
                "ry": _svg_float(node.attrib.get("ry")),
            }
        if shape is None:
            continue
        shape["stroke"] = stroke
        shape["fill"] = fill
        shape["stroke_width"] = stroke_width
        shapes.append(shape)

    root.attrib["width"] = str(safe_width)
    root.attrib["height"] = str(safe_height)
    safe_svg = ET.tostring(root, encoding="unicode")
    return shapes, safe_svg, True, reason


# 将 SVG 形状列表渲染为绿色轮廓画布（供轮廓提取使用）
def _render_green_canvas_from_shapes(shapes: list[dict], width: int, height: int) -> np.ndarray:
    canvas = np.zeros((max(1, int(height)), max(1, int(width)), 3), dtype=np.uint8)
    for shape in shapes:
        stroke = shape.get("stroke")
        fill = shape.get("fill")
        thickness = int(shape.get("stroke_width", 2) or 2)
        tag = str(shape.get("tag") or "")
        if tag == "rect":
            x1 = int(round(float(shape.get("x", 0.0))))
            y1 = int(round(float(shape.get("y", 0.0))))
            x2 = int(round(float(shape.get("x", 0.0)) + float(shape.get("width", 0.0))))
            y2 = int(round(float(shape.get("y", 0.0)) + float(shape.get("height", 0.0))))
            if fill is not None:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, thickness=-1)
            if stroke is not None:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), stroke, thickness=thickness)
        elif tag in {"polygon", "polyline"}:
            pts = np.array(shape.get("points") or [], dtype=np.int32)
            if pts.size <= 0:
                continue
            pts = pts.reshape((-1, 1, 2))
            if fill is not None and tag == "polygon":
                cv2.fillPoly(canvas, [pts], fill)
            if stroke is not None:
                cv2.polylines(canvas, [pts], isClosed=(tag == "polygon"), color=stroke, thickness=thickness)
        elif tag == "line" and stroke is not None:
            p1 = (int(round(float(shape.get("x1", 0.0)))), int(round(float(shape.get("y1", 0.0)))))
            p2 = (int(round(float(shape.get("x2", 0.0)))), int(round(float(shape.get("y2", 0.0)))))
            cv2.line(canvas, p1, p2, stroke, thickness=thickness)
        elif tag == "circle":
            center = (int(round(float(shape.get("cx", 0.0)))), int(round(float(shape.get("cy", 0.0)))))
            radius = max(1, int(round(float(shape.get("r", 0.0)))))
            if fill is not None:
                cv2.circle(canvas, center, radius, fill, thickness=-1)
            if stroke is not None:
                cv2.circle(canvas, center, radius, stroke, thickness=thickness)
        elif tag == "ellipse":
            center = (int(round(float(shape.get("cx", 0.0)))), int(round(float(shape.get("cy", 0.0)))))
            axes = (
                max(1, int(round(float(shape.get("rx", 0.0))))),
                max(1, int(round(float(shape.get("ry", 0.0))))),
            )
            if fill is not None:
                cv2.ellipse(canvas, center, axes, 0, 0, 360, fill, thickness=-1)
            if stroke is not None:
                cv2.ellipse(canvas, center, axes, 0, 0, 360, stroke, thickness=thickness)
    return canvas


# 从绿色画布中提取连通区域的检测框和质量分数
def _extract_green_bboxes(
    green_canvas: np.ndarray,
    width: int,
    height: int,
    hsv_profile: str = "neon",
) -> tuple[list[list[float]], np.ndarray, int, int, float, str]:
    hsv = cv2.cvtColor(green_canvas, cv2.COLOR_BGR2HSV)
    profile = (hsv_profile or "neon").strip().lower()
    if profile == "soft":
        lower = np.array([35, 35, 35], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)
    else:
        lower = np.array([40, 80, 80], dtype=np.uint8)
        upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    green_pixels = int(cv2.countNonZero(mask))
    if green_pixels <= 0:
        return [], mask, 0, 0, 0.0, "SVG颜色不合规：未提取到亮绿色区域"

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[float]] = []
    min_area = max(4.0, float(width * height) * 0.0002)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w_box, h_box = cv2.boundingRect(contour)
        boxes.append([float(x), float(y), float(x + w_box), float(y + h_box)])
    if not boxes:
        return [], mask, green_pixels, len(contours), 0.0, "SVG颜色不合规：未提取到亮绿色区域"
    quality = round(min(1.0, green_pixels / max(1.0, float(width * height) * 0.02)), 4)
    return boxes, mask, green_pixels, len(contours), quality, ""


# 从二值掩码中提取目标对象（按 label_task 决定输出 bbox/polygon/obb）
def _extract_green_objects_from_mask(mask: np.ndarray, width: int, height: int, label_task: str) -> list[dict]:
    effective_task = normalize_label_task(label_task)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: list[dict] = []
    min_area = max(4.0, float(width * height) * 0.0002)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        if effective_task == "detect":
            x, y, w_box, h_box = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(x + w_box), float(y + h_box)]
            points = [[float(x), float(y)], [float(x + w_box), float(y)], [float(x + w_box), float(y + h_box)], [float(x), float(y + h_box)]]
        elif effective_task == "obb":
            rect = cv2.minAreaRect(contour)
            pts = cv2.boxPoints(rect).tolist()
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            points = [[float(p[0]), float(p[1])] for p in pts]
        else:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, max(1.0, 0.01 * peri), True)
            pts = approx.reshape((-1, 2)).tolist()
            if len(pts) < 3:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                pts = [[x, y], [x + w_box, y], [x + w_box, y + h_box], [x, y + h_box]]
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            points = [[float(p[0]), float(p[1])] for p in pts]
        objects.append({"bbox": bbox, "points": points, "shape_type": effective_task, "confidence": 1.0})
    return objects


class QwenAnnotationService:
    # 根据 YOLO 标签行在原图上绘制叠加图并保存
    @staticmethod
    def _overlay_from_yolo_lines(
        src_path: Path,
        out_path: Path,
        yolo_lines: list[str],
        class_id_to_name: dict[int, str],
        label_task: str = "detect",
    ) -> None:
        img = cv2.imread(str(src_path))
        if img is None:
            return
        h, w = img.shape[:2]
        overlay_items: list[dict] = []
        for line in yolo_lines:
            parsed = parse_label_line(line, normalize_label_task(label_task))
            if not parsed:
                continue
            overlay_items.append(shape_to_overlay_item(parsed, w, h, class_id_to_name.get(parsed.class_id, f"class_{parsed.class_id}")))
        draw_overlay(src_path, out_path, overlay_items)

    # 带尺寸重试的 Qwen 检测：若 size_check 失败则重试指定次数
    @staticmethod
    def _detect_with_size_retry(
        *,
        description: str,
        image_name: str,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
        refs: list[tuple[str, bytes, str]],
        qwen_model: str,
        api_key: str,
        label_task: str,
        strict_size_check: bool,
        size_retry: int,
    ) -> dict:
        max_retry = max(0, int(size_retry or 0))
        attempt_desc = description
        last_resp: dict | None = None
        for attempt in range(max_retry + 1):
            resp = asyncio.run(
                QwenService.detect_objects_on_image(
                    description=attempt_desc,
                    image_name=image_name,
                    image_bytes=image_bytes,
                    image_width=image_width,
                    image_height=image_height,
                    reference_images=refs,
                    qwen_model=qwen_model,
                    api_key=api_key,
                    label_task=label_task,
                )
            )
            last_resp = resp
            if bool(resp.get("size_check_passed", False)):
                return resp
            if attempt < max_retry:
                sent_size = list(resp.get("sent_image_size") or [])
                send_w = int(sent_size[0]) if len(sent_size) == 2 else int(image_width)
                send_h = int(sent_size[1]) if len(sent_size) == 2 else int(image_height)
                attempt_desc = (
                    f"{description}\n"
                    f"重要：本次发送给你的目标图尺寸是 {send_w}x{send_h}，"
                    f"原图尺寸是 {image_width}x{image_height}。"
                    "你必须返回与发送尺寸完全一致的 image_size，并使用该发送尺寸下的像素坐标。"
                )

        if strict_size_check:
            size_echo = (last_resp or {}).get("size_echo", [])
            raise RuntimeError(f"尺寸校验失败，期望 {image_width}x{image_height}，返回 {size_echo}")
        return last_resp or {"objects": [], "size_check_passed": False, "size_echo": [], "raw_object_count": 0}

    # 主入口：运行 Qwen 大模型打标任务，将结果写回图片集并生成叠加图
    @staticmethod
    def run_qwen_annotate_job(
        state: AppState,
        job_ctx: JobContext,
        imageset_id: str,
        description: str,
        reference_files: list[str] | None = None,
        qwen_model: str = "",
        api_key: str = "",
        label_task: str = "detect",
        label_mode: str = "append",
        update_imageset_labels: bool = True,
        round_tag: str = "",
        operator: str = "anonymous",
        save_overlays: bool = True,
        strict_size_check: bool = True,
        size_retry: int = 1,
        precision_mode: str = "strict",
        sample_count: int = 3,
        max_calibration_error_px: float = 8.0,
        min_consensus_rate: float = 0.67,
        duplicate_iou: float = 0.98,
        reject_on_invalid_svg: bool = True,
        green_hsv_profile: str = "neon",
    ) -> dict:
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise ValueError("imageset_id 不存在")
        if not imageset.images:
            raise ValueError("图片集为空")
        if label_mode not in {"replace", "append"}:
            raise ValueError("label_mode 仅支持 replace 或 append")
        label_task = normalize_label_task(label_task)
        precision_mode = (precision_mode or "strict").strip().lower()
        if precision_mode not in {"strict", "fast"}:
            raise ValueError("precision_mode 仅支持 strict 或 fast")

        sample_count = max(1, int(sample_count or 1))
        max_calibration_error_px = max(0.0, float(max_calibration_error_px or 0.0))
        min_consensus_rate = min(max(float(min_consensus_rate or 0.67), 0.0), 1.0)
        duplicate_iou = min(max(float(duplicate_iou or 0.98), 0.5), 0.9999)
        reject_on_invalid_svg = bool(reject_on_invalid_svg)
        green_hsv_profile = (green_hsv_profile or "neon").strip().lower() or "neon"
        effective_sample_count_run = sample_count if precision_mode == "strict" else 1

        description = (description or "").strip()
        reference_files = list(reference_files or [])
        if not description and not reference_files:
            raise ValueError("请提供文字描述或参考图片")

        refs: list[tuple[str, bytes, str]] = []
        for p in reference_files:
            path = Path(p)
            if not path.exists() or not path.is_file():
                continue
            refs.append((path.name, path.read_bytes(), ""))

        output_root = OUTPUTS_DIR / job_ctx.job_id
        images_dir = output_root / "images"
        labels_dir = output_root / "labels"
        overlays_dir = output_root / "overlays"
        labels_before_dir = output_root / "labels_before"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        labels_before_dir.mkdir(parents=True, exist_ok=True)
        imageset_labels_dir = Path(imageset.dir_path) / "labels"
        imageset_labels_dir.mkdir(parents=True, exist_ok=True)
        label_task = AnnotationService._ensure_imageset_label_task(imageset, label_task, label_mode, state)
        labels_before_index: dict[str, bool] = {}

        manifest_path = output_root / "manifest.csv"
        headers = [
            "image_id", "source_image", "output_image", "label_file",
            "existing_boxes", "new_boxes", "final_boxes", "label_mode",
            "size_check_passed", "raw_object_count", "consensus_rate",
            "quality_score", "extract_quality_score", "calibration_error_px",
            "svg_parse_ok", "green_pixels", "contour_count", "refine_rounds",
            "svg_overlay_file", "mask_file", "reject_reason", "qwen_note", "status", "error",
        ]

        existing_class_name_map = read_used_imageset_class_names(Path(imageset.dir_path))
        class_id_to_name: dict[int, str] = dict(existing_class_name_map)
        class_key_to_id: dict[str, int] = {
            _normalize_label_key(name): class_id
            for class_id, name in existing_class_name_map.items()
            if str(name or "").strip() and not str(name).startswith("class_")
        }

        # In append mode, scan existing labels to find max class ID in use,
        # and build name→ID map for smart reuse (same name reuses same ID).
        next_class_id = 0
        existing_name_to_id: dict[str, int] = dict(class_key_to_id)
        if label_mode == "append":
            for _img in imageset.images:
                _lp = imageset_labels_dir / f"{Path(_img.filename).stem}.txt"
                for _line in AnnotationService._read_label_lines(_lp):
                    _parsed = parse_label_line(_line, label_task)
                    if _parsed:
                        next_class_id = max(next_class_id, int(_parsed.class_id) + 1)
        for existing_id in existing_class_name_map:
            next_class_id = max(next_class_id, int(existing_id) + 1)
        total = len(imageset.images)
        total_existing_boxes = 0
        total_new_boxes = 0
        total_final_boxes = 0
        size_check_failed_images = 0
        quality_rejected_images = 0
        quality_score_values: list[float] = []
        consensus_rate_values: list[float] = []
        extract_quality_values: list[float] = []
        calibration_error_values: list[float] = []
        svg_parse_failed_images = 0
        used_final_class_name_map: dict[int, str] = {}

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for idx, image in enumerate(imageset.images, start=1):
                job_ctx.raise_if_cancelled()
                source_path = Path(imageset.dir_path) / image.rel_path
                output_image = images_dir / image.filename
                output_label = labels_dir / f"{Path(image.filename).stem}.txt"
                output_overlay = overlays_dir / image.filename
                image_label_path = imageset_labels_dir / f"{Path(image.filename).stem}.txt"

                try:
                    shutil.copy2(source_path, output_image)
                    img = cv2.imread(str(source_path))
                    if img is None:
                        raise RuntimeError(f"无法读取图片: {source_path}")
                    h, w = img.shape[:2]
                    image_bytes = source_path.read_bytes()

                    use_strict_precision = precision_mode == "strict"
                    effective_samples = effective_sample_count_run if use_strict_precision else 1
                    min_votes = max(1, int(math.ceil(effective_samples * min_consensus_rate)))
                    sample_objects: list[list[dict]] = []
                    qwen_notes: list[str] = []
                    size_ok_all = True
                    total_raw_objects = 0
                    svg_parse_ok = 0
                    green_pixels = 0
                    contour_count = 0
                    extract_quality_score = 0.0
                    calibration_error_px = 0.0
                    calibration_sample_errors: list[float] = []
                    refine_rounds = 0
                    svg_overlay_path = output_root / f"{Path(image.filename).stem}.svg"
                    mask_path = output_root / f"{Path(image.filename).stem}_mask.png"
                    safe_svg_text = ""
                    best_mask: np.ndarray | None = None

                    for _sample_idx in range(effective_samples):
                        qwen_resp = QwenAnnotationService._detect_with_size_retry(
                            description=description,
                            image_name=image.filename,
                            image_bytes=image_bytes,
                            image_width=w,
                            image_height=h,
                            refs=refs,
                            qwen_model=qwen_model,
                            api_key=api_key,
                            label_task=label_task,
                            strict_size_check=bool(strict_size_check),
                            size_retry=size_retry,
                        )
                        qwen_notes.append(str(qwen_resp.get("note") or ""))
                        size_ok = bool(qwen_resp.get("size_check_passed", False))
                        size_ok_all = size_ok_all and size_ok
                        if not size_ok:
                            size_check_failed_images += 1

                        objects = list(qwen_resp.get("objects") or [])
                        calibration_points = list(qwen_resp.get("calibration_points") or [])
                        svg_overlay = str(qwen_resp.get("svg_overlay") or "").strip()
                        raw_count = int(qwen_resp.get("raw_object_count", len(objects)))

                        # 仿射纠正主入口：
                        # 这里不能只改 bbox，必须把 segment polygon / obb 四点一起做坐标校正。
                        # 若后续改成别的坐标解析策略，请保留这一步，否则 AI 输出会再次出现“整体偏移”。
                        if calibration_points:
                            affine, sample_calibration_error, used_markers = _estimate_affine_from_markers(
                                _build_calibration_markers(w, h),
                                calibration_points,
                            )
                            if math.isfinite(sample_calibration_error):
                                calibration_sample_errors.append(sample_calibration_error)
                            if affine is not None and used_markers >= 3 and sample_calibration_error <= max_calibration_error_px:
                                objects = _apply_affine_to_objects(objects, affine, w, h, label_task)
                            elif sample_calibration_error > max_calibration_error_px:
                                qwen_notes.append(f"偏移校准超阈值:{sample_calibration_error:.1f}px")

                        if not objects and svg_overlay:
                            shapes, safe_svg, svg_ok, svg_reason = _sanitize_svg_overlay(
                                svg_overlay=svg_overlay,
                                width=w,
                                height=h,
                                reject_on_invalid_svg=reject_on_invalid_svg,
                            )
                            if not svg_ok:
                                svg_parse_failed_images += 1
                                if reject_on_invalid_svg:
                                    raise RuntimeError(svg_reason or "SVG解析失败")
                            else:
                                svg_parse_ok = 1
                                safe_svg_text = safe_svg
                                canvas = _render_green_canvas_from_shapes(shapes, w, h)
                                boxes, mask, gpixels, contours, eqs, extract_reason = _extract_green_bboxes(
                                    green_canvas=canvas,
                                    width=w,
                                    height=h,
                                    hsv_profile=green_hsv_profile,
                                )
                                if extract_reason:
                                    svg_parse_failed_images += 1
                                    if reject_on_invalid_svg:
                                        raise RuntimeError(extract_reason)
                                green_pixels = max(green_pixels, gpixels)
                                contour_count = max(contour_count, contours)
                                if eqs >= extract_quality_score:
                                    extract_quality_score = eqs
                                    best_mask = mask
                                label_name = str(qwen_resp.get("label") or "object").strip() or "object"
                                objects = _extract_green_objects_from_mask(mask, w, h, label_task)
                                for obj in objects:
                                    obj["label"] = label_name
                                raw_count = len(objects)
                                refine_rounds = max(refine_rounds, 1)

                        total_raw_objects += raw_count
                        sample_objects.append(objects)

                    if calibration_sample_errors:
                        calibration_error_px = float(np.mean(calibration_sample_errors))

                    merged_objects, best_vote = _merge_consensus_objects(
                        sample_objects,
                        effective_samples,
                        min_votes if use_strict_precision else 1,
                        iou_threshold=0.45,
                    )

                    consensus_rate = float(best_vote if merged_objects else 0.0)
                    quality_score = round(max(consensus_rate, extract_quality_score), 4)
                    reject_reason = ""

                    if use_strict_precision and consensus_rate < min_consensus_rate and merged_objects:
                        reject_reason = f"一致性不足: {consensus_rate:.2f} < {min_consensus_rate:.2f}"
                    elif not merged_objects and total_raw_objects == 0:
                        reject_reason = "AI 未检测到目标"

                    # Convert to YOLO
                    yolo_lines: list[str] = []
                    for obj in merged_objects:
                        label_name = str(obj.get("label") or "").strip()
                        bbox = obj.get("bbox") or []
                        if not label_name or not isinstance(bbox, list) or len(bbox) != 4:
                            continue
                        key = _normalize_label_key(label_name)
                        if key not in class_key_to_id:
                            # Reuse existing class ID if same name already exists
                            reused_id = existing_name_to_id.get(key)
                            if reused_id is not None:
                                cls_id = reused_id
                            else:
                                cls_id = next_class_id
                                next_class_id += 1
                            class_key_to_id[key] = cls_id
                            class_id_to_name[cls_id] = label_name
                        cls_id = class_key_to_id[key]
                        if label_task == "detect":
                            x1, y1, x2, y2 = [float(v) for v in bbox]
                            yolo_lines.append(xyxy_to_yolo(cls_id, x1, y1, x2, y2, w, h))
                        else:
                            points = obj.get("points") or []
                            if not isinstance(points, list) or not points:
                                points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
                            yolo_lines.append(points_to_shape(cls_id, points, w, h, label_task).format_yolo())
                    yolo_lines = _dedupe_yolo_lines_iou(yolo_lines, label_task=label_task, iou_threshold=duplicate_iou)

                    # Backup + merge with existing labels
                    if image.filename not in labels_before_index:
                        labels_before_index[image.filename] = image_label_path.exists()
                        if image_label_path.exists():
                            shutil.copy2(image_label_path, labels_before_dir / f"{Path(image.filename).stem}.txt")
                    existing_lines = AnnotationService._read_label_lines(image_label_path)
                    if label_mode == "append":
                        final_lines = _dedupe_yolo_lines_iou(
                            AnnotationService._merge_label_lines(existing_lines, yolo_lines),
                            label_task=label_task,
                            iou_threshold=duplicate_iou,
                        )
                    else:
                        final_lines = _dedupe_yolo_lines_iou(yolo_lines, label_task=label_task, iou_threshold=duplicate_iou)
                    final_lines = sort_label_lines(final_lines, label_task)

                    for line in final_lines:
                        parsed = parse_label_line(line, label_task)
                        if parsed is None:
                            continue
                        class_id = parsed.class_id
                        name = class_id_to_name.get(class_id, f"class_{class_id}")
                        used_final_class_name_map[class_id] = name

                    row_status = "ok"
                    row_error = ""
                    if reject_reason:
                        quality_rejected_images += 1
                        row_status = "rejected"
                        row_error = reject_reason
                        final_lines = existing_lines[:]
                        output_label.write_text("\n".join(final_lines) if final_lines else "", encoding="utf-8")
                    else:
                        output_label.write_text("\n".join(final_lines), encoding="utf-8")
                        if update_imageset_labels:
                            image_label_path.write_text("\n".join(final_lines), encoding="utf-8")

                    if safe_svg_text:
                        svg_overlay_path.write_text(safe_svg_text, encoding="utf-8")
                    if best_mask is not None:
                        cv2.imwrite(str(mask_path), best_mask)

                    if save_overlays:
                        QwenAnnotationService._overlay_from_yolo_lines(
                            source_path, output_overlay, final_lines, class_id_to_name, label_task=label_task,
                        )

                    note_text = " | ".join([x for x in qwen_notes if x][:3])
                    writer.writerow({
                        "image_id": image.id,
                        "source_image": str(source_path),
                        "output_image": str(output_image),
                        "label_file": str(output_label),
                        "existing_boxes": len(existing_lines),
                        "new_boxes": len(yolo_lines),
                        "final_boxes": len(final_lines),
                        "label_mode": label_mode,
                        "size_check_passed": int(size_ok_all),
                        "raw_object_count": total_raw_objects,
                        "consensus_rate": round(consensus_rate, 4),
                        "quality_score": quality_score,
                        "extract_quality_score": round(extract_quality_score, 4),
                        "calibration_error_px": round(calibration_error_px, 4),
                        "svg_parse_ok": svg_parse_ok,
                        "green_pixels": green_pixels,
                        "contour_count": contour_count,
                        "refine_rounds": max(refine_rounds, effective_samples if svg_parse_ok else 0),
                        "svg_overlay_file": str(svg_overlay_path) if safe_svg_text else "",
                        "mask_file": str(mask_path) if best_mask is not None else "",
                        "reject_reason": reject_reason,
                        "qwen_note": note_text,
                        "status": row_status,
                        "error": row_error,
                    })

                    total_existing_boxes += len(existing_lines)
                    total_new_boxes += len(yolo_lines)
                    total_final_boxes += len(final_lines)
                    consensus_rate_values.append(consensus_rate)
                    quality_score_values.append(quality_score)
                    extract_quality_values.append(extract_quality_score)
                    calibration_error_values.append(calibration_error_px)
                except Exception as exc:  # noqa: BLE001
                    if "尺寸校验失败" in str(exc):
                        size_check_failed_images += 1
                    output_label.write_text("", encoding="utf-8")
                    writer.writerow({
                        "image_id": image.id,
                        "source_image": str(source_path),
                        "output_image": str(output_image),
                        "label_file": str(output_label),
                        "existing_boxes": 0, "new_boxes": 0, "final_boxes": 0,
                        "label_mode": label_mode,
                        "size_check_passed": 0, "raw_object_count": 0,
                        "consensus_rate": 0.0, "quality_score": 0.0,
                        "extract_quality_score": 0.0,
                        "calibration_error_px": 0.0,
                        "svg_parse_ok": 0,
                        "green_pixels": 0,
                        "contour_count": 0,
                        "refine_rounds": 0,
                        "svg_overlay_file": "",
                        "mask_file": "",
                        "reject_reason": str(exc) or "执行异常",
                        "qwen_note": "",
                        "status": "error", "error": str(exc),
                    })

                job_ctx.set_progress(idx, total, f"AI 打标中: {idx}/{total}")

        # Save indexes and metadata
        labels_before_index_path = output_root / "labels_before_index.json"
        labels_before_index_path.write_text(
            json.dumps({"imageset_id": imageset_id, "job_id": job_ctx.job_id, "files": labels_before_index},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Rebuild classes.txt + data.yaml from actual label files
        class_names = rebuild_classes_txt(
            imageset_labels_dir=imageset_labels_dir,
            imageset_dir=Path(imageset.dir_path),
            new_class_names=used_final_class_name_map,
            label_task=label_task,
        )
        used_ids_after_rebuild = sorted(collect_used_class_stats(Path(imageset.dir_path)).keys())
        final_class_name_map = {
            class_id: class_names[class_id]
            for class_id in used_ids_after_rebuild
            if 0 <= class_id < len(class_names)
        }
        # Also write to job output dir
        classes_txt_path = output_root / "classes.txt"
        classes_txt_path.write_text("\n".join(class_names), encoding="utf-8")
        class_map_path = output_root / "label_map.json"
        class_map_path.write_text(
            json.dumps({
                "id_to_name": {str(i): final_class_name_map[i] for i in sorted(final_class_name_map.keys())},
                "name_to_id": {name: i for i, name in final_class_name_map.items()},
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        zip_path = OUTPUTS_DIR / f"{job_ctx.job_id}.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=output_root)

        final_name_map = {i: final_class_name_map[i] for i in sorted(final_class_name_map.keys())}
        cvat_artifacts = AnnotationService._build_cvat_exports(
            output_root=output_root, images_dir=images_dir, labels_dir=labels_dir,
            model_classes=class_names, selected_ids=list(range(len(class_names))),
            job_id=job_ctx.job_id, final_class_name_map=final_name_map, label_task=label_task,
        )

        avg_quality_score = float(np.mean(quality_score_values)) if quality_score_values else 0.0
        avg_consensus_rate = float(np.mean(consensus_rate_values)) if consensus_rate_values else 0.0
        avg_extract_quality_score = float(np.mean(extract_quality_values)) if extract_quality_values else 0.0
        avg_calibration_error_px = float(np.mean(calibration_error_values)) if calibration_error_values else 0.0

        run_meta_path = output_root / "run_meta.json"
        run_meta_path.write_text(
            json.dumps({
                "job_id": job_ctx.job_id,
                "pipeline": "qwen_zero_shot",
                "pipeline_variant": "qwen_zero_shot_v2",
                "imageset_id": imageset_id,
                "description": description,
                "reference_files": [Path(x).name for x in reference_files],
                "qwen_model": qwen_model,
                "label_task": label_task,
                "label_mode": label_mode,
                "update_imageset_labels": bool(update_imageset_labels),
                "save_overlays": bool(save_overlays),
                "strict_size_check": bool(strict_size_check),
                "size_retry": max(0, int(size_retry or 0)),
                "operator": (operator or "anonymous").strip() or "anonymous",
                "round_tag": (round_tag or "").strip(),
                "class_names": class_names,
                "final_class_name_map": final_class_name_map,
                "size_check_failed_images": size_check_failed_images,
                "precision_mode": precision_mode,
                "sample_count": effective_sample_count_run,
                "max_calibration_error_px": max_calibration_error_px,
                "min_consensus_rate": min_consensus_rate,
                "duplicate_iou": duplicate_iou,
                "quality_rejected_images": quality_rejected_images,
                "svg_parse_failed_images": svg_parse_failed_images,
                "render_mode": "svg" if avg_extract_quality_score > 0 else "bbox",
                "reject_on_invalid_svg": reject_on_invalid_svg,
                "green_hsv_profile": green_hsv_profile,
                "avg_quality_score": round(avg_quality_score, 4),
                "avg_extract_quality_score": round(avg_extract_quality_score, 4),
                "avg_calibration_error_px": round(avg_calibration_error_px, 4),
                "avg_consensus_rate": round(avg_consensus_rate, 4),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {
            "job_id": job_ctx.job_id,
            "pipeline": "qwen_zero_shot",
            "pipeline_variant": "qwen_zero_shot_v2",
            "imageset_id": imageset_id,
            "operator": (operator or "anonymous").strip() or "anonymous",
            "round_tag": (round_tag or "").strip(),
            "label_task": label_task,
            "label_mode": label_mode,
            "update_imageset_labels": bool(update_imageset_labels),
            "total_images": total,
            "total_existing_boxes": total_existing_boxes,
            "total_new_boxes": total_new_boxes,
            "total_final_boxes": total_final_boxes,
            "total_boxes": total_final_boxes,
            "class_count": len(class_names),
            "class_names": class_names,
            "final_class_name_map": final_class_name_map,
            "size_check_failed_images": size_check_failed_images,
            "strict_size_check": bool(strict_size_check),
            "size_retry": max(0, int(size_retry or 0)),
            "output_root": str(output_root),
            "precision_mode": precision_mode,
            "sample_count": effective_sample_count_run,
            "max_calibration_error_px": max_calibration_error_px,
            "min_consensus_rate": min_consensus_rate,
            "duplicate_iou": duplicate_iou,
            "quality_rejected_images": quality_rejected_images,
            "svg_parse_failed_images": svg_parse_failed_images,
            "render_mode": "svg" if avg_extract_quality_score > 0 else "bbox",
            "reject_on_invalid_svg": reject_on_invalid_svg,
            "green_hsv_profile": green_hsv_profile,
            "avg_quality_score": round(avg_quality_score, 4),
            "avg_extract_quality_score": round(avg_extract_quality_score, 4),
            "avg_calibration_error_px": round(avg_calibration_error_px, 4),
            "avg_consensus_rate": round(avg_consensus_rate, 4),
            "artifacts": {
                "manifest_csv": f"/data/outputs/{job_ctx.job_id}/manifest.csv",
                "run_meta_json": f"/data/outputs/{job_ctx.job_id}/run_meta.json",
                "images_dir": f"/data/outputs/{job_ctx.job_id}/images",
                "labels_dir": f"/data/outputs/{job_ctx.job_id}/labels",
                "overlays_dir": f"/data/outputs/{job_ctx.job_id}/overlays",
                "classes_txt": f"/data/outputs/{job_ctx.job_id}/classes.txt",
                "label_map_json": f"/data/outputs/{job_ctx.job_id}/label_map.json",
                "labels_before_dir": f"/data/outputs/{job_ctx.job_id}/labels_before",
                "labels_before_index_json": f"/data/outputs/{job_ctx.job_id}/labels_before_index.json",
                "zip_file": f"/data/outputs/{job_ctx.job_id}.zip",
                "preview_page_url": f"/front/preview.html?job_id={job_ctx.job_id}",
                "rollback_api": f"/api/annotate/jobs/{job_ctx.job_id}/rollback",
                "cleanup_api": f"/api/jobs/{job_ctx.job_id}/artifacts",
                "can_rollback": bool(update_imageset_labels),
                **cvat_artifacts,
            },
        }
