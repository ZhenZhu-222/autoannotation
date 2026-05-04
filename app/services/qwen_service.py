# ============================================================
# Qwen 视觉大模型服务
# 职责：调用 Qwen-VL API 进行智能类别推荐 / 图片标注
# ============================================================
from __future__ import annotations

import base64
import json
import mimetypes
import re
from typing import Iterable, NamedTuple

import cv2
import httpx
import numpy as np

from app.core.config import (
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_DEFAULT_MODEL,
    QWEN_MAX_REF_IMAGES,
    QWEN_SEND_CANVAS_SIZE,
    QWEN_TIMEOUT_SECONDS,
)
from app.services.label_format import normalize_label_task


# 根据文件名猜测 MIME类型
def _mime_type_from_name(filename: str, fallback: str = "image/jpeg") -> str:
    detected = mimetypes.guess_type(filename)[0]
    return detected or fallback


# 将图片字节编码为 Base64 data URL（用于 Qwen API 上传图片）
def _image_data_url(name: str, content: bytes, content_type: str = "") -> str:
    mime = content_type or _mime_type_from_name(name)
    encoded = base64.b64encode(content).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


class PreparedSendImage(NamedTuple):
    data: bytes
    width: int
    height: int
    resize_scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int


# 从 Qwen 响应内容中提取文本（字符串/列表/字典均支持）
def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    out.append(text.strip())
        return "\n".join(out)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""


# 尝试解析 AI 返回的 JSON，支持纯 JSON 和文本嵌入 JSON 两种格式
def _parse_json_answer(text: str) -> dict:
    stripped = (text or "").strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# 将 xyxy 检测框裁剪到图片范围内，无效框返回 None
def _clip_box(bbox: list[float], width: int, height: int) -> list[float] | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in bbox]
    except (TypeError, ValueError):
        return None
    if width <= 1 or height <= 1:
        return None
    x1 = min(max(0.0, x1), float(width - 1))
    y1 = min(max(0.0, y1), float(height - 1))
    x2 = min(max(0.0, x2), float(width - 1))
    y2 = min(max(0.0, y2), float(height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


# 验证 xyxy 检测框是否合法（x2>x1 且 y2>y1）
def _valid_bbox(bbox: list[float]) -> list[float] | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


# 从多边形点列表计算包围矩形
def _bbox_from_polygon(points, width: int, height: int, clip: bool = True) -> list[float] | None:
    parsed_points = _points_from_polygon(points, width, height, clip=clip)
    if len(parsed_points) < 3:
        return None
    xs = [x for x, _ in parsed_points]
    ys = [y for _, y in parsed_points]
    raw = _valid_bbox([min(xs), min(ys), max(xs), max(ys)])
    if raw is None:
        return None
    if not clip:
        return raw
    return _clip_box(raw, width, height)


# 解析多边形点列表（支持展平、字典、元组多种格式）
def _points_from_polygon(points, width: int, height: int, clip: bool = True) -> list[tuple[float, float]]:
    parsed_points: list[tuple[float, float]] = []
    if not isinstance(points, list):
        return parsed_points

    if points and all(isinstance(x, (int, float)) for x in points):
        if len(points) < 6 or len(points) % 2 != 0:
            return parsed_points
        for i in range(0, len(points), 2):
            point = _parse_point([points[i], points[i + 1]], width, height) if clip else (float(points[i]), float(points[i + 1]))
            if point is not None:
                parsed_points.append(point)
        return parsed_points

    for point in points:
        parsed = _parse_point(point, width, height) if clip else None
        if parsed is not None:
            parsed_points.append(parsed)
            continue
        if not clip and isinstance(point, dict):
            try:
                parsed_points.append((float(point["x"]), float(point["y"])))
            except (KeyError, TypeError, ValueError):
                continue
        elif not clip and isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                parsed_points.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return parsed_points


# 从任意层级的 payload 中提取 xyxy 检测框（支持 bbox/xyxy/xywh/多边形 key）
def _bbox_from_any(payload, width: int, height: int, clip: bool = True) -> list[float] | None:
    if isinstance(payload, list):
        if len(payload) == 4 and all(isinstance(x, (int, float)) for x in payload):
            raw = _valid_bbox([float(x) for x in payload])
            if raw is None:
                return None
            if not clip:
                return raw
            return _clip_box(raw, width, height)
        return _bbox_from_polygon(payload, width, height, clip=clip)

    if not isinstance(payload, dict):
        return None

    if "bbox" in payload:
        return _bbox_from_any(payload.get("bbox"), width, height, clip=clip)

    # xyxy object
    if all(k in payload for k in ("x1", "y1", "x2", "y2")):
        try:
            raw = _valid_bbox(
                [
                    float(payload["x1"]),
                    float(payload["y1"]),
                    float(payload["x2"]),
                    float(payload["y2"]),
                ]
            )
            if raw is None:
                return None
            if not clip:
                return raw
            return _clip_box(raw, width, height)
        except (TypeError, ValueError):
            return None

    # xywh object
    if all(k in payload for k in ("x", "y", "w", "h")):
        try:
            x = float(payload["x"])
            y = float(payload["y"])
            w = float(payload["w"])
            h = float(payload["h"])
        except (TypeError, ValueError):
            return None
        raw = _valid_bbox([x, y, x + w, y + h])
        if raw is None:
            return None
        if not clip:
            return raw
        return _clip_box(raw, width, height)

    for key in ("polygon", "points", "segmentation", "mask"):
        if key in payload:
            box = _bbox_from_any(payload[key], width, height, clip=clip)
            if box is not None:
                return box
    return None


# 将 xyxy 检测框转换为公顺时针四角点列表
def _rect_points_from_bbox(bbox: list[float]) -> list[list[float]]:
    if len(bbox) != 4:
        return []
    x1, y1, x2, y2 = [float(x) for x in bbox]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


# 从多边形点计算最小外接旋转矩形（OBB）四角点
def _obb_from_points(points: list[tuple[float, float]]) -> list[list[float]]:
    if len(points) == 4:
        return [[float(x), float(y)] for x, y in points]
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    if contour.shape[0] < 3:
        return []
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return [[float(p[0]), float(p[1])] for p in box.tolist()]


# 从任意 payload 提取多边形点列表，无多边形时降级为检测框四角
def _points_from_any(payload, width: int, height: int, clip: bool = True) -> list[list[float]]:
    if isinstance(payload, list):
        points = _points_from_polygon(payload, width, height, clip=clip)
        return [[float(x), float(y)] for x, y in points]

    if not isinstance(payload, dict):
        return []

    for key in ("points", "polygon", "segmentation", "mask"):
        if key in payload:
            points = _points_from_any(payload[key], width, height, clip=clip)
            if points:
                return points

    bbox = _bbox_from_any(payload, width, height, clip=clip)
    return _rect_points_from_bbox(bbox or [])


# 解析校准锁点列表（支持内嵌字典和列表格式）
def _parse_calibration_points(payload, width: int, height: int, clip: bool = True) -> list[dict]:
    items = payload
    if isinstance(items, dict):
        items = items.get("calibration_points") or items.get("markers") or items.get("points") or []
    if not isinstance(items, list):
        return []

    parsed: list[dict] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        marker_id = str(item.get("id") or item.get("name") or item.get("key") or "").strip().lower()
        if not marker_id or marker_id in seen_ids:
            continue
        if clip:
            point = _parse_point(item, width, height)
        else:
            point = None
            try:
                if "xy" in item:
                    raw_xy = item.get("xy") or []
                    point = (float(raw_xy[0]), float(raw_xy[1]))
                else:
                    point = (float(item["x"]), float(item["y"]))
            except (KeyError, TypeError, ValueError, IndexError):
                point = None
        if point is None:
            continue
        parsed.append({"id": marker_id, "x": float(point[0]), "y": float(point[1])})
        seen_ids.add(marker_id)
    return parsed


# 解析单个坐标点（字典/列表/元组均支持），自动裁剪到图片范围
def _parse_point(payload, width: int, height: int) -> tuple[float, float] | None:
    if isinstance(payload, dict):
        if "xy" in payload:
            return _parse_point(payload.get("xy"), width, height)
        if all(k in payload for k in ("x", "y")):
            try:
                x = float(payload["x"])
                y = float(payload["y"])
            except (TypeError, ValueError):
                return None
            x = min(max(0.0, x), float(width - 1))
            y = min(max(0.0, y), float(height - 1))
            return x, y
        return None

    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        try:
            x = float(payload[0])
            y = float(payload[1])
        except (TypeError, ValueError):
            return None
        x = min(max(0.0, x), float(width - 1))
        y = min(max(0.0, y), float(height - 1))
        return x, y
    return None


# 将对象列表中所有 bbox 和 points 按缩放比例缩放
def _scale_bboxes(objects: list[dict], sx: float, sy: float) -> list[dict]:
    for obj in objects:
        b = obj.get("bbox")
        if isinstance(b, list) and len(b) == 4:
            obj["bbox"] = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        points = obj.get("points")
        if isinstance(points, list):
            scaled = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    scaled.append([float(point[0]) * sx, float(point[1]) * sy])
            obj["points"] = scaled
    return objects


# 将校准锁点按比例缩放
def _scale_calibration_points(points: list[dict], sx: float, sy: float) -> list[dict]:
    for item in points:
        try:
            item["x"] = float(item.get("x", 0.0)) * sx
            item["y"] = float(item.get("y", 0.0)) * sy
        except (TypeError, ValueError):
            continue
    return points


# 将校准锁点转换为像素坐标（支持归一化/百分比/1000-scale 等多种格式）
def _denormalize_calibration_points(
    points: list[dict], width: int, height: int, size_echo: list | None = None,
) -> list[dict]:
    if not points:
        return points

    all_coords: list[float] = []
    for item in points:
        try:
            all_coords.extend([abs(float(item.get("x", 0.0))), abs(float(item.get("y", 0.0)))])
        except (TypeError, ValueError):
            continue
    if not all_coords:
        return points
    max_coord = max(all_coords)

    if max_coord <= 1.0:
        return _scale_calibration_points(points, float(width), float(height))

    if isinstance(size_echo, list) and len(size_echo) == 2:
        try:
            echo_w, echo_h = float(size_echo[0]), float(size_echo[1])
            if echo_w > 0 and echo_h > 0:
                rw, rh = width / echo_w, height / echo_h
                if abs(rw - 1.0) > 0.05 or abs(rh - 1.0) > 0.05:
                    return _scale_calibration_points(points, rw, rh)
        except (TypeError, ValueError):
            pass

    # NOTE:
    # calibration_points 必须和 objects 保持“同一原始坐标系”，后续仿射纠正才有意义。
    # 这里不能像历史逻辑那样根据 marker 的最大坐标去猜测性放大，
    # 否则 marker 会被提前拉回原图尺寸，而 polygon / obb 仍停留在模型内部坐标系，
    # 最终会出现“detect 看着还行，但 segment/obb 整体漂移”的问题。
    #
    # 允许的缩放只保留三类明确语义：
    # 1) 0~1 归一化坐标；
    # 2) image_size 明确与真实发送尺寸不同；
    # 3) 百分比 / 1000-scale 这类常见格式。
    #
    # 只要 calibration_points 已经是像素坐标（即便是内部缩放后的像素坐标），
    # 就必须原样保留，交给 qwen_annotation_service 的仿射拟合去恢复。
    max_dim = max(width, height)
    if max_coord > 1.5 and max_coord <= max_dim * 1.1:
        return points

    if max_coord <= 100.0 and max_dim > 100:
        return _scale_calibration_points(points, width / 100.0, height / 100.0)

    if max_coord <= 1000.0:
        return _scale_calibration_points(points, width / 1000.0, height / 1000.0)

    return points


# 将发送画布的单点坐标反向映射回原始图坐标系
def _map_send_point_to_original(
    x: float,
    y: float,
    prepared: PreparedSendImage,
    original_width: int,
    original_height: int,
) -> tuple[float, float]:
    scale = max(float(prepared.resize_scale), 1e-9)
    ox = (float(x) - float(prepared.pad_left)) / scale
    oy = (float(y) - float(prepared.pad_top)) / scale
    ox = min(max(0.0, ox), float(max(1, original_width) - 1))
    oy = min(max(0.0, oy), float(max(1, original_height) - 1))
    return ox, oy


# 将发送画布的点列表反向映射回原始图坐标系
def _map_send_points_to_original(
    points: list,
    prepared: PreparedSendImage,
    original_width: int,
    original_height: int,
) -> list[list[float]]:
    mapped: list[list[float]] = []
    for point in points or []:
        try:
            if isinstance(point, dict):
                x, y = float(point["x"]), float(point["y"])
            else:
                x, y = float(point[0]), float(point[1])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        ox, oy = _map_send_point_to_original(x, y, prepared, original_width, original_height)
        mapped.append([ox, oy])
    return mapped


# 将发送画布的 xyxy 检测框反向映射回原始图坐标系
def _map_send_bbox_to_original(
    bbox: list[float],
    prepared: PreparedSendImage,
    original_width: int,
    original_height: int,
) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    corners = [
        _map_send_point_to_original(x1, y1, prepared, original_width, original_height),
        _map_send_point_to_original(x2, y1, prepared, original_width, original_height),
        _map_send_point_to_original(x2, y2, prepared, original_width, original_height),
        _map_send_point_to_original(x1, y2, prepared, original_width, original_height),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    mapped = [min(xs), min(ys), max(xs), max(ys)]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        return None
    return mapped


# 将发送画布的全部对象反向映射回原始图坐标系
def _map_send_objects_to_original(
    objects: list[dict],
    prepared: PreparedSendImage,
    original_width: int,
    original_height: int,
) -> list[dict]:
    mapped_objects: list[dict] = []
    for obj in objects:
        bbox = _map_send_bbox_to_original(obj.get("bbox") or [], prepared, original_width, original_height)
        points = _map_send_points_to_original(obj.get("points") or [], prepared, original_width, original_height)
        if bbox is None and points:
            bbox = _bbox_from_polygon(points, original_width, original_height, clip=True)
        if bbox is None:
            continue
        mapped_objects.append({**obj, "bbox": bbox, "points": points})
    return mapped_objects


# 将发送画布的校准锁点反向映射回原始图坐标系
def _map_send_calibration_to_original(
    points: list[dict],
    prepared: PreparedSendImage,
    original_width: int,
    original_height: int,
) -> list[dict]:
    mapped: list[dict] = []
    for item in points:
        try:
            marker_id = str(item.get("id") or "").strip()
            x, y = float(item["x"]), float(item["y"])
        except (KeyError, TypeError, ValueError):
            continue
        ox, oy = _map_send_point_to_original(x, y, prepared, original_width, original_height)
        mapped.append({"id": marker_id, "x": ox, "y": oy})
    return mapped


def _denormalize_objects(
    objects: list[dict], width: int, height: int, size_echo: list | None = None,
) -> list[dict]:
    """Convert bbox coordinates to pixel coordinates.

    We ask Qwen for normalized 0-1 coordinates in the prompt. This function handles
    that case plus fallbacks for when Qwen returns other coordinate systems.
    """
    if not objects:
        return objects

    all_coords: list[float] = []
    for obj in objects:
        bbox = obj.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            all_coords.extend(abs(float(c)) for c in bbox)
    if not all_coords:
        return objects
    max_coord = max(all_coords)

    # Case 1: Normalized 0-1 (this is what we ask for in the prompt)
    if max_coord <= 1.0:
        return _scale_bboxes(objects, float(width), float(height))

    # Case 2: size_echo differs from real size — use it as scale reference
    if isinstance(size_echo, list) and len(size_echo) == 2:
        try:
            echo_w, echo_h = float(size_echo[0]), float(size_echo[1])
            if echo_w > 0 and echo_h > 0:
                rw, rh = width / echo_w, height / echo_h
                if abs(rw - 1.0) > 0.05 or abs(rh - 1.0) > 0.05:
                    return _scale_bboxes(objects, rw, rh)
        except (TypeError, ValueError):
            pass

    max_dim = max(width, height)

    # Case 3: Coords look like valid pixel coords for this image
    if max_coord > 1.5 and max_coord <= max_dim * 1.1:
        # Extra safety: if image is large (>1500px) but all coords are suspiciously
        # small (<45% of image), Qwen likely returned internal-resolution coords.
        # Detect and scale up.
        if max_dim > 1500 and max_coord < max_dim * 0.45:
            max_x = max(abs(float(obj.get("bbox", [0])[0])) for obj in objects if isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4)
            max_x = max(max_x, max(abs(float(obj.get("bbox", [0, 0, 0])[2])) for obj in objects if isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4))
            max_y = max(abs(float(obj.get("bbox", [0, 0])[1])) for obj in objects if isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4)
            max_y = max(max_y, max(abs(float(obj.get("bbox", [0, 0, 0, 0])[3])) for obj in objects if isinstance(obj.get("bbox"), list) and len(obj["bbox"]) == 4))
            if max_x > 0 and max_y > 0:
                # Estimate scale: add 5% buffer since max coord isn't necessarily at image edge
                sx = width / (max_x * 1.05)
                sy = height / (max_y * 1.05)
                # Only apply if both scales are similar (indicating uniform downscale)
                if sx > 1.5 and sy > 1.5 and abs(sx - sy) / max(sx, sy) < 0.15:
                    return _scale_bboxes(objects, sx, sy)
        return objects

    # Case 4: Percentage 0-100
    if max_coord <= 100.0 and max_dim > 100:
        return _scale_bboxes(objects, width / 100.0, height / 100.0)

    # Case 5: 1000-scale (Qwen-VL grounding format)
    if max_coord <= 1000.0:
        return _scale_bboxes(objects, width / 1000.0, height / 1000.0)

    return objects


class QwenService:
    # AI 返回空类别时，尝试从文字描述中关键词匹配类别
    @staticmethod
    def _fallback_from_text(description: str, class_names: list[str]) -> list[int]:
        desc = (description or "").lower()
        if not desc:
            return []
        tokens = [x for x in re.split(r"[\s,，。;；:/]+", desc) if len(x) >= 2]
        if not tokens:
            return []
        picked = []
        for idx, name in enumerate(class_names):
            low = name.lower()
            if any(t in low or low in t for t in tokens):
                picked.append(idx)
        return picked[:8]

    # 主入口：调用 Qwen 根据文字描述和参考图片建议应选类别 ID
    @staticmethod
    async def suggest_classes(
        description: str,
        class_names: list[str],
        images: Iterable[tuple[str, bytes, str]] | None = None,
        qwen_model: str = "",
        api_key: str = "",
    ) -> dict:
        key = (api_key or "").strip() or QWEN_API_KEY
        if not key:
            raise ValueError("未配置 AI API Key，请在页面填写或设置对应环境变量")
        if not class_names:
            raise ValueError("当前模型没有可用类别")

        model_name = (qwen_model or "").strip() or QWEN_DEFAULT_MODEL
        images = list(images or [])[:QWEN_MAX_REF_IMAGES]

        class_lines = "\n".join(f"{i}: {name}" for i, name in enumerate(class_names))
        system_prompt = (
            "你是自动打标系统的类别选择助手。"
            "你必须只从候选类别中选择可能需要打标的类别ID，并返回严格JSON。"
        )
        user_content = [
            {
                "type": "text",
                "text": (
                    "任务: 根据用户描述和参考图片，选择需要保留的类别ID。\n"
                    f"用户描述: {description or '（无文字描述，仅参考图片）'}\n"
                    "候选类别:\n"
                    f"{class_lines}\n"
                    "输出格式要求: 只返回JSON对象，格式为 "
                    '{"class_ids":[int,...],"reason":"简短原因"}。'
                ),
            }
        ]
        for name, data, content_type in images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(name, data, content_type)},
                }
            )

        text = await QwenService._chat_json(system_prompt, user_content, model_name, key)
        parsed = _parse_json_answer(text)

        raw_ids = parsed.get("class_ids", [])
        selected_ids: list[int] = []
        if isinstance(raw_ids, list):
            for item in raw_ids:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(class_names):
                    selected_ids.append(idx)
        selected_ids = sorted(set(selected_ids))
        if not selected_ids:
            selected_ids = QwenService._fallback_from_text(description, class_names)

        selected_classes = [{"id": i, "name": class_names[i]} for i in selected_ids]
        return {
            "selected_class_ids": selected_ids,
            "selected_classes": selected_classes,
            "reason": str(parsed.get("reason") or "").strip() or "AI 已返回类别建议",
            "provider_model": model_name,
            "image_count": len(images),
            "raw_text": text,
        }

    # 调用 Qwen JSON 模式对话，返回模型回复文本
    @staticmethod
    async def _chat_json(system_prompt: str, user_content: list[dict], model_name: str, key: str) -> str:
        payload = {
            "model": model_name,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        url = f"{QWEN_API_BASE.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=QWEN_TIMEOUT_SECONDS) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise ValueError(f"AI 请求失败: HTTP {resp.status_code} {resp.text[:200]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("AI 返回为空")
        text = _extract_text(choices[0].get("message", {}).get("content"))
        if not text.strip():
            raise ValueError("AI 返回内容为空")
        return text

    # 将图片缩放并内嵌入固定正方画布（letterbox），保证 Qwen 坐标稳定
    @staticmethod
    def _resize_image_bytes(image_bytes: bytes, canvas_size: int = QWEN_SEND_CANVAS_SIZE) -> PreparedSendImage:
        """Letterbox image into a fixed square canvas for stable Qwen coordinates."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码图片")
        h, w = img.shape[:2]
        safe_canvas = max(256, int(canvas_size or QWEN_SEND_CANVAS_SIZE))
        scale = safe_canvas / max(1, max(w, h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        new_w = min(max(1, new_w), safe_canvas)
        new_h = min(max(1, new_h), safe_canvas)
        interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        canvas = np.full((safe_canvas, safe_canvas, 3), 114, dtype=np.uint8)
        pad_left = (safe_canvas - new_w) // 2
        pad_top = (safe_canvas - new_h) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise ValueError("图片编码失败")
        return PreparedSendImage(
            data=buf.tobytes(),
            width=safe_canvas,
            height=safe_canvas,
            resize_scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            resized_width=new_w,
            resized_height=new_h,
        )

    # 主入口：调用 Qwen 对单张图片进行目标检测，返回检测框+校准锁点+原始文本
    @staticmethod
    async def detect_objects_on_image(
        description: str,
        image_name: str,
        image_bytes: bytes,
        image_width: int,
        image_height: int,
        reference_images: Iterable[tuple[str, bytes, str]] | None = None,
        qwen_model: str = "",
        api_key: str = "",
        focus_hint: str = "",
        label_task: str = "detect",
    ) -> dict:
        key = (api_key or "").strip() or QWEN_API_KEY
        if not key:
            raise ValueError("未配置 AI API Key，请在页面填写或设置对应环境变量")
        model_name = (qwen_model or "").strip() or QWEN_DEFAULT_MODEL
        refs = list(reference_images or [])[:QWEN_MAX_REF_IMAGES]

        # ============================================================
        # !!! AI 坐标偏移/缩放修正关键点，后续升级不要删 !!!
        # Qwen 只看到固定 1280x1280（默认）letterbox 画布：
        # 1) prompt 中所有 bbox / polygon / obb / calibration_points 都基于固定画布；
        # 2) 后端先反向 letterbox 映射回原图比例；
        # 3) qwen_annotation_service 再用 calibration_points 做残余仿射纠正。
        # AABB / OBB / SEG 必须共用这条链路，否则不同图片尺寸会再次产生整体偏移。
        # ============================================================
        prepared_image = QwenService._resize_image_bytes(image_bytes)
        send_bytes, send_w, send_h = prepared_image.data, prepared_image.width, prepared_image.height
        task = normalize_label_task(label_task)
        content_x1 = float(prepared_image.pad_left)
        content_y1 = float(prepared_image.pad_top)
        content_x2 = float(prepared_image.pad_left + max(1, prepared_image.resized_width) - 1)
        content_y2 = float(prepared_image.pad_top + max(1, prepared_image.resized_height) - 1)
        calibration_markers = [
            {"id": "tl", "x": content_x1, "y": content_y1},
            {"id": "tr", "x": content_x2, "y": content_y1},
            {"id": "bl", "x": content_x1, "y": content_y2},
            {"id": "br", "x": content_x2, "y": content_y2},
            {"id": "center", "x": (content_x1 + content_x2) / 2.0, "y": (content_y1 + content_y2) / 2.0},
        ]
        calibration_json = json.dumps(calibration_markers, ensure_ascii=False)

        system_prompt = (
            "你是专业视觉标注员。根据任务描述，检测目标图中所有目标物体，"
            "直接输出每个目标的标签和像素坐标。返回严格JSON。"
        )
        hint = (focus_hint or "").strip()
        hint_line = f"\n{hint}" if hint else ""
        user_content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "任务说明:\n"
                    f"{description or '请尽可能标出图中主要目标'}\n"
                    "你会先看到0~N张参考图，再看到1张目标图。\n"
                    "仅在目标图上检测目标物体。\n"
                    f"目标图尺寸: width={send_w}, height={send_h}\n"
                    f"{hint_line}\n"
                    "输出要求: 只返回 JSON 对象，格式如下：\n"
                    '{"image_size":[width,height],'
                    '"calibration_points":[{"id":"tl","x":0,"y":0},...],'
                    '"objects":[{"label":"类别名","bbox":[x1,y1,x2,y2],"points":[[x,y],...],"confidence":0.9}]}\n'
                    "坐标规则:\n"
                    f"1) 所有坐标都必须基于目标图尺寸 {send_w}x{send_h} 的像素坐标\n"
                    f"2) 坐标不得超出图片边界 (0,0) 到 ({send_w},{send_h})\n"
                    f"3) 当前输出任务类型是 {task}\n"
                    f"4) detect: 填 bbox；segment: 填 points(>=3)；obb: 填 points(恰好4个点，按顺时针或逆时针)\n"
                    "5) calibration_points 必须返回这 5 个锚点的坐标，id 固定为 tl/tr/bl/br/center：\n"
                    f"{calibration_json}\n"
                    "6) confidence 是你对检测结果的置信度(0-1)\n"
                    '7) 若无目标，仍需返回 calibration_points，并将 objects 设为空列表: {"image_size":[w,h],"calibration_points":[...],"objects":[]}'
                ),
            }
        ]

        for idx, (name, data, content_type) in enumerate(refs, start=1):
            user_content.append({"type": "text", "text": f"参考样例图 {idx}: {name}"})
            user_content.append(
                {"type": "image_url", "image_url": {"url": _image_data_url(name, data, content_type)}}
            )

        user_content.append({"type": "text", "text": f"目标图: {image_name}"})
        user_content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_name, send_bytes, "")}})

        text = await QwenService._chat_json(system_prompt, user_content, model_name, key)
        parsed = _parse_json_answer(text)

        # Size check — compare against the SENT size, not original
        size_echo = parsed.get("image_size")
        size_check_passed = False
        if isinstance(size_echo, list) and len(size_echo) == 2:
            try:
                size_check_passed = int(size_echo[0]) == send_w and int(size_echo[1]) == send_h
            except (TypeError, ValueError):
                size_check_passed = False

        calibration_points = _parse_calibration_points(parsed.get("calibration_points"), send_w, send_h, clip=False)
        calibration_points = _denormalize_calibration_points(calibration_points, send_w, send_h, size_echo=size_echo)

        # Parse objects — bbox coords are in send_w x send_h pixel space
        raw_objects = parsed.get("objects", [])
        objects: list[dict] = []
        if isinstance(raw_objects, list):
            for item in raw_objects:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                if not label:
                    continue
                bbox = _bbox_from_any(item, send_w, send_h, clip=False)
                if bbox is None:
                    continue
                points = _points_from_any(item, send_w, send_h, clip=False)
                if task == "segment" and len(points) < 3:
                    points = _rect_points_from_bbox(bbox)
                if task == "obb":
                    parsed_points = [(float(p[0]), float(p[1])) for p in points]
                    points = _obb_from_points(parsed_points) if parsed_points else _rect_points_from_bbox(bbox)
                if task == "detect":
                    points = _rect_points_from_bbox(bbox)
                conf = 0.0
                try:
                    conf = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
                except (TypeError, ValueError):
                    pass
                objects.append({
                    "label": label,
                    "bbox": bbox,
                    "points": points,
                    "shape_type": task,
                    "confidence": round(conf, 4),
                })

        # Denormalize coords to send_w x send_h pixel space
        objects = _denormalize_objects(objects, send_w, send_h, size_echo=size_echo)

        # Map from fixed send canvas back to ORIGINAL image size.
        # Keep calibration_points and object points in the same coordinate space;
        # qwen_annotation_service will then apply the residual affine correction.
        objects = _map_send_objects_to_original(objects, prepared_image, image_width, image_height)
        calibration_points = _map_send_calibration_to_original(
            calibration_points,
            prepared_image,
            image_width,
            image_height,
        )

        # Clip to original image bounds
        clipped: list[dict] = []
        for obj in objects:
            box = _clip_box(obj["bbox"], image_width, image_height)
            if box is not None:
                obj["bbox"] = box
                clipped_points = _points_from_any({"points": obj.get("points", [])}, image_width, image_height, clip=True)
                if task == "segment" and len(clipped_points) < 3:
                    clipped_points = _rect_points_from_bbox(box)
                if task == "obb":
                    clipped_points = _obb_from_points([(float(p[0]), float(p[1])) for p in clipped_points]) if clipped_points else _rect_points_from_bbox(box)
                obj["points"] = clipped_points
                clipped.append(obj)
        objects = clipped
        calibration_points = _parse_calibration_points(calibration_points, image_width, image_height, clip=True)

        label_text = str(parsed.get("label") or "").strip()
        if not label_text and objects:
            label_text = objects[0]["label"]

        return {
            "objects": objects,
            "label": label_text,
            "note": str(parsed.get("note") or "").strip(),
            "provider_model": model_name,
            "raw_text": text,
            "size_check_passed": size_check_passed,
            "size_echo": size_echo if isinstance(size_echo, list) else [],
            "sent_image_size": [send_w, send_h],
            "calibration_points": calibration_points,
            "raw_object_count": len(raw_objects) if isinstance(raw_objects, list) else 0,
        }
