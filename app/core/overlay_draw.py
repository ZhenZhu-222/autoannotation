# ============================================================
# 图片叠加绘制模块
# 职责：在图片上绘制检测框 + 标签文字
# 思路：
#   1. 优先用 Pillow 绘制（支持中文字体）
#   2. Pillow 不可用时回退到 OpenCV（不支持中文）
# ============================================================
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # noqa: BLE001
    Image = None
    ImageDraw = None
    ImageFont = None


# ---------- 坐标工具 ----------
# 将坐标限制在图片范围内，返回整数像素坐标
def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    xi = int(max(0, min(width - 1, round(float(x)))))
    yi = int(max(0, min(height - 1, round(float(y)))))
    return xi, yi


# ---------- 字体加载：按优先级搜索系统中文字体 ----------
# 返回字体文件候选列表（自定义 > 项目内置 > 系统中文字体）
def _font_candidates() -> list[str]:
    custom_path = os.environ.get("AUTOANNOTATION_FONT_PATH", "").strip()
    candidates = []
    if custom_path:
        candidates.append(custom_path)
    candidates.extend(
        [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]
    )
    return candidates


# 加载字体（带 LRU 缓存），按优先级搜索系统中文字体
@lru_cache(maxsize=6)
def _load_font(size: int):
    if ImageFont is None:
        return None
    for font_path in _font_candidates():
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size=size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


# ---------- 类别色板：固定 3 色轮换（和前端精修页保持一致） ----------
_CLASS_PALETTE_BGR = [
    (235, 99, 37),    # blue-ish (matches frontend #2563EB)
    (129, 185, 16),   # green-ish (matches frontend #10B981)
    (153, 72, 236),   # pink-ish (matches frontend #EC4899)
]


# 根据标签名自动分配颜色（相同标签始终得到相同颜色）
def _color_for_label(label: str, label_color_map: dict[str, int]) -> tuple[int, int, int]:
    key = str(label or "").strip() or "__default__"
    if key not in label_color_map:
        seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(key))
        label_color_map[key] = seed % len(_CLASS_PALETTE_BGR)
    idx = label_color_map[key] % len(_CLASS_PALETTE_BGR)
    return _CLASS_PALETTE_BGR[idx]


# ---------- OpenCV 绘制（回退方案，不支持中文） ----------
# 用 OpenCV 绘制检测框（回退方案，不支持中文字体）
def _draw_with_opencv(
    image: np.ndarray,
    boxes: list[dict],
    *,
    box_color_bgr: tuple[int, int, int] | None,
    text_color_bgr: tuple[int, int, int],
    line_width: int,
    font_scale: float,
    text_thickness: int,
) -> np.ndarray:
    out = image.copy()
    label_color_map: dict[str, int] = {}
    for item in boxes:
        x1 = int(item["x1"])
        y1 = int(item["y1"])
        x2 = int(item["x2"])
        y2 = int(item["y2"])
        label = str(item.get("label", "")).strip()
        color_key = str(item.get("color_key") or label).strip()
        color = box_color_bgr or _color_for_label(color_key, label_color_map)
        shape_type = str(item.get("shape_type") or "detect").strip() or "detect"
        points = item.get("points") or []
        if shape_type in {"segment", "obb"} and points:
            pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=line_width)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)
        if label:
            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - (line_width * 2 + 4))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color_bgr,
                text_thickness,
            )
    return out


# ---------- 主入口：在图片上绘制检测框叠加层 ----------
# 主入口：在图片上绘制检测框叠加层，优先用 Pillow（支持中文），回退到 OpenCV
def draw_overlay(
    src_path: Path,
    out_path: Path,
    items: list[dict],
    *,
    box_color_bgr: tuple[int, int, int] | None = None,
    text_color_bgr: tuple[int, int, int] = (255, 255, 255),
) -> None:
    image = cv2.imread(str(src_path))
    if image is None:
        return

    height, width = image.shape[:2]
    line_width = max(2, min(10, int(round(max(width, height) / 500.0))))
    text_thickness = max(1, line_width - 1)
    font_scale = max(0.55, min(1.15, 0.45 + line_width * 0.1))
    normalized: list[dict] = []
    for raw in items:
        x1, y1 = _clamp_point(raw.get("x1", 0), raw.get("y1", 0), width, height)
        x2, y2 = _clamp_point(raw.get("x2", 0), raw.get("y2", 0), width, height)
        points = []
        for point in raw.get("points") or []:
            try:
                px, py = _clamp_point(point[0], point[1], width, height)
            except Exception:  # noqa: BLE001
                continue
            points.append((px, py))
        if not points and (x2 <= x1 or y2 <= y1):
            continue
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        normalized.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "points": points,
                "shape_type": str(raw.get("shape_type") or "detect").strip() or "detect",
                "label": str(raw.get("label") or "").strip(),
            }
        )

    if not normalized:
        cv2.imwrite(str(out_path), image)
        return

    can_use_pillow = Image is not None and ImageDraw is not None and ImageFont is not None
    font_size = max(16, min(44, int(round(max(width, height) / 180.0))))
    font = _load_font(font_size) if can_use_pillow else None
    if not can_use_pillow or font is None:
        out = _draw_with_opencv(
            image,
            normalized,
            box_color_bgr=box_color_bgr,
            text_color_bgr=text_color_bgr,
            line_width=line_width,
            font_scale=font_scale,
            text_thickness=text_thickness,
        )
        cv2.imwrite(str(out_path), out)
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(canvas)
    text_color_rgb = (text_color_bgr[2], text_color_bgr[1], text_color_bgr[0])
    label_color_map: dict[str, int] = {}

    for item in normalized:
        x1 = item["x1"]
        y1 = item["y1"]
        x2 = item["x2"]
        y2 = item["y2"]
        label = item["label"]
        color_key = str(item.get("color_key") or label).strip()
        shape_type = str(item.get("shape_type") or "detect").strip() or "detect"
        points = item.get("points") or []

        if box_color_bgr is not None:
            color_bgr = box_color_bgr
        else:
            color_bgr = _color_for_label(color_key, label_color_map)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        if shape_type in {"segment", "obb"} and points:
            draw.line([*points, points[0]], fill=color_rgb, width=line_width, joint="curve")
        else:
            draw.rectangle([(x1, y1), (x2, y2)], outline=color_rgb, width=line_width)

        if not label:
            continue

        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = max(1, text_bbox[2] - text_bbox[0])
        th = max(1, text_bbox[3] - text_bbox[1])
        tx = max(0, min(x1, width - tw - 4))
        ty = y1 - th - (line_width * 2 + 2)
        if ty < 0:
            ty = min(y1 + 4, height - th - 2)

        draw.rectangle([(tx, ty), (tx + tw + 4, ty + th + 4)], fill=color_rgb)
        draw.text((tx + 2, ty + 1), label, fill=text_color_rgb, font=font)

    out = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out)
