# ============================================================
# 图片精细调整服务 (Refine)
# 职责：加载/保存/回滚单张图片的标注框编辑
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from app.core.state import AppState
from app.core.utils import now_iso
from app.services.annotation_service import AnnotationService, rebuild_classes_txt, xyxy_to_yolo
from app.services.class_name_service import collect_used_class_stats, read_current_imageset_class_names, resolve_imageset_class_names
from app.services.label_format import normalize_label_task, parse_label_line, points_to_shape, sort_label_lines
from app.services.task_manager import TaskManager

REFINE_BACKUP_DIRNAME = ".refine_backups"
MIN_BOX_SIZE_PX = 2.0


# 将类别名标准化（小写+去前后空格），用于同名匹配
def _normalize_name(name: str) -> str:
    return str(name or "").strip().casefold()


# 判断名称是否是占位符（class_N 格式）
def _is_placeholder_name(class_id: int, name: str) -> bool:
    return str(name or "").strip().lower() == f"class_{int(class_id)}"


# 将多个类别名映射合并，真实名称优先覆盖占位符
def _merge_name_maps(*maps: dict[int, str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for source in maps:
        for class_id, raw_name in (source or {}).items():
            name = str(raw_name or "").strip()
            if not name:
                continue
            existing = result.get(class_id, "")
            if not existing or (_is_placeholder_name(class_id, existing) and not _is_placeholder_name(class_id, name)):
                result[class_id] = name
    return result


# 根据 image_id 获取对应的 imageset 和 image 记录
def _resolve_image_records(state: AppState, image_id: str):
    pair = state.image_index.get(image_id)
    if not pair:
        raise KeyError("image not found")
    imageset_id, image = pair
    imageset = state.imagesets.get(imageset_id)
    if not imageset:
        raise KeyError("imageset not found")
    return imageset, image


# 根据图片文件名计算对应的标签文件路径
def _label_path(imageset_dir: Path, filename: str) -> Path:
    return imageset_dir / "labels" / f"{Path(filename).stem}.txt"


# 计算指定图片的精修备份文件路径
def _backup_path(imageset_dir: Path, image_id: str) -> Path:
    return imageset_dir / REFINE_BACKUP_DIRNAME / f"{image_id}.json"


# 读取图片尺寸，返回 (width, height)
def _load_image_size(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path.name}")
    height, width = image.shape[:2]
    return int(width), int(height)


# 将一行 YOLO 标签转为前端编辑器所需的 box dict（像素坐标，含类别名）
def _yolo_line_to_box(line: str, width: int, height: int, class_names: dict[int, str], box_id: str, label_task: str) -> dict[str, Any] | None:
    target_task = normalize_label_task(label_task)
    parsed = parse_label_line(line, target_task)
    if parsed is None and target_task != "detect":
        # Legacy repair: older UI versions could save 5-column detect rows
        # into a segment/obb imageset. Show them as rectangular polygons so
        # the next save rewrites them in the imageset's real task format.
        fallback = parse_label_line(line, "detect")
        if fallback is not None:
            x1, y1, x2, y2 = fallback.bbox_pixels(width, height)
            rect_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            try:
                parsed = points_to_shape(fallback.class_id, rect_points, width, height, target_task)
            except ValueError:
                parsed = None
    if parsed is None:
        return None
    points = parsed.points_pixels(width, height)
    x1, y1, x2, y2 = parsed.bbox_pixels(width, height)
    return {
        "box_id": box_id,
        "class_id": parsed.class_id,
        "class_name": class_names.get(parsed.class_id, f"class_{parsed.class_id}"),
        "shape_type": parsed.shape_type,
        "points": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "x1": round(float(x1), 3),
        "y1": round(float(y1), 3),
        "x2": round(float(x2), 3),
        "y2": round(float(y2), 3),
    }


# 读取指定图片的精修备份快照，不存在返回空字典
def _read_backup(imageset_dir: Path, image_id: str) -> dict[str, Any]:
    backup_file = _backup_path(imageset_dir, image_id)
    if not backup_file.exists():
        return {}
    try:
        return json.loads(backup_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


# 构建精修编辑器的类别名映射（当前图片集 + 历史任务合并）
def _build_editor_class_map(
    *,
    imageset_id: str,
    imageset_dir: Path,
    state: AppState,
    tasks: TaskManager | None,
) -> dict[int, str]:
    current_map = read_current_imageset_class_names(imageset_dir)
    resolved_map = {
        int(class_id): name
        for class_id, name in resolve_imageset_class_names(
            imageset_id=imageset_id,
            imageset_dir=imageset_dir,
            state=state,
            tasks=tasks,
        ).items()
    }
    used_ids = set(collect_used_class_stats(imageset_dir).keys())
    merged = _merge_name_maps(current_map, resolved_map)
    out: dict[int, str] = {}
    for class_id, name in sorted(merged.items()):
        if class_id in used_ids or not _is_placeholder_name(class_id, name):
            out[class_id] = name
    for class_id in sorted(used_ids):
        out.setdefault(class_id, merged.get(class_id, f"class_{class_id}"))
    return out


# 加载单张图片的精修编辑器数据（图片路径、标注框、类别选项、审核状态等）
def load_refine_image_payload(
    *,
    state: AppState,
    image_id: str,
    tasks: TaskManager | None = None,
) -> dict[str, Any]:
    imageset, image = _resolve_image_records(state, image_id)
    imageset_dir = Path(imageset.dir_path)
    image_path = imageset_dir / image.rel_path
    if not image_path.exists():
        raise FileNotFoundError("图片文件不存在")

    width, height = _load_image_size(image_path)
    label_path = _label_path(imageset_dir, image.filename)
    class_names = _build_editor_class_map(
        imageset_id=imageset.id,
        imageset_dir=imageset_dir,
        state=state,
        tasks=tasks,
    )
    label_task = normalize_label_task(getattr(imageset, "label_task", "detect"))

    boxes: list[dict[str, Any]] = []
    if label_path.exists():
        for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            box = _yolo_line_to_box(line, width, height, class_names, box_id=f"box_{idx}", label_task=label_task)
            if box is not None:
                boxes.append(box)

    backup = _read_backup(imageset_dir, image.id)
    return {
        "image_id": image.id,
        "imageset_id": imageset.id,
        "imageset_name": imageset.name,
        "filename": image.filename,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "image_width": width,
        "image_height": height,
        "boxes": boxes,
        "label_task": label_task,
        "class_options": [
            {"class_id": class_id, "name": name}
            for class_id, name in sorted(class_names.items())
        ],
        "label_exists": label_path.exists(),
        "review_status": image.review_status,
        "reviewer": image.reviewer,
        "review_note": image.review_note,
        "reviewed_at": image.reviewed_at,
        "can_rollback": bool(backup),
        "last_backup_at": str(backup.get("saved_at") or ""),
    }


# 解析模板提交的标注框类别，自动分配新类别 ID，返回解析后的框和最终类别映射
def _resolve_box_class_ids(
    *,
    raw_boxes: list[Any],
    new_classes: list[Any],
    current_class_map: dict[int, str],
    used_ids: set[int],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    combined_map = dict(current_class_map)
    next_id = max({-1, *combined_map.keys(), *used_ids}) + 1
    normalized_to_id = {
        _normalize_name(name): class_id
        for class_id, name in combined_map.items()
        if str(name or "").strip()
    }

    pending_names: list[str] = []
    for item in new_classes or []:
        name = str(getattr(item, "name", "") or "").strip()
        if name:
            pending_names.append(name)
    for item in raw_boxes or []:
        box_class_id = getattr(item, "class_id", None)
        box_class_name = str(getattr(item, "class_name", "") or "").strip()
        if box_class_id is None and box_class_name:
            pending_names.append(box_class_name)

    allocated_new_names: dict[int, str] = {}
    for name in pending_names:
        norm = _normalize_name(name)
        if not norm:
            continue
        if norm in normalized_to_id:
            continue
        normalized_to_id[norm] = next_id
        combined_map[next_id] = name
        allocated_new_names[next_id] = name
        next_id += 1

    out_boxes: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_boxes or []):
        class_id = getattr(item, "class_id", None)
        class_name = str(getattr(item, "class_name", "") or "").strip()
        if class_id is None:
            norm = _normalize_name(class_name)
            if not norm or norm not in normalized_to_id:
                raise ValueError(f"第 {idx + 1} 个框缺少有效类别")
            class_id = normalized_to_id[norm]
        else:
            class_id = int(class_id)
            if class_id < 0:
                raise ValueError(f"第 {idx + 1} 个框类别 ID 不能为负数")
            if class_id not in combined_map:
                combined_map[class_id] = class_name or f"class_{class_id}"
            class_name = combined_map[class_id]
        out_boxes.append(
            {
                "box_id": str(getattr(item, "box_id", "") or f"box_{idx}"),
                "class_id": class_id,
                "class_name": combined_map.get(class_id, class_name or f"class_{class_id}"),
                "shape_type": str(getattr(item, "shape_type", "detect") or "detect"),
                "points": [list(p) for p in (getattr(item, "points", None) or [])],
                "x1": float(getattr(item, "x1")),
                "y1": float(getattr(item, "y1")),
                "x2": float(getattr(item, "x2")),
                "y2": float(getattr(item, "y2")),
            }
        )
    return out_boxes, combined_map


# 保存单张图片的精修标注，写备份快照，更新 classes.txt 和图片集 label_task
def save_refine_image_labels(
    *,
    state: AppState,
    image_id: str,
    boxes: list[Any],
    new_classes: list[Any],
    operator: str = "anonymous",
    tasks: TaskManager | None = None,
) -> dict[str, Any]:
    imageset, image = _resolve_image_records(state, image_id)
    imageset_dir = Path(imageset.dir_path)
    image_path = imageset_dir / image.rel_path
    if not image_path.exists():
        raise FileNotFoundError("图片文件不存在")
    width, height = _load_image_size(image_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = _label_path(imageset_dir, image.filename)

    current_map = _build_editor_class_map(
        imageset_id=imageset.id,
        imageset_dir=imageset_dir,
        state=state,
        tasks=tasks,
    )
    label_task = normalize_label_task(getattr(imageset, "label_task", "detect"))
    used_ids = set(collect_used_class_stats(imageset_dir).keys())
    resolved_boxes, final_class_map = _resolve_box_class_ids(
        raw_boxes=boxes,
        new_classes=new_classes,
        current_class_map=current_map,
        used_ids=used_ids,
    )

    backup_file = _backup_path(imageset_dir, image.id)
    backup_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file.write_text(
        json.dumps(
            {
                "image_id": image.id,
                "imageset_id": imageset.id,
                "filename": image.filename,
                "label_exists": label_path.exists(),
                "label_text": label_path.read_text(encoding="utf-8") if label_path.exists() else "",
                "class_names": current_map,
                "saved_at": now_iso(),
                "operator": (operator or "anonymous").strip() or "anonymous",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines: list[str] = []
    shape_types = {normalize_label_task(box.get("shape_type", label_task), label_task) for box in resolved_boxes}
    if len(shape_types) > 1:
        raise ValueError("同一张图片暂不支持混合 detect/segment/obb 形状")
    if shape_types:
        label_task = next(iter(shape_types))
    for idx, box in enumerate(resolved_boxes):
        shape_type = normalize_label_task(box.get("shape_type", label_task), label_task)
        raw_points = box.get("points") or []
        if shape_type == "detect":
            x1, x2 = sorted([float(box["x1"]), float(box["x2"])])
            y1, y2 = sorted([float(box["y1"]), float(box["y2"])])
            x1 = min(max(0.0, x1), width)
            x2 = min(max(0.0, x2), width)
            y1 = min(max(0.0, y1), height)
            y2 = min(max(0.0, y2), height)
            if (x2 - x1) < MIN_BOX_SIZE_PX or (y2 - y1) < MIN_BOX_SIZE_PX:
                raise ValueError(f"第 {idx + 1} 个框过小，至少保留 {MIN_BOX_SIZE_PX:.0f}px")
            lines.append(xyxy_to_yolo(int(box["class_id"]), x1, y1, x2, y2, width, height))
            continue
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError(f"第 {idx + 1} 个形状缺少 points")
        shape = points_to_shape(int(box["class_id"]), raw_points, width, height, shape_type)
        x1, y1, x2, y2 = shape.bbox_pixels(width, height)
        if (x2 - x1) < MIN_BOX_SIZE_PX or (y2 - y1) < MIN_BOX_SIZE_PX:
            raise ValueError(f"第 {idx + 1} 个形状过小，至少保留 {MIN_BOX_SIZE_PX:.0f}px")
        lines.append(shape.format_yolo())

    if lines:
        lines = sort_label_lines(lines, label_task)
        label_path.write_text("\n".join(lines), encoding="utf-8")
    else:
        label_path.unlink(missing_ok=True)

    rebuild_classes_txt(
        imageset_labels_dir=labels_dir,
        imageset_dir=imageset_dir,
        new_class_names=final_class_map,
        label_task=label_task,
    )
    imageset.label_task = label_task
    state._save_imageset(imageset)
    payload = load_refine_image_payload(state=state, image_id=image_id, tasks=tasks)
    payload["saved"] = True
    return payload


# 回滚单张图片的精修标注，还原到备份快照里的标签内容
def rollback_refine_image_labels(
    *,
    state: AppState,
    image_id: str,
    tasks: TaskManager | None = None,
) -> dict[str, Any]:
    imageset, image = _resolve_image_records(state, image_id)
    imageset_dir = Path(imageset.dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = _label_path(imageset_dir, image.filename)
    backup = _read_backup(imageset_dir, image.id)
    if not backup:
        raise FileNotFoundError("未找到该图片的精修回滚快照")

    if bool(backup.get("label_exists")):
        label_path.write_text(str(backup.get("label_text") or ""), encoding="utf-8")
    else:
        label_path.unlink(missing_ok=True)

    backup_class_names = {
        int(class_id): str(name or "").strip()
        for class_id, name in (backup.get("class_names") or {}).items()
        if str(name or "").strip()
    }
    current_map = _build_editor_class_map(
        imageset_id=imageset.id,
        imageset_dir=imageset_dir,
        state=state,
        tasks=tasks,
    )
    rebuild_classes_txt(
        imageset_labels_dir=labels_dir,
        imageset_dir=imageset_dir,
        new_class_names=_merge_name_maps(backup_class_names, current_map),
    )
    payload = load_refine_image_payload(state=state, image_id=image_id, tasks=tasks)
    payload["rolled_back"] = True
    return payload
