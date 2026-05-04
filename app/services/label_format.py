# ============================================================
# YOLO 多任务标签格式工具
# 职责：统一 detect / segment / obb 标签解析、格式化和 overlay 转换
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal


LabelTask = Literal["detect", "segment", "obb"]
VALID_LABEL_TASKS: set[str] = {"detect", "segment", "obb"}


@dataclass(frozen=True)
class LabelTaskSpec:
    task: LabelTask
    min_points: int
    fixed_points: int | None = None
    line_value_count: int | None = None


LABEL_TASK_SPECS: dict[LabelTask, LabelTaskSpec] = {
    "detect": LabelTaskSpec(task="detect", min_points=2, fixed_points=2, line_value_count=4),
    "segment": LabelTaskSpec(task="segment", min_points=3),
    "obb": LabelTaskSpec(task="obb", min_points=4, fixed_points=4, line_value_count=8),
}


# 将任意字符串标准化为合法的 label_task（detect/segment/obb）
def normalize_label_task(task: str | None, default: LabelTask = "detect") -> LabelTask:
    text = str(task or "").strip().lower()
    if text in VALID_LABEL_TASKS:
        return text  # type: ignore[return-value]
    return default


# 获取指定任务类型的规格定义（最少点数、固定点数等）
def get_label_task_spec(task: str | None, default: LabelTask = "detect") -> LabelTaskSpec:
    return LABEL_TASK_SPECS[normalize_label_task(task, default)]


# 将浮点数格式化为 6 位小数字符串（用于写入标签文件）
def format_float(value: float) -> str:
    return f"{float(value):.6f}"


# 将数值限制在 [low, high] 范围内
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(max(float(value), low), high)


# 将像素坐标归一化到 [0, 1] 范围
def _normalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    safe_w = max(1.0, float(width))
    safe_h = max(1.0, float(height))
    return _clamp(float(x) / safe_w), _clamp(float(y) / safe_h)


# 将归一化坐标还原为像素坐标
def _denormalize_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return float(x) * max(1, int(width)), float(y) * max(1, int(height))


@dataclass
class LabelShape:
    class_id: int
    shape_type: LabelTask
    points: list[tuple[float, float]] = field(default_factory=list)

    # 将当前形状输出为 YOLO 格式的标签行字符串
    def format_yolo(self) -> str:
        if self.shape_type == "detect":
            if len(self.points) < 2:
                raise ValueError("detect label requires 2 bbox corner points")
            x1, y1, x2, y2 = self.bbox_norm()
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            values = [cx, cy, bw, bh]
        else:
            spec = get_label_task_spec(self.shape_type)
            if len(self.points) < spec.min_points:
                raise ValueError(f"{self.shape_type} label requires at least {spec.min_points} points")
            points = self.points[:spec.fixed_points] if spec.fixed_points else self.points
            values = [coord for point in points for coord in point]
        return " ".join([str(int(self.class_id)), *[format_float(v) for v in values]])

    # 返回归一化坐标的包围框 (x1, y1, x2, y2)
    def bbox_norm(self) -> tuple[float, float, float, float]:
        if not self.points:
            return 0.0, 0.0, 0.0, 0.0
        xs = [_clamp(p[0]) for p in self.points]
        ys = [_clamp(p[1]) for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    # 返回像素坐标的包围框 (x1, y1, x2, y2)
    def bbox_pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox_norm()
        px1, py1 = _denormalize_point(x1, y1, width, height)
        px2, py2 = _denormalize_point(x2, y2, width, height)
        return px1, py1, px2, py2

    # 返回所有点的像素坐标列表
    def points_pixels(self, width: int, height: int) -> list[tuple[float, float]]:
        return [_denormalize_point(x, y, width, height) for x, y in self.points]

    # 返回一个相同形状但更换了 class_id 的新对象
    def with_class_id(self, class_id: int) -> "LabelShape":
        return LabelShape(class_id=int(class_id), shape_type=self.shape_type, points=list(self.points))


# 解析一行 YOLO 标签文本，返回 LabelShape，无效行返回 None
def parse_label_line(line: str, label_task: str | None = None) -> LabelShape | None:
    parts = str(line or "").strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
    except (TypeError, ValueError):
        return None
    if class_id < 0:
        return None

    task = normalize_label_task(label_task, default="detect") if label_task else ""
    if task == "detect" or (not task and len(nums) == 4):
        if len(nums) != 4:
            return None
        cx, cy, bw, bh = [_clamp(v) for v in nums]
        x1 = _clamp(cx - bw / 2.0)
        y1 = _clamp(cy - bh / 2.0)
        x2 = _clamp(cx + bw / 2.0)
        y2 = _clamp(cy + bh / 2.0)
        return LabelShape(class_id=class_id, shape_type="detect", points=[(x1, y1), (x2, y2)])

    if len(nums) % 2 != 0:
        return None
    points = [(_clamp(nums[i]), _clamp(nums[i + 1])) for i in range(0, len(nums), 2)]

    if task == "obb" or (not task and len(points) == 4):
        if len(points) != 4:
            return None
        return LabelShape(class_id=class_id, shape_type="obb", points=points)

    if task == "segment" or not task:
        if len(points) < 3:
            return None
        return LabelShape(class_id=class_id, shape_type="segment", points=points)

    return None


def sort_label_lines(lines: Iterable[str], label_task: str | None = None) -> list[str]:
    """Group labels by class id, then keep a stable top-to-bottom/left-to-right order.

    This keeps same-class labels (person, helmet, etc.) together instead of letting
    model output order or manual editing order reshuffle the file each save.
    """
    sortable: list[tuple[tuple[int, float, float, int], str]] = []
    invalid: list[tuple[int, str]] = []
    for idx, raw in enumerate(lines):
        text = str(raw or "").strip()
        if not text:
            continue
        shape = parse_label_line(text, label_task)
        if not shape:
            invalid.append((idx, text))
            continue
        x1, y1, _, _ = shape.bbox_norm()
        sortable.append(((int(shape.class_id), float(y1), float(x1), idx), text))
    sortable.sort(key=lambda item: item[0])
    return [text for _, text in sortable] + [text for _, text in sorted(invalid, key=lambda item: item[0])]


# 将像素坐标格式的检测框 (xyxy) 转为 LabelShape
def xyxy_to_shape(cls_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> LabelShape:
    safe_w = max(1, int(width))
    safe_h = max(1, int(height))
    left = min(max(0.0, float(x1)), safe_w)
    top = min(max(0.0, float(y1)), safe_h)
    right = min(max(0.0, float(x2)), safe_w)
    bottom = min(max(0.0, float(y2)), safe_h)
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    return LabelShape(
        class_id=int(cls_id),
        shape_type="detect",
        points=[_normalize_point(left, top, safe_w, safe_h), _normalize_point(right, bottom, safe_w, safe_h)],
    )


# 将像素坐标点列表转为指定任务类型的 LabelShape
def points_to_shape(
    cls_id: int,
    points: Iterable[tuple[float, float] | list[float]],
    width: int,
    height: int,
    label_task: str,
) -> LabelShape:
    task = normalize_label_task(label_task)
    spec = get_label_task_spec(task)
    normalized = [_normalize_point(float(p[0]), float(p[1]), width, height) for p in points]
    if task == "detect":
        if not normalized:
            raise ValueError("detect requires points")
        xs = [p[0] for p in normalized]
        ys = [p[1] for p in normalized]
        return LabelShape(class_id=int(cls_id), shape_type="detect", points=[(min(xs), min(ys)), (max(xs), max(ys))])
    if task == "obb":
        if len(normalized) != spec.fixed_points:
            raise ValueError(f"obb requires exactly {spec.fixed_points} points")
        return LabelShape(class_id=int(cls_id), shape_type="obb", points=normalized)
    if len(normalized) < spec.min_points:
        raise ValueError(f"segment requires at least {spec.min_points} points")
    return LabelShape(class_id=int(cls_id), shape_type="segment", points=normalized)


# 将 xyxy 格式模型输出直接转为 YOLO 标签行字符串
def xyxy_to_yolo(cls_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    return xyxy_to_shape(cls_id, x1, y1, x2, y2, width, height).format_yolo()


# 将 LabelShape 转换为前端 overlay 请求所需的 dict 格式
def shape_to_overlay_item(shape: LabelShape, width: int, height: int, label: str = "") -> dict:
    x1, y1, x2, y2 = shape.bbox_pixels(width, height)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "points": shape.points_pixels(width, height) if shape.shape_type in {"segment", "obb"} else [],
        "shape_type": shape.shape_type,
        "label": label,
        "color_key": f"class_{int(shape.class_id)}",
    }


# 根据单行标签的字段数量推断它属于哪个任务类型
def label_line_to_task(line: str, default: LabelTask = "detect") -> LabelTask:
    parts = str(line or "").strip().split()
    if len(parts) == 1 + LABEL_TASK_SPECS["detect"].line_value_count:
        return "detect"
    if len(parts) == 1 + LABEL_TASK_SPECS["obb"].line_value_count:
        return "obb"
    if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
        return "segment"
    return default


# 从多行标签中推断整个文件属于哪个任务类型
def infer_label_task_from_lines(lines: Iterable[str], default: LabelTask = "detect") -> LabelTask:
    tasks: set[str] = set()
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        tasks.add(label_line_to_task(text, default=default))
        if "segment" in tasks:
            return "segment"
    if "obb" in tasks:
        return "obb"
    return default


# 扫描整个目录的标签文件，推断该图片集的任务类型
def infer_label_task_from_dir(labels_dir: Path, default: LabelTask = "detect") -> LabelTask:
    if not labels_dir.exists():
        return default
    task = default
    for label_file in sorted(labels_dir.glob("*.txt")):
        if label_file.name == "classes.txt":
            continue
        try:
            inferred = infer_label_task_from_lines(label_file.read_text(encoding="utf-8").splitlines(), default=default)
        except Exception:  # noqa: BLE001
            continue
        if inferred == "segment":
            return "segment"
        if inferred == "obb":
            task = "obb"
    return task
