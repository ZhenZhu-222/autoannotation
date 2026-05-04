# ============================================================
# YOLO 推理封装
# 职责：加载 YOLO 模型、执行预测、自动选择设备 (GPU/MPS/CPU)
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Protocol, runtime_checkable

import numpy as np

from app.services.label_format import normalize_label_task


# ---------- 检测结果数据结构 ----------
@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    shape_type: str = "detect"
    points: list[tuple[float, float]] = field(default_factory=list)


@runtime_checkable
class Predictor(Protocol):
    class_names: List[str]
    task: str

    def predict(self, image_path: str, conf: float, iou: float, device: str = "") -> List[Detection]:
        ...


PredictorFactory = Callable[[str], Predictor]


# 将 PyTorch tensor 或其他类型转换为 numpy 数组
def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


# ---------- YOLO 预测器 ----------
class YoloPredictor:
    # 加载 YOLO 模型，解析类别名和任务类型
    def __init__(self, model_path: str) -> None:
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(model_path)
        self.class_names = self._parse_class_names(getattr(self.model, "names", {}))
        self.task = normalize_label_task(getattr(self.model, "task", "") or _infer_task_from_path(model_path))

    # 将模型的 names 属性（dict 或 list）解析为类别名列表
    @staticmethod
    def _parse_class_names(raw: Any) -> List[str]:
        if isinstance(raw, dict):
            def _sort_key(value):
                try:
                    return int(value)
                except Exception:  # noqa: BLE001
                    return str(value)

            return [str(raw[k]) for k in sorted(raw.keys(), key=_sort_key)]
        if isinstance(raw, list):
            return [str(v) for v in raw]
        return []

    # 对单张图片运行模型推理，根据任务类型自动分配解析方法
    def predict(self, image_path: str, conf: float, iou: float, device: str = "") -> List[Detection]:
        resolved_device = resolve_device(device)
        kwargs = {"source": image_path, "conf": conf, "iou": iou, "verbose": False, "device": resolved_device}
        results = self.model.predict(**kwargs)
        if not results:
            return []

        one = results[0]
        if self.task == "obb":
            return self._parse_obb_result(one)
        if self.task == "segment":
            return self._parse_segment_result(one)

        return self._parse_detect_result(one)

    # 根据 class_id 获取类别名称，越界时返回字符串形式的 ID
    def _class_name(self, cls_id: int) -> str:
        return self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else str(cls_id)

    # 解析目标检测结果（xyxy 检测框）为 Detection 列表
    def _parse_detect_result(self, one: Any) -> List[Detection]:
        boxes = getattr(one, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        clss = _to_numpy(boxes.cls)

        out: List[Detection] = []
        for idx in range(int(xyxy.shape[0])):
            x1, y1, x2, y2 = [float(v) for v in xyxy[idx].tolist()]
            cls_id = int(clss[idx]) if idx < len(clss) else -1
            conf_v = float(confs[idx]) if idx < len(confs) else 0.0
            out.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self._class_name(cls_id),
                    conf=conf_v,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    shape_type="detect",
                )
            )
        return out

    # 解析语义分割结果（实例掩码多边形）为 Detection 列表
    def _parse_segment_result(self, one: Any) -> List[Detection]:
        boxes = getattr(one, "boxes", None)
        masks = getattr(one, "masks", None)
        if boxes is None or len(boxes) == 0:
            return []
        if masks is None or getattr(masks, "xy", None) is None:
            raise RuntimeError("模型任务为 segment，但预测结果缺少 masks 输出")

        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        clss = _to_numpy(boxes.cls)
        masks_xy = list(getattr(masks, "xy", []) or [])
        if len(masks_xy) < int(xyxy.shape[0]):
            raise RuntimeError("模型任务为 segment，但 masks 数量少于检测框数量")

        out: List[Detection] = []
        for idx in range(int(xyxy.shape[0])):
            pts = np.asarray(masks_xy[idx], dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[idx].tolist()]
            cls_id = int(clss[idx]) if idx < len(clss) else -1
            conf_v = float(confs[idx]) if idx < len(confs) else 0.0
            out.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self._class_name(cls_id),
                    conf=conf_v,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    shape_type="segment",
                    points=[(float(p[0]), float(p[1])) for p in pts.tolist()],
                )
            )
        return out

    # 解析旋转检测框结果（OBB，四个角点）为 Detection 列表
    def _parse_obb_result(self, one: Any) -> List[Detection]:
        obb = getattr(one, "obb", None)
        if obb is None or len(obb) == 0:
            boxes = getattr(one, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                raise RuntimeError("模型任务为 obb，但预测结果缺少 OBB 输出")
            return []

        raw_points = _to_numpy(obb.xyxyxyxy)
        confs = _to_numpy(obb.conf)
        clss = _to_numpy(obb.cls)
        if raw_points.ndim == 2 and raw_points.shape[1] == 8:
            raw_points = raw_points.reshape((-1, 4, 2))

        out: List[Detection] = []
        for idx in range(int(raw_points.shape[0])):
            pts = np.asarray(raw_points[idx], dtype=float).reshape((-1, 2))
            if pts.shape[0] != 4:
                continue
            xs = pts[:, 0].tolist()
            ys = pts[:, 1].tolist()
            cls_id = int(clss[idx]) if idx < len(clss) else -1
            conf_v = float(confs[idx]) if idx < len(confs) else 0.0
            out.append(
                Detection(
                    cls_id=cls_id,
                    cls_name=self._class_name(cls_id),
                    conf=conf_v,
                    x1=float(min(xs)),
                    y1=float(min(ys)),
                    x2=float(max(xs)),
                    y2=float(max(ys)),
                    shape_type="obb",
                    points=[(float(p[0]), float(p[1])) for p in pts.tolist()],
                )
            )
        return out


# ---------- 工厂方法与工具函数 ----------
_DEFAULT_PREDICTOR_FACTORY: PredictorFactory = YoloPredictor
_PREDICTOR_FACTORIES: dict[str, PredictorFactory] = {
    ".pt": YoloPredictor,
    ".onnx": YoloPredictor,
}


# 注册自定义模型格式的预测器工厂（扩展点，支持新模型格式）
def register_predictor_factory(suffixes: str | list[str] | tuple[str, ...], factory: PredictorFactory) -> None:
    items = [suffixes] if isinstance(suffixes, str) else list(suffixes)
    for suffix in items:
        clean = str(suffix or "").strip().lower()
        if not clean:
            continue
        if not clean.startswith("."):
            clean = f".{clean}"
        _PREDICTOR_FACTORIES[clean] = factory


# 根据模型文件后缀返回对应的预测器工厂
def get_predictor_factory(model_path: str) -> PredictorFactory:
    suffix = Path(str(model_path or "")).suffix.lower()
    return _PREDICTOR_FACTORIES.get(suffix, _DEFAULT_PREDICTOR_FACTORY)


# 创建并加载指定模型文件的预测器实例
def create_predictor(model_path: str) -> Predictor:
    return get_predictor_factory(model_path)(model_path)


# 根据模型文件名推断任务类型（包含 obb/seg 字样则判断为对应类型）
def _infer_task_from_path(model_path: str) -> str:
    name = str(model_path or "").lower()
    if "obb" in name:
        return "obb"
    if "-seg" in name or "_seg" in name or "segment" in name:
        return "segment"
    return "detect"


# 解析并自动降级设备选择（自动优先 GPU > MPS > CPU，无 CUDA 时降级）
def resolve_device(device: str = "") -> str:
    text = (device or "").strip()
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        has_cuda = False
        has_mps = False
    # 用户显式指定了设备
    if text:
        # 指定了 GPU 编号但没有 CUDA，自动降级
        if text.replace(",", "").isdigit() and not has_cuda:
            return "mps" if has_mps else "cpu"
        if text == "mps" and not has_mps:
            return "0" if has_cuda else "cpu"
        return text
    # 自动检测
    if has_cuda:
        return "0"
    if has_mps:
        return "mps"
    return "cpu"


# 从模型文件中提取类别名列表（失败时返回空列表）
def extract_classes_from_model_file(model_path: str) -> List[str]:
    try:
        predictor = create_predictor(model_path)
        return predictor.class_names
    except Exception:  # noqa: BLE001
        return []


# 从模型文件中提取任务类型（失败时从文件名推断）
def extract_task_from_model_file(model_path: str) -> str:
    try:
        predictor = create_predictor(model_path)
        return predictor.task
    except Exception:  # noqa: BLE001
        return normalize_label_task(_infer_task_from_path(model_path))
