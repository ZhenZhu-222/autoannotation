# ============================================================
# YOLO 打标任务请求 DTO
# ============================================================
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ClassItem


class ResolveMappingRequest(BaseModel):
    model_id: str
    imageset_id: str
    selected_class_ids: list[int] = Field(default_factory=list)
    target_classes: list[str] = Field(default_factory=list)
    label_mode: Literal["replace", "append"] = "append"
    label_task: Literal["detect", "segment", "obb"] = "detect"


class CreateAnnotateJobRequest(BaseModel):
    model_id: str
    imageset_id: str
    selected_class_ids: list[int] = Field(default_factory=list)
    conf: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.45, ge=0, le=1)
    device: str = ""
    save_overlays: bool = True
    label_task: Literal["detect", "segment", "obb"] = "detect"
    label_mode: Literal["replace", "append"] = "append"
    update_imageset_labels: bool = True
    round_tag: str = ""
    operator: str = "anonymous"
    class_id_overrides: dict[str, int] = Field(default_factory=dict)
    target_classes: list[str] = Field(default_factory=list)
    mapping_confirmed: bool = False


# ============================================================
# 打标模块响应 DTO
# ============================================================
# ---------- 类别映射解析响应 ----------
class ResolveMappingResponse(BaseModel):
    items: list[ClassItem]


# ---------- 产物查询响应 ----------
class AnnotateArtifactsResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    artifacts: dict = Field(default_factory=dict)
    output_root: str = ""
    summary: dict = Field(default_factory=dict)


# ---------- 回滚响应 ----------
class AnnotateRollbackResponse(BaseModel):
    job_id: str
    status: str
    imageset_id: str = ""
    restored_files: int = 0
    removed_files: int = 0
    untouched_files: int = 0


# ---------- 预览单项 ----------
class AnnotatePreviewItem(BaseModel):
    image_id: str = ""
    filename: str = ""
    source_url: str = ""
    overlay_url: str = ""
    label_url: str = ""
    svg_overlay_url: str = ""
    mask_url: str = ""
    existing_boxes: int = 0
    new_boxes: int = 0
    final_boxes: int = 0
    size_check_passed: int = 1
    svg_parse_ok: int = 0
    green_pixels: int = 0
    contour_count: int = 0
    calibration_error_px: float = 0.0
    consensus_rate: float = 0.0
    quality_score: float = 0.0
    extract_quality_score: float = 0.0
    reject_reason: str = ""
    refine_rounds: int = 0
    qwen_note: str = ""
    status: str = ""
    error: str = ""


# ---------- 预览分页响应 ----------
class AnnotatePreviewResponse(BaseModel):
    job_id: str
    status: str
    job_type: str
    page: int
    page_size: int
    total: int
    summary: dict = Field(default_factory=dict)
    class_names: dict[int, str] = Field(default_factory=dict)
    label_task: str = "detect"
    items: list[AnnotatePreviewItem] = Field(default_factory=list)
