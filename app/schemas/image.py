# ============================================================
# 图片操作相关请求 DTO
# ============================================================
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.assignments import AssigneeItem


# ---------- 批量删除 ----------
class BatchDeleteImagesRequest(BaseModel):
    image_ids: list[str] = Field(default_factory=list)


# ---------- 图片审核 ----------
class UpdateImageReviewRequest(BaseModel):
    review_status: Literal["todo", "in_progress", "reviewed", "rejected", "accepted"]
    reviewer: str = ""
    review_note: str = ""


# ---------- 图片精细调整（框编辑） ----------
class RefineClassDraft(BaseModel):
    name: str = Field(..., min_length=1)


class RefineBoxPayload(BaseModel):
    box_id: str = ""
    class_id: int | None = None
    class_name: str = ""
    shape_type: Literal["detect", "segment", "obb"] = "detect"
    points: list[list[float]] = Field(default_factory=list)
    x1: float
    y1: float
    x2: float
    y2: float


class SaveImageRefineRequest(BaseModel):
    boxes: list[RefineBoxPayload] = Field(default_factory=list)
    new_classes: list[RefineClassDraft] = Field(default_factory=list)
    operator: str = "anonymous"


# ---------- 类别 ID 重映射 ----------
class RemapClassesRequest(BaseModel):
    id_mapping: dict[str, int] = Field(..., description="old_id (str) → new_id (int)")


# ============================================================
# 图片集/图片模块响应 DTO
# ============================================================


# ---------- 图片集列表 ----------
class ImagesetItem(BaseModel):
    imageset_id: str
    name: str
    source: str
    image_count: int
    label_count: int
    max_class_id: int = -1
    label_task: str = "detect"
    created_at: str
    creator_id: str = ""
    creator_username: str = ""
    assignee_ids: list[str] = Field(default_factory=list)
    assignee_usernames: list[str] = Field(default_factory=list)
    assignee_states: dict[str, dict] = Field(default_factory=dict)
    assignee_history: list[AssigneeItem] = Field(default_factory=list)
    assignment_history_count: int = 0
    can_operate: bool = True


class ListImagesetsResponse(BaseModel):
    items: list[ImagesetItem]


# ---------- 图片列表单项 ----------
class ImageItem(BaseModel):
    image_id: str
    filename: str
    url: str
    label_exists: bool
    label_url: str = ""
    created_at: str
    review_status: str | None = None
    reviewer: str = ""
    review_note: str = ""
    reviewed_at: str = ""


class ListImagesResponse(BaseModel):
    imageset_id: str
    name: str
    source: str
    label_task: str = "detect"
    page: int
    page_size: int
    total: int
    has_next: bool
    items: list[ImageItem]
    class_names: dict[str, str] = Field(default_factory=dict)
    latest_job_id: str = ""


# ---------- 图片集重命名 ----------
class ImagesetRenameResponse(BaseModel):
    imageset_id: str
    name: str


# ---------- 图片集删除 ----------
class ImagesetDeleteResponse(BaseModel):
    imageset_id: str
    deleted: bool


# ---------- 图片删除 ----------
class ImageDeleteDetail(BaseModel):
    image: bool = False
    label: bool = False


class ImageDeleteResponse(BaseModel):
    image_id: str
    imageset_id: str
    deleted: ImageDeleteDetail


# ---------- 批量删除 ----------
class BatchDeleteImagesResponse(BaseModel):
    results: list[dict] = Field(default_factory=list)


# ---------- 图片审核 ----------
class ImageReviewResponse(BaseModel):
    image_id: str
    imageset_id: str
    review_status: str
    reviewer: str = ""
    review_note: str = ""
    reviewed_at: str = ""


class ReviewSummaryResponse(BaseModel):
    imageset_id: str
    total: int
    counts: dict[str, int] = Field(default_factory=dict)


# ---------- 文件夹上传 ----------
class UploadFolderResponse(BaseModel):
    imageset_id: str
    name: str
    image_count: int
    labels_imported: int
    labels_unmatched: int
    labels_unmatched_examples: list[str] = Field(default_factory=list)
    duplicate_label_name_count: int = 0
    class_names_imported: int = 0
    class_name_sources: list[str] = Field(default_factory=list)
    label_task: str = "detect"


# ---------- 类别映射查询 ----------
class ClassMappingEntry(BaseModel):
    class_id: int
    name: str
    count: int


class ClassMappingResponse(BaseModel):
    imageset_id: str
    entries: list[ClassMappingEntry]


# ---------- 类别重映射结果 ----------
class RemapClassesResponse(BaseModel):
    imageset_id: str
    files_modified: int
    lines_modified: int
    new_class_count: int


# ---------- 精细调整编辑器响应 ----------
class RefineBoxResponseItem(BaseModel):
    box_id: str = ""
    class_id: int | None = None
    class_name: str = ""
    shape_type: str = "detect"
    points: list[list[float]] = Field(default_factory=list)
    x1: float = 0
    y1: float = 0
    x2: float = 0
    y2: float = 0


class RefineClassOption(BaseModel):
    class_id: int
    name: str


class RefineImageResponse(BaseModel):
    image_id: str
    imageset_id: str
    imageset_name: str
    filename: str
    image_url: str = ""
    label_url: str = ""
    image_width: int
    image_height: int
    boxes: list[RefineBoxResponseItem] = Field(default_factory=list)
    label_task: str = "detect"
    class_options: list[RefineClassOption] = Field(default_factory=list)
    label_exists: bool = False
    review_status: str | None = None
    reviewer: str = ""
    review_note: str = ""
    reviewed_at: str = ""
    can_rollback: bool = False
    last_backup_at: str = ""
    saved: bool | None = None
    rolled_back: bool | None = None
