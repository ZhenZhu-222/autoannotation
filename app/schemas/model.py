# ============================================================
# 模型模块响应 DTO
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ClassItem


# ---------- 模型上传结果 ----------
class ModelUploadResponse(BaseModel):
    model_id: str
    name: str
    model_type: str
    class_source: str
    task: str
    classes: list[str] = Field(default_factory=list)
    model_path: str
    created_at: str
    creator_id: str = ""


# ---------- 模型列表项 ----------
class ModelItem(BaseModel):
    model_id: str
    name: str
    model_type: str
    class_source: str
    task: str
    classes_count: int
    created_at: str
    creator_id: str = ""
    creator_username: str = ""


class ListModelsResponse(BaseModel):
    items: list[ModelItem]


# ---------- 模型类别查询 ----------
class ModelClassesResponse(BaseModel):
    model_id: str
    name: str
    task: str
    classes: list[ClassItem]


# ---------- 模型重命名 ----------
class ModelRenameResponse(BaseModel):
    model_id: str
    name: str


# ---------- 模型删除 ----------
class ModelDeleteResponse(BaseModel):
    model_id: str
    deleted: bool


# ---------- classes 文件解析 ----------
class ParseClassesResponse(BaseModel):
    count: int
    classes: list[ClassItem]
