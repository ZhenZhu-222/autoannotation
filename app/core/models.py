# ============================================================
# 内部数据模型定义 (Entity 层)
# 职责：用 dataclass 定义所有业务实体的字段结构
# 说明：仅定义数据结构，不包含业务逻辑
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------- 用户记录 ----------
@dataclass
class UserRecord:
    id: str
    username: str
    password_hash: str
    role: str
    enabled: bool = True
    created_at: str = ""
    last_login_at: str = ""
    deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- 视频记录 ----------
@dataclass
class VideoRecord:
    id: str
    filename: str
    path: str
    created_at: str

    # 序列化为字典（用于 API 响应）
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- 图片记录（属于某个图片集） ----------
@dataclass
class ImageRecord:
    id: str
    imageset_id: str
    filename: str
    rel_path: str
    created_at: str
    review_status: str = "todo"
    reviewer: str = ""
    review_note: str = ""
    reviewed_at: str = ""

    # 序列化为字典（用于 API 响应）
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- 图片集记录 ----------
@dataclass
class ImageSetRecord:
    id: str
    name: str
    source: str
    dir_path: str
    created_at: str
    images: List[ImageRecord] = field(default_factory=list)
    label_task: str = "detect"
    creator_id: str = ""
    assignee_ids: List[str] = field(default_factory=list)
    assignee_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assignee_history: List[Dict[str, Any]] = field(default_factory=list)

    # 序列化为字典（含嵌套的图片列表）
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["images"] = [img.to_dict() for img in self.images]
        return data


# ---------- 模型记录 ----------
@dataclass
class ModelRecord:
    id: str
    name: str
    model_type: str
    model_path: str
    classes: List[str]
    class_source: str
    created_at: str
    task: str = "detect"
    creator_id: str = ""

    # 序列化为字典（用于 API 响应）
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- 任务记录（打标/训练等异步任务） ----------
@dataclass
class JobRecord:
    id: str
    job_type: str
    status: str
    payload: Dict[str, Any]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress: float = 0.0
    progress_text: str = ""
    cancel_requested: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    operator_id: str = ""

    # 序列化为字典（用于 API 响应）
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
