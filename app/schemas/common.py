# ============================================================
# 通用响应 DTO
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- 通用子类型 ----------
class ClassItem(BaseModel):
    id: int
    name: str


# ---------- 通用操作结果 ----------
class OkResponse(BaseModel):
    ok: bool
    detail: str = ""


# ---------- 任务提交结果（annotate / extract / qwen_annotate 共用） ----------
class JobSubmitResponse(BaseModel):
    id: str
    job_type: str
    status: str
    payload: dict = Field(default_factory=dict)
    created_at: str
    operator_id: str = ""


# ---------- 任务取消结果 ----------
class JobCancelResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    cancel_requested: bool
    progress: float = 0.0
    progress_text: str = ""
    finished_at: str | None = None


# ---------- 任务状态响应（打标/训练/抽帧等异步任务通用） ----------
class JobStatusResponse(BaseModel):
    id: str
    job_type: str
    status: str
    progress: float = 0.0
    progress_text: str = ""
    payload: dict = Field(default_factory=dict)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    result: dict | None = None
    error: str | None = None
    operator_id: str = ""
