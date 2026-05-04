# ============================================================
# 任务管理响应 DTO
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- 任务列表响应（items 为 job.to_dict() 结果） ----------
class JobListResponse(BaseModel):
    items: list[dict] = Field(default_factory=list)


# ---------- 历史记录列表响应 ----------
class JobHistoryListResponse(BaseModel):
    items: list[dict] = Field(default_factory=list)


# ---------- 业务流水单 ----------
class WorkLedgerListResponse(BaseModel):
    items: list[dict] = Field(default_factory=list)


# ---------- 清空业务流水单响应 ----------
class ClearWorkLedgerResponse(BaseModel):
    cleared: bool = True
    cleared_at: str = ""
    cleared_by: str = ""
    hidden_rows: int = 0


# ---------- 产物清理明细 ----------
class ArtifactsRemovedDetail(BaseModel):
    removed_files: int = 0
    removed_dirs: int = 0


# ---------- 图片集清理明细 ----------
class ImagesetRemovedDetail(BaseModel):
    imageset_id: str = ""
    imageset_deleted: bool = False


# ---------- 产物删除响应 ----------
class DeleteJobArtifactsResponse(BaseModel):
    job_id: str
    removed_files: int = 0
    removed_dirs: int = 0


# ---------- 历史删除响应（含产物 + 图片集级联） ----------
class DeleteJobHistoryResponse(BaseModel):
    job_id: str
    removed_history_rows: int = 0
    removed_active_job: bool = False
    job_type: str = ""
    artifacts_removed: ArtifactsRemovedDetail = Field(default_factory=ArtifactsRemovedDetail)
    imageset_removed: ImagesetRemovedDetail = Field(default_factory=ImagesetRemovedDetail)
