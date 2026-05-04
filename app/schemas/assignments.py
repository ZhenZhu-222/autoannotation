# ============================================================
# 分派模块 DTO（分派 / 提交验收 / 审核 / 响应）
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- 请求 ----------
# 设置完整分派集合（覆盖式，而不是追加）
class AssignImagesetRequest(BaseModel):
    user_ids: list[str] = Field(default_factory=list)


# operator 提交验收
class AssignmentSubmitRequest(BaseModel):
    submit_note: str = ""


# admin 审核验收（approved=true 通过，false 驳回）
class AssignmentReviewRequest(BaseModel):
    approved: bool = True
    review_note: str = ""


# ---------- 响应 ----------
# 单个分派人员信息（含状态流转字段）
class AssigneeItem(BaseModel):
    id: str
    username: str
    role: str
    enabled: bool = True
    status: str = "assigned"
    assigned_at: str = ""
    submitted_at: str = ""
    reviewed_at: str = ""
    submit_note: str = ""
    review_note: str = ""
    reviewer_username: str = ""


# 分派列表响应（含当前分派 + 历史流水）
class AssigneeListResponse(BaseModel):
    imageset_id: str
    assignees: list[AssigneeItem] = Field(default_factory=list)
    history: list[AssigneeItem] = Field(default_factory=list)
