# ============================================================
# 视频抽帧任务请求 DTO
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field


class CreateExtractJobRequest(BaseModel):
    video_id: str
    sample_every_seconds: float = Field(default=1.0, gt=0)
    imageset_name: str | None = None
    operator: str = "anonymous"
