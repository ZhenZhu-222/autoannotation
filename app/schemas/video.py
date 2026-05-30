# ============================================================
# 视频模块 DTO
# ============================================================
from __future__ import annotations

from pydantic import BaseModel


# ---------- 视频上传/导入响应 ----------
class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    path: str
    created_at: str
