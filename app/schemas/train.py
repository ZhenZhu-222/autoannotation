# ============================================================
# 训练模块 DTO
# ============================================================
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    imageset_id: str
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    img_size: int = Field(default=640, ge=32, le=1920)
    base_model: str = Field(default="yolo11n.pt")
    base_model_id: str = Field(default="")
    task: Literal["detect", "segment", "obb"] = "detect"
    device: str = Field(default="")
    save_to_system: bool = Field(default=True)
    operator: str = Field(default="anonymous")


# ---------- 训练任务提交响应 ----------
class TrainJobSubmitResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    created_at: str
