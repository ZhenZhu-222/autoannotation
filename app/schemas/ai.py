# ============================================================
# AI 智能推荐响应 DTO
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ClassItem




class AiSuggestClassesResponse(BaseModel):
    selected_class_ids: list[int] = Field(default_factory=list)
    selected_classes: list[ClassItem] = Field(default_factory=list)
    reason: str = ""
    provider_model: str = ""
    image_count: int = 0
    raw_text: str = ""
    model_id: str = ""
    model_name: str = ""
    class_count: int = 0
