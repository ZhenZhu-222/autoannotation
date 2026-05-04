# ============================================================
# AI 路由 (Controller)
# 职责：Qwen 大模型智能推荐类别
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_state
from app.core.auth import require_login
from app.core.config import MAX_IMAGE_UPLOAD_BYTES
from app.core.state import AppState
from app.core.utils import read_upload_with_limit
from app.schemas.ai import AiSuggestClassesResponse
from app.services.qwen_service import QwenService

router = APIRouter(prefix="/api", tags=["ai"])


# Qwen 大模型智能推荐类别（传入文字描述或参考图片，让 AI 从模型类别里挑选合适的打标类别）
@router.post("/ai/qwen/suggest-classes")
async def qwen_suggest_classes(
    model_id: str = Form(...),
    description: str = Form(default=""),
    qwen_model: str = Form(default=""),
    api_key: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> AiSuggestClassesResponse:
    _ = current_user
    model = state.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="model_id 不存在")
    if not description.strip() and not files:
        raise HTTPException(status_code=400, detail="请至少提供文字描述或参考图片")

    image_payload = []
    for f in files:
        try:
            content = await read_upload_with_limit(f, MAX_IMAGE_UPLOAD_BYTES)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        image_payload.append((f.filename or "ref.jpg", content, f.content_type or ""))

    try:
        suggestion = await QwenService.suggest_classes(
            description=description,
            class_names=model.classes,
            images=image_payload,
            qwen_model=qwen_model,
            api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"AI 服务异常: {exc}") from None

    suggestion.update(
        {
            "model_id": model.id,
            "model_name": model.name,
            "class_count": len(model.classes),
        }
    )
    return suggestion
