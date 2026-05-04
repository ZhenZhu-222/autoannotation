# ============================================================
# 模型路由 (Controller)
# 职责：模型上传 / 列表 / 查询类别 / 删除 / 解析 classes 文件
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_state
from app.core.auth import require_admin, require_login
from app.core.config import MAX_CLASSES_FILE_BYTES, MAX_MODEL_UPLOAD_BYTES
from app.core.state import AppState
from app.core.utils import read_upload_with_limit
from app.schemas.model import (
    ListModelsResponse,
    ModelClassesResponse,
    ModelDeleteResponse,
    ModelRenameResponse,
    ModelUploadResponse,
    ParseClassesResponse,
)
from app.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["models"])


# 上传模型文件（.pt/.onnx 等）和可选的 classes.txt，保存到系统
@router.post("/models/upload")
async def upload_model(
    model_file: UploadFile = File(...),
    classes_file: UploadFile | None = File(default=None),
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ModelUploadResponse:
    try:
        model_bytes = await read_upload_with_limit(model_file, MAX_MODEL_UPLOAD_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None
    classes_bytes = None
    if classes_file:
        try:
            classes_bytes = await read_upload_with_limit(classes_file, MAX_CLASSES_FILE_BYTES)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None

    return ModelService.save_model_upload(
        state=state,
        model_filename=model_file.filename or "model.pt",
        model_bytes=model_bytes,
        classes_filename=classes_file.filename if classes_file else None,
        classes_bytes=classes_bytes,
        creator_id=current_user.id,
    )


# 列出所有已上传的模型（重新扫描磁盘后返回列表）
@router.get("/models")
def list_models(
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ListModelsResponse:
    _ = current_user
    state.rescan_models()
    items = []
    for model in state.list_models():
        creator = state.get_user(getattr(model, "creator_id", ""))
        items.append(
            {
                "model_id": model.id,
                "name": model.name,
                "model_type": model.model_type,
                "class_source": model.class_source,
                "task": model.task,
                "classes_count": len(model.classes),
                "created_at": model.created_at,
                "creator_id": getattr(model, "creator_id", ""),
                "creator_username": creator.username if creator else "已删除用户",
            }
        )
    return {"items": items}


# 查询指定模型的类别列表（用于打标前选择要打哪些类别）
@router.get("/models/{model_id}/classes")
def get_model_classes(
    model_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ModelClassesResponse:
    _ = current_user
    model = state.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="model 不存在")
    return {
        "model_id": model.id,
        "name": model.name,
        "task": model.task,
        "classes": [{"id": idx, "name": name} for idx, name in enumerate(model.classes)],
    }


# 重命名指定模型
@router.patch("/models/{model_id}/name")
def rename_model(
    model_id: str,
    body: dict,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ModelRenameResponse:
    _ = current_user
    new_name = (body.get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="名称不能为空")
    try:
        return state.rename_model(model_id, new_name)
    except KeyError:
        raise HTTPException(status_code=404, detail="model 不存在") from None


# 删除指定模型（同时删除磁盘文件）
@router.delete("/models/{model_id}")
def delete_model(
    model_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> ModelDeleteResponse:
    _ = current_user
    try:
        return state.delete_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="model 不存在") from None


# 解析上传的 classes.txt 文件，返回类别列表（用于上传模型时预览类别）
@router.post("/classes/parse")
async def parse_classes_file(
    classes_file: UploadFile = File(...),
    current_user=Depends(require_login),
) -> ParseClassesResponse:
    _ = current_user
    try:
        content = await read_upload_with_limit(classes_file, MAX_CLASSES_FILE_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None
    classes = ModelService.parse_classes_file(classes_file.filename or "classes.txt", content)
    return {
        "count": len(classes),
        "classes": [{"id": idx, "name": name} for idx, name in enumerate(classes)],
    }
