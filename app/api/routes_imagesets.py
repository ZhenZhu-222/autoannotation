# ============================================================
# 图片集路由 (Controller)
# 职责：图片集增删改查 / 图片审核 / 精细调整 / 下载 / 类别 ID 重映射
# 规范：仅参数接收与异常转换，逻辑委托 service
# ============================================================
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_admin, require_login
from app.core.config import MAX_IMAGE_UPLOAD_BYTES
from app.core.rbac import can_operate_imageset
from app.core.state import AppState
from app.core.utils import stream_upload_to_file
from app.schemas.image import (
    BatchDeleteImagesRequest,
    BatchDeleteImagesResponse,
    ClassMappingResponse,
    ImageDeleteResponse,
    ImageReviewResponse,
    ImagesetDeleteResponse,
    ImagesetRenameResponse,
    ListImagesetsResponse,
    ListImagesResponse,
    RefineImageResponse,
    RemapClassesRequest,
    RemapClassesResponse,
    ReviewSummaryResponse,
    SaveImageRefineRequest,
    UpdateImageReviewRequest,
    UploadFolderResponse,
)
from app.services.imageset_service import ImageSetService
from app.services.refine_service import load_refine_image_payload, rollback_refine_image_labels, save_refine_image_labels
from app.services.remap_service import RemapService
from app.services.task_manager import TaskManager

router = APIRouter(prefix="/api", tags=["imagesets"])


def _ensure_imageset_access(state: AppState, imageset_id: str, user) -> None:
    imageset = state.get_imageset(imageset_id)
    if not imageset:
        raise HTTPException(status_code=404, detail="imageset 不存在")
    if not can_operate_imageset(user, imageset):
        raise HTTPException(status_code=403, detail="无权访问该图片集")


def _ensure_image_access(state: AppState, image_id: str, user) -> None:
    owner = state.image_index.get(image_id)
    if not owner:
        raise HTTPException(status_code=404, detail="image 不存在")
    imageset_id, _ = owner
    _ensure_imageset_access(state, imageset_id, user)


# 列出所有图片集（名称、图片数、标签数等基本信息）
@router.get("/imagesets")
def list_imagesets(
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ListImagesetsResponse:
    return ImageSetService.list_imagesets(state, user=current_user)


# 分页查询指定图片集内的图片列表（支持按有无标签、审核状态、关键词过滤）
@router.get("/imagesets/{imageset_id}/images")
def list_images(
    imageset_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=120, ge=1, le=2000),
    has_label: bool | None = Query(default=None),
    review_status: str = Query(default=""),
    keyword: str = Query(default=""),
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> ListImagesResponse:
    _ensure_imageset_access(state, imageset_id, current_user)
    try:
        return ImageSetService.list_images(
            imageset_id=imageset_id, page=page, page_size=page_size,
            has_label=has_label, review_status=review_status, keyword=keyword,
            state=state, tasks=tasks,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None


# 获取单张图片的精修编辑器数据（图片路径、现有标注框、类别选项、审核状态等）
@router.get("/images/{image_id}/refine")
def get_image_refine_payload(
    image_id: str,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> RefineImageResponse:
    _ensure_image_access(state, image_id, current_user)
    try:
        payload = load_refine_image_payload(state=state, image_id=image_id, tasks=tasks)
    except KeyError:
        raise HTTPException(status_code=404, detail="image 不存在") from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return ImageSetService.refine_payload_to_response(payload)


# 保存单张图片的精修结果（提交编辑后的标注框，同时写备份快照）
@router.post("/images/{image_id}/refine")
def save_image_refine_payload(
    image_id: str,
    req: SaveImageRefineRequest,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> RefineImageResponse:
    _ensure_image_access(state, image_id, current_user)
    try:
        payload = save_refine_image_labels(
            state=state, image_id=image_id,
            boxes=req.boxes, new_classes=req.new_classes,
            operator=current_user.username, tasks=tasks,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="image 不存在") from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return ImageSetService.refine_payload_to_response(payload)


# 回滚单张图片的精修结果（还原到上次保存前的备份标注）
@router.post("/images/{image_id}/refine/rollback")
def rollback_image_refine_payload(
    image_id: str,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> RefineImageResponse:
    _ensure_image_access(state, image_id, current_user)
    try:
        payload = rollback_refine_image_labels(state=state, image_id=image_id, tasks=tasks)
    except KeyError:
        raise HTTPException(status_code=404, detail="image 不存在") from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return ImageSetService.refine_payload_to_response(payload)


# 重命名图片集
@router.patch("/imagesets/{imageset_id}/rename")
def rename_imageset(
    imageset_id: str,
    name: str = Form(...),
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ImagesetRenameResponse:
    _ensure_imageset_access(state, imageset_id, current_user)
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="名称不能为空")
    try:
        return state.rename_imageset(imageset_id, name)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None


# 删除图片集（同时删除磁盘目录和所有图片/标签文件）
@router.delete("/imagesets/{imageset_id}")
def delete_imageset(
    imageset_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> ImagesetDeleteResponse:
    _ = current_user
    try:
        return state.delete_imageset(imageset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None


# 下载图片集为 ZIP 包（打包图片和标签文件）
@router.get("/imagesets/{imageset_id}/download")
def download_imageset(
    imageset_id: str,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_admin),
):
    _ = current_user
    try:
        zip_file = ImageSetService.download_imageset(imageset_id, state, tasks)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    imageset = state.get_imageset(imageset_id)
    return FileResponse(
        path=str(zip_file),
        filename=f"{imageset.name}.zip",
        media_type="application/zip",
        background=None,
    )


# 删除单张图片（同时删除对应标签文件）
@router.delete("/images/{image_id}")
def delete_image(
    image_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> ImageDeleteResponse:
    _ = current_user
    try:
        return state.delete_image(image_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="image 不存在") from None


# 更新单张图片的审核状态（通过/拒绝/待审，记录审核人和备注）
@router.patch("/images/{image_id}/review")
def update_image_review(
    image_id: str,
    req: UpdateImageReviewRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ImageReviewResponse:
    _ensure_image_access(state, image_id, current_user)
    try:
        return state.update_image_review(
            image_id=image_id, review_status=req.review_status,
            reviewer=current_user.username, review_note=req.review_note,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="image 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# 查询图片集的审核汇总（各状态图片数量统计）
@router.get("/imagesets/{imageset_id}/review-summary")
def get_imageset_review_summary(
    imageset_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ReviewSummaryResponse:
    _ensure_imageset_access(state, imageset_id, current_user)
    try:
        return state.imageset_review_summary(imageset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None


# 批量删除图片（一次提交多个 image_id 删除）
@router.post("/images/delete-batch")
def delete_images_batch(
    req: BatchDeleteImagesRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> BatchDeleteImagesResponse:
    _ = current_user
    return state.delete_images(req.image_ids)


# 批量上传图片文件夹（支持同时上传图片和标签文件，自动创建图片集）
@router.post("/imagesets/upload-folder")
async def upload_folder_images(
    files: list[UploadFile] = File(...),
    imageset_name: str | None = Form(default=None),
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> UploadFolderResponse:
    if not files:
        raise HTTPException(status_code=400, detail="未上传文件")
    with TemporaryDirectory(prefix="api_upload_folder_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        staged_files: list[tuple[str, Path]] = []
        for idx, file in enumerate(files):
            staged = tmp_path / f"{idx:06d}.bin"
            try:
                await stream_upload_to_file(file, staged, MAX_IMAGE_UPLOAD_BYTES)
            except ValueError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from None
            staged_files.append((file.filename or "image.jpg", staged))
        return ImageSetService.create_from_staged_paths(
            state=state, files=staged_files, name=imageset_name, creator_id=current_user.id,
        )


# 查询图片集当前的类别 ID→名称映射表（用于重映射前预览）
@router.get("/imagesets/{imageset_id}/class-mapping")
def get_class_mapping(
    imageset_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ClassMappingResponse:
    _ensure_imageset_access(state, imageset_id, current_user)
    try:
        return RemapService.get_class_mapping(imageset_id, state)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None


# 重映射图片集的类别 ID（将旧 ID 批量替换成新 ID，同时修改所有标签文件）
@router.post("/imagesets/{imageset_id}/remap-classes")
def remap_classes(
    imageset_id: str,
    req: RemapClassesRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> RemapClassesResponse:
    _ensure_imageset_access(state, imageset_id, current_user)
    try:
        return RemapService.remap_classes(imageset_id, req.id_mapping, state)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
