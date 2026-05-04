# ============================================================
# YOLO 打标路由 (Controller)
# 职责：创建打标任务 / 查询任务 / 预览结果 / 回滚标签 / 产物查看
# 规范：仅参数接收与异常转换，逻辑委托 service
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_login
from app.core.rbac import can_operate_imageset, can_view_job
from app.core.state import AppState
from app.schemas.annotate import (
    AnnotateArtifactsResponse,
    AnnotatePreviewResponse,
    AnnotateRollbackResponse,
    CreateAnnotateJobRequest,
    ResolveMappingRequest,
    ResolveMappingResponse,
)
from app.schemas.common import JobStatusResponse, JobSubmitResponse
from app.services.annotate_preview_service import AnnotatePreviewService
from app.services.annotation_service import AnnotationService
from app.services.task_manager import TaskManager

router = APIRouter(prefix="/api", tags=["annotate"])


def _ensure_imageset_access(state: AppState, imageset_id: str, user) -> None:
    imageset = state.get_imageset(imageset_id)
    if not imageset:
        raise HTTPException(status_code=404, detail="imageset_id 不存在")
    if not can_operate_imageset(user, imageset):
        raise HTTPException(status_code=403, detail="无权操作该图片集")


def _ensure_job_access(tasks: TaskManager, job_id: str, user):
    job = tasks.get(job_id)
    if job:
        if not can_view_job(user, job):
            raise HTTPException(status_code=403, detail="无权查看该任务")
        return job
    history = tasks.get_history_entry(job_id)
    if history:
        if not can_view_job(user, history):
            raise HTTPException(status_code=403, detail="无权查看该任务")
        return history
    raise HTTPException(status_code=404, detail="job 不存在")


# 计算模型类别 ID→名称的映射表，前端用于渲染「选择要打哪些类别」的下拉框
@router.post("/annotate/resolve-mapping")
def resolve_mapping(
    req: ResolveMappingRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> ResolveMappingResponse:
    """计算目标类别 ID→name 映射表，前端用于渲染映射下拉框。"""
    _ensure_imageset_access(state, req.imageset_id, current_user)
    try:
        candidate = AnnotationService.resolve_mapping(
            state=state,
            model_id=req.model_id,
            imageset_id=req.imageset_id,
            selected_class_ids=req.selected_class_ids,
            target_classes=req.target_classes if req.target_classes else None,
            label_mode=req.label_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    # 返回有序列表，前端直接渲染
    items = [{"id": cid, "name": name} for cid, name in sorted(candidate.items())]
    return {"items": items}


# 创建 YOLO 自动打标任务（用指定模型对指定图片集运行推理并写入标签）
@router.post("/annotate/jobs")
def create_annotate_job(
    req: CreateAnnotateJobRequest,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobSubmitResponse:
    if not state.get_model(req.model_id):
        raise HTTPException(status_code=404, detail="model_id 不存在")
    if not state.get_imageset(req.imageset_id):
        raise HTTPException(status_code=404, detail="imageset_id 不存在")
    _ensure_imageset_access(state, req.imageset_id, current_user)
    if not req.mapping_confirmed:
        raise HTTPException(status_code=400, detail='请先完成类别映射并点击"确认映射（必做）"')

    payload = req.model_dump()
    payload["operator"] = current_user.username

    def runner(ctx):
        return AnnotationService.run_annotate_job(
            state=state, job_ctx=ctx,
            model_id=req.model_id, imageset_id=req.imageset_id,
            selected_class_ids=req.selected_class_ids, target_classes=req.target_classes,
            conf=req.conf, iou=req.iou, device=req.device,
            save_overlays=req.save_overlays, label_mode=req.label_mode,
            update_imageset_labels=req.update_imageset_labels,
            round_tag=req.round_tag, class_id_overrides=req.class_id_overrides,
            operator=current_user.username,
            label_task=req.label_task,
        )

    job = tasks.submit(
        job_type="annotate",
        payload=payload,
        runner=runner,
        operator_id=current_user.id,
        operator_username=current_user.username,
    )
    return {"id": job.id, "job_type": job.job_type, "status": job.status, "payload": job.payload, "created_at": job.created_at}


# 查询指定打标任务的实时状态（进度、状态、错误信息）
@router.get("/annotate/jobs/{job_id}")
def get_annotate_job(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobStatusResponse:
    job = _ensure_job_access(tasks, job_id, current_user)
    job_type = job.job_type if hasattr(job, "job_type") else str(job.get("job_type") or "")
    if job_type != "annotate":
        raise HTTPException(status_code=400, detail="job 类型不是 annotate")
    if isinstance(job, dict):
        raise HTTPException(status_code=404, detail="job 不在运行中")
    return job.to_dict()


# 获取打标任务产物（overlay 图片路径、manifest.csv、输出目录）
@router.get("/annotate/jobs/{job_id}/artifacts")
def get_annotate_artifacts(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> AnnotateArtifactsResponse:
    _ensure_job_access(tasks, job_id, current_user)
    try:
        payload = AnnotatePreviewService.get_artifacts(job_id, tasks)
        if current_user.role != "admin":
            artifacts = dict(payload.get("artifacts") or {})
            for key in ("zip_file", "cvat_ultralytics_zip", "cleanup_api"):
                artifacts.pop(key, None)
            payload["artifacts"] = artifacts
        return payload
    except KeyError:
        raise HTTPException(status_code=404, detail="job 不存在") from None
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


# 回滚打标结果（将图片集标签还原到打标前的备份状态）
@router.post("/annotate/jobs/{job_id}/rollback")
def rollback_annotate_labels(
    job_id: str,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> AnnotateRollbackResponse:
    job = _ensure_job_access(tasks, job_id, current_user)
    job_payload = job.payload if hasattr(job, "payload") else dict(job.get("payload") or {})
    imageset_id = str((job_payload or {}).get("imageset_id") or "")
    if imageset_id:
        _ensure_imageset_access(state, imageset_id, current_user)
    try:
        return AnnotatePreviewService.rollback_labels(job_id, tasks, state)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None


# 分页预览打标结果（展示图片+标签框，支持关键词过滤和只看有框）
@router.get("/annotate/jobs/{job_id}/preview")
def get_annotate_preview(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    only_with_boxes: bool = Query(default=False),
    keyword: str = Query(default=""),
    prefer_current_labels: bool = Query(default=False),
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> AnnotatePreviewResponse:
    _ensure_job_access(tasks, job_id, current_user)
    try:
        return AnnotatePreviewService.build_preview(
            job_id=job_id, page=page, page_size=page_size,
            only_with_boxes=only_with_boxes, keyword=keyword,
            prefer_current_labels=prefer_current_labels,
            state=state, tasks=tasks,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="job 不存在") from None
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
