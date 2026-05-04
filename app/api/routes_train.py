# ============================================================
# 训练路由 (Controller)
# 职责：YOLO 训练任务创建 / 查询 / 模型下载
# ============================================================
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_admin, require_login
from app.core.rbac import can_operate_imageset, can_view_job
from app.core.config import SUPPORTED_MODEL_EXTENSIONS
from app.core.state import AppState
from app.schemas.common import JobStatusResponse
from app.schemas.train import TrainJobSubmitResponse, TrainRequest
from app.services.task_manager import TaskManager
from app.services.training_service import TrainingService

router = APIRouter(prefix="/api", tags=["train"])


# 创建 YOLO 训练任务（用指定图片集训练自定义模型）
@router.post("/train/jobs")
def create_train_job(
    req: TrainRequest,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> TrainJobSubmitResponse:
    imageset = state.get_imageset(req.imageset_id)
    if not imageset:
        raise HTTPException(status_code=404, detail="imageset_id not found")
    if not can_operate_imageset(current_user, imageset):
        raise HTTPException(status_code=403, detail="无权操作该图片集")

    try:
        TrainingService.validate_trainable_imageset(Path(imageset.dir_path), req.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    payload = req.model_dump()

    def runner(ctx):
        return TrainingService.run_train_job(
            state=state,
            job_ctx=ctx,
            imageset_id=req.imageset_id,
            epochs=req.epochs,
            batch_size=req.batch_size,
            img_size=req.img_size,
            base_model=req.base_model,
            base_model_id=req.base_model_id,
            task=req.task,
            device=req.device,
            save_to_system=req.save_to_system,
            operator=current_user.username,
            creator_id=current_user.id,
        )

    job = tasks.submit(
        job_type="train",
        payload=payload,
        runner=runner,
        operator_id=current_user.id,
        operator_username=current_user.username,
    )
    return {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "created_at": job.created_at,
    }


# 查询训练任务实时状态（进度、loss、是否完成）
@router.get("/train/jobs/{job_id}")
def get_train_job(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobStatusResponse:
    job = tasks.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.job_type != "train":
        raise HTTPException(status_code=400, detail="job type is not train")
    if not can_view_job(current_user, job):
        raise HTTPException(status_code=403, detail="无权查看该任务")
    return job.to_dict()


# ---- Model download ----


def _model_download_filename(model_name: str, model_path: Path, model_type: str) -> str:
    name = Path((model_name or model_path.name or "model").replace("\\", "/")).name or "model"
    actual_suffix = model_path.suffix.lower()
    if actual_suffix not in SUPPORTED_MODEL_EXTENSIONS:
        model_type_suffix = f".{(model_type or '').strip().lower().lstrip('.')}"
        actual_suffix = model_type_suffix if model_type_suffix in SUPPORTED_MODEL_EXTENSIONS else ".pt"

    current_suffix = Path(name).suffix.lower()
    if current_suffix == actual_suffix:
        return name
    if current_suffix in SUPPORTED_MODEL_EXTENSIONS:
        return f"{Path(name).stem}{actual_suffix}"
    return f"{name}{actual_suffix}"


# 下载训练好的模型文件（管理员操作）
@router.get("/models/{model_id}/download")
def download_model(
    request: Request,
    model_id: str,
    admin_pwd: str = Query(default=""),
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
):
    _ = (request, admin_pwd, current_user)
    model = state.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="model not found")
    model_path = Path(model.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="model file missing")
    return FileResponse(
        path=str(model_path),
        filename=_model_download_filename(model.name, model_path, model.model_type),
        media_type="application/octet-stream",
    )
