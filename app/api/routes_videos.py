# ============================================================
# 视频路由 (Controller)
# 职责：视频上传 / 本地导入 / 抽帧任务创建与查询
# ============================================================
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_login
from app.core.config import MAX_VIDEO_UPLOAD_BYTES, SUPPORTED_VIDEO_EXTENSIONS, UPLOAD_VIDEOS_DIR
from app.core.state import AppState
from app.core.utils import ensure_unique_path, sanitize_filename, stream_upload_to_file
from app.schemas.common import JobStatusResponse, JobSubmitResponse
from app.schemas.extract import CreateExtractJobRequest
from app.schemas.video import VideoUploadResponse
from app.services.extract_service import ExtractService
from app.services.task_manager import TaskManager

router = APIRouter(prefix="/api", tags=["videos"])


# 上传视频文件到服务器（流式写入，支持大文件）
@router.post("/videos/upload")
async def upload_video(
    file: UploadFile = File(...),
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> VideoUploadResponse:
    _ = current_user
    filename = sanitize_filename(file.filename or "video.mp4", "video.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="视频格式不支持")

    output = ensure_unique_path(UPLOAD_VIDEOS_DIR / filename)
    try:
        await stream_upload_to_file(file, output, MAX_VIDEO_UPLOAD_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from None

    record = state.register_video(filename=filename, path=str(output))
    return {"video_id": record.id, "filename": record.filename, "path": record.path, "created_at": record.created_at}


# 创建视频抽帧任务（按秒数间隔抽帧，生成图片集）
@router.post("/extract/jobs")
def create_extract_job(
    req: CreateExtractJobRequest,
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobSubmitResponse:
    if not state.get_video(req.video_id):
        raise HTTPException(status_code=404, detail="video_id 不存在")

    payload = req.model_dump()

    def runner(ctx):
        return ExtractService.run_extract_job(
            state=state,
            job_ctx=ctx,
            video_id=req.video_id,
            sample_every_seconds=req.sample_every_seconds,
            imageset_name=req.imageset_name,
            operator=current_user.username,
            creator_id=current_user.id,
        )

    job = tasks.submit(
        job_type="extract",
        payload=payload,
        runner=runner,
        operator_id=current_user.id,
        operator_username=current_user.username,
    )
    return {"id": job.id, "job_type": job.job_type, "status": job.status, "payload": job.payload, "created_at": job.created_at}


# 查询抽帧任务状态（进度、是否完成、生成的图片集 ID）
@router.get("/extract/jobs/{job_id}")
def get_extract_job(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobStatusResponse:
    job = tasks.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    if job.job_type != "extract":
        raise HTTPException(status_code=400, detail="job 类型不是 extract")
    if current_user.role != "admin" and job.operator_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权查看该任务")
    return job.to_dict()
