# ============================================================
# 视频路由 (Controller)
# 职责：视频上传 / 本地导入 / 抽帧任务创建与查询
# ============================================================
from __future__ import annotations

import ipaddress
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_login
from app.core.config import MAX_VIDEO_UPLOAD_BYTES, SUPPORTED_VIDEO_EXTENSIONS, UPLOAD_VIDEOS_DIR
from app.core.state import AppState
from app.core.utils import ensure_unique_path, sanitize_filename, stream_upload_to_file
from app.schemas.common import JobStatusResponse, JobSubmitResponse
from app.schemas.extract import CreateExtractJobRequest
from app.schemas.video import ImportLocalVideoRequest, VideoUploadResponse
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


# 从本机/内网本地路径导入视频（仅允许本地或内网 IP 调用）
@router.post("/videos/import-local")
def import_local_video(
    req: ImportLocalVideoRequest,
    request: Request,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> VideoUploadResponse:
    _ = current_user
    # Docker 内部署时，浏览器走宿主机访问，IP 会是 172.17.0.1 等网桥地址
    client_host = (request.client.host if request.client else "") or ""
    _ALWAYS_ALLOW = {"127.0.0.1", "::1", "localhost", "testclient"}
    is_local = client_host in _ALWAYS_ALLOW
    if not is_local:
        try:
            is_local = ipaddress.ip_address(client_host).is_private
        except ValueError:
            is_local = False
    if not is_local:
        raise HTTPException(status_code=403, detail="仅允许本机或内网导入本地路径")

    src = Path(req.path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail="本地视频路径不存在")

    filename = sanitize_filename(src.name or "video.mp4", "video.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="视频格式不支持")

    size = src.stat().st_size
    if size > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"文件大小超过限制 ({MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)}MB)")

    output = ensure_unique_path(UPLOAD_VIDEOS_DIR / filename)
    try:
        shutil.copy2(src, output)
    except OSError as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"导入本地视频失败: {exc}") from exc

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
