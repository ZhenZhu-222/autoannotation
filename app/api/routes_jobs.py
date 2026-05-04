# ============================================================
# 任务路由 (Controller)
# 职责：任务列表 / 查询 / 取消 / 历史记录 / 产物清理
# 规范：仅参数接收与异常转换，逻辑委托 service
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import get_state, get_task_manager
from app.core.auth import require_admin, require_login
from app.core.rbac import can_view_job
from app.core.state import AppState
from app.schemas.common import JobCancelResponse, JobStatusResponse
from app.schemas.jobs import (
    ClearWorkLedgerResponse,
    DeleteJobArtifactsResponse,
    DeleteJobHistoryResponse,
    JobHistoryListResponse,
    JobListResponse,
    WorkLedgerListResponse,
)
from app.services.job_cleanup_service import JobCleanupService
from app.services.task_manager import TaskManager
from app.services.work_ledger_service import WorkLedgerService

router = APIRouter(prefix="/api", tags=["jobs"])


# 查询当前正在运行或队列中的所有任务（实时任务，不包含历史）
@router.get("/jobs")
def list_jobs(
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobListResponse:
    rows = tasks.list_jobs()
    if current_user.role != "admin":
        rows = [job for job in rows if job.operator_id == current_user.id]
    return {"items": [job.to_dict() for job in rows]}


# 查询历史任务记录（已完成/失败/取消），支持按类型和操作人过滤
@router.get("/jobs/history")
def list_job_history(
    limit: int = Query(default=200, ge=1, le=2000),
    job_type: str = Query(default=""),
    operator: str = Query(default=""),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobHistoryListResponse:
    if current_user.role == "admin":
        return {"items": tasks.list_history(limit=limit, job_type=job_type, operator=operator)}
    return {"items": tasks.list_history(limit=limit, job_type=job_type, operator_id=current_user.id)}


# 查询业务流水单：把分派/验收记录和自动任务记录合成一张时间线
@router.get("/jobs/ledger")
def list_work_ledger(
    limit: int = Query(default=200, ge=1, le=2000),
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> WorkLedgerListResponse:
    return WorkLedgerService.list_entries(
        state=state,
        tasks=tasks,
        current_user=current_user,
        limit=limit,
    )


# 管理员格式化业务流水单：只隐藏旧流水，不删除数据集/图片/模型/任务产物
@router.delete("/jobs/ledger")
def clear_work_ledger(
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_admin),
) -> ClearWorkLedgerResponse:
    return WorkLedgerService.clear_visible_entries(
        state=state,
        tasks=tasks,
        current_user=current_user,
    )


# 查询单个任务的实时状态（进度、状态、结果等）
@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobStatusResponse:
    job = tasks.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job 不存在")
    if not can_view_job(current_user, job):
        raise HTTPException(status_code=403, detail="无权查看该任务")
    return job.to_dict()


# 取消指定任务（设置取消标志，任务线程检测到后自行中止）
@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_login),
) -> JobCancelResponse:
    existing = tasks.get(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="job 不存在")
    if not can_view_job(current_user, existing):
        raise HTTPException(status_code=403, detail="无权取消该任务")
    try:
        job = tasks.cancel(job_id, reason="用户取消任务")
    except KeyError:
        raise HTTPException(status_code=404, detail="job 不存在") from None
    return {
        "job_id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "cancel_requested": bool(job.cancel_requested),
        "progress": job.progress,
        "progress_text": job.progress_text,
        "finished_at": job.finished_at,
    }


# 删除指定任务的产物文件（overlay 图、manifest.csv 等，释放磁盘空间）
@router.delete("/jobs/{job_id}/artifacts")
def delete_job_artifacts(
    job_id: str,
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_admin),
) -> DeleteJobArtifactsResponse:
    _ = current_user
    try:
        return JobCleanupService.delete_artifacts_for_job(job_id, tasks)
    except KeyError:
        raise HTTPException(status_code=404, detail="job 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# 删除历史记录（管理员操作，可级联删除产物文件和图片集）
@router.delete("/jobs/history/{job_id}")
def delete_job_history_entry(
    request: Request,
    job_id: str,
    with_artifacts: bool = Query(default=True),
    with_imageset: bool = Query(default=True),
    admin_pwd: str = Query(default=""),
    state: AppState = Depends(get_state),
    tasks: TaskManager = Depends(get_task_manager),
    current_user=Depends(require_admin),
) -> DeleteJobHistoryResponse:
    _ = (request, admin_pwd, current_user)
    try:
        return JobCleanupService.delete_history_with_cleanup(
            job_id, tasks, state, with_artifacts=with_artifacts, with_imageset=with_imageset,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="运行记录不存在") from None
