# ============================================================
# 业务流水单服务
# 职责：把“分派/提交验收/验收”和“自动任务运行记录”合成一张可读流水
# 说明：这里不是安全审计系统；它解决管理页面要看“谁在什么时间做了什么活”。
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core import config
from app.core.models import ImageSetRecord, JobRecord, UserRecord
from app.core.rbac import can_view_job
from app.core.state import AppState
from app.core.utils import now_iso
from app.services.task_manager import TaskManager


JOB_ACTION_LABELS = {
    "extract": "视频抽帧",
    "annotate": "权重打标",
    "qwen_annotate": "AI打标",
    "train": "权重训练",
}

JOB_STATUS_LABELS = {
    "queued": "排队中",
    "running": "运行中",
    "succeeded": "已完成",
    "failed": "失败",
    "cancelled": "已取消",
    "canceled": "已取消",
}

ASSIGNMENT_STATUS_LABELS = {
    "assigned": "已分派",
    "submitted": "提交验收",
    "accepted": "已验收",
    "rejected": "已驳回",
    "canceled": "已取消",
    "todo": "待处理",
    "in_progress": "处理中",
    "reviewed": "已审核",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ledger_state_file() -> Path:
    return config.SYSTEM_DIR / "work_ledger_state.json"


def _load_hidden_before() -> str:
    state_file = _ledger_state_file()
    if not state_file.exists():
        return ""
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return _clean(payload.get("hidden_before"))


def _save_hidden_before(value: str, actor_username: str = "") -> None:
    state_file = _ledger_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    # “格式化流水”只写一个时间水位线，隐藏旧流水；不删除图片集、模型、任务产物和精修备份。
    state_file.write_text(
        json.dumps(
            {
                "hidden_before": value,
                "cleared_at": value,
                "cleared_by": actor_username,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _imageset_name(state: AppState, imageset_id: str) -> str:
    imageset = state.get_imageset(imageset_id)
    return imageset.name if imageset else ""


def _assignment_visible(current_user: UserRecord, imageset: ImageSetRecord, assignee_id: str) -> bool:
    if current_user.role == "admin":
        return True
    # operator 只看自己的流水；自己创建的数据集也允许看到。
    return assignee_id == current_user.id or imageset.creator_id == current_user.id


def _entry(
    *,
    event_id: str,
    event_type: str,
    action: str,
    status: str,
    status_label: str,
    occurred_at: str,
    actor_id: str = "",
    actor_username: str = "",
    target_user_id: str = "",
    target_username: str = "",
    imageset_id: str = "",
    imageset_name: str = "",
    job_id: str = "",
    job_type: str = "",
    note: str = "",
    detail: str = "",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "event_type": event_type,
        "action": action,
        "status": status,
        "status_label": status_label,
        "occurred_at": occurred_at,
        "actor_id": actor_id,
        "actor_username": actor_username,
        "target_user_id": target_user_id,
        "target_username": target_username,
        "imageset_id": imageset_id,
        "imageset_name": imageset_name,
        "job_id": job_id,
        "job_type": job_type,
        "note": note,
        "detail": detail,
    }


def _append_assignment_events(
    entries: list[dict[str, Any]],
    *,
    state: AppState,
    current_user: UserRecord,
    imageset: ImageSetRecord,
    assignee_id: str,
    assignment: dict[str, Any],
    assignee_username: str,
) -> None:
    if not _assignment_visible(current_user, imageset, assignee_id):
        return

    assigned_at = _clean(assignment.get("assigned_at"))
    assigned_by_id = _clean(assignment.get("assigned_by_id"))
    assigned_by_username = _clean(assignment.get("assigned_by_username")) or "管理员/系统"
    if assigned_at:
        entries.append(_entry(
            event_id=f"assignment:{imageset.id}:{assignee_id}:assigned:{assigned_at}",
            event_type="assignment",
            action="分派任务",
            status="assigned",
            status_label=ASSIGNMENT_STATUS_LABELS["assigned"],
            occurred_at=assigned_at,
            actor_id=assigned_by_id,
            actor_username=assigned_by_username,
            target_user_id=assignee_id,
            target_username=assignee_username,
            imageset_id=imageset.id,
            imageset_name=imageset.name,
        ))

    submitted_at = _clean(assignment.get("submitted_at"))
    if submitted_at:
        entries.append(_entry(
            event_id=f"assignment:{imageset.id}:{assignee_id}:submitted:{submitted_at}",
            event_type="assignment",
            action="提交验收",
            status="submitted",
            status_label=ASSIGNMENT_STATUS_LABELS["submitted"],
            occurred_at=submitted_at,
            actor_id=assignee_id,
            actor_username=assignee_username,
            target_user_id=assignee_id,
            target_username=assignee_username,
            imageset_id=imageset.id,
            imageset_name=imageset.name,
            note=_clean(assignment.get("submit_note")),
        ))

    final_status = _clean(assignment.get("status"))
    reviewed_at = _clean(assignment.get("reviewed_at"))
    if final_status in {"accepted", "rejected", "canceled"} and reviewed_at:
        entries.append(_entry(
            event_id=f"assignment:{imageset.id}:{assignee_id}:{final_status}:{reviewed_at}",
            event_type="assignment",
            action=ASSIGNMENT_STATUS_LABELS[final_status],
            status=final_status,
            status_label=ASSIGNMENT_STATUS_LABELS[final_status],
            occurred_at=reviewed_at,
            actor_id=_clean(assignment.get("reviewer_id")),
            actor_username=_clean(assignment.get("reviewer_username")) or "管理员/系统",
            target_user_id=assignee_id,
            target_username=assignee_username,
            imageset_id=imageset.id,
            imageset_name=imageset.name,
            note=_clean(assignment.get("review_note")),
        ))


def _append_image_review_events(
    entries: list[dict[str, Any]],
    *,
    current_user: UserRecord,
    imageset: ImageSetRecord,
) -> None:
    if current_user.role != "admin" and imageset.creator_id != current_user.id and current_user.id not in imageset.assignee_ids:
        return
    for image in imageset.images:
        if not image.reviewed_at:
            continue
        status = _clean(image.review_status)
        entries.append(_entry(
            event_id=f"image_review:{image.id}:{image.reviewed_at}",
            event_type="image_review",
            action="图片审核",
            status=status,
            status_label=ASSIGNMENT_STATUS_LABELS.get(status, status or "已记录"),
            occurred_at=image.reviewed_at,
            actor_username=image.reviewer or "未知",
            imageset_id=imageset.id,
            imageset_name=imageset.name,
            note=image.review_note,
            detail=image.filename,
        ))


def _append_latest_refine_events(
    entries: list[dict[str, Any]],
    *,
    current_user: UserRecord,
    imageset: ImageSetRecord,
) -> None:
    if current_user.role != "admin" and imageset.creator_id != current_user.id and current_user.id not in imageset.assignee_ids:
        return
    backup_dir = Path(imageset.dir_path) / ".refine_backups"
    if not backup_dir.exists():
        return
    for backup_file in backup_dir.glob("*.json"):
        try:
            payload = json.loads(backup_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        saved_at = _clean(payload.get("saved_at"))
        if not saved_at:
            continue
        entries.append(_entry(
            event_id=f"refine:{_clean(payload.get('image_id'))}:{saved_at}",
            event_type="refine",
            action="数据精修保存",
            status="saved",
            status_label="已保存",
            occurred_at=saved_at,
            actor_username=_clean(payload.get("operator")) or "anonymous",
            imageset_id=imageset.id,
            imageset_name=imageset.name,
            detail=_clean(payload.get("filename")),
        ))


def _job_entry_from_record(state: AppState, job: JobRecord) -> dict[str, Any]:
    payload = dict(job.payload or {})
    imageset_id = _clean(payload.get("imageset_id"))
    imageset_name = _clean(payload.get("imageset_name")) or _imageset_name(state, imageset_id)
    action = JOB_ACTION_LABELS.get(job.job_type, job.job_type)
    status_label = JOB_STATUS_LABELS.get(job.status, job.status)
    return _entry(
        event_id=f"job:{job.id}:{job.status}",
        event_type="job",
        action=action,
        status=job.status,
        status_label=status_label,
        occurred_at=job.finished_at or job.started_at or job.created_at,
        actor_id=job.operator_id,
        actor_username=_clean(payload.get("operator_username")) or _clean(payload.get("operator")) or "anonymous",
        imageset_id=imageset_id,
        imageset_name=imageset_name,
        job_id=job.id,
        job_type=job.job_type,
        detail=job.progress_text or "",
        note=job.error or "",
    )


def _job_entry_from_history(state: AppState, row: dict[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("payload") or {})
    summary = dict(row.get("result_summary") or {})
    imageset_id = _clean(summary.get("imageset_id")) or _clean(payload.get("imageset_id"))
    imageset_name = _clean(summary.get("imageset_name")) or _clean(payload.get("imageset_name")) or _imageset_name(state, imageset_id)
    job_type = _clean(row.get("job_type"))
    status = _clean(row.get("status"))
    action = JOB_ACTION_LABELS.get(job_type, job_type or "运行任务")
    status_label = JOB_STATUS_LABELS.get(status, status or "已记录")
    detail_parts = []
    if summary.get("saved_images") is not None:
        detail_parts.append(f"抽帧 {summary.get('saved_images')} 张")
    if summary.get("total_final_boxes") is not None:
        detail_parts.append(f"最终框 {summary.get('total_final_boxes')}")
    if summary.get("saved_model_id"):
        detail_parts.append(f"模型 {summary.get('saved_model_id')}")
    return _entry(
        event_id=f"job_history:{row.get('job_id')}:{row.get('finished_at') or row.get('created_at')}",
        event_type="job",
        action=action,
        status=status,
        status_label=status_label,
        occurred_at=_clean(row.get("finished_at")) or _clean(row.get("created_at")),
        actor_id=_clean(row.get("operator_id")),
        actor_username=_clean(row.get("operator_username")) or _clean(row.get("operator")) or "anonymous",
        imageset_id=imageset_id,
        imageset_name=imageset_name,
        job_id=_clean(row.get("job_id")),
        job_type=job_type,
        detail="；".join(detail_parts),
        note=_clean(row.get("error")),
    )


class WorkLedgerService:
    @staticmethod
    def list_entries(
        *,
        state: AppState,
        tasks: TaskManager,
        current_user: UserRecord,
        limit: int = 200,
        include_hidden: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        entries: list[dict[str, Any]] = []

        # 图片集分派流水：分派、提交验收、验收通过/驳回、取消分派。
        for imageset in state.imagesets.values():
            states = dict(getattr(imageset, "assignee_states", {}) or {})
            for assignee_id, assignment in states.items():
                assignee = state.get_user(assignee_id)
                _append_assignment_events(
                    entries,
                    state=state,
                    current_user=current_user,
                    imageset=imageset,
                    assignee_id=assignee_id,
                    assignment=dict(assignment or {}),
                    assignee_username=assignee.username if assignee else "已删除用户",
                )
            for archived in getattr(imageset, "assignee_history", []) or []:
                assignee_id = _clean(archived.get("user_id"))
                _append_assignment_events(
                    entries,
                    state=state,
                    current_user=current_user,
                    imageset=imageset,
                    assignee_id=assignee_id,
                    assignment=dict(archived or {}),
                    assignee_username=_clean(archived.get("username")) or "已删除用户",
                )
            _append_image_review_events(entries, current_user=current_user, imageset=imageset)
            _append_latest_refine_events(entries, current_user=current_user, imageset=imageset)

        # 运行任务流水：自动打标、AI 打标、训练、抽帧。
        for job in tasks.list_jobs():
            if can_view_job(current_user, job):
                entries.append(_job_entry_from_record(state, job))
        history_limit = max(limit, 500)
        if current_user.role == "admin":
            history = tasks.list_history(limit=history_limit)
        else:
            history = tasks.list_history(limit=history_limit, operator_id=current_user.id)
        for row in history:
            entries.append(_job_entry_from_history(state, row))

        hidden_before = "" if include_hidden else _load_hidden_before()
        entries = [
            item for item in entries
            if _clean(item.get("occurred_at")) and (not hidden_before or _clean(item.get("occurred_at")) > hidden_before)
        ]
        entries.sort(key=lambda item: _clean(item.get("occurred_at")), reverse=True)
        return {"items": entries[:limit]}

    @staticmethod
    def clear_visible_entries(
        *,
        state: AppState,
        tasks: TaskManager,
        current_user: UserRecord,
    ) -> dict[str, Any]:
        before = WorkLedgerService.list_entries(
            state=state,
            tasks=tasks,
            current_user=current_user,
            limit=2000,
        )
        cleared_at = now_iso()
        _save_hidden_before(cleared_at, current_user.username)
        return {
            "cleared": True,
            "cleared_at": cleared_at,
            "cleared_by": current_user.username,
            "hidden_rows": len(before["items"]),
        }
