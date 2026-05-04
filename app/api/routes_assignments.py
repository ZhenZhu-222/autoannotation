# ============================================================
# 分派路由 (Controller)
# 职责：图片集分派 CRUD / 验收提交 / 验收审核
# 说明：分派对象只能是 enabled 的 operator；admin 天然全局可见，不进分派列表
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_state
from app.core.auth import require_admin, require_login
from app.core.rbac import can_operate_imageset
from app.core.state import AppState
from app.schemas.assignments import (
    AssigneeListResponse,
    AssignmentReviewRequest,
    AssignmentSubmitRequest,
    AssignImagesetRequest,
)

router = APIRouter(prefix="/api", tags=["assignments"])


def _assignees_payload(state: AppState, imageset_id: str) -> dict:
    """把图片集 assignee_ids 转成前端可读的用户列表。"""
    imageset = state.get_imageset(imageset_id)
    if not imageset:
        raise KeyError("imageset not found")
    items = []
    states = getattr(imageset, "assignee_states", {}) or {}
    for user_id in imageset.assignee_ids:
        user = state.get_user(user_id)
        if not user or user.deleted:
            continue
        assignment = states.get(user_id) or {}
        items.append({
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "enabled": user.enabled,
            "status": assignment.get("status") or "assigned",
            "assigned_at": assignment.get("assigned_at") or "",
            "submitted_at": assignment.get("submitted_at") or "",
            "reviewed_at": assignment.get("reviewed_at") or "",
            "submit_note": assignment.get("submit_note") or "",
            "review_note": assignment.get("review_note") or "",
            "reviewer_username": assignment.get("reviewer_username") or "",
        })
    history = []
    for assignment in getattr(imageset, "assignee_history", []) or []:
        history.append({
            "id": assignment.get("user_id") or "",
            "username": assignment.get("username") or "已删除用户",
            "role": "operator",
            "enabled": False,
            "status": assignment.get("status") or "",
            "assigned_at": assignment.get("assigned_at") or "",
            "submitted_at": assignment.get("submitted_at") or "",
            "reviewed_at": assignment.get("reviewed_at") or "",
            "submit_note": assignment.get("submit_note") or "",
            "review_note": assignment.get("review_note") or "",
            "reviewer_username": assignment.get("reviewer_username") or "",
        })
    return {"imageset_id": imageset_id, "assignees": items, "history": history}


@router.get("/imagesets/{imageset_id}/assignees")
def list_assignees(
    imageset_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> AssigneeListResponse:
    # admin 可以看所有图片集分派；operator 只能看自己有权操作的图片集。
    imageset = state.get_imageset(imageset_id)
    if not imageset:
        raise HTTPException(status_code=404, detail="imageset 不存在")
    if current_user.role != "admin" and not can_operate_imageset(current_user, imageset):
        raise HTTPException(status_code=403, detail="无权查看该图片集分派")
    return _assignees_payload(state, imageset_id)


@router.post("/imagesets/{imageset_id}/assignees")
def set_assignees(
    imageset_id: str,
    req: AssignImagesetRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> AssigneeListResponse:
    # 保存完整分派集合，而不是追加单个用户，前端多选保存时更简单。
    try:
        state.set_imageset_assignees(
            imageset_id,
            req.user_ids,
            actor_id=current_user.id,
            actor_username=current_user.username,
        )
        return _assignees_payload(state, imageset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# 移除单个分派人员，会把该人员转入历史流水
@router.delete("/imagesets/{imageset_id}/assignees/{user_id}")
def remove_assignee(
    imageset_id: str,
    user_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> AssigneeListResponse:
    try:
        state.remove_imageset_assignee(
            imageset_id,
            user_id,
            actor_id=current_user.id,
            actor_username=current_user.username,
        )
        return _assignees_payload(state, imageset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None


# operator 完成标注后主动提交验收；提交不移除访问权，等 admin 审核
@router.post("/imagesets/{imageset_id}/assignees/{user_id}/submit")
def submit_assignment_review(
    imageset_id: str,
    user_id: str,
    req: AssignmentSubmitRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> AssigneeListResponse:
    if current_user.role != "admin" and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="只能提交自己的任务验收")
    imageset = state.get_imageset(imageset_id)
    if not imageset:
        raise HTTPException(status_code=404, detail="imageset 不存在")
    if current_user.role != "admin" and not can_operate_imageset(current_user, imageset):
        raise HTTPException(status_code=403, detail="无权提交该图片集验收")
    try:
        state.submit_imageset_assignment(imageset_id, user_id, req.submit_note)
        return _assignees_payload(state, imageset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


# admin 审核：通过则移出分派列表并归档；驳回则保留让 operator 返工
@router.post("/imagesets/{imageset_id}/assignees/{user_id}/review")
def review_assignment(
    imageset_id: str,
    user_id: str,
    req: AssignmentReviewRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> AssigneeListResponse:
    try:
        state.review_imageset_assignment(
            imageset_id,
            user_id,
            approved=req.approved,
            review_note=req.review_note,
            reviewer_id=current_user.id,
        )
        return _assignees_payload(state, imageset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="imageset 不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
