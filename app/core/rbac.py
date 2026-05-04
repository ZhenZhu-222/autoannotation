# ============================================================
# RBAC 权限判定
# 职责：判断用户对图片集 / 任务的操作权和可见性
# 规则：admin 全通；operator 只能操作自己创建的或被分派的资源
# ============================================================
from __future__ import annotations

from app.core.models import ImageSetRecord, JobRecord, UserRecord


def can_operate_imageset(user: UserRecord, imageset: ImageSetRecord) -> bool:
    """图片集操作权判断。

    admin 不需要进入 assignee_ids，天然拥有全部图片集权限。
    operator 只能操作自己创建的图片集，或 admin 分派给他的图片集。
    """
    if user.role == "admin":
        return True
    return imageset.creator_id == user.id or user.id in imageset.assignee_ids


def can_view_job(user: UserRecord, job: JobRecord | dict) -> bool:
    """任务可见性判断：admin 看全部，operator 只看自己创建的任务。"""
    if user.role == "admin":
        return True
    if isinstance(job, JobRecord):
        return job.operator_id == user.id
    return str(job.get("operator_id") or "") == user.id


def user_display_name(user: UserRecord | None) -> str:
    if not user:
        return "已删除用户"
    return user.username
