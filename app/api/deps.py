# ============================================================
# 依赖注入（类似 Spring @Autowired）
# 职责：从请求中提取全局 AppState 和 TaskManager 实例
# ============================================================
from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.state import AppState
from app.services.task_manager import TaskManager


def get_state(request: Request) -> AppState:
    state = getattr(request.app.state, "app_state", None)
    if state is None:
        raise HTTPException(status_code=500, detail="state not initialized")
    return state


def get_task_manager(request: Request) -> TaskManager:
    task_manager = getattr(request.app.state, "task_manager", None)
    if task_manager is None:
        raise HTTPException(status_code=500, detail="task manager not initialized")
    return task_manager
