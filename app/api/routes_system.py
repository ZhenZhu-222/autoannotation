# ============================================================
# 系统路由 (Controller)
# 职责：登录 / 登出 / 会话查询 / 用户管理 / License 安装
# ============================================================
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile

from app.api.deps import get_state
from app.core.auth import (
    create_session_token,
    get_current_user,
    get_license_status,
    install_license,
    is_authenticated,
    require_admin,
    require_login,
    unlock_license,
    verify_credentials,
)
from app.core.config import LOGIN_REQUIRED, SESSION_COOKIE_NAME, SESSION_TTL_HOURS, get_security_warnings
from app.core.passwords import verify_password
from app.core.state import AppState
from app.schemas.system import (
    LicenseInstallResponse,
    LicenseStatusResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    ReadinessResponse,
    SessionResponse,
    UnlockRequest,
    UnlockResponse,
)
from app.schemas.users import (
    PasswordChangeRequest,
    PasswordResetRequest,
    UserCreateRequest,
    UserListResponse,
    UserMutationResponse,
    UserResponse,
    UserUpdateRequest,
)
from app.services.system_readiness_service import SystemReadinessService

router = APIRouter(prefix="/api/system", tags=["system"])
alias_router = APIRouter(prefix="/api", tags=["system-alias"])


# 工具函数：把 UserRecord 转成前端可安全返回的字典（不含 password_hash）
def _user_response(user) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "enabled": user.enabled,
        "created_at": user.created_at,
        "last_login_at": user.last_login_at,
        "deleted": user.deleted,
    }


def build_session_payload(request: Request) -> dict:
    """构造当前会话信息。

    /api/system/session 和 /api/auth/me 都走这里，避免能力位计算分叉。
    """
    current = get_current_user(request)
    license_status = get_license_status().to_dict()
    is_admin = bool(current and current.role == "admin")
    capabilities = {
        "can_delete": is_admin,
        "can_download": is_admin,
        "can_assign": is_admin,
        "can_manage_users": is_admin,
        "can_install_license": is_admin,
    }
    return {
        "login_required": LOGIN_REQUIRED,
        "authenticated": is_authenticated(request),
        "username": current.username if current else "",
        "user": {
            "user_id": current.id,
            "username": current.username,
            "role": current.role,
        } if current else None,
        "capabilities": capabilities,
        "license": license_status,
        "security_warnings": get_security_warnings(),
    }


# 查询当前会话状态（登录态 + 能力位 + License）
@router.get("/session")
def get_session(request: Request) -> SessionResponse:
    return build_session_payload(request)


# /api/auth/me 是 /api/system/session 的别名，前端两个地址都能用
@alias_router.get("/auth/me")
def auth_me(request: Request) -> SessionResponse:
    return build_session_payload(request)


# 系统自检，交付验收用
@router.get("/readiness")
def get_readiness() -> ReadinessResponse:
    return SystemReadinessService.build()


# 登录：验证用户名密码 → 签发 session token → 写入 cookie
@router.post("/login")
def login(req: LoginRequest, response: Response, state: AppState = Depends(get_state)) -> LoginResponse:
    user = verify_credentials(req.username.strip(), req.password, state)
    if not user:
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        response.status_code = 401
        return {"ok": False, "detail": "账户名或密码错误"}
    token = create_session_token(user)
    state.mark_user_login(user.id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_HOURS * 3600,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"ok": True, "license": get_license_status().to_dict()}


# 登出：清除 session cookie
@router.post("/logout")
def logout(response: Response) -> LogoutResponse:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}


# ---------- License 管理 ----------
# 查询当前 License 状态（licensed / trial / expired / locked）
@router.get("/license/status")
def license_status() -> LicenseStatusResponse:
    return get_license_status().to_dict()


@alias_router.get("/license/status")
def license_status_alias() -> LicenseStatusResponse:
    return get_license_status().to_dict()


# 上传安装正式授权文件（admin-only，先验后写，不会覆盖合法 license）
@router.post("/license/install")
async def install_license_file(
    file: UploadFile = File(...),
    current_user=Depends(require_admin),
) -> LicenseInstallResponse:
    _ = current_user
    content = await file.read()
    status = install_license(content)
    return {"ok": True, "license": status.to_dict()}


@alias_router.post("/license/install")
async def install_license_file_alias(
    file: UploadFile = File(...),
    current_user=Depends(require_admin),
) -> LicenseInstallResponse:
    _ = current_user
    content = await file.read()
    status = install_license(content)
    return {"ok": True, "license": status.to_dict()}


# 旧版静态密钥解锁入口（已废弃，返回 410）
@router.post("/license/unlock")
def unlock(req: UnlockRequest, request: Request) -> UnlockResponse:
    current = require_admin(request)
    status = unlock_license(req.key.strip(), current.username)
    return {"ok": True, "license": status.to_dict()}


# ---------- 用户管理 CRUD（admin-only，软删除用户不会出现在列表中） ----------
# 列出所有活跃用户
@router.get("/users")
@alias_router.get("/users")
def list_users(
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> UserListResponse:
    _ = current_user
    return {"items": [_user_response(user) for user in state.list_users()]}


# 创建新用户（用户名唯一只检查未删除用户，允许重建同名账号）
@router.post("/users")
@alias_router.post("/users")
def create_user(
    req: UserCreateRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> UserMutationResponse:
    _ = current_user
    try:
        user = state.create_user(req.username, req.password, req.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"ok": True, "user": _user_response(user)}


@router.patch("/users/me/password")
@alias_router.patch("/users/me/password")
def change_own_password(
    req: PasswordChangeRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_login),
) -> UserMutationResponse:
    # 改自己的密码必须验证旧密码，防止 session 被别人拿到后直接改密。
    user = state.get_user(current_user.id)
    if not user or not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=403, detail="旧密码错误")
    try:
        updated = state.reset_password(current_user.id, req.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"ok": True, "user": _user_response(updated)}


@router.post("/users/{user_id}/password")
@alias_router.post("/users/{user_id}/password")
def reset_user_password(
    user_id: str,
    req: PasswordResetRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> UserMutationResponse:
    # admin 重置别人密码不需要旧密码；但不能用这个接口绕过自己的旧密码校验。
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="请使用 /users/me/password 修改自己的密码")
    try:
        user = state.reset_password(user_id, req.new_password)
    except KeyError:
        raise HTTPException(status_code=404, detail="用户不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"ok": True, "user": _user_response(user)}


@router.patch("/users/{user_id}")
@alias_router.patch("/users/{user_id}")
def update_user(
    user_id: str,
    req: UserUpdateRequest,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> UserMutationResponse:
    # 防止唯一 admin 把自己禁用或降级后把系统锁死。
    if user_id == current_user.id and (req.enabled is False or req.role == "operator"):
        raise HTTPException(status_code=400, detail="不能禁用或降级自己")
    try:
        user = state.update_user(user_id, username=req.username, role=req.role, enabled=req.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail="用户不存在") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"ok": True, "user": _user_response(user)}


# 软删除用户（同时从所有图片集的 assignee_ids 移除，防止僵尸分派）
@router.delete("/users/{user_id}")
@alias_router.delete("/users/{user_id}")
def delete_user(
    user_id: str,
    state: AppState = Depends(get_state),
    current_user=Depends(require_admin),
) -> UserMutationResponse:
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能删除自己")
    try:
        user = state.soft_delete_user(user_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="用户不存在") from None
    return {"ok": True, "user": _user_response(user)}
