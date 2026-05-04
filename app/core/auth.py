# ============================================================
# 鉴权与授权模块
# 职责：登录验证 / Session Token 签发与解析 / RBAC 依赖 / License 状态兼容导出
# ============================================================
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import HTTPException, Request

from app.core.config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    LOGIN_REQUIRED,
    SESSION_SECRET,
    SESSION_TTL_HOURS,
)
from app.core.license import LicenseStatus, get_license_status, install_license
from app.core.models import UserRecord
from app.core.passwords import verify_password


# ---------- Base64URL 编解码（Token 序列化用） ----------
def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("utf-8").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


# HMAC-SHA256 签名，防止 token 被篡改
def _sign(value: str) -> str:
    digest = hmac.new(SESSION_SECRET.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(digest)


# ---------- Session Token 签发与解析 ----------
# 把用户信息编码成 "base64url(payload).HMAC" 格式的 token
def create_session_token(user: UserRecord | str) -> str:
    if isinstance(user, UserRecord):
        payload = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role,
            "iat": int(time.time()),
            "exp": int(time.time()) + SESSION_TTL_HOURS * 3600,
        }
    else:
        payload = {
            "user_id": "user_admin",
            "username": str(user or ADMIN_USERNAME),
            "role": "admin",
            "iat": int(time.time()),
            "exp": int(time.time()) + SESSION_TTL_HOURS * 3600,
        }
    body = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{body}.{_sign(body)}"


# 解析并验证 token：校验 HMAC + 过期时间 + 必要字段
def parse_session_token(token: str) -> dict[str, Any] | None:
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp") or 0) < int(time.time()):
        return None
    if not payload.get("user_id") or not payload.get("username"):
        return None
    return payload


# ---------- 请求上下文工具 ----------
# 从请求中取 AppState 实例
def _state_from_request(request: Request):
    return getattr(request.app.state, "app_state", None)


# LOGIN_REQUIRED=false 时兜底返回内置 admin
def _admin_user_from_state(request: Request) -> UserRecord:
    state = _state_from_request(request)
    if state is not None:
        user = state.get_user("user_admin")
        if user:
            return user
    return UserRecord(
        id="user_admin",
        username=ADMIN_USERNAME,
        password_hash="",
        role="admin",
        enabled=True,
        deleted=False,
    )


# 从 cookie 中读取并解析 session token
def _cookie_payload(request: Request) -> dict[str, Any] | None:
    cookie_name = getattr(request.app.state, "session_cookie_name", "autoannotation_session")
    token = request.cookies.get(cookie_name, "")
    return parse_session_token(token)


# ---------- 当前用户识别 ----------
# 从 session cookie 解析用户 → 去 AppState 查最新状态 → 返回 UserRecord
def get_current_user(request: Request) -> UserRecord | None:
    if not LOGIN_REQUIRED:
        return _admin_user_from_state(request)
    payload = _cookie_payload(request)
    if not payload:
        return None
    state = _state_from_request(request)
    if state is None:
        return None
    user = state.get_user(str(payload.get("user_id") or ""))
    if not user or user.deleted or not user.enabled:
        return None
    if user.username != str(payload.get("username") or ""):
        return None
    return user


def is_authenticated(request: Request) -> bool:
    if not LOGIN_REQUIRED:
        return True
    return get_current_user(request) is not None


def get_current_username(request: Request) -> str:
    user = get_current_user(request)
    return user.username if user else ""


# ---------- RBAC 依赖注入 ----------
# 任何已登录用户
def require_login(request: Request) -> UserRecord:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


# 必须是 admin
def require_admin(request: Request) -> UserRecord:
    user = require_login(request)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


# admin 或 operator
def require_operator_or_admin(request: Request) -> UserRecord:
    user = require_login(request)
    if user.role not in {"admin", "operator"}:
        raise HTTPException(status_code=403, detail="无权限")
    return user


# ---------- 登录凭证验证 ----------
# 有 AppState 时查 users 表 + PBKDF2；无 state 时退回配置文件 admin
def verify_credentials(username: str, password: str, state=None) -> UserRecord | None:
    if state is None:
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return UserRecord(
                id="user_admin",
                username=ADMIN_USERNAME,
                password_hash="",
                role="admin",
                enabled=True,
                deleted=False,
            )
        return None
    user = state.get_user_by_username(username)
    if not user or user.deleted or not user.enabled:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# ---------- 旧版兼容 ----------
# V3 静态密钥解锁已废弃，直接返回 410
def unlock_license(key: str, operator: str) -> LicenseStatus:
    _ = (key, operator)
    raise HTTPException(status_code=410, detail="旧版静态密钥解锁已下架，请上传授权文件")
