# ============================================================
# 用户模块 DTO（创建 / 更新 / 密码 / 响应）
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field


# ---------- 请求 ----------
class UserCreateRequest(BaseModel):
    username: str = Field(default="")
    password: str = Field(default="")
    role: str = Field(default="operator")


class UserUpdateRequest(BaseModel):
    username: str | None = None
    role: str | None = None
    enabled: bool | None = None


# 普通用户修改自己密码，必须验旧密码
class PasswordChangeRequest(BaseModel):
    old_password: str = Field(default="")
    new_password: str = Field(default="")


# admin 重置别人密码，不需要旧密码
class PasswordResetRequest(BaseModel):
    new_password: str = Field(default="")


# ---------- 响应 ----------
class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    enabled: bool = True
    created_at: str = ""
    last_login_at: str = ""
    deleted: bool = False


class UserListResponse(BaseModel):
    items: list[UserResponse] = Field(default_factory=list)


class UserMutationResponse(BaseModel):
    ok: bool
    user: UserResponse | None = None
