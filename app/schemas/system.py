# ============================================================
# 系统模块 DTO（登录 / License）
# ============================================================
from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(default="")
    password: str = Field(default="")


class UnlockRequest(BaseModel):
    key: str = Field(default="")


# ============================================================
# 系统模块响应 DTO
# ============================================================


# ---------- License 状态 ----------
class LicenseStatusResponse(BaseModel):
    mode: str = "trial"
    activated: bool = False
    customer: str = ""
    license_id: str = ""
    trial_days_left: int = 0
    machine_fingerprint: str = ""
    error: str = ""
    issued_at: str = ""
    trial_started_at: str = ""
    trial_expires_at: str = ""
    locked: bool
    unlocked: bool = False
    start_date: str = ""
    expires_at: str = ""
    unlocked_at: str = ""
    unlocked_by: str = ""


class UserSessionInfo(BaseModel):
    user_id: str = ""
    username: str = ""
    role: str = ""


class CapabilitiesResponse(BaseModel):
    can_delete: bool = False
    can_download: bool = False
    can_assign: bool = False
    can_manage_users: bool = False
    can_install_license: bool = False


# ---------- 会话查询 ----------
class SessionResponse(BaseModel):
    login_required: bool
    authenticated: bool
    username: str = ""
    user: UserSessionInfo | None = None
    capabilities: CapabilitiesResponse = Field(default_factory=CapabilitiesResponse)
    license: LicenseStatusResponse
    security_warnings: list[str] = Field(default_factory=list)


# ---------- 登录响应 ----------
class LoginResponse(BaseModel):
    ok: bool
    detail: str = ""
    license: LicenseStatusResponse | None = None


# ---------- 登出响应 ----------
class LogoutResponse(BaseModel):
    ok: bool


# ---------- License 解锁响应 ----------
class UnlockResponse(BaseModel):
    ok: bool
    license: LicenseStatusResponse


class LicenseInstallResponse(BaseModel):
    ok: bool
    license: LicenseStatusResponse


# ---------- 系统自检 ----------
class ReadinessCheckItem(BaseModel):
    name: str
    ok: bool


class ReadinessChecks(BaseModel):
    writable_dirs: list[ReadinessCheckItem] = Field(default_factory=list)
    front_files: list[ReadinessCheckItem] = Field(default_factory=list)
    license: LicenseStatusResponse | None = None
    qwen_configured: bool = False
    source_artifacts: list[str] = Field(default_factory=list)


class ReadinessAcceptance(BaseModel):
    minimum_test_command: str = ""
    frontend_check_command: str = ""


class ReadinessResponse(BaseModel):
    status: str
    operational: bool
    production_ready: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: ReadinessChecks = Field(default_factory=ReadinessChecks)
    acceptance: ReadinessAcceptance = Field(default_factory=ReadinessAcceptance)
