# ============================================================
# License 授权管理
# 职责：RSA 签名验证 / 14 天试用期 / 授权安装
# 链路：
#   1. 先查 license.dat（正式授权，指纹 + RSA 验签）
#   2. 没有正式授权 → trial_state.dat（14 天试用，HMAC 防篡改）
#   3. 试用到期 → locked，仅允许登录 + 安装 license
# ============================================================
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.core import config
from app.core.license_pubkey import PUBLIC_KEY_PEM
from app.core.machine_fingerprint import compute_fingerprint


TRIAL_DAYS = 14
_TRIAL_HMAC_SECRET = "autoannotation-v4-offline-trial-state"


@dataclass
class LicensePayload:
    """license.dat 中被签名保护的正式授权字段。"""
    version: int
    customer: str
    fingerprint: str
    issued_at: str
    license_id: str
    signature: str = ""

    def canonical(self) -> str:
        """生成签名原文。

        签发端和校验端必须使用完全一致的拼接顺序，否则签名会校验失败。
        """
        return "|".join(
            [
                str(int(self.version)),
                self.customer,
                self.fingerprint,
                self.issued_at,
                self.license_id,
            ]
        )


@dataclass
class LicenseStatus:
    """前端和中间件统一使用的授权状态。"""
    mode: str
    activated: bool
    customer: str = ""
    license_id: str = ""
    trial_days_left: int = 0
    machine_fingerprint: str = ""
    error: str = ""
    issued_at: str = ""
    trial_started_at: str = ""
    trial_expires_at: str = ""

    @property
    def locked(self) -> bool:
        return self.mode in {"expired", "locked"} or not self.activated

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "activated": self.activated,
            "customer": self.customer,
            "license_id": self.license_id,
            "trial_days_left": self.trial_days_left,
            "machine_fingerprint": self.machine_fingerprint,
            "error": self.error,
            "issued_at": self.issued_at,
            "trial_started_at": self.trial_started_at,
            "trial_expires_at": self.trial_expires_at,
            # Backward-compatible fields used by older UI/tests.
            "locked": self.locked,
            "unlocked": self.activated,
            "start_date": self.trial_started_at,
            "expires_at": self.trial_expires_at,
            "unlocked_at": "",
            "unlocked_by": "",
        }


# ---------- 时间工具 ----------
def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# 安全写文件：先写临时文件再原子替换，权限设为 600
def _write_private_file(path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


# ---------- RSA 签名验证 ----------
# 加载内嵌公钥（私钥不在仓库中）
def _load_public_key():
    try:
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_public_key(PUBLIC_KEY_PEM.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"License 公钥加载失败: {exc}") from exc


def _verify_signature(payload: LicensePayload) -> bool:
    """用内嵌公钥验证 license.dat 是否确实由私钥签发。"""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        signature = base64.b64decode(payload.signature.encode("utf-8"), validate=True)
        public_key = _load_public_key()
        public_key.verify(
            signature,
            payload.canonical().encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


def _parse_license(raw: bytes) -> LicensePayload:
    """解析上传或磁盘里的 license.dat，并做基础字段校验。"""
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("授权文件不是有效 JSON") from exc
    try:
        payload = LicensePayload(
            version=int(data.get("version") or 0),
            customer=str(data.get("customer") or "").strip(),
            fingerprint=str(data.get("fingerprint") or "").strip(),
            issued_at=str(data.get("issued_at") or "").strip(),
            license_id=str(data.get("license_id") or "").strip(),
            signature=str(data.get("signature") or "").strip(),
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError("授权文件字段无效") from exc
    if payload.version != 1:
        raise ValueError("授权文件版本不支持")
    if not payload.customer or not payload.fingerprint or not payload.issued_at or not payload.license_id:
        raise ValueError("授权文件缺少必要字段")
    if not payload.signature:
        raise ValueError("授权文件缺少签名")
    return payload


# ---------- 磁盘 IO ----------
# 读取磁盘上的 license.dat，不存在则返回 None
def _load_license() -> tuple[LicensePayload | None, str]:
    path = config.LICENSE_FILE
    if not path.exists():
        return None, ""
    try:
        payload = _parse_license(path.read_bytes())
    except ValueError as exc:
        return None, str(exc)
    return payload, ""


def _trial_hmac(state: dict[str, Any], fingerprint: str) -> str:
    """给 trial_state.dat 做 HMAC，防止用户直接改 started_at 延长试用。"""
    message = "|".join(
        [
            str(state.get("version") or 1),
            str(state.get("fingerprint") or ""),
            str(state.get("started_at") or ""),
            str(state.get("last_seen_at") or ""),
        ]
    )
    secret = f"{_TRIAL_HMAC_SECRET}:{fingerprint}".encode("utf-8")
    return hmac.new(secret, message.encode("utf-8"), hashlib.sha256).hexdigest()


def _write_trial(state: dict[str, Any], fingerprint: str) -> None:
    payload = dict(state)
    payload["hmac"] = _trial_hmac(payload, fingerprint)
    _write_private_file(
        config.TRIAL_STATE_FILE,
        json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
    )


def _load_or_init_trial(fingerprint: str) -> tuple[dict[str, Any] | None, str]:
    """读取试用状态；首次启动没有 trial_state.dat 时自动创建 14 天试用。"""
    path = config.TRIAL_STATE_FILE
    now_text = _iso(_now())
    if not path.exists():
        state = {
            "version": 1,
            "fingerprint": fingerprint,
            "started_at": now_text,
            "last_seen_at": now_text,
        }
        _write_trial(state, fingerprint)
        return state, ""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, "试用状态损坏"
    expected = _trial_hmac(state, fingerprint)
    if not hmac.compare_digest(str(state.get("hmac") or ""), expected):
        return None, "试用状态校验失败"
    if str(state.get("fingerprint") or "") != fingerprint:
        return None, "试用状态不属于当前机器"
    return state, ""


# ---------- 试用期状态计算 ----------
def _trial_status(fingerprint: str) -> LicenseStatus:
    """计算试用是否仍有效，并检测系统时间倒退。"""
    state, error = _load_or_init_trial(fingerprint)
    if not state:
        return LicenseStatus(
            mode="locked",
            activated=False,
            customer="试用用户",
            machine_fingerprint=fingerprint,
            error=error,
        )
    started = _parse_time(str(state.get("started_at") or ""))
    last_seen = _parse_time(str(state.get("last_seen_at") or ""))
    now = _now()
    if not started or not last_seen:
        return LicenseStatus(
            mode="locked",
            activated=False,
            customer="试用用户",
            machine_fingerprint=fingerprint,
            error="试用状态时间无效",
        )
    if last_seen.timestamp() - now.timestamp() > 24 * 3600:
        return LicenseStatus(
            mode="locked",
            activated=False,
            customer="试用用户",
            machine_fingerprint=fingerprint,
            error="系统时间倒退超过 24 小时",
        )
    elapsed_days = max(0, int((now - started).total_seconds() // 86400))
    days_left = TRIAL_DAYS - elapsed_days
    expires = started.timestamp() + TRIAL_DAYS * 86400
    expires_at = datetime.fromtimestamp(expires, tz=timezone.utc)
    if now > last_seen:
        state["last_seen_at"] = _iso(now)
        _write_trial(state, fingerprint)
    if days_left > 0:
        return LicenseStatus(
            mode="trial",
            activated=True,
            customer="试用用户",
            trial_days_left=days_left,
            machine_fingerprint=fingerprint,
            trial_started_at=_iso(started),
            trial_expires_at=_iso(expires_at),
        )
    return LicenseStatus(
        mode="expired",
        activated=False,
        customer="试用用户",
        trial_days_left=days_left,
        machine_fingerprint=fingerprint,
        trial_started_at=_iso(started),
        trial_expires_at=_iso(expires_at),
        error="试用期已到期",
    )


def get_license_status() -> LicenseStatus:
    """授权状态主入口。

    中间件每次请求都会调这个函数，判断系统当前是「正式授权 / 试用中 / 已锁定」。
    步骤：
      ① 计算当前机器指纹
      ② 尝试从磁盘读 license.dat
      ③ 有 license.dat → 校验指纹 + RSA 签名
      ④ 没有 license.dat → 进入 14 天试用逻辑
    """
    # ① 计算当前机器的硬件指纹（用于和 license 中绑定的指纹比对）
    fingerprint = compute_fingerprint()

    # ② 尝试从磁盘读取 license.dat 文件
    licensed, load_error = _load_license()

    if licensed:
        # ③-a 指纹比对：license 里记录的机器指纹必须和当前机器一致
        #      防止把别人的 license 拷过来用
        if licensed.fingerprint != fingerprint:
            return LicenseStatus(
                mode="locked",
                activated=False,
                machine_fingerprint=fingerprint,
                error="授权文件不匹配当前机器",
            )
        # ③-b RSA 签名验证：用内嵌公钥验证 license 内容未被篡改
        #      私钥只在发行方手里，用户改不了 license 内容
        if not _verify_signature(licensed):
            return LicenseStatus(
                mode="locked",
                activated=False,
                machine_fingerprint=fingerprint,
                error="授权文件签名无效",
            )
        # ③-c 全部校验通过 → 正式授权，系统完全解锁
        return LicenseStatus(
            mode="licensed",
            activated=True,
            customer=licensed.customer,
            license_id=licensed.license_id,
            machine_fingerprint=fingerprint,
            issued_at=licensed.issued_at,
        )

    # license.dat 存在但解析失败（格式损坏、字段缺失等）→ 直接锁定
    if load_error:
        return LicenseStatus(
            mode="locked",
            activated=False,
            machine_fingerprint=fingerprint,
            error=load_error,
        )

    # ④ 没有 license.dat → 进入 14 天试用期逻辑（HMAC 防篡改）
    return _trial_status(fingerprint)


def install_license(file_content: bytes) -> LicenseStatus:
    """安装正式授权文件。

    步骤：
      ① 解析上传的 JSON 内容为 LicensePayload
      ② 比对机器指纹
      ③ RSA 验签
      ④ 全部通过后才写入磁盘（先验后写，坏文件不落盘）
      ⑤ 重新读取返回最新状态
    """
    # ① 计算当前机器指纹
    fingerprint = compute_fingerprint()

    # ② 解析上传内容：JSON 格式 + 必要字段检查
    try:
        payload = _parse_license(file_content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ③ 指纹比对：确保 license 是给当前这台机器签发的
    if payload.fingerprint != fingerprint:
        raise HTTPException(status_code=400, detail="授权文件不匹配当前机器")

    # ④ RSA 签名验证：确保 license 未被篡改
    if not _verify_signature(payload):
        raise HTTPException(status_code=400, detail="授权文件签名无效")

    # ⑤ 全部校验通过，写入磁盘（原子替换，权限 600）
    data = json.dumps(
        {
            "version": payload.version,
            "customer": payload.customer,
            "fingerprint": payload.fingerprint,
            "issued_at": payload.issued_at,
            "license_id": payload.license_id,
            "signature": payload.signature,
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    _write_private_file(config.LICENSE_FILE, data)

    # ⑥ 重新走一遍 get_license_status，返回安装后的最新状态
    return get_license_status()
