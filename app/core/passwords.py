# ============================================================
# 密码哈希工具
# 职责：PBKDF2-SHA256 加密 / 验证
# 说明：密码不存明文，登录时重新计算再常量时间比对
# ============================================================
from __future__ import annotations

import hashlib
import hmac
import os


_SCHEME = "pbkdf2_sha256"
_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    """把明文密码变成不可逆哈希。

    这里不用把密码原文写进 users/index.json。登录时重新按同样算法算一次，
    再和保存的哈希做常量时间比较。
    """
    salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        (password or "").encode("utf-8"),
        bytes.fromhex(salt),
        _ITERATIONS,
    ).hex()
    return f"{_SCHEME}${_ITERATIONS}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """校验用户输入的密码是否匹配保存的 PBKDF2 哈希。"""
    try:
        scheme, iterations_raw, salt, expected = str(password_hash or "").split("$", 3)
        iterations = int(iterations_raw)
    except Exception:
        return False
    if scheme != _SCHEME or not salt or not expected:
        return False
    try:
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            (password or "").encode("utf-8"),
            bytes.fromhex(salt),
            iterations,
        ).hex()
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)
