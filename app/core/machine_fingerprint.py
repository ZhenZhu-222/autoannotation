# ============================================================
# 机器指纹计算
# 职责：计算当前机器的稳定唯一标识，用于 License 绑定
# 策略：每个平台只读 1 个最稳定的标识 → SHA-256
#       读不到就报错，不 fallback 到不稳定的 MAC/hostname
#       Docker/虚拟机可通过环境变量 AUTOANNOTATION_MACHINE_ID 显式指定
# ============================================================
from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path


_CACHE_VALUE = ""


# 运行系统命令，超时 2 秒，失败返回空字符串
def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
    except Exception:
        return ""


# 每个平台只取 1 个最稳定的硬件标识
def _get_primary_id() -> str:
    system = platform.system().lower()

    if "darwin" in system:
        # macOS：IOPlatformUUID，主板级别，除非换主板否则不变
        raw = _run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
        match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', raw)
        if match:
            return match.group(1).strip()

    elif "linux" in system:
        # Linux：/etc/machine-id，系统安装时生成，不变
        try:
            value = Path("/etc/machine-id").read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                return value
        except Exception:
            pass

    elif "windows" in system:
        # Windows：csproduct UUID，BIOS 级别，不变
        raw = _run(["wmic", "csproduct", "get", "UUID"])
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if len(lines) >= 2:
            return lines[1]

    return ""


def compute_fingerprint() -> str:
    """计算当前机器指纹。

    优先级：
    1. 环境变量 AUTOANNOTATION_MACHINE_ID（Docker/虚拟机用）
    2. 平台唯一硬件标识（macOS IOPlatformUUID / Linux machine-id / Windows csproduct UUID）
    3. 都没有 → 抛异常，不 fallback
    """
    global _CACHE_VALUE

    # 进程内缓存，避免重复调用系统命令
    if _CACHE_VALUE:
        return _CACHE_VALUE

    # 优先使用显式配置（Docker/虚拟机场景）
    configured = os.getenv("AUTOANNOTATION_MACHINE_ID", "").strip()
    if configured:
        _CACHE_VALUE = hashlib.sha256(configured.lower().encode("utf-8")).hexdigest()
        return _CACHE_VALUE

    # 读平台唯一标识
    primary = _get_primary_id()
    if not primary:
        raise RuntimeError(
            "无法获取机器标识，请设置环境变量 AUTOANNOTATION_MACHINE_ID 或联系支持"
        )

    _CACHE_VALUE = hashlib.sha256(primary.strip().lower().encode("utf-8")).hexdigest()
    return _CACHE_VALUE
