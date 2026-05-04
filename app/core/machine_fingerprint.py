# ============================================================
# 机器指纹计算
# 职责：计算当前机器的稳定唯一标识，用于 License 绑定
# 链路：硬件指标 → AUTOANNOTATION_MACHINE_ID 环境变量 → MAC+hostname 兆底
# ============================================================
from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
import subprocess
import uuid
from pathlib import Path


_CACHE_KEY: tuple[str, str, int, int, int, int] | None = None
_CACHE_VALUE = ""


# 安全读取文本文件，权限不足或不存在时返回空字符串
def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


# 运行系统命令，超时 2 秒，失败返回空字符串
def _run(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
    except Exception:
        return ""


# Linux 指标：UUID / 主板序列号 / machine-id / CPU 序列号
def _linux_indicators() -> list[str]:
    values = [
        _read_text("/sys/class/dmi/id/product_uuid"),
        _read_text("/sys/class/dmi/id/board_serial"),
        _read_text("/sys/class/dmi/id/product_serial"),
        _read_text("/etc/machine-id"),
    ]
    cpuinfo = _read_text("/proc/cpuinfo")
    match = re.search(r"(?im)^serial\s*:\s*(.+)$", cpuinfo)
    if match:
        values.append(match.group(1).strip())
    return values


# macOS 指标：IOPlatformUUID / 硬件序列号
def _mac_indicators() -> list[str]:
    raw = _run(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
    values: list[str] = []
    match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', raw)
    if match:
        values.append(match.group(1))
    serial = _run(["system_profiler", "SPHardwareDataType"])
    match = re.search(r"(?im)Serial Number.*:\s*(.+)$", serial)
    if match:
        values.append(match.group(1).strip())
    return values


# Windows 指标：产品 UUID / 主板序列号 / 处理器 ID
def _windows_indicators() -> list[str]:
    values: list[str] = []
    for command in (
        ["wmic", "csproduct", "get", "UUID"],
        ["wmic", "baseboard", "get", "SerialNumber"],
        ["wmic", "cpu", "get", "ProcessorId"],
    ):
        raw = _run(command)
        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if len(lines) >= 2:
            values.append(lines[1])
    return values


# 获取主网卡 MAC 地址（过滤随机生成的虚拟 MAC）
def _primary_mac() -> str:
    node = uuid.getnode()
    if node and (node >> 40) % 2 == 0:
        return f"{node:012x}"
    return ""


# 把多个指标值拼接后 SHA-256，产生稳定指纹字符串
def _stable_hash(values: list[str]) -> str:
    clean = [str(v).strip().lower() for v in values if str(v or "").strip()]
    raw = "::".join(clean).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_fingerprint() -> str:
    """计算当前机器指纹。

    顺序是：
    1. 尽量读硬件/系统稳定标识；
    2. 如果硬件指标全失败，优先使用 AUTOANNOTATION_MACHINE_ID；
    3. 最后退到 MAC + hostname。

    Docker/虚拟机部署时硬件信息经常不可用，所以第 2 步很重要。
    """
    global _CACHE_KEY, _CACHE_VALUE

    system = platform.system().lower()
    configured = os.getenv("AUTOANNOTATION_MACHINE_ID", "").strip()
    # 指纹会被 License / trial HMAC 反复使用，必须在一个进程内保持稳定。
    # macOS 的 ioreg/system_profiler 偶尔超时；如果每次请求都重算，可能前后
    # 算出不同结果，导致系统误判 trial_state.dat 被篡改而锁定。
    cache_key = (
        system,
        configured,
        id(_linux_indicators),
        id(_mac_indicators),
        id(_windows_indicators),
        id(_primary_mac),
    )
    if _CACHE_KEY == cache_key and _CACHE_VALUE:
        return _CACHE_VALUE

    if "linux" in system:
        values = _linux_indicators()
    elif "darwin" in system:
        values = _mac_indicators()
    elif "windows" in system:
        values = _windows_indicators()
    else:
        values = []

    mac = _primary_mac()
    if mac:
        values.append(mac)
    values = [v for v in values if str(v or "").strip()]
    if values:
        _CACHE_KEY = cache_key
        _CACHE_VALUE = _stable_hash(values)
        return _CACHE_VALUE

    if configured:
        _CACHE_KEY = cache_key
        _CACHE_VALUE = _stable_hash([configured])
        return _CACHE_VALUE

    _CACHE_KEY = cache_key
    _CACHE_VALUE = _stable_hash([_primary_mac(), socket.gethostname()])
    return _CACHE_VALUE
