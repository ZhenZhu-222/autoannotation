# ============================================================
# 通用工具函数
# 职责：提供 ID 生成 / 时间格式化 / 文件名安全化 / 上传流处理 / 路径转换
# ============================================================
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")


# ---------- 时间与 ID ----------
# 返回当前 UTC 时间的 ISO 8601 字符串
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 生成带前缀的唯一 ID（prefix_随机12位hex）
def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------- 文件名安全化：去除特殊字符，保留扩展名 ----------
def sanitize_filename(name: str, fallback: str = "file") -> str:
    raw = (name or "").strip()
    if not raw:
        raw = fallback
    raw = raw.replace("\\", "/")
    raw = raw.split("/")[-1]

    p = Path(raw)
    stem = p.stem
    suffix = p.suffix

    safe_stem = _slug_re.sub("_", stem).strip("._")
    if not safe_stem:
        fb_stem = Path(fallback).stem
        safe_stem = _slug_re.sub("_", fb_stem).strip("._") or "file"

    ext = "".join(ch for ch in suffix.lower() if ch.isalnum())
    safe_suffix = f".{ext}" if ext else ""

    if not safe_suffix:
        fb_ext = "".join(ch for ch in Path(fallback).suffix.lower() if ch.isalnum())
        safe_suffix = f".{fb_ext}" if fb_ext else ""

    return f"{safe_stem}{safe_suffix}"


# ---------- URL 路径安全解析：防止目录穿越 ----------
def safe_data_path(data_url: str, base_dir: Path) -> Path | None:
    """Resolve a /data/ URL to a safe filesystem path within base_dir."""
    text = (data_url or "").strip()
    if not text.startswith("/data/"):
        return None
    relative = text.removeprefix("/data/").lstrip("/")
    if not relative:
        return None
    resolved = (base_dir / relative).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        return None
    return resolved


# ---------- 上传文件流处理（带大小限制） ----------
async def read_upload_with_limit(upload_file, max_bytes: int) -> bytes:
    """Read an UploadFile up to max_bytes. Raises ValueError if exceeded."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload_file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"文件大小超过限制 ({max_bytes // (1024 * 1024)}MB)")
        chunks.append(chunk)
    return b"".join(chunks)


async def stream_upload_to_file(upload_file, dest: Path, max_bytes: int) -> int:
    """Stream an UploadFile to disk, aborting if it exceeds max_bytes."""
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                break
            out.write(chunk)
    if total > max_bytes:
        dest.unlink(missing_ok=True)
        raise ValueError(f"文件大小超过限制 ({max_bytes // (1024 * 1024)}MB)")
    return total


# ---------- /data/ URL 与磁盘路径互转 ----------
def to_data_url(path: Path, data_dir: Path) -> str:
    """将磁盘绝对路径转为 /data/ 开头的 URL。"""
    raw = str(path)
    path_candidates = [Path(raw), Path(raw).resolve()]
    if raw.startswith("/private/"):
        path_candidates.append(Path(raw.removeprefix("/private")))
    elif raw.startswith("/var/"):
        path_candidates.append(Path("/private" + raw))

    base_raw = str(data_dir)
    base_candidates = [Path(base_raw), Path(base_raw).resolve()]
    if base_raw.startswith("/private/"):
        base_candidates.append(Path(base_raw.removeprefix("/private")))
    elif base_raw.startswith("/var/"):
        base_candidates.append(Path("/private" + base_raw))

    for cand in path_candidates:
        for base in base_candidates:
            try:
                rel = cand.relative_to(base).as_posix()
                return f"/data/{rel}"
            except ValueError:
                continue
    return ""


def data_url_to_path(data_url: str, data_dir: Path) -> Path | None:
    """safe_data_path 的别名，保持向后兼容。"""
    return safe_data_path(data_url, data_dir)


# ---------- 安全类型转换 ----------
# 安全转换为 int，失败时返回 default
def safe_int(val: str | int | float | None, default: int = 0) -> int:
    try:
        return int(val or default)
    except (TypeError, ValueError):
        return default


# 安全转换为 float，失败时返回 default
def safe_float(val: str | int | float | None, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


# 将字符串/bool/None 解析为布尔值（支持 1/true/yes/on/y）
def as_bool(value: str | bool | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on", "y"}


# ---------- 文件去重：若已存在则加后缀序号 ----------
# 若路径已存在则自动追加序号后缀，确保返回不冲突的路径
def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1
