# ============================================================
# 全局配置中心
# 职责：读取 runtime.json + 环境变量，导出所有目录路径和运行参数
# 优先级：环境变量 > runtime.json > 默认值
# ============================================================
from __future__ import annotations

import json
import os
from pathlib import Path

# ---------- 项目根目录 ----------
PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_DIR / "config" / "runtime.json"


def _load_runtime_config() -> tuple[Path | None, dict]:
    raw_path = os.getenv("AUTOANNOTATION_CONFIG", "").strip()
    config_path = Path(raw_path).expanduser() if raw_path else DEFAULT_CONFIG_FILE
    if not config_path.is_absolute():
        config_path = (PROJECT_DIR / config_path).resolve()
    if not config_path.exists():
        return None, {}
    try:
        return config_path, json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return config_path, {}


# ---------- 加载运行时配置文件 ----------
CONFIG_FILE, RUNTIME_CONFIG = _load_runtime_config()
CONFIG_BASE_DIR = PROJECT_DIR.resolve()


def _config_value(name: str, default: str) -> str:
    env_name = f"AUTOANNOTATION_{name.upper()}"
    if os.getenv(env_name, "").strip():
        return os.getenv(env_name, "").strip()
    raw = RUNTIME_CONFIG.get(name)
    if raw is None:
        return default
    text = str(raw).strip()
    return text or default


def resolve_runtime_path(value: str | Path, *, base_dir: Path | None = None) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    root = (base_dir or CONFIG_BASE_DIR).resolve()
    return (root / raw).resolve()


# ---------- 核心目录 ----------
DATA_DIR = resolve_runtime_path(_config_value("data_dir", "./data"))
FRONT_DIR = resolve_runtime_path(_config_value("front_dir", "./front"))
WEIGHTS_DIR = PROJECT_DIR / "config" / "weights"

# ---------- 数据子目录 ----------
UPLOAD_VIDEOS_DIR = DATA_DIR / "uploads" / "videos"
QWEN_REFS_DIR = DATA_DIR / "uploads" / "qwen_refs"
IMAGESETS_DIR = DATA_DIR / "imagesets"
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
HISTORY_DIR = DATA_DIR / "history"
JOBS_HISTORY_FILE = HISTORY_DIR / "jobs_history.jsonl"
SYSTEM_DIR = DATA_DIR / "system"
LICENSE_FILE = SYSTEM_DIR / "license.dat"
TRIAL_STATE_FILE = SYSTEM_DIR / "trial_state.dat"
USERS_DIR = DATA_DIR / "users"
USERS_INDEX_FILE = USERS_DIR / "index.json"

# ---------- 并发与推理默认参数 ----------
MAX_WORKERS = max(2, int(os.getenv("AUTOANNOTATION_MAX_WORKERS", "8")))

DEFAULT_EXTRACT_SECONDS = float(os.getenv("AUTOANNOTATION_DEFAULT_EXTRACT_SECONDS", "1.0"))
DEFAULT_CONFIDENCE = float(os.getenv("AUTOANNOTATION_DEFAULT_CONF", "0.25"))
DEFAULT_IOU = float(os.getenv("AUTOANNOTATION_DEFAULT_IOU", "0.45"))
DEFAULT_DEVICE = os.getenv("AUTOANNOTATION_DEFAULT_DEVICE", "")

# ---------- Qwen 大模型配置 ----------
QWEN_API_BASE = os.getenv("AUTOANNOTATION_QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_DEFAULT_MODEL = os.getenv("AUTOANNOTATION_QWEN_MODEL", "qwen-vl-max-latest")
QWEN_API_KEY = os.getenv("AUTOANNOTATION_QWEN_API_KEY", "")
QWEN_TIMEOUT_SECONDS = float(os.getenv("AUTOANNOTATION_QWEN_TIMEOUT", "45"))
QWEN_MAX_REF_IMAGES = max(0, int(os.getenv("AUTOANNOTATION_QWEN_MAX_REF_IMAGES", "6")))
QWEN_SEND_CANVAS_SIZE = max(256, int(os.getenv("AUTOANNOTATION_QWEN_SEND_CANVAS_SIZE", "1280")))
# ---------- 登录与鉴权 ----------
DEFAULT_ADMIN_USERNAME = "root"
DEFAULT_ADMIN_PASSWORD = "herefly"

ADMIN_USERNAME = os.getenv("AUTOANNOTATION_ADMIN_USERNAME", DEFAULT_ADMIN_USERNAME).strip() or DEFAULT_ADMIN_USERNAME
ADMIN_PASSWORD = os.getenv("AUTOANNOTATION_ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD).strip() or DEFAULT_ADMIN_PASSWORD

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}

# ---------- 会话与 Cookie ----------
LOGIN_REQUIRED = _env_bool("AUTOANNOTATION_LOGIN_REQUIRED", True)
SESSION_COOKIE_NAME = os.getenv("AUTOANNOTATION_SESSION_COOKIE", "autoannotation_session").strip() or "autoannotation_session"
SESSION_TTL_HOURS = max(1, int(os.getenv("AUTOANNOTATION_SESSION_TTL_HOURS", "168")))
_SESSION_SECRET_ENV = os.getenv("AUTOANNOTATION_SESSION_SECRET", "").strip()
SESSION_SECRET = _SESSION_SECRET_ENV or f"autoannotation::{ADMIN_USERNAME}::{ADMIN_PASSWORD}::{PROJECT_DIR}"

# ---------- License 授权 ----------

# ---------- CORS 跨域 ----------
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("AUTOANNOTATION_CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]

# ---------- 上传大小限制 ----------
MAX_MODEL_UPLOAD_BYTES = int(os.getenv("AUTOANNOTATION_MAX_MODEL_MB", "500")) * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = int(os.getenv("AUTOANNOTATION_MAX_VIDEO_MB", "2000")) * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = int(os.getenv("AUTOANNOTATION_MAX_IMAGE_MB", "20")) * 1024 * 1024
MAX_CLASSES_FILE_BYTES = 2 * 1024 * 1024

# ---------- 支持的文件扩展名 ----------
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".m4v", ".webm"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx"}
SUPPORTED_CLASSES_EXTENSIONS = {".txt", ".json", ".yaml", ".yml"}


def get_security_warnings() -> list[str]:
    warnings: list[str] = []
    if LOGIN_REQUIRED and ADMIN_PASSWORD == DEFAULT_ADMIN_PASSWORD:
        warnings.append("管理员密码仍为默认值，请设置 AUTOANNOTATION_ADMIN_PASSWORD")
    if not _SESSION_SECRET_ENV:
        warnings.append("会话密钥未显式配置，请设置 AUTOANNOTATION_SESSION_SECRET")
    if "*" in CORS_ORIGINS:
        warnings.append("CORS 当前允许任意来源，生产环境请设置 AUTOANNOTATION_CORS_ORIGINS")
    return warnings


# ---------- 目录初始化（启动时调用） ----------
def ensure_dirs() -> None:
    for path in [
        DATA_DIR,
        UPLOAD_VIDEOS_DIR,
        QWEN_REFS_DIR,
        IMAGESETS_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        HISTORY_DIR,
        SYSTEM_DIR,
        USERS_DIR,
        FRONT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# ---------- 路径转换工具 ----------
def to_data_relative_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def from_data_relative_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (DATA_DIR / candidate).resolve()
