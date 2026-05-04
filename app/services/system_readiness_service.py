# ============================================================
# 系统交付自检服务
# 职责：汇总运行目录、前端资源、安全默认值和发布产物状态
# ============================================================
from __future__ import annotations

from pathlib import Path

import app.core.config as config
from app.core.auth import get_license_status


REQUIRED_FRONT_FILES = [
    "index.html",
    "style.css",
    "ui_utils.js",
    "api.js",
    "dataset_upload.js",
    "imageset_preview.js",
    "gallery.js",
    "class_mapping.js",
    "annotate.js",
    "qwen_annotate.js",
    "jobs_history.js",
    "refine.js",
    "train.js",
    "app.js",
    "preview.html",
    "preview.js",
]

SOURCE_ARTIFACT_PATTERNS = [
    "*.tar",
    "*.tar.gz",
    "*.zip",
    "*.pt",
    "*.onnx",
    "dist*/autoannotation*.tar",
    "dist*/autoannotation*.tar.gz",
    "dist*/autoannotation*.zip",
]


# 将绝对路径转为相对项目根目录的可读路径（用于错误信息展示）
def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(config.PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# 检测目录是否可写（尝试创建目录并写临时文件）
def _check_writable_dir(path: Path) -> dict:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".readiness_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"name": _rel(path), "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"name": _rel(path), "ok": False, "detail": str(exc)}


# 扫描项目目录中是否残留了模型/发布包大文件（正式交付前应移出）
def _find_source_artifacts() -> list[str]:
    found: set[str] = set()
    for pattern in SOURCE_ARTIFACT_PATTERNS:
        for path in config.PROJECT_DIR.glob(pattern):
            if not path.is_file():
                continue
            if ".venv" in path.parts or ".git" in path.parts:
                continue
            found.add(_rel(path))
    return sorted(found)


class SystemReadinessService:
    # 汇总目录可写性、前端文件完整性、License 状态、Qwen 配置等，输出系统就绪报告
    @staticmethod
    def build() -> dict:
        writable_dirs = [
            _check_writable_dir(path)
            for path in [
                config.DATA_DIR,
                config.UPLOAD_VIDEOS_DIR,
                config.QWEN_REFS_DIR,
                config.IMAGESETS_DIR,
                config.MODELS_DIR,
                config.OUTPUTS_DIR,
                config.HISTORY_DIR,
                config.SYSTEM_DIR,
            ]
        ]
        front_files = [
            {"name": name, "ok": (config.FRONT_DIR / name).exists()}
            for name in REQUIRED_FRONT_FILES
        ]
        artifact_files = _find_source_artifacts()
        security_warnings = config.get_security_warnings()
        capability_warnings: list[str] = []
        if not config.QWEN_API_KEY:
            capability_warnings.append("未配置 AUTOANNOTATION_QWEN_API_KEY，AI 打标能力不可用")

        errors = [
            f"目录不可写: {item['name']}"
            for item in writable_dirs
            if not item["ok"]
        ]
        errors.extend(
            f"前端资源缺失: {item['name']}"
            for item in front_files
            if not item["ok"]
        )
        release_warnings = [
            "源码目录存在发布包/模型大文件，正式源码交付前请移出版本控制: "
            + ", ".join(artifact_files[:10])
        ] if artifact_files else []

        warnings = [*security_warnings, *capability_warnings, *release_warnings]
        operational = not errors
        production_ready = operational and not security_warnings and not artifact_files
        status = "ok" if production_ready else ("needs_attention" if operational else "failed")

        return {
            "status": status,
            "operational": operational,
            "production_ready": production_ready,
            "errors": errors,
            "warnings": warnings,
            "checks": {
                "writable_dirs": writable_dirs,
                "front_files": front_files,
                "license": get_license_status().to_dict(),
                "qwen_configured": bool(config.QWEN_API_KEY),
                "source_artifacts": artifact_files,
            },
            "acceptance": {
                "minimum_test_command": "PYTHONPATH=. pytest -q",
                "frontend_check_command": "node --check front/ui_utils.js front/api.js front/dataset_upload.js front/imageset_preview.js front/gallery.js front/class_mapping.js front/annotate.js front/qwen_annotate.js front/jobs_history.js front/refine.js front/train.js front/preview.js front/app.js",
                "full_script": "scripts/commercial_acceptance.sh",
            },
        }
