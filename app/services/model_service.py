# ============================================================
# 模型服务
# 职责：模型上传保存 / classes 文件解析 / 同名模型去重替换
# ============================================================
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from fastapi import HTTPException

from app.core.config import MODELS_DIR, SUPPORTED_CLASSES_EXTENSIONS, SUPPORTED_MODEL_EXTENSIONS
from app.core.state import AppState
from app.core.utils import ensure_unique_path, sanitize_filename
from app.services.inference import extract_classes_from_model_file, extract_task_from_model_file


class ModelService:
    # 解析上传的 classes 文件（txt/json/yaml），返回类别名称列表
    @staticmethod
    def parse_classes_file(filename: str, content: bytes) -> list[str]:
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_CLASSES_EXTENSIONS:
            raise HTTPException(status_code=400, detail="classes 文件格式仅支持 txt/json/yaml/yml")

        text = content.decode("utf-8")
        if suffix == ".txt":
            classes = [line.strip() for line in text.splitlines() if line.strip()]
            if not classes:
                raise HTTPException(status_code=400, detail="classes txt 为空")
            return classes

        if suffix == ".json":
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)

        if isinstance(payload, list):
            classes = [str(x) for x in payload]
        elif isinstance(payload, dict):
            names = payload.get("names", payload)
            if isinstance(names, list):
                classes = [str(x) for x in names]
            elif isinstance(names, dict):
                def _sort_key(value):
                    try:
                        return int(value)
                    except Exception:  # noqa: BLE001
                        return str(value)

                classes = [str(names[k]) for k in sorted(names.keys(), key=_sort_key)]
            else:
                raise HTTPException(status_code=400, detail="classes 文件结构无效")
        else:
            raise HTTPException(status_code=400, detail="classes 文件结构无效")

        classes = [c.strip() for c in classes if c and str(c).strip()]
        if not classes:
            raise HTTPException(status_code=400, detail="classes 文件为空")
        return classes

    # 保存上传的模型文件到磁盘，解析类别和任务类型，若同名模型已存在则替换
    @staticmethod
    def save_model_upload(
        state: AppState,
        model_filename: str,
        model_bytes: bytes,
        classes_filename: str | None,
        classes_bytes: bytes | None,
        creator_id: str = "",
    ) -> dict:
        clean_name = sanitize_filename(model_filename, "model")
        suffix = Path(clean_name).suffix.lower()
        if suffix not in SUPPORTED_MODEL_EXTENSIONS:
            raise HTTPException(status_code=400, detail="模型只支持 .pt 或 .onnx")

        with TemporaryDirectory(prefix="tmp_model_", dir=MODELS_DIR) as temp_dir:
            model_path = ensure_unique_path(Path(temp_dir) / clean_name)
            model_path.write_bytes(model_bytes)

            classes = extract_classes_from_model_file(str(model_path))
            task = extract_task_from_model_file(str(model_path))
            class_source = "model"

            if not classes:
                if suffix == ".onnx":
                    if not classes_filename or not classes_bytes:
                        raise HTTPException(
                            status_code=400,
                            detail="ONNX 未解析到类别，请上传 classes 文件（txt/json/yaml）",
                        )
                    classes = ModelService.parse_classes_file(classes_filename, classes_bytes)
                    class_source = "classes_file"
                elif classes_filename and classes_bytes:
                    classes = ModelService.parse_classes_file(classes_filename, classes_bytes)
                    class_source = "classes_file"
                else:
                    raise HTTPException(status_code=400, detail="模型未解析到类别")

            # 同名模型去重：如果已有同名 + 同类型模型，替换旧的
            existing = None
            for m in state.list_models():
                if m.name == clean_name and m.model_type == suffix.lstrip("."):
                    existing = m
                    break

            if existing:
                # 替换旧模型文件和元数据
                old_dir = MODELS_DIR / existing.id
                if old_dir.exists():
                    import shutil
                    shutil.rmtree(old_dir, ignore_errors=True)
                old_dir.mkdir(parents=True, exist_ok=True)
                final_model_path = old_dir / clean_name
                final_model_path.write_bytes(model_bytes)
                with state._lock:
                    existing.classes = classes
                    existing.class_source = class_source
                    existing.model_path = str(final_model_path)
                    existing.task = task
                    if creator_id:
                        existing.creator_id = creator_id
                    state._save_model(existing)
                record = existing
                updated = existing
            else:
                record = state.register_model(
                    name=clean_name,
                    model_type=suffix.lstrip("."),
                    model_path="",
                    classes=classes,
                    class_source=class_source,
                    task=task,
                    creator_id=creator_id,
                )

                final_model_dir = MODELS_DIR / record.id
                final_model_dir.mkdir(parents=True, exist_ok=True)
                final_model_path = final_model_dir / clean_name
                final_model_path.write_bytes(model_bytes)

                # 补写实际模型路径
                updated = state.models[record.id]
                updated.model_path = str(final_model_path)
                updated.task = task
                state._save_model(updated)

            return {
                "model_id": updated.id,
                "name": updated.name,
                "model_type": updated.model_type,
                "class_source": updated.class_source,
                "task": updated.task,
                "classes": updated.classes,
                "model_path": updated.model_path,
                "created_at": updated.created_at,
                "creator_id": updated.creator_id,
            }
