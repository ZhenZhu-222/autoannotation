# ============================================================
# 图片集服务
# 职责：创建图片集 / 列表查询 / 图片列表 / 下载打包
# ============================================================
from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from fastapi import HTTPException

from app.core.config import DATA_DIR
from app.core.state import AppState
from app.core.utils import safe_data_path, to_data_url
from app.services.class_name_service import resolve_imageset_class_names
from app.services.dataset_import_service import DatasetImportService


class ImageSetService:
    # 从已暂存的文件路径列表创建图片集（支持同时导入图片和标签）
    @staticmethod
    def create_from_staged_paths(
        state: AppState,
        files: list[tuple[str, Path]],
        name: str | None = None,
        creator_id: str = "",
    ) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="未上传图片文件")

        with TemporaryDirectory(prefix="upload_imageset_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            try:
                prepared = DatasetImportService.prepare_upload(files, tmp_path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None

            if not prepared.image_files:
                raise HTTPException(status_code=400, detail="没有可用的图片文件；请确认目录中包含 jpg/png/bmp/webp 图片")

            imageset = state.create_imageset(
                name=name or "manual_upload",
                source="upload-folder",
                image_files=prepared.image_files,
                creator_id=creator_id,
            )
            applied = DatasetImportService.apply_to_imageset(imageset, prepared)
            state._save_imageset(imageset)

            return {
                "imageset_id": imageset.id,
                "name": imageset.name,
                "image_count": len(imageset.images),
                "labels_imported": applied.labels_imported,
                "labels_unmatched": len(applied.labels_unmatched),
                "labels_unmatched_examples": applied.labels_unmatched[:20],
                "duplicate_label_name_count": applied.duplicate_label_name_count,
                "class_names_imported": applied.class_names_imported,
                "class_name_sources": applied.class_name_sources,
                "label_task": applied.label_task,
                "creator_id": imageset.creator_id,
            }

    # ---------- 图片集列表 ----------
    # 返回所有图片集的摘要列表（含图片数、标签数、最大 class_id）
    @staticmethod
    def list_imagesets(state: AppState, user=None) -> dict:
        items = []
        for imageset in state.list_imagesets():
            if user is not None:
                from app.core.rbac import can_operate_imageset
                if not can_operate_imageset(user, imageset):
                    continue
            labels_dir = Path(imageset.dir_path) / "labels"
            label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
            label_count = len([f for f in label_files if f.name != "classes.txt"])
            # Scan for max class ID across all labels
            max_cid = -1
            for lf in label_files:
                if lf.name == "classes.txt":
                    continue
                try:
                    for line in lf.read_text(encoding="utf-8").splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            if cid > max_cid:
                                max_cid = cid
                except Exception:  # noqa: BLE001
                    pass
            creator = state.get_user(getattr(imageset, "creator_id", ""))
            assignee_names = []
            for user_id in getattr(imageset, "assignee_ids", []):
                assignee = state.get_user(user_id)
                if assignee:
                    assignee_names.append(assignee.username)
            assignee_history = []
            for assignment in getattr(imageset, "assignee_history", []) or []:
                assignee_history.append({
                    "id": assignment.get("user_id") or "",
                    "username": assignment.get("username") or "已删除用户",
                    "role": "operator",
                    "enabled": False,
                    "status": assignment.get("status") or "",
                    "assigned_at": assignment.get("assigned_at") or "",
                    "submitted_at": assignment.get("submitted_at") or "",
                    "reviewed_at": assignment.get("reviewed_at") or "",
                    "submit_note": assignment.get("submit_note") or "",
                    "review_note": assignment.get("review_note") or "",
                    "reviewer_username": assignment.get("reviewer_username") or "",
                })
            items.append({
                "imageset_id": imageset.id,
                "name": imageset.name,
                "source": imageset.source,
                "image_count": len(imageset.images),
                "label_count": label_count,
                "max_class_id": max_cid,
                "label_task": getattr(imageset, "label_task", "detect"),
                "created_at": imageset.created_at,
                "creator_id": getattr(imageset, "creator_id", ""),
                "creator_username": creator.username if creator else "已删除用户",
                "assignee_ids": list(getattr(imageset, "assignee_ids", [])),
                "assignee_usernames": assignee_names,
                "assignee_states": dict(getattr(imageset, "assignee_states", {}) or {}),
                "assignee_history": assignee_history,
                "assignment_history_count": len(getattr(imageset, "assignee_history", []) or []),
                "can_operate": True,
            })
        return {"items": items}

    # ---------- 图片列表（分页/筛选） ----------
    # 分页查询指定图片集内的图片，支持按标签/审核状态/关键词过滤
    @staticmethod
    def list_images(
        imageset_id: str,
        page: int,
        page_size: int,
        has_label: bool | None,
        review_status: str,
        keyword: str,
        state: AppState,
        tasks=None,
    ) -> dict:
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise KeyError("imageset 不存在")

        review_status_filter = (review_status or "").strip().lower()
        keyword_filter = (keyword or "").strip().lower()
        items_all = []
        for image in imageset.images:
            path = Path(imageset.dir_path) / image.rel_path
            label_path = Path(imageset.dir_path) / "labels" / f"{Path(image.filename).stem}.txt"
            label_exists = label_path.exists()
            if has_label is not None and bool(label_exists) is not bool(has_label):
                continue
            if review_status_filter and str(image.review_status or "").strip().lower() != review_status_filter:
                continue
            if keyword_filter and keyword_filter not in str(image.filename).lower():
                continue
            items_all.append({
                "image_id": image.id,
                "filename": image.filename,
                "url": to_data_url(path, DATA_DIR),
                "label_exists": label_exists,
                "label_url": to_data_url(label_path, DATA_DIR) if label_exists else "",
                "created_at": image.created_at,
                "review_status": image.review_status,
                "reviewer": image.reviewer,
                "review_note": image.review_note,
                "reviewed_at": image.reviewed_at,
            })

        total = len(items_all)
        start = (page - 1) * page_size
        end = start + page_size

        class_names = resolve_imageset_class_names(
            imageset_id=imageset_id, imageset_dir=Path(imageset.dir_path),
            state=state, tasks=tasks,
        )
        latest_job_id = ImageSetService._find_latest_job(imageset_id, tasks) if tasks else ""

        return {
            "imageset_id": imageset.id,
            "name": imageset.name,
            "source": imageset.source,
            "label_task": getattr(imageset, "label_task", "detect"),
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end < total,
            "items": items_all[start:end],
            "class_names": class_names,
            "latest_job_id": latest_job_id,
        }

    # 查找该图片集最近一次有效的标注任务 ID（产物仍在磁盘上）
    @staticmethod
    def _find_latest_job(imageset_id: str, tasks) -> str:
        """Find the most recent succeeded annotation job whose gallery artifacts still exist."""
        if tasks is None:
            return ""
        for entry in tasks.list_history(limit=500):
            if entry.get("status") != "succeeded":
                continue
            jt = entry.get("job_type", "")
            if jt not in ("annotate", "qwen_annotate"):
                continue
            summary = entry.get("result_summary", {})
            if summary.get("imageset_id") == imageset_id:
                artifacts = summary.get("artifacts") or {}
                manifest_path = safe_data_path(str(artifacts.get("manifest_csv") or ""), DATA_DIR)
                if not manifest_path or not manifest_path.exists():
                    continue
                return entry.get("job_id", "")
        return ""

    # ---------- 下载打包 ----------
    @staticmethod
    def download_imageset(imageset_id: str, state: AppState, tasks=None) -> Path:
        """生成 ZIP 压缩包并返回路径。imageset 不存在抛 KeyError。"""
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise KeyError("imageset 不存在")
        imageset_dir = Path(imageset.dir_path)
        if not imageset_dir.exists():
            raise FileNotFoundError("数据集目录不存在")

        class_names = resolve_imageset_class_names(
            imageset_id=imageset_id, imageset_dir=imageset_dir, state=state, tasks=tasks,
        )
        if class_names:
            names_dict = {int(k): v for k, v in sorted(class_names.items(), key=lambda x: int(x[0]))}
            data_yaml = {"path": ".", "train": "images", "val": "images", "names": names_dict}
            if getattr(imageset, "label_task", "detect") != "detect":
                data_yaml["task"] = getattr(imageset, "label_task", "detect")
            (imageset_dir / "data.yaml").write_text(
                yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8",
            )
            classes_txt = imageset_dir / "labels" / "classes.txt"
            if not classes_txt.exists():
                labels_dir = imageset_dir / "labels"
                labels_dir.mkdir(parents=True, exist_ok=True)
                classes_txt.write_text(
                    "\n".join(names_dict[i] for i in sorted(names_dict.keys())), encoding="utf-8",
                )

        zip_path = imageset_dir.parent / f"{imageset.name}_{imageset_id}"
        shutil.make_archive(str(zip_path), "zip", root_dir=imageset_dir)
        return Path(f"{zip_path}.zip")

    # ---------- refine 输出转换 ----------
    # 将精修服务返回的 payload 转换为 API 响应格式（image_path → image_url）
    @staticmethod
    def refine_payload_to_response(payload: dict) -> dict:
        image_path = Path(payload.pop("image_path", ""))
        label_path = Path(payload.pop("label_path", ""))
        return {
            **payload,
            "image_url": to_data_url(image_path, DATA_DIR) if image_path.exists() else "",
            "label_url": to_data_url(label_path, DATA_DIR) if label_path.exists() else "",
        }

    # ---------- 从文件列表创建 ----------
    # 从内存字节列表创建图片集（先暂存到临时目录再调用 create_from_staged_paths）
    @staticmethod
    def create_from_uploaded_files(
        state: AppState,
        files: list[tuple[str, bytes]],
        name: str | None = None,
        creator_id: str = "",
    ) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="未上传图片文件")

        with TemporaryDirectory(prefix="upload_imageset_staged_") as tmp_dir:
            staged_dir = Path(tmp_dir)
            staged_files: list[tuple[str, Path]] = []
            for idx, (filename, content) in enumerate(files):
                staged_path = staged_dir / f"{idx:06d}.bin"
                staged_path.write_bytes(content)
                staged_files.append((filename, staged_path))
            return ImageSetService.create_from_staged_paths(
                state=state, files=staged_files, name=name, creator_id=creator_id,
            )
