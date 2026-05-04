# ============================================================
# 任务产物清理服务
# 职责：解析任务摘要 / 清理产物文件 / 级联删除图片集
# ============================================================
from __future__ import annotations

import shutil
from pathlib import Path

from app.core.config import DATA_DIR
from app.core.state import AppState
from app.core.utils import safe_data_path
from app.services.task_manager import TaskManager


class JobCleanupService:

    @staticmethod
    def resolve_job_summary(job_id: str, tasks: TaskManager) -> tuple[dict, str]:
        """从活跃任务或历史记录中获取任务摘要和类型。未找到抛 KeyError。"""
        record = tasks.get(job_id)
        if record and record.result:
            return dict(record.result), str(record.job_type)
        history = tasks.get_history_entry(job_id)
        if not history:
            raise KeyError("job 不存在")
        return dict(history.get("result_summary") or {}), str(history.get("job_type") or "")

    @staticmethod
    def cleanup_artifacts(job_id: str, summary: dict) -> dict[str, int]:
        """清理标注任务的产物文件和目录。"""
        artifacts = dict(summary.get("artifacts") or {})
        candidates: set[Path] = set()
        for _, value in artifacts.items():
            if not isinstance(value, str):
                continue
            path = safe_data_path(value, DATA_DIR)
            if path is None:
                continue
            candidates.add(path)

        output_root = DATA_DIR / "outputs" / job_id
        candidates.add(output_root)

        removed_files = 0
        removed_dirs = 0
        for path in sorted(candidates, key=lambda p: len(str(p)), reverse=True):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed_dirs += 1
            else:
                path.unlink(missing_ok=True)
                removed_files += 1
        return {"removed_files": removed_files, "removed_dirs": removed_dirs}

    @staticmethod
    def delete_imageset_from_summary(summary: dict, state: AppState) -> dict:
        """根据任务摘要中的 imageset_id 级联删除图片集。"""
        imageset_id = str(summary.get("imageset_id") or "").strip()
        if not imageset_id:
            return {"imageset_id": "", "imageset_deleted": False}
        if not state.get_imageset(imageset_id):
            return {"imageset_id": imageset_id, "imageset_deleted": False}
        try:
            state.delete_imageset(imageset_id)
        except KeyError:
            return {"imageset_id": imageset_id, "imageset_deleted": False}
        return {"imageset_id": imageset_id, "imageset_deleted": True}

    @staticmethod
    def delete_artifacts_for_job(job_id: str, tasks: TaskManager) -> dict:
        """删除标注任务的产物。非标注类型抛 ValueError。"""
        summary, job_type = JobCleanupService.resolve_job_summary(job_id, tasks)
        if job_type not in {"annotate", "qwen_annotate"}:
            raise ValueError("仅标注任务支持产物清理")
        removed = JobCleanupService.cleanup_artifacts(job_id, summary)
        return {"job_id": job_id, **removed}

    @staticmethod
    def delete_history_with_cleanup(
        job_id: str,
        tasks: TaskManager,
        state: AppState,
        with_artifacts: bool = True,
        with_imageset: bool = True,
    ) -> dict:
        """删除历史记录，可选同时清理产物和图片集。未找到抛 KeyError。"""
        summary, job_type = JobCleanupService.resolve_job_summary(job_id, tasks)
        artifacts_removed = {"removed_files": 0, "removed_dirs": 0}
        if with_artifacts and job_type in {"annotate", "qwen_annotate"}:
            artifacts_removed = JobCleanupService.cleanup_artifacts(job_id, summary)
        imageset_removed = {"imageset_id": "", "imageset_deleted": False}
        if with_imageset:
            imageset_removed = JobCleanupService.delete_imageset_from_summary(summary, state)
        history_deleted = tasks.delete_history_entry(job_id, delete_active_job=False)
        return {
            **history_deleted,
            "job_type": job_type,
            "artifacts_removed": artifacts_removed,
            "imageset_removed": imageset_removed,
        }
