# ============================================================
# 任务管理器
# 职责：异步任务提交 / 进度更新 / 取消 / 历史记录持久化
# 思路：
#   1. 线程池执行任务，Lock 保护共享状态
#   2. 任务完成后写入 JSONL 历史文件
#   3. JobContext 提供进度回调和取消检测
# ============================================================
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict

from app.core.config import JOBS_HISTORY_FILE, MAX_WORKERS
from app.core.models import JobRecord
from app.core.utils import now_iso, new_id


JobRunner = Callable[["JobContext"], Dict[str, Any]]


class JobCancelledError(RuntimeError):
    pass


# ---------- 任务上下文：传给 runner 用于报告进度和检查取消 ----------
class JobContext:
    def __init__(self, manager: "TaskManager", job_id: str) -> None:
        self._manager = manager
        self.job_id = job_id

    # 更新任务进度，同时检查是否已被取消
    def set_progress(self, current: int, total: int, text: str = "") -> None:
        self.raise_if_cancelled()
        if total <= 0:
            progress = 0.0
        else:
            progress = max(0.0, min(1.0, current / total))
        self._manager.update_progress(self.job_id, progress, text)

    # 查询当前任务是否已被请求取消
    def is_cancel_requested(self) -> bool:
        return self._manager.is_cancel_requested(self.job_id)

    # 若已被取消则抛出 JobCancelledError，实现协作式取消
    def raise_if_cancelled(self) -> None:
        if self.is_cancel_requested():
            raise JobCancelledError("任务已取消")


# ---------- 任务管理器主体 ----------
class TaskManager:
    # ---------- start / stop / get ----------
    def __init__(self, max_workers: int = MAX_WORKERS, history_file: Path | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._history_file = Path(history_file or JOBS_HISTORY_FILE)
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []
        self._load_history()

    # 启动时从 JSONL 文件加载历史记录到内存
    def _load_history(self) -> None:
        if not self._history_file.exists():
            return
        for line in self._history_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self._history.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # 从任务结果中提取关键字段写入历史文件（过滤掉大体积数据）
    def _result_summary(self, job: JobRecord) -> dict[str, Any]:
        if not job.result:
            return {}
        keys = [
            "video_id",
            "imageset_id",
            "imageset_name",
            "saved_images",
            "model_id",
            "total_images",
            "total_boxes",
            "total_existing_boxes",
            "total_new_boxes",
            "total_final_boxes",
            "label_mode",
            "round_tag",
            "operator",
            "operator_id",
            "operator_username",
            "update_imageset_labels",
            "class_id_overrides",
            "final_class_name_map",
            "class_names",
            "target_classes",
            "pipeline",
            "size_check_failed_images",
            "strict_size_check",
            "size_retry",
            "precision_mode",
            "sample_count",
            "max_calibration_error_px",
            "min_consensus_rate",
            "duplicate_iou",
            "quality_rejected_images",
            "svg_parse_failed_images",
            "render_mode",
            "reject_on_invalid_svg",
            "green_hsv_profile",
            "avg_quality_score",
            "avg_extract_quality_score",
            "avg_calibration_error_px",
            "avg_consensus_rate",
            "requested_device",
            "resolved_device",
            "artifacts",
            "saved_model_id",
            "trained_classes",
            "class_count",
            "epochs",
            "batch_size",
            "img_size",
        ]
        return {k: job.result[k] for k in keys if k in job.result}

    # 将已完成任务追加到内存和 JSONL 历史文件
    def _append_history(self, job: JobRecord) -> None:
        payload = dict(job.payload or {})
        record = {
            "job_id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "operator": str(payload.get("operator") or "anonymous"),
            "operator_id": str(job.operator_id or payload.get("operator_id") or ""),
            "operator_username": str(payload.get("operator_username") or payload.get("operator") or "anonymous"),
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "progress": job.progress,
            "progress_text": job.progress_text,
            "cancel_requested": bool(job.cancel_requested),
            "payload": payload,
            "result_summary": self._result_summary(job),
            "error": job.error or "",
        }
        self._history.append(record)
        with self._history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 将内存中的历史记录全量覆写到 JSONL 文件
    def _rewrite_history_file(self) -> None:
        with self._history_file.open("w", encoding="utf-8") as f:
            for row in self._history:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- 任务提交与执行 ----------
    # 提交新任务到线程池，返回已入队的 JobRecord
    def submit(
        self,
        job_type: str,
        payload: Dict[str, Any],
        runner: JobRunner,
        operator_id: str = "",
        operator_username: str = "",
    ) -> JobRecord:
        payload = dict(payload or {})
        if operator_id:
            payload["operator_id"] = operator_id
        if operator_username:
            payload["operator_username"] = operator_username
            payload["operator"] = operator_username
        job = JobRecord(
            id=new_id("job"),
            job_type=job_type,
            status="queued",
            payload=payload,
            created_at=now_iso(),
            progress=0.0,
            operator_id=operator_id,
        )
        with self._lock:
            self._jobs[job.id] = job

        self._executor.submit(self._run_job, job.id, runner)
        return replace(job)

    # 内部执行器：在线程池中运行任务，处理完成/取消/失败三种状态
    def _run_job(self, job_id: str, runner: JobRunner) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "cancelled":
                return
            job.status = "running"
            job.started_at = now_iso()

        ctx = JobContext(self, job_id)
        try:
            result = runner(ctx)
            with self._lock:
                job = self._jobs[job_id]
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.error = "任务已取消"
                    job.finished_at = now_iso()
                    self._append_history(job)
                    return
                job.status = "succeeded"
                job.progress = 1.0
                if not job.progress_text:
                    job.progress_text = "完成"
                job.result = result
                job.finished_at = now_iso()
                self._append_history(job)
        except JobCancelledError as exc:
            with self._lock:
                job = self._jobs[job_id]
                job.status = "cancelled"
                job.error = str(exc) or "任务已取消"
                job.finished_at = now_iso()
                if not job.progress_text:
                    job.progress_text = "已取消"
                self._append_history(job)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = now_iso()
                self._append_history(job)

    # ---------- 进度查询与取消 ----------
    # 更新指定任务的进度百分比和描述文字
    def update_progress(self, job_id: str, progress: float, text: str = "") -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.progress = progress
            if text:
                job.progress_text = text

    # 获取指定任务的当前状态快照，不存在返回 None
    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return replace(job)

    # 查询指定任务是否已被请求取消
    def is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            return bool(job.cancel_requested)

    # 请求取消指定任务，已入队未开始的任务直接标记已取消
    def cancel(self, job_id: str, reason: str = "任务已取消") -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError("job not found")
            if job.status in {"succeeded", "failed", "cancelled"}:
                return replace(job)
            job.cancel_requested = True
            job.progress_text = "取消中..."
            if job.status == "queued":
                job.status = "cancelled"
                job.error = reason
                job.finished_at = now_iso()
                self._append_history(job)
            return replace(job)

    # ---------- 列表与历史 ----------
    # 返回内存中所有活跃任务的安全副本列表
    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return [replace(j) for j in self._jobs.values()]

    # 分页查询历史任务，支持按类型和操作者过滤
    def list_history(
        self,
        limit: int = 200,
        job_type: str = "",
        operator: str = "",
        operator_id: str = "",
    ) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 200), 2000))
        job_type = (job_type or "").strip().lower()
        operator = (operator or "").strip().lower()
        operator_id = (operator_id or "").strip()
        with self._lock:
            rows = list(reversed(self._history))
            if job_type:
                rows = [x for x in rows if str(x.get("job_type", "")).lower() == job_type]
            if operator_id:
                rows = [x for x in rows if str(x.get("operator_id") or "") == operator_id]
            if operator:
                rows = [
                    x for x in rows
                    if operator in str(x.get("operator_username") or x.get("operator") or "").lower()
                ]
            return rows[:limit]

    # 查询单条历史记录，不存在返回 None
    def get_history_entry(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            for item in reversed(self._history):
                if item.get("job_id") == job_id:
                    return dict(item)
            return None

    # 删除历史记录，可选同时删除内存中的活跃任务
    def delete_history_entry(self, job_id: str, delete_active_job: bool = False) -> dict[str, Any]:
        with self._lock:
            before = len(self._history)
            self._history = [item for item in self._history if item.get("job_id") != job_id]
            removed = before - len(self._history)
            if removed <= 0:
                raise KeyError("history job not found")
            self._rewrite_history_file()

            active_removed = False
            if delete_active_job and job_id in self._jobs:
                self._jobs.pop(job_id, None)
                active_removed = True

            return {
                "job_id": job_id,
                "removed_history_rows": removed,
                "removed_active_job": active_removed,
            }
