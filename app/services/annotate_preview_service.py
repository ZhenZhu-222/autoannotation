# ============================================================
# 打标预览 / 回滚 / 产物查询服务
# 职责：从 routes_annotate.py 提取的全部业务逻辑
#   1. 解析任务上下文（活跃 + 历史）
#   2. 构建预览列表（manifest 解析 + overlay 渲染）
#   3. 构建产物摘要
#   4. 回滚标签到上一轮快照
# ============================================================
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import cv2

from app.core.config import DATA_DIR
from app.core.overlay_draw import draw_overlay
from app.core.state import AppState
from app.core.utils import safe_data_path, safe_int, safe_float, to_data_url
from app.services.annotation_service import AnnotationService
from app.services.class_name_service import resolve_imageset_class_names
from app.services.label_format import infer_label_task_from_dir, normalize_label_task, parse_label_line, shape_to_overlay_item
from app.services.task_manager import TaskManager


# ---------- 摘要字段白名单（避免 route 层逐字段复制） ----------
_SUMMARY_KEYS = [
    "total_images", "total_existing_boxes", "total_new_boxes",
    "total_final_boxes", "total_boxes",
    "label_mode", "update_imageset_labels", "round_tag", "operator",
    "class_id_overrides", "target_classes", "mapping_confirmed",
    "size_check_failed_images", "strict_size_check", "size_retry",
    "precision_mode", "sample_count",
    "max_calibration_error_px", "min_consensus_rate", "duplicate_iou",
    "quality_rejected_images", "svg_parse_failed_images",
    "render_mode", "reject_on_invalid_svg", "green_hsv_profile",
    "avg_quality_score", "avg_extract_quality_score",
    "avg_calibration_error_px", "avg_consensus_rate",
    "pipeline", "requested_device", "resolved_device",
    "label_task",
]


class AnnotatePreviewService:

    # ---------- 工具方法 ----------

    # 将本地路径转换为前端可访问的数据 URL
    @staticmethod
    def _url(path: Path) -> str:
        return to_data_url(path, DATA_DIR)

    # 从任务结果中安全提取摘要字段（仅保留白名单字段）
    @staticmethod
    def extract_summary(source: dict) -> dict:
        """从任务结果中安全提取摘要字段。"""
        return {k: source.get(k) for k in _SUMMARY_KEYS if k in source}

    # ---------- 任务上下文解析 ----------

    # 从活跃任务或历史记录中解析打标任务上下文（找不到抛 KeyError）
    @staticmethod
    def resolve_job_context(job_id: str, tasks: TaskManager) -> dict:
        """从活跃任务或历史记录中解析打标任务上下文。
        找不到抛 KeyError, 类型不对抛 TypeError, 未完成抛 ValueError。
        """
        job = tasks.get(job_id)
        if job:
            if job.job_type not in {"annotate", "qwen_annotate"}:
                raise TypeError("job 类型不是 annotate/qwen_annotate")
            if job.status != "succeeded" or not job.result:
                raise ValueError("任务尚未成功完成")
            artifacts = dict(job.result.get("artifacts", {}))
            artifacts.setdefault("preview_page_url", f"/front/preview.html?job_id={job.id}")
            return {
                "job_id": job.id,
                "status": job.status,
                "job_type": job.job_type,
                "summary_source": job.result,
                "artifacts": artifacts,
                "output_root": Path(job.result.get("output_root", "")),
            }

        history = tasks.get_history_entry(job_id)
        if not history:
            raise KeyError("job 不存在")
        history_job_type = str(history.get("job_type", ""))
        if history_job_type not in {"annotate", "qwen_annotate"}:
            raise TypeError("job 类型不是 annotate/qwen_annotate")
        if history.get("status") != "succeeded":
            raise ValueError("任务尚未成功完成")

        summary_source = dict(history.get("result_summary", {}))
        artifacts = dict(summary_source.get("artifacts", {}))
        artifacts.setdefault("preview_page_url", f"/front/preview.html?job_id={job_id}")
        manifest_path = safe_data_path(artifacts.get("manifest_csv", ""), DATA_DIR)
        output_root = manifest_path.parent if manifest_path else (DATA_DIR / "outputs" / job_id)
        return {
            "job_id": job_id,
            "status": "succeeded",
            "job_type": history_job_type,
            "summary_source": summary_source,
            "artifacts": artifacts,
            "output_root": output_root,
        }

    # ---------- 产物查询 ----------

    # 查询任务产物信息（叠加图/标签包/回滚状态等）
    @staticmethod
    def get_artifacts(job_id: str, tasks: TaskManager) -> dict:
        ctx = AnnotatePreviewService.resolve_job_context(job_id, tasks)
        summary_source = ctx["summary_source"]
        artifacts = dict(ctx["artifacts"])
        can_rollback = bool(summary_source.get("update_imageset_labels", False)) and (
            Path(ctx["output_root"]) / "labels_before_index.json"
        ).exists()
        artifacts.setdefault("rollback_api", f"/api/annotate/jobs/{ctx['job_id']}/rollback")
        artifacts.setdefault("cleanup_api", f"/api/jobs/{ctx['job_id']}/artifacts")
        artifacts["can_rollback"] = can_rollback

        return {
            "job_id": ctx["job_id"],
            "status": ctx["status"],
            "job_type": ctx["job_type"],
            "artifacts": artifacts,
            "output_root": str(ctx["output_root"]),
            "summary": AnnotatePreviewService.extract_summary(summary_source),
        }

    # ---------- 回滚 ----------

    # 将图片集标签回滚到本次任务前的快照
    @staticmethod
    def rollback_labels(job_id: str, tasks: TaskManager, state: AppState) -> dict:
        ctx = AnnotatePreviewService.resolve_job_context(job_id, tasks)
        summary_source = ctx["summary_source"]
        if not summary_source.get("update_imageset_labels", False):
            raise ValueError("该任务未回写图片集 labels，无需回滚")

        output_root = Path(ctx["output_root"])
        rollback_result = AnnotatePreviewService._rollback_imageset_labels(output_root, state)
        return {"job_id": ctx["job_id"], "status": "rolled_back", **rollback_result}

    # 实际执行回滚：按 labels_before_index.json 恢复或删除标签文件
    @staticmethod
    def _rollback_imageset_labels(output_root: Path, state: AppState) -> dict:
        index_path = output_root / "labels_before_index.json"
        backup_dir = output_root / "labels_before"
        if not index_path.exists():
            raise FileNotFoundError("未找到该任务的回滚快照（labels_before_index.json）")

        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise RuntimeError("回滚快照损坏，无法解析") from None

        imageset_id = str(payload.get("imageset_id") or "").strip()
        if not imageset_id:
            raise RuntimeError("回滚快照缺少 imageset_id")
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise KeyError("回滚目标图片集不存在")

        file_map = payload.get("files", {})
        if not isinstance(file_map, dict) or not file_map:
            raise ValueError("回滚快照中没有可恢复文件")

        labels_dir = Path(imageset.dir_path) / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        restored = 0
        removed = 0
        untouched = 0
        for raw_filename, existed in file_map.items():
            stem = Path(str(raw_filename)).stem
            if not stem:
                continue
            target = labels_dir / f"{stem}.txt"
            backup = backup_dir / f"{stem}.txt"
            if bool(existed):
                if backup.exists():
                    shutil.copy2(backup, target)
                else:
                    target.write_text("", encoding="utf-8")
                restored += 1
            else:
                if target.exists():
                    target.unlink()
                    removed += 1
                else:
                    untouched += 1

        return {
            "imageset_id": imageset_id,
            "restored_files": restored,
            "removed_files": removed,
            "untouched_files": untouched,
        }

    # ---------- overlay 渲染 ----------

    # 根据标签文件实时渲染叠加图（用于预览时按当前标签重绘）
    @staticmethod
    def _render_overlay_from_label(src_path: Path, out_path: Path, label_path: Path, class_names: dict[int, str], label_task: str = "detect") -> bool:
        if not src_path.exists() or not label_path.exists():
            return False
        image = cv2.imread(str(src_path))
        if image is None:
            return False
        height, width = image.shape[:2]
        items: list[dict] = []
        try:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parsed = parse_label_line(line, normalize_label_task(label_task))
                if not parsed:
                    continue
                items.append(shape_to_overlay_item(parsed, width, height, class_names.get(parsed.class_id, f"class_{parsed.class_id}")))
        except Exception:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        draw_overlay(src_path, out_path, items)
        return out_path.exists()

    # 构建最新渲染的叠加图 URL（有变更则自动重绘）
    @staticmethod
    def _build_resolved_overlay_url(*, output_root: Path, filename: str, source_path: Path, label_path: Path, class_names: dict[int, str], label_task: str = "detect") -> str:
        if not filename or not class_names or not source_path.exists() or not label_path.exists():
            return ""
        resolved_path = output_root / "overlays_resolved" / filename
        needs_refresh = not resolved_path.exists()
        if not needs_refresh:
            try:
                resolved_mtime = resolved_path.stat().st_mtime
                for candidate in (source_path, label_path):
                    if candidate.exists() and candidate.stat().st_mtime > resolved_mtime:
                        needs_refresh = True
                        break
            except Exception:
                needs_refresh = True
        if needs_refresh:
            ok = AnnotatePreviewService._render_overlay_from_label(source_path, resolved_path, label_path, class_names, label_task=label_task)
            if not ok:
                return ""
        return to_data_url(resolved_path, DATA_DIR)

    # ---------- 路径解析 ----------

    # 从任务结果或 run_meta.json 中提取 imageset_id
    @staticmethod
    def _resolve_imageset_id(summary_source: dict, output_root: Path) -> str:
        imageset_id = str(summary_source.get("imageset_id") or "").strip()
        if imageset_id:
            return imageset_id
        run_meta_path = output_root / "run_meta.json"
        if not run_meta_path.exists():
            return ""
        try:
            payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(payload.get("imageset_id") or "").strip()

    # 按优先级解析图片源文件路径（当前图片集 > 任务输出目录）
    @staticmethod
    def _resolve_source_path(*, source_path: Path, filename: str, output_root: Path, imageset_dir: Path | None, prefer_current_labels: bool = False) -> Path:
        candidates: list[Path] = []
        if filename and imageset_dir is not None and prefer_current_labels:
            candidates.append(imageset_dir / "images" / filename)
        if source_path.exists():
            candidates.append(source_path)
        if filename and imageset_dir is not None and not prefer_current_labels:
            candidates.append(imageset_dir / "images" / filename)
        if filename:
            candidates.append(output_root / "images" / filename)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return source_path

    # 按优先级解析标签文件路径（当前图片集 > 任务输出目录）
    @staticmethod
    def _resolve_label_path(*, label_path: Path, filename: str, output_root: Path, imageset_dir: Path | None, prefer_current_labels: bool = False) -> Path:
        stem = Path(filename).stem if filename else ""
        candidates: list[Path] = []
        if stem and imageset_dir is not None and prefer_current_labels:
            candidates.append(imageset_dir / "labels" / f"{stem}.txt")
        if label_path.exists():
            candidates.append(label_path)
        if stem:
            candidates.append(output_root / "labels" / f"{stem}.txt")
            if imageset_dir is not None and not prefer_current_labels:
                candidates.append(imageset_dir / "labels" / f"{stem}.txt")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return label_path

    # ---------- 预览列表构建（主入口） ----------

    # 主入口：构建任务预览列表（解析 manifest.csv，渲染叠加图，返回分页结果）
    @staticmethod
    def build_preview(
        job_id: str,
        page: int,
        page_size: int,
        only_with_boxes: bool,
        keyword: str,
        prefer_current_labels: bool,
        state: AppState,
        tasks: TaskManager,
    ) -> dict:
        ctx = AnnotatePreviewService.resolve_job_context(job_id, tasks)
        summary_source = ctx["summary_source"]
        output_root = Path(ctx["output_root"]).resolve()
        manifest_path = output_root / "manifest.csv"
        if not output_root.exists() or not manifest_path.exists():
            raise FileNotFoundError("任务产物不存在或已被清理")

        overlays_dir = output_root / "overlays"
        keyword_lower = (keyword or "").strip().lower()

        # 解析类别名
        preview_class_names: dict[int, str] = {}
        imageset_id = AnnotatePreviewService._resolve_imageset_id(summary_source, output_root)
        imageset_dir: Path | None = None
        preview_label_task = normalize_label_task(summary_source.get("label_task", "detect"))
        if imageset_id:
            imageset = state.get_imageset(imageset_id)
            if imageset:
                imageset_dir = Path(imageset.dir_path)
                preview_label_task = normalize_label_task(getattr(imageset, "label_task", "") or preview_label_task or infer_label_task_from_dir(imageset_dir / "labels", "detect"))
                try:
                    class_name_map = resolve_imageset_class_names(
                        imageset_id=imageset_id, imageset_dir=imageset_dir, state=state, tasks=tasks,
                    )
                    preview_class_names = {int(cid): name for cid, name in class_name_map.items()}
                except Exception:
                    preview_class_names = {}

        url = AnnotatePreviewService._url
        rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = Path(row.get("output_image") or row.get("source_image") or "").name
                if keyword_lower and keyword_lower not in filename.lower():
                    continue
                existing_boxes = safe_int(row.get("existing_boxes"), 0)
                new_boxes = safe_int(row.get("new_boxes"), 0)
                final_boxes = safe_int(row.get("final_boxes"), 0)
                if only_with_boxes and final_boxes <= 0:
                    continue

                source_path = AnnotatePreviewService._resolve_source_path(
                    source_path=Path(row.get("source_image", "")), filename=filename,
                    output_root=output_root, imageset_dir=imageset_dir, prefer_current_labels=prefer_current_labels,
                )
                overlay_path = overlays_dir / filename if filename else Path("")
                label_path = AnnotatePreviewService._resolve_label_path(
                    label_path=Path(row.get("label_file", "")), filename=filename,
                    output_root=output_root, imageset_dir=imageset_dir, prefer_current_labels=prefer_current_labels,
                )
                svg_overlay_path = Path(row.get("svg_overlay_file", ""))
                mask_path = Path(row.get("mask_file", ""))
                overlay_url = url(overlay_path) if overlay_path.exists() else ""
                resolved_overlay_url = AnnotatePreviewService._build_resolved_overlay_url(
                    output_root=output_root, filename=filename,
                    source_path=source_path, label_path=label_path, class_names=preview_class_names, label_task=preview_label_task,
                )
                if resolved_overlay_url:
                    overlay_url = resolved_overlay_url

                rows.append({
                    "image_id": row.get("image_id", ""),
                    "filename": filename,
                    "source_url": url(source_path),
                    "overlay_url": overlay_url,
                    "label_url": url(label_path) if label_path.exists() else "",
                    "svg_overlay_url": url(svg_overlay_path) if svg_overlay_path.exists() else "",
                    "mask_url": url(mask_path) if mask_path.exists() else "",
                    "existing_boxes": existing_boxes,
                    "new_boxes": new_boxes,
                    "final_boxes": final_boxes,
                    "size_check_passed": safe_int(row.get("size_check_passed"), 1),
                    "svg_parse_ok": safe_int(row.get("svg_parse_ok"), 0),
                    "green_pixels": safe_int(row.get("green_pixels"), 0),
                    "contour_count": safe_int(row.get("contour_count"), 0),
                    "calibration_error_px": safe_float(row.get("calibration_error_px"), 0.0),
                    "consensus_rate": safe_float(row.get("consensus_rate"), 0.0),
                    "quality_score": safe_float(row.get("quality_score"), 0.0),
                    "extract_quality_score": safe_float(row.get("extract_quality_score"), 0.0),
                    "reject_reason": row.get("reject_reason", ""),
                    "refine_rounds": safe_int(row.get("refine_rounds"), 0),
                    "qwen_note": row.get("qwen_note", ""),
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                })

        total = len(rows)
        start = (page - 1) * page_size
        end = start + page_size

        return {
            "job_id": ctx["job_id"],
            "status": ctx["status"],
            "job_type": ctx["job_type"],
            "page": page,
            "page_size": page_size,
            "total": total,
            "summary": AnnotatePreviewService.extract_summary(summary_source),
            "class_names": preview_class_names,
            "label_task": preview_label_task,
            "items": rows[start:end],
        }
