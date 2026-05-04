# ============================================================
# YOLO 训练服务
# 职责：校验图片集 / 执行 YOLO 训练 / 生成预览 / 保存训练结果
# ============================================================
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from app.core.config import MODELS_DIR, OUTPUTS_DIR, WEIGHTS_DIR
from app.core.overlay_draw import draw_overlay
from app.core.state import AppState
from app.services.class_name_service import collect_used_class_stats, read_used_imageset_class_names
from app.services.inference import Detection, create_predictor, resolve_device, extract_classes_from_model_file, extract_task_from_model_file
from app.services.label_format import infer_label_task_from_dir, normalize_label_task, parse_label_line
from app.services.task_manager import JobContext


class TrainingService:

    PREVIEW_CONF_CANDIDATES = (0.25, 0.1, 0.05, 0.01)
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # 提取检测结果的多边形点列表，无多边形时降级为外接矩形四角
    @staticmethod
    def _det_points_or_bbox(det) -> list[tuple[float, float]]:
        raw_points = getattr(det, "points", None)
        if isinstance(raw_points, list) and raw_points:
            points: list[tuple[float, float]] = []
            for point in raw_points:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError, IndexError):
                    continue
            if points:
                return points
        return [
            (float(getattr(det, "x1", 0.0)), float(getattr(det, "y1", 0.0))),
            (float(getattr(det, "x2", 0.0)), float(getattr(det, "y1", 0.0))),
            (float(getattr(det, "x2", 0.0)), float(getattr(det, "y2", 0.0))),
            (float(getattr(det, "x1", 0.0)), float(getattr(det, "y2", 0.0))),
        ]

    # 将训练后模型的预测结果绘制成叠加图
    @staticmethod
    def _draw_prediction_overlay(src_path: Path, out_path: Path, detections: list[Detection]) -> None:
        overlay_items = [
            {
                "x1": float(getattr(det, "x1", 0.0)),
                "y1": float(getattr(det, "y1", 0.0)),
                "x2": float(getattr(det, "x2", 0.0)),
                "y2": float(getattr(det, "y2", 0.0)),
                "points": TrainingService._det_points_or_bbox(det),
                "shape_type": normalize_label_task(getattr(det, "shape_type", "detect")),
                "label": f"{getattr(det, 'cls_name', 'object')} {float(getattr(det, 'conf', 0.0)):.2f}",
                "color_key": f"class_{int(getattr(det, 'cls_id', 0))}",
            }
            for det in detections
        ]
        draw_overlay(src_path, out_path, overlay_items)

    # 将预测结果格式化为可读的文本字符串（含 conf/坐标）
    @staticmethod
    def _format_prediction_text(detections: list[Detection], preview_conf: float | None = None) -> str:
        prefix = f"预览阈值 conf={preview_conf:.3f}\n" if preview_conf is not None else ""
        if not detections:
            return prefix + "无预测"
        return prefix + "\n".join(
            f"{int(getattr(det, 'cls_id', 0))} {getattr(det, 'cls_name', 'object')} "
            f"conf={float(getattr(det, 'conf', 0.0)):.3f} "
            f"{normalize_label_task(getattr(det, 'shape_type', 'detect'))}=("
            f"{float(getattr(det, 'x1', 0.0)):.1f}, {float(getattr(det, 'y1', 0.0)):.1f}, "
            f"{float(getattr(det, 'x2', 0.0)):.1f}, {float(getattr(det, 'y2', 0.0)):.1f})"
            for det in detections
        )

    # 逐步降低 conf 阈值进行预测，直到有检测结果为止（保证预览图不为空）
    @staticmethod
    def _predict_preview_detections(
        predictor,
        source_path: Path,
        *,
        device: str,
        iou: float,
        conf_candidates: tuple[float, ...],
    ) -> tuple[list[Detection], float]:
        last_detections: list[Detection] = []
        last_conf = conf_candidates[-1]
        for conf in conf_candidates:
            detections = predictor.predict(str(source_path), conf=conf, iou=iou, device=device)
            last_detections = detections
            last_conf = conf
            if detections:
                return detections, conf
        return last_detections, last_conf

    # 对图片集中若干样本图片运行预测，生成预览叠加图列表
    @staticmethod
    def _build_prediction_preview(
        *,
        imageset,
        imageset_dir: Path,
        model_path: Path,
        output_dir: Path,
        device: str,
        max_images: int = 4,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[dict]:
        preview_items: list[dict] = []
        try:
            predictor = create_predictor(str(model_path))
        except Exception:
            return preview_items

        preview_root = output_dir / "prediction_preview"
        overlays_dir = preview_root / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        conf_candidates = tuple(dict.fromkeys((conf, *TrainingService.PREVIEW_CONF_CANDIDATES)))

        for image in list(imageset.images)[:max_images]:
            source_path = imageset_dir / image.rel_path
            if not source_path.exists():
                continue
            try:
                detections, used_conf = TrainingService._predict_preview_detections(
                    predictor,
                    source_path,
                    device=device,
                    iou=iou,
                    conf_candidates=conf_candidates,
                )
            except Exception:
                continue

            overlay_path = overlays_dir / image.filename
            try:
                if detections:
                    TrainingService._draw_prediction_overlay(source_path, overlay_path, detections)
                else:
                    shutil.copy2(source_path, overlay_path)
            except Exception:
                continue

            preview_items.append(
                {
                    "filename": image.filename,
                    "source_url": f"/data/imagesets/{imageset.id}/{image.rel_path}",
                    "overlay_url": f"/data/outputs/{output_dir.name}/prediction_preview/overlays/{image.filename}",
                    "prediction_count": len(detections),
                    "prediction_text": TrainingService._format_prediction_text(detections, used_conf),
                    "preview_conf": used_conf,
                }
            )

        return preview_items
    # 验证所有标签文件格式是否与指定任务类型匹配
    @staticmethod
    def _validate_label_task(imageset_dir: Path, task: str) -> str:
        labels_dir = imageset_dir / "labels"
        effective_task = normalize_label_task(task, infer_label_task_from_dir(labels_dir, "detect"))
        for label_file in sorted(labels_dir.glob("*.txt")):
            if label_file.name == "classes.txt":
                continue
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
            except Exception:  # noqa: BLE001
                continue
            for idx, line in enumerate(lines, start=1):
                text = str(line or "").strip()
                if not text:
                    continue
                if parse_label_line(text, effective_task) is None:
                    raise ValueError(f"{label_file.name}:{idx} 与训练任务 {effective_task} 不匹配")
        return effective_task

    # 验证图片集是否可训练（有标签、class_id 连续），返回已用 ID 列表和任务类型
    @staticmethod
    def validate_trainable_imageset(imageset_dir: Path, task: str = "detect") -> tuple[list[int], str]:
        labels_dir = imageset_dir / "labels"
        if not labels_dir.exists() or not list(labels_dir.glob("*.txt")):
            raise ValueError("dataset has no labels; annotate first")

        used_ids = sorted(collect_used_class_stats(imageset_dir).keys())
        if not used_ids:
            raise ValueError("dataset has no labels; annotate first")
        effective_task = TrainingService._validate_label_task(imageset_dir, task)

        expected_ids = list(range(len(used_ids)))
        if used_ids != expected_ids:
            current = ", ".join(str(class_id) for class_id in used_ids)
            expected = f"0..{len(used_ids) - 1}"
            raise ValueError(
                "当前图片集继续打标没问题，但 YOLO 训练要求类 ID 连续。"
                f"当前实际使用的 IDs: {current}；训练应为连续编号 {expected}。"
                "请先到“类 ID 调整”里整理为连续编号后再训练。"
            )
        return used_ids, effective_task

    # 生成训练用 data.yaml，验证图片集可训练性并写入类别名
    @staticmethod
    def _prepare_train_data_yaml(imageset_dir: Path, output_dir: Path, task: str) -> Path:
        images_dir = imageset_dir / "images"
        if not images_dir.exists():
            raise ValueError(f"dataset images dir not found: {images_dir}")
        if not any(path.is_file() and path.suffix.lower() in TrainingService.IMAGE_SUFFIXES for path in images_dir.iterdir()):
            raise ValueError(f"dataset images dir is empty: {images_dir}")

        used_ids, effective_task = TrainingService.validate_trainable_imageset(imageset_dir, task)
        class_names = read_used_imageset_class_names(imageset_dir)
        names = {class_id: class_names.get(class_id, f"class_{class_id}") for class_id in used_ids}
        if not names:
            raise ValueError("no classes.txt or data.yaml; annotate first")

        output_dir.mkdir(parents=True, exist_ok=True)
        data_yaml_path = output_dir / "data.yaml"
        data_yaml = {
            "path": str(imageset_dir.resolve()),
            "train": str(images_dir.resolve()),
            "val": str(images_dir.resolve()),
            "names": names,
        }
        if effective_task != "detect":
            data_yaml["task"] = effective_task
        data_yaml_path.write_text(
            yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return data_yaml_path

    # 主入口：执行 YOLO 训练任务，保存模型，生成预测预览
    @staticmethod
    def run_train_job(
        state: AppState,
        job_ctx: JobContext,
        imageset_id: str,
        epochs: int = 50,
        batch_size: int = 16,
        img_size: int = 640,
        base_model: str = "yolo11n.pt",
        base_model_id: str = "",
        task: str = "detect",
        device: str = "",
        save_to_system: bool = True,
        operator: str = "anonymous",
        creator_id: str = "",
    ) -> dict:
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise ValueError("imageset_id not found")
        if not imageset.images:
            raise ValueError("dataset is empty")

        imageset_dir = Path(imageset.dir_path)
        requested_task = normalize_label_task(task, getattr(imageset, "label_task", "detect"))
        _, dataset_task = TrainingService.validate_trainable_imageset(imageset_dir, requested_task)
        if getattr(imageset, "label_task", "") != dataset_task:
            imageset.label_task = dataset_task
            state._save_imageset(imageset)

        resolved_base = base_model
        base_model_task = requested_task
        if base_model_id:
            model_rec = state.get_model(base_model_id)
            if model_rec and Path(model_rec.model_path).exists():
                resolved_base = model_rec.model_path
                base_model_task = normalize_label_task(getattr(model_rec, "task", "") or dataset_task)
            elif model_rec:
                raise ValueError(f"基础模型文件不存在: {model_rec.model_path}")
            else:
                raise ValueError(f"系统模型 {base_model_id} 不存在")
        elif not Path(resolved_base).is_absolute():
            candidate = WEIGHTS_DIR / resolved_base
            if candidate.exists():
                resolved_base = str(candidate)
                base_model_task = extract_task_from_model_file(str(candidate))
            else:
                raise ValueError(
                    f"预训练权重 {resolved_base} 不存在于 {WEIGHTS_DIR}，"
                    "请先上传权重文件或选择已有系统模型"
                )
        else:
            base_model_task = extract_task_from_model_file(str(resolved_base))

        if base_model_task != dataset_task:
            raise ValueError(f"基础模型任务为 {base_model_task}，但当前数据集标签任务为 {dataset_task}")

        resolved_device = resolve_device(device)
        output_dir = OUTPUTS_DIR / job_ctx.job_id
        data_yaml_path = TrainingService._prepare_train_data_yaml(imageset_dir, output_dir, dataset_task)

        job_ctx.set_progress(0, 100, "loading model...")

        from ultralytics import YOLO  # lazy import

        model = YOLO(resolved_base)

        job_ctx.set_progress(5, 100, f"start training: {epochs} epochs")

        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            pct = 5 + int(90 * epoch / max(1, epochs))
            job_ctx.set_progress(pct, 100, f"epoch {epoch}/{epochs}")

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        # Docker 默认 /dev/shm 只有 64MB，多 worker 会卡死；CPU 训练也不需要多 worker
        train_workers = 0 if resolved_device == "cpu" else 2

        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            task=dataset_task,
            device=resolved_device,
            workers=train_workers,
            project=str(output_dir),
            name="train",
            exist_ok=True,
            verbose=False,
        )
        _ = results

        job_ctx.set_progress(95, 100, "saving results...")

        train_dir = output_dir / "train"
        weights_dir = train_dir / "weights"
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        result_pt = best_pt if best_pt.exists() else (last_pt if last_pt.exists() else None)

        if not result_pt:
            raise RuntimeError("training finished but no weight file found")

        output_pt = output_dir / "best.pt"
        shutil.copy2(result_pt, output_pt)

        trained_classes = extract_classes_from_model_file(str(output_pt))
        trained_task = extract_task_from_model_file(str(output_pt))

        saved_model_id = ""
        if save_to_system:
            model_name = f"{imageset.name}_trained.pt"
            record = state.register_model(
                name=model_name,
                model_type="pt",
                model_path="",
                classes=trained_classes,
                class_source="model",
                task=trained_task,
                creator_id=creator_id,
            )
            model_dir = MODELS_DIR / record.id
            model_dir.mkdir(parents=True, exist_ok=True)
            final_path = model_dir / model_name
            shutil.copy2(output_pt, final_path)
            record.model_path = str(final_path)
            state._save_model(record)
            saved_model_id = record.id

        prediction_preview = TrainingService._build_prediction_preview(
            imageset=imageset,
            imageset_dir=imageset_dir,
            model_path=output_pt,
            output_dir=output_dir,
            device=resolved_device,
        )

        meta = {
            "job_id": job_ctx.job_id,
            "job_type": "train",
            "imageset_id": imageset_id,
            "imageset_name": imageset.name,
            "base_model": resolved_base,
            "base_model_id": base_model_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "task": dataset_task,
            "device": resolved_device,
            "operator": (operator or "anonymous").strip() or "anonymous",
            "operator_id": creator_id,
            "operator_username": (operator or "anonymous").strip() or "anonymous",
            "save_to_system": save_to_system,
            "saved_model_id": saved_model_id,
            "trained_classes": trained_classes,
            "result_pt": str(output_pt),
            "prediction_preview": prediction_preview,
        }
        (output_dir / "run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        job_ctx.set_progress(100, 100, "done")

        return {
            "job_id": job_ctx.job_id,
            "imageset_id": imageset_id,
            "imageset_name": imageset.name,
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "task": dataset_task,
            "device": resolved_device,
            "operator": meta["operator"],
            "operator_id": creator_id,
            "operator_username": meta["operator"],
            "saved_model_id": saved_model_id,
            "trained_classes": trained_classes,
            "class_count": len(trained_classes),
            "prediction_preview": prediction_preview,
            "artifacts": {
                "best_pt": f"/data/outputs/{job_ctx.job_id}/best.pt",
                "train_dir": f"/data/outputs/{job_ctx.job_id}/train",
                "run_meta_json": f"/data/outputs/{job_ctx.job_id}/run_meta.json",
                "prediction_preview_dir": f"/data/outputs/{job_ctx.job_id}/prediction_preview",
                "cleanup_api": f"/api/jobs/{job_ctx.job_id}/artifacts",
            },
        }
