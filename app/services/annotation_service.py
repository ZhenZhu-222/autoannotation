# ============================================================
# YOLO 打标服务
# 职责：使用 YOLO 模型对图片集进行自动标注、生成叠加图、回写标签
# ============================================================
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import List

import cv2
import yaml

from app.core.config import DEFAULT_CONFIDENCE, DEFAULT_DEVICE, DEFAULT_IOU, OUTPUTS_DIR
from app.core.overlay_draw import draw_overlay
from app.core.state import AppState
from app.services.class_name_service import collect_used_class_stats, read_used_imageset_class_names
from app.services.inference import Detection, create_predictor, resolve_device
from app.services.label_format import (
    LabelShape,
    infer_label_task_from_dir,
    normalize_label_task,
    parse_label_line,
    points_to_shape,
    shape_to_overlay_item,
    sort_label_lines,
    xyxy_to_yolo as _xyxy_to_yolo,
)
from app.services.task_manager import JobContext


def rebuild_classes_txt(
    imageset_labels_dir: Path,
    imageset_dir: Path,
    new_class_names: dict[int, str] | None = None,
    label_task: str = "detect",
) -> list[str]:
    """Regenerate classes.txt + data.yaml from actual label files.

    Scans every label .txt to find which class IDs are actually used,
    then builds a dense classes.txt covering 0..max_used_id.

    Name resolution priority for actually-used IDs:
      1. new_class_names (from current annotation round)
      2. existing classes.txt (from previous rounds)
      3. fallback "class_{i}"

    For UNUSED IDs inside the dense 0..max_used_id range, always write
    placeholder names so stale semantic names do not linger forever.
    """
    new_class_names = new_class_names or {}

    # Read existing classes.txt for names from prior rounds
    existing_names: dict[int, str] = {}
    cls_txt_path = imageset_labels_dir / "classes.txt"
    if cls_txt_path.exists():
        for idx, line in enumerate(cls_txt_path.read_text(encoding="utf-8").splitlines()):
            name = line.strip()
            if name:
                existing_names[idx] = name

    # Scan ALL label files to find actually-used class IDs
    used_ids: set[int] = set()
    if imageset_labels_dir.exists():
        for lf in imageset_labels_dir.glob("*.txt"):
            if lf.name == "classes.txt":
                continue
            try:
                for line in lf.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        used_ids.add(int(float(parts[0])))
            except Exception:  # noqa: BLE001
                pass

    if not used_ids:
        # No labels at all — clean up
        if cls_txt_path.exists():
            cls_txt_path.unlink()
        data_yaml_path = imageset_dir / "data.yaml"
        if data_yaml_path.exists():
            data_yaml_path.unlink()
        return []

    max_id = max(used_ids)

    # Build name map: only used IDs keep semantic names; holes are placeholders
    names: dict[int, str] = {}
    for i in range(max_id + 1):
        if i not in used_ids:
            names[i] = f"class_{i}"
        elif i in new_class_names:
            names[i] = new_class_names[i]
        elif i in existing_names:
            names[i] = existing_names[i]
        else:
            names[i] = f"class_{i}"

    # Write classes.txt
    cls_lines = [names[i] for i in range(max_id + 1)]
    cls_txt_path.write_text("\n".join(cls_lines), encoding="utf-8")

    # Write data.yaml (only actually-used IDs in names dict)
    data_yaml = {
        "path": ".",
        "train": "images",
        "val": "images",
        "names": {i: names[i] for i in sorted(used_ids)},
    }
    effective_task = normalize_label_task(label_task, infer_label_task_from_dir(imageset_labels_dir, "detect"))
    if effective_task != "detect":
        data_yaml["task"] = effective_task
    (imageset_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )

    return cls_lines


# xyxy_to_yolo 的兼容导出别名，供 refine_service 等模块直接 import
def xyxy_to_yolo(cls_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    return _xyxy_to_yolo(cls_id, x1, y1, x2, y2, width, height)


class AnnotationService:
    # 提取检测结果的多边形点列表（segment/obb），无多边形时降级为外接矩形四角
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

    # 读取标签文件的有效行列表（空行已过滤）
    @staticmethod
    def _read_label_lines(label_path: Path) -> list[str]:
        if not label_path.exists():
            return []
        lines = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item:
                lines.append(item)
        return lines

    # 合并已有标签行和新标签行，去重（append 模式核心逻辑）
    @staticmethod
    def _merge_label_lines(existing: list[str], new: list[str]) -> list[str]:
        out: list[str] = []
        seen = set()
        for line in [*existing, *new]:
            text = line.strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    # 对标签行按 class_id / y / x 排序（统一标签文件顺序）
    @staticmethod
    def _sort_label_lines(lines: list[str], label_task: str) -> list[str]:
        return sort_label_lines(lines, label_task)

    # 解析单行 YOLO 标签为 LabelShape（parse_label_line 的包装）
    @staticmethod
    def _parse_label_line(line: str, label_task: str = "detect") -> LabelShape | None:
        return parse_label_line(line, label_task=label_task)

    # 确认并统一图片集的 label_task，append 模式下不允许跨任务类型合并
    @staticmethod
    def _ensure_imageset_label_task(imageset, label_task: str, label_mode: str, state: AppState) -> str:
        imageset_dir = Path(imageset.dir_path)
        labels_dir = imageset_dir / "labels"
        inferred = infer_label_task_from_dir(labels_dir, getattr(imageset, "label_task", "detect"))
        current = normalize_label_task(getattr(imageset, "label_task", "") or inferred, inferred)
        target = normalize_label_task(label_task, current)
        has_existing_labels = bool(collect_used_class_stats(imageset_dir))
        if has_existing_labels and label_mode == "append" and current != target:
            raise ValueError(
                f"当前图片集标签任务为 {current}，append 模式不能追加 {target}。请使用 replace 覆盖或新建图片集。"
            )
        if label_mode == "replace" or not has_existing_labels or getattr(imageset, "label_task", "") != target:
            imageset.label_task = target
            state._save_imageset(imageset)
        return target

    # ---------- 类别映射计算 — single source of truth ----------
    @staticmethod
    def resolve_mapping(
        state: AppState,
        model_id: str,
        imageset_id: str,
        selected_class_ids: list[int],
        target_classes: list[str] | None = None,
        label_mode: str = "append",
    ) -> dict[int, str]:
        """计算目标类别 ID→name 映射表。

        前端展示和后端打标都调用此方法，保证 ID 分配逻辑唯一。
        返回 {target_class_id: class_name} 字典。
        """
        model = state.get_model(model_id)
        if not model:
            raise ValueError("model_id 不存在")
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise ValueError("imageset_id 不存在")

        imageset_labels_dir = Path(imageset.dir_path) / "labels"

        # 1. 计算 offset：append 模式下从现有最大 class_id + 1 开始
        tc_offset = 0
        if label_mode == "append" and imageset_labels_dir.exists():
            max_cid = -1
            for lf in imageset_labels_dir.glob("*.txt"):
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
            if max_cid >= 0:
                tc_offset = max_cid + 1

        # 2. 构建现有 name→ID 查找表（用于同名复用）
        existing_used_class_name_map = read_used_imageset_class_names(Path(imageset.dir_path))
        existing_name_to_id: dict[str, int] = {
            name.lower(): class_id
            for class_id, name in existing_used_class_name_map.items()
            if name and not name.startswith("class_")
        }

        # 3. 分配目标类别 ID
        normalized_target_classes = [str(x).strip() for x in (target_classes or []) if str(x).strip()]
        selected_ids = sorted(set(int(x) for x in selected_class_ids))

        candidate: dict[int, str] = {}
        if normalized_target_classes:
            next_new_id = tc_offset
            for name in normalized_target_classes:
                existing_id = existing_name_to_id.get(name.lower())
                if existing_id is not None:
                    candidate[existing_id] = name
                else:
                    candidate[next_new_id] = name
                    next_new_id += 1
        else:
            next_new_id = tc_offset
            for src_id in selected_ids:
                cls_name = model.classes[src_id] if 0 <= src_id < len(model.classes) else f"class_{src_id}"
                existing_id = existing_name_to_id.get(cls_name.lower())
                if existing_id is not None:
                    candidate[existing_id] = cls_name
                else:
                    candidate[next_new_id] = cls_name
                    next_new_id += 1

        return candidate

    # 生成 CVAT 兼容的导出目录和压缩包（CVAT YOLO 1.1 和 Ultralytics YOLO 格式）
    @staticmethod
    def _build_cvat_exports(
        output_root: Path,
        images_dir: Path,
        labels_dir: Path,
        model_classes: list[str],
        selected_ids: list[int],
        job_id: str,
        final_class_name_map: dict[int, str] | None = None,
        label_task: str = "detect",
    ) -> dict:
        class_ids = set()
        effective_task = normalize_label_task(label_task)
        parsed_labels: dict[str, list[LabelShape]] = {}
        for label_file in labels_dir.glob("*.txt"):
            rows = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parsed = AnnotationService._parse_label_line(line, effective_task)
                if parsed is None:
                    continue
                rows.append(parsed)
                class_ids.add(parsed.class_id)
            parsed_labels[label_file.stem] = rows

        if not class_ids:
            class_ids = set(selected_ids)
        if not class_ids:
            class_ids = {0}

        sorted_ids = sorted(class_ids)
        id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        name_map = {
            old_id: (
                (final_class_name_map or {}).get(old_id)
                or (model_classes[old_id] if 0 <= old_id < len(model_classes) else f"class_{old_id}")
            )
            for old_id in sorted_ids
        }

        def remap_rows(rows):
            out = []
            for shape in rows:
                if shape.class_id not in id_map:
                    continue
                out.append(shape.with_class_id(id_map[shape.class_id]).format_yolo())
            return out

        # CVAT YOLO 1.1
        cvat11_artifacts = {}
        if effective_task == "detect":
            cvat11_dir = output_root / "cvat_yolo11"
            cvat11_data = cvat11_dir / "obj_train_data"
            cvat11_data.mkdir(parents=True, exist_ok=True)

            train_lines = []
            for image_path in images_dir.glob("*"):
                if not image_path.is_file():
                    continue
                dst_img = cvat11_data / image_path.name
                shutil.copy2(image_path, dst_img)

                rows = parsed_labels.get(image_path.stem, [])
                (cvat11_data / f"{image_path.stem}.txt").write_text(
                    "\n".join(remap_rows(rows)),
                    encoding="utf-8",
                )
                train_lines.append(f"obj_train_data/{image_path.name}")

            (cvat11_dir / "train.txt").write_text("\n".join(train_lines), encoding="utf-8")
            (cvat11_dir / "obj.names").write_text(
                "\n".join(name_map[old_id] for old_id in sorted_ids),
                encoding="utf-8",
            )
            (cvat11_dir / "obj.data").write_text(
                "\n".join(
                    [
                        f"classes = {len(sorted_ids)}",
                        "train = train.txt",
                        "names = obj.names",
                        "backup = backup/",
                    ]
                ),
                encoding="utf-8",
            )
            (cvat11_dir / "label_id_map.json").write_text(
                json.dumps({"original_to_cvat": id_map, "class_names": name_map}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            cvat11_zip = output_root.parent / f"{job_id}_cvat_yolo11.zip"
            if cvat11_zip.exists():
                cvat11_zip.unlink()
            shutil.make_archive(str(cvat11_zip.with_suffix("")), "zip", root_dir=cvat11_dir)
            cvat11_artifacts = {
                "cvat_yolo11_dir": f"/data/outputs/{job_id}/cvat_yolo11",
                "cvat_yolo11_zip": f"/data/outputs/{job_id}_cvat_yolo11.zip",
                "label_id_map": f"/data/outputs/{job_id}/cvat_yolo11/label_id_map.json",
            }

        # CVAT Ultralytics YOLO
        ultra_dir = output_root / "cvat_ultralytics"
        ultra_images = ultra_dir / "images"
        ultra_labels = ultra_dir / "labels"
        ultra_images.mkdir(parents=True, exist_ok=True)
        ultra_labels.mkdir(parents=True, exist_ok=True)

        for image_path in images_dir.glob("*"):
            if not image_path.is_file():
                continue
            shutil.copy2(image_path, ultra_images / image_path.name)
            rows = parsed_labels.get(image_path.stem, [])
            (ultra_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(remap_rows(rows)),
                encoding="utf-8",
            )

        yaml_data = {
            "path": ".",
            "train": "images",
            "val": "images",
            "names": {id_map[old_id]: name_map[old_id] for old_id in sorted_ids},
        }
        if effective_task != "detect":
            yaml_data["task"] = effective_task
        (ultra_dir / "data.yaml").write_text(yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (ultra_dir / "label_id_map.json").write_text(
            json.dumps({"original_to_cvat": id_map, "class_names": name_map}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        ultra_zip = output_root.parent / f"{job_id}_cvat_ultralytics.zip"
        if ultra_zip.exists():
            ultra_zip.unlink()
        shutil.make_archive(str(ultra_zip.with_suffix("")), "zip", root_dir=ultra_dir)

        return {
            "cvat_ultralytics_dir": f"/data/outputs/{job_id}/cvat_ultralytics",
            "cvat_ultralytics_zip": f"/data/outputs/{job_id}_cvat_ultralytics.zip",
            **cvat11_artifacts,
        }

    # 将检测结果绘制成叠加图并保存到指定路径
    @staticmethod
    def _draw_overlay(src_path: Path, out_path: Path, detections: List[Detection]) -> None:
        overlay_items = [
            {
                "x1": float(getattr(det, "x1", 0.0)),
                "y1": float(getattr(det, "y1", 0.0)),
                "x2": float(getattr(det, "x2", 0.0)),
                "y2": float(getattr(det, "y2", 0.0)),
                "points": AnnotationService._det_points_or_bbox(det),
                "shape_type": normalize_label_task(getattr(det, "shape_type", "detect")),
                "label": f"{getattr(det, 'cls_name', 'object')} {float(getattr(det, 'conf', 0.0)):.2f}",
                "color_key": f"class_{int(getattr(det, 'cls_id', 0))}",
            }
            for det in detections
        ]
        draw_overlay(src_path, out_path, overlay_items)

    # 主入口：运行 YOLO 自动打标任务，将结果写回图片集，生成叠加图和 CVAT 导出包
    @staticmethod
    def run_annotate_job(
        state: AppState,
        job_ctx: JobContext,
        model_id: str,
        imageset_id: str,
        selected_class_ids: list[int],
        target_classes: list[str] | None = None,
        conf: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        device: str = DEFAULT_DEVICE,
        save_overlays: bool = True,
        label_mode: str = "append",
        update_imageset_labels: bool = True,
        round_tag: str = "",
        class_id_overrides: dict[str, int] | None = None,
        operator: str = "anonymous",
        label_task: str = "",
    ) -> dict:
        model = state.get_model(model_id)
        if not model:
            raise ValueError("model_id 不存在")

        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise ValueError("imageset_id 不存在")

        if not imageset.images:
            raise ValueError("图片集为空")

        predictor = create_predictor(model.model_path)
        resolved_device = resolve_device(device)
        effective_label_task = normalize_label_task(label_task or getattr(model, "task", "") or getattr(predictor, "task", "detect"))
        model_task = normalize_label_task(getattr(model, "task", "") or getattr(predictor, "task", "detect"))
        if label_task and normalize_label_task(label_task) != model_task:
            raise ValueError(f"所选模型任务为 {model_task}，不能按 {normalize_label_task(label_task)} 打标")

        selected_ids = sorted(set(int(x) for x in selected_class_ids))
        if not selected_ids:
            raise ValueError("请先完成类别映射，并至少选择 1 个来源类别")

        normalized_target_classes = [str(x).strip() for x in (target_classes or []) if str(x).strip()]
        if target_classes is not None and target_classes and not normalized_target_classes:
            raise ValueError("target_classes 为空，请先加载有效的新场景类别")

        id_overrides: dict[int, int] = {}
        for raw_src, raw_dst in (class_id_overrides or {}).items():
            try:
                src = int(raw_src)
                dst = int(raw_dst)
            except Exception:  # noqa: BLE001
                continue
            if src < 0 or dst < 0:
                continue
            id_overrides[src] = dst

        imageset_labels_dir = Path(imageset.dir_path) / "labels"
        effective_label_task = AnnotationService._ensure_imageset_label_task(imageset, effective_label_task, label_mode, state)
        existing_used_class_name_map = read_used_imageset_class_names(Path(imageset.dir_path))

        # 统一调用 resolve_mapping — single source of truth
        # 只作为新分配 ID 的名字查找表，不做越界校验
        # override 可以自由映射到任意 ID，名字通过下面的兜底链查找
        candidate_class_name_map = AnnotationService.resolve_mapping(
            state=state,
            model_id=model_id,
            imageset_id=imageset_id,
            selected_class_ids=selected_class_ids,
            target_classes=target_classes,
            label_mode=label_mode,
        )

        # override 反查表：dst_id → src_id，用于 override 到 candidate 外 ID 时
        # 从 model.classes[src] 取兜底名字（保持 src 语义）
        reverse_overrides: dict[int, int] = {dst: src for src, dst in id_overrides.items()}

        # 本轮输出 ID 集合：对每个 selected src，写入的最终 dst
        # 仅在 append 模式下，这些 dst 若已在 existing 中需保留原名（归并语义）
        current_output_ids: set[int] = {
            int(id_overrides.get(s, s)) for s in selected_ids
        }

        if label_mode not in {"replace", "append"}:
            raise ValueError("label_mode 仅支持 replace 或 append")

        output_root = OUTPUTS_DIR / job_ctx.job_id
        images_dir = output_root / "images"
        labels_dir = output_root / "labels"
        overlays_dir = output_root / "overlays"
        labels_before_dir = output_root / "labels_before"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir.mkdir(parents=True, exist_ok=True)
        labels_before_dir.mkdir(parents=True, exist_ok=True)
        imageset_labels_dir.mkdir(parents=True, exist_ok=True)
        labels_before_index: dict[str, bool] = {}

        manifest_path = output_root / "manifest.csv"
        headers = [
            "image_id",
            "source_image",
            "output_image",
            "label_file",
            "existing_boxes",
            "new_boxes",
            "final_boxes",
            "label_mode",
            "status",
            "error",
        ]

        total = len(imageset.images)
        total_existing_boxes = 0
        total_new_boxes = 0
        total_final_boxes = 0
        used_final_class_name_map: dict[int, str] = {}

        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for idx, image in enumerate(imageset.images, start=1):
                job_ctx.raise_if_cancelled()
                source_path = Path(imageset.dir_path) / image.rel_path
                output_image = images_dir / image.filename
                output_label = labels_dir / f"{Path(image.filename).stem}.txt"
                output_overlay = overlays_dir / image.filename
                image_label_path = imageset_labels_dir / f"{Path(image.filename).stem}.txt"

                try:
                    shutil.copy2(source_path, output_image)
                    dets = predictor.predict(str(source_path), conf=conf, iou=iou, device=resolved_device)
                    filtered = [d for d in dets if d.cls_id in selected_ids]

                    img = cv2.imread(str(source_path))
                    if img is None:
                        raise RuntimeError(f"无法读取图片: {source_path}")
                    h, w = img.shape[:2]

                    yolo_lines: list[str] = []
                    for det in filtered:
                        dst_id = id_overrides.get(det.cls_id, det.cls_id)
                        if effective_label_task == "detect":
                            shape = points_to_shape(
                                dst_id,
                                [
                                    (float(getattr(det, "x1", 0.0)), float(getattr(det, "y1", 0.0))),
                                    (float(getattr(det, "x2", 0.0)), float(getattr(det, "y2", 0.0))),
                                ],
                                w,
                                h,
                                "detect",
                            )
                        else:
                            shape = points_to_shape(dst_id, AnnotationService._det_points_or_bbox(det), w, h, effective_label_task)
                        yolo_lines.append(shape.format_yolo())
                    if image.filename not in labels_before_index:
                        labels_before_index[image.filename] = image_label_path.exists()
                        if image_label_path.exists():
                            shutil.copy2(image_label_path, labels_before_dir / f"{Path(image.filename).stem}.txt")
                    existing_lines = AnnotationService._read_label_lines(image_label_path)

                    if label_mode == "append":
                        final_lines = AnnotationService._merge_label_lines(existing_lines, yolo_lines)
                    else:
                        final_lines = yolo_lines
                    final_lines = AnnotationService._sort_label_lines(final_lines, effective_label_task)

                    for line in final_lines:
                        parsed_shape = AnnotationService._parse_label_line(line, effective_label_task)
                        if parsed_shape is None:
                            continue
                        class_id = parsed_shape.class_id
                        # 查名字优先级（按 class_id 来源区分）：
                        #   1. 纯 existing 保留（不在本轮写入集合）→ existing 名字，不查 model
                        #   2. append 归并到已有 ID → existing 名字（合并语义，避免污染）
                        #   3. 本轮新写入 → candidate > override src 兜底 > fallback
                        existing_name = existing_used_class_name_map.get(class_id)
                        if class_id not in current_output_ids:
                            # 1: 纯 existing 保留，仅用 existing 名字（不借用 model/candidate）
                            name = existing_name or f"class_{class_id}"
                        elif label_mode == "append" and existing_name:
                            # 2: append 归并到已有 ID
                            name = existing_name
                        else:
                            # 3: 本轮新写入（replace / append 到全新 ID）
                            src_for_dst = reverse_overrides.get(class_id, class_id)
                            src_fallback_name = (
                                model.classes[src_for_dst]
                                if 0 <= src_for_dst < len(model.classes)
                                else ""
                            )
                            name = (
                                candidate_class_name_map.get(class_id)
                                or src_fallback_name
                                or f"class_{class_id}"
                            )
                        used_final_class_name_map[class_id] = name

                    output_label.write_text("\n".join(final_lines), encoding="utf-8")
                    if update_imageset_labels:
                        image_label_path.write_text("\n".join(final_lines), encoding="utf-8")

                    if save_overlays:
                        AnnotationService._draw_overlay(source_path, output_overlay, filtered)

                    writer.writerow(
                        {
                            "image_id": image.id,
                            "source_image": str(source_path),
                            "output_image": str(output_image),
                            "label_file": str(output_label),
                            "existing_boxes": len(existing_lines),
                            "new_boxes": len(yolo_lines),
                            "final_boxes": len(final_lines),
                            "label_mode": label_mode,
                            "status": "ok",
                            "error": "",
                        }
                    )
                    total_existing_boxes += len(existing_lines)
                    total_new_boxes += len(yolo_lines)
                    total_final_boxes += len(final_lines)
                except Exception as exc:  # noqa: BLE001
                    output_label.write_text("", encoding="utf-8")
                    writer.writerow(
                        {
                            "image_id": image.id,
                            "source_image": str(source_path),
                            "output_image": str(output_image),
                            "label_file": str(output_label),
                            "existing_boxes": 0,
                            "new_boxes": 0,
                            "final_boxes": 0,
                            "label_mode": label_mode,
                            "status": "error",
                            "error": str(exc),
                        }
                    )

                job_ctx.set_progress(idx, total, f"打标中: {idx}/{total}")

        labels_before_index_path = output_root / "labels_before_index.json"
        labels_before_index_path.write_text(
            json.dumps(
                {
                    "imageset_id": imageset_id,
                    "job_id": job_ctx.job_id,
                    "files": labels_before_index,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        zip_path = OUTPUTS_DIR / f"{job_ctx.job_id}.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=output_root)
        cvat_artifacts = AnnotationService._build_cvat_exports(
            output_root=output_root,
            images_dir=images_dir,
            labels_dir=labels_dir,
            model_classes=normalized_target_classes or model.classes,
            selected_ids=selected_ids,
            job_id=job_ctx.job_id,
            final_class_name_map=used_final_class_name_map,
            label_task=effective_label_task,
        )

        # Rebuild classes.txt + data.yaml from actual label files
        rebuilt_class_names = rebuild_classes_txt(
            imageset_labels_dir=imageset_labels_dir,
            imageset_dir=Path(imageset.dir_path),
            new_class_names=used_final_class_name_map,
            label_task=effective_label_task,
        )
        used_ids_after_rebuild = sorted(collect_used_class_stats(Path(imageset.dir_path)).keys())
        final_class_name_map = {
            class_id: rebuilt_class_names[class_id]
            for class_id in used_ids_after_rebuild
            if 0 <= class_id < len(rebuilt_class_names)
        }

        run_meta_path = output_root / "run_meta.json"
        run_meta_path.write_text(
            json.dumps(
                {
                    "job_id": job_ctx.job_id,
                    "round_tag": (round_tag or "").strip(),
                    "operator": (operator or "anonymous").strip() or "anonymous",
                    "model_id": model_id,
                    "model_name": model.name,
                    "imageset_id": imageset_id,
                    "selected_class_ids": selected_ids,
                    "mapping_confirmed": True,
                    "label_task": effective_label_task,
                    "label_mode": label_mode,
                    "update_imageset_labels": bool(update_imageset_labels),
                    "save_overlays": bool(save_overlays),
                    "conf": float(conf),
                    "iou": float(iou),
                    "device": str(device),
                    "resolved_device": resolved_device,
                    "class_id_overrides": id_overrides,
                    "final_class_name_map": final_class_name_map,
                    "class_names": rebuilt_class_names,
                    "target_classes": normalized_target_classes,
                    "labels_before_index_json": str(labels_before_index_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "job_id": job_ctx.job_id,
            "round_tag": (round_tag or "").strip(),
            "operator": (operator or "anonymous").strip() or "anonymous",
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": selected_ids,
            "mapping_confirmed": True,
            "label_task": effective_label_task,
            "class_id_overrides": id_overrides,
            "final_class_name_map": final_class_name_map,
            "class_names": rebuilt_class_names,
            "target_classes": normalized_target_classes,
            "label_mode": label_mode,
            "update_imageset_labels": bool(update_imageset_labels),
            "requested_device": str(device),
            "resolved_device": resolved_device,
            "total_images": total,
            "total_existing_boxes": total_existing_boxes,
            "total_new_boxes": total_new_boxes,
            "total_final_boxes": total_final_boxes,
            "total_boxes": total_final_boxes,
            "output_root": str(output_root),
            "artifacts": {
                "manifest_csv": f"/data/outputs/{job_ctx.job_id}/manifest.csv",
                "run_meta_json": f"/data/outputs/{job_ctx.job_id}/run_meta.json",
                "images_dir": f"/data/outputs/{job_ctx.job_id}/images",
                "labels_dir": f"/data/outputs/{job_ctx.job_id}/labels",
                "overlays_dir": f"/data/outputs/{job_ctx.job_id}/overlays",
                "labels_before_dir": f"/data/outputs/{job_ctx.job_id}/labels_before",
                "labels_before_index_json": f"/data/outputs/{job_ctx.job_id}/labels_before_index.json",
                "zip_file": f"/data/outputs/{job_ctx.job_id}.zip",
                "preview_page_url": f"/front/preview.html?job_id={job_ctx.job_id}",
                "rollback_api": f"/api/annotate/jobs/{job_ctx.job_id}/rollback",
                "cleanup_api": f"/api/jobs/{job_ctx.job_id}/artifacts",
                "can_rollback": bool(update_imageset_labels),
                **cvat_artifacts,
            },
        }
