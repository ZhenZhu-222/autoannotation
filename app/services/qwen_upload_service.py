# ============================================================
# Qwen 参考图上传预处理服务
# 职责：图片归一化（统一转 JPEG）/ 参考图暂存 / 参数校验
# ============================================================
from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import MAX_IMAGE_UPLOAD_BYTES, QWEN_REFS_DIR
from app.core.utils import as_bool, ensure_unique_path, new_id, read_upload_with_limit, sanitize_filename
from app.services.label_format import normalize_label_task


class QwenUploadService:

    @staticmethod
    def normalize_ref_image(filename: str, content: bytes) -> tuple[str, bytes]:
        """将参考图片归一化为 JPEG 格式。无效图片抛 ValueError。"""
        if not content:
            raise ValueError("empty-image")
        safe_name = sanitize_filename(filename or "ref.jpg", "ref.jpg")
        stem = Path(safe_name).stem or "ref"

        arr = np.frombuffer(content, dtype=np.uint8)
        if arr.size > 0:
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    return f"{stem}.jpg", encoded.tobytes()

        try:
            with Image.open(BytesIO(content)) as pil:
                rgb = pil.convert("RGB")
                out = BytesIO()
                rgb.save(out, format="JPEG", quality=95)
                return f"{stem}.jpg", out.getvalue()
        except (UnidentifiedImageError, OSError, ValueError):
            raise ValueError("invalid-image") from None

    @staticmethod
    async def stage_ref_images(all_files: list) -> tuple[Path, list[str], list[str]]:
        """将上传的参考图暂存到 QWEN_REFS_DIR 下，返回 (upload_dir, ref_paths, invalid_refs)。"""
        upload_dir = QWEN_REFS_DIR / new_id("qref")
        upload_dir.mkdir(parents=True, exist_ok=True)
        ref_paths: list[str] = []
        invalid_refs: list[str] = []
        for ref in all_files:
            try:
                content = await read_upload_with_limit(ref, MAX_IMAGE_UPLOAD_BYTES)
            except ValueError:
                invalid_refs.append(ref.filename or "unnamed")
                continue
            if not content:
                invalid_refs.append(ref.filename or "unnamed")
                continue
            try:
                filename, normalized = QwenUploadService.normalize_ref_image(ref.filename or "ref.jpg", content)
            except ValueError:
                invalid_refs.append(ref.filename or "unnamed")
                continue
            out_path = ensure_unique_path(upload_dir / filename)
            out_path.write_bytes(normalized)
            ref_paths.append(str(out_path))
        return upload_dir, ref_paths, invalid_refs

    @staticmethod
    def validate_and_build_params(
        label_mode,
        update_imageset_labels,
        save_overlays,
        strict_size_check,
        precision_mode,
        sample_count,
        max_calibration_error_px,
        min_consensus_rate,
        duplicate_iou,
        reject_on_invalid_svg,
        green_hsv_profile,
        size_retry,
        label_task,
    ):
        """校验并规范化 Qwen 打标参数。参数非法抛 ValueError。"""
        if label_mode not in {"append", "replace"}:
            raise ValueError("label_mode 仅支持 append 或 replace")
        run_precision_mode = (precision_mode or "strict").strip().lower()
        if run_precision_mode not in {"strict", "fast"}:
            raise ValueError("precision_mode 仅支持 strict 或 fast")
        return {
            "update_imageset_labels": as_bool(update_imageset_labels, True),
            "save_overlays": as_bool(save_overlays, True),
            "strict_size_check": as_bool(strict_size_check, True),
            "precision_mode": run_precision_mode,
            "sample_count": max(1, min(int(sample_count or 1), 7)),
            "max_calibration_error_px": max(0.0, float(max_calibration_error_px or 0.0)),
            "min_consensus_rate": min(max(float(min_consensus_rate or 0.67), 0.0), 1.0),
            "duplicate_iou": min(max(float(duplicate_iou or 0.98), 0.5), 0.9999),
            "reject_on_invalid_svg": as_bool(reject_on_invalid_svg, True),
            "green_hsv_profile": (green_hsv_profile or "neon").strip().lower() or "neon",
            "size_retry": max(0, int(size_retry or 0)),
            "label_task": normalize_label_task(label_task),
        }
