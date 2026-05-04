# ============================================================
# 数据集导入服务
# 职责：解析上传目录中的图片 / 标签 / 类别名配置，并写入平台图片集目录
# ============================================================
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable

import yaml

from app.core.config import SUPPORTED_IMAGE_EXTENSIONS
from app.core.utils import sanitize_filename
from app.services.label_format import infer_label_task_from_dir, infer_label_task_from_lines, normalize_label_task


CLASS_NAME_FILENAMES = {"classes.txt", "names.txt", "obj.names"}
DATASET_CONFIG_FILENAMES = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}
JSON_CLASS_METADATA_FILENAMES = {"classes.json", "names.json", "class_names.json", "data.json", "dataset.json"}
PATH_IMAGE_DIRS = {"images", "image", "imgs", "img", "jpegimages"}
PATH_LABEL_DIRS = {"labels", "label", "labeltxt", "annotations"}


@dataclass
class PreparedDatasetUpload:
    image_files: list[Path] = field(default_factory=list)
    label_payload: dict[str, str] = field(default_factory=dict)
    image_match_keys: dict[str, str] = field(default_factory=dict)
    duplicate_label_stems: set[str] = field(default_factory=set)
    class_names: dict[int, str] = field(default_factory=dict)
    class_name_sources: list[str] = field(default_factory=list)
    label_task: str = ""


@dataclass
class AppliedDatasetUpload:
    labels_imported: int
    labels_unmatched: list[str]
    duplicate_label_name_count: int
    class_names_imported: int
    class_name_sources: list[str]
    label_task: str


# 将文件名转为统一正斜杠路径的各级分量
def _posix_parts(filename: str) -> list[str]:
    normalized = str(filename or "").replace("\\", "/").strip("/")
    return [part for part in PurePosixPath(normalized).parts if part and part not in {".", ".."}]


# 返回文件名的小写基名（去掉目录层级）
def basename_lower(filename: str) -> str:
    parts = _posix_parts(filename)
    return (parts[-1] if parts else filename).lower()


# 生成图片或标签文件的匹配键（剔除 images/labels 目录前缀，小写 stem）
def dataset_match_key(filename: str, kind: str) -> str:
    parts = _posix_parts(filename)
    if not parts:
        return Path(filename or "").stem.casefold()

    markers = PATH_IMAGE_DIRS if kind == "image" else PATH_LABEL_DIRS
    start_idx = 0
    for idx, part in enumerate(parts[:-1]):
        if part.lower() in markers:
            start_idx = idx + 1
            break

    rel_parts = list(parts[start_idx:])
    rel_parts[-1] = PurePosixPath(rel_parts[-1]).stem
    return PurePosixPath(*rel_parts).as_posix().casefold()


# 将 list 或 dict 格式的类别数据统一转为 {id: name} 字典
def coerce_name_map(payload) -> dict[int, str]:
    if isinstance(payload, list):
        return {
            idx: str(raw_name or "").strip()
            for idx, raw_name in enumerate(payload)
            if str(raw_name or "").strip()
        }
    if isinstance(payload, dict):
        out: dict[int, str] = {}
        for raw_key, raw_name in payload.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                continue
            name = str(raw_name or "").strip()
            if name:
                out[key] = name
        return out
    return {}


# 判断类别名是否是占位符（class_N 格式）
def is_placeholder_class_name(class_id: int, name: str) -> bool:
    return str(name or "").strip().lower() == f"class_{int(class_id)}"


# 将新类别名合并入目标字典，高优先级及真实名优先覆盖占位符
def merge_class_names(
    target: dict[int, str],
    target_priority: dict[int, int],
    incoming: dict[int, str],
    priority: int,
) -> None:
    for class_id, raw_name in incoming.items():
        if class_id < 0:
            continue
        name = str(raw_name or "").strip()
        if not name:
            continue
        existing = target.get(class_id, "")
        existing_priority = target_priority.get(class_id, -1)
        if (
            not existing
            or priority > existing_priority
            or (is_placeholder_class_name(class_id, existing) and not is_placeholder_class_name(class_id, name))
        ):
            target[class_id] = name
            target_priority[class_id] = priority


# 解析类别元数据文件（txt/yaml/json），返回 {id: name} 字典和任务类型
def parse_class_metadata(filename: str, text: str) -> tuple[dict[int, str], str]:
    suffix = Path(basename_lower(filename)).suffix
    basename = basename_lower(filename)
    if basename in CLASS_NAME_FILENAMES:
        names = coerce_name_map(text.splitlines())
        if not names:
            raise ValueError(f"类别名文件为空: {filename}")
        return names, ""

    if suffix not in {".yaml", ".yml", ".json"}:
        return {}, ""

    try:
        payload = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"类别配置解析失败: {filename}") from exc

    if isinstance(payload, dict):
        names = coerce_name_map(payload.get("names", payload))
        label_task = normalize_label_task(str(payload.get("task") or ""), default="detect") if payload.get("task") else ""
    else:
        names = coerce_name_map(payload)
        label_task = ""

    if not names:
        raise ValueError(f"类别配置未包含有效 names: {filename}")
    return names, label_task


# 判断文件是否为类别元数据文件（classes.txt/data.yaml/names.json 等）
def is_class_metadata_file(filename: str) -> bool:
    basename = basename_lower(filename)
    if basename in CLASS_NAME_FILENAMES or basename in DATASET_CONFIG_FILENAMES:
        return True
    if basename in JSON_CLASS_METADATA_FILENAMES:
        return True
    return False


# 将类别名写入 classes.txt 和 data.yaml
def write_class_metadata(imageset_dir: Path, class_names: dict[int, str], label_task: str) -> None:
    if not class_names:
        return
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    max_class_id = max(class_names.keys())
    dense_names = {
        class_id: class_names.get(class_id, f"class_{class_id}")
        for class_id in range(max_class_id + 1)
    }
    (labels_dir / "classes.txt").write_text(
        "\n".join(dense_names[class_id] for class_id in range(max_class_id + 1)),
        encoding="utf-8",
    )
    data_yaml = {"path": ".", "train": "images", "val": "images", "names": dense_names}
    effective_task = normalize_label_task(label_task, "detect")
    if effective_task != "detect":
        data_yaml["task"] = effective_task
    (imageset_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


class DatasetImportService:
    # 预处理上传文件列表：分类图片、标签和类别元数据，写入临时目录并返回结构化结果
    @staticmethod
    def prepare_upload(files: Iterable[tuple[str, Path]], temp_dir: Path) -> PreparedDatasetUpload:
        prepared = PreparedDatasetUpload()
        class_name_priority: dict[int, int] = {}
        seq = 0

        for filename, staged_path in files:
            suffix = Path(basename_lower(filename)).suffix
            clean = sanitize_filename(filename, "label.txt" if suffix == ".txt" else "image.jpg")
            if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
                if is_class_metadata_file(filename):
                    text = staged_path.read_text(encoding="utf-8", errors="ignore")
                    incoming_names, incoming_task = parse_class_metadata(filename, text)
                    priority = 2 if suffix in {".yaml", ".yml", ".json"} else 1
                    merge_class_names(prepared.class_names, class_name_priority, incoming_names, priority)
                    if incoming_names:
                        prepared.class_name_sources.append(str(filename))
                    if incoming_task:
                        prepared.label_task = incoming_task
                elif suffix == ".txt":
                    stem = dataset_match_key(filename, "label")
                    if stem in prepared.label_payload:
                        prepared.duplicate_label_stems.add(stem)
                    prepared.label_payload[stem] = staged_path.read_text(encoding="utf-8", errors="ignore")
                continue

            out_path = temp_dir / f"{seq:06d}_{clean}"
            out_path.write_bytes(staged_path.read_bytes())
            prepared.image_files.append(out_path)
            prepared.image_match_keys[out_path.stem] = dataset_match_key(filename, "image")
            seq += 1

        return prepared

    # 将预处理的数据应用到图片集：写入标签文件、更新类别名和任务类型
    @staticmethod
    def apply_to_imageset(imageset, prepared: PreparedDatasetUpload) -> AppliedDatasetUpload:
        if prepared.label_task:
            imageset.label_task = prepared.label_task
        has_explicit_label_task = bool(prepared.label_task)

        imported_labels = 0
        matched_stems: set[str] = set()
        imageset_dir = Path(imageset.dir_path)
        labels_dir = imageset_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image in imageset.images:
            stem = Path(image.filename).stem
            label_stem = prepared.image_match_keys.get(stem, stem)
            text = prepared.label_payload.get(label_stem)
            if text is None:
                continue
            (labels_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
            imported_labels += 1
            matched_stems.add(label_stem)
            if not has_explicit_label_task:
                imageset.label_task = infer_label_task_from_lines(text.splitlines(), default=imageset.label_task)

        if imported_labels and not has_explicit_label_task:
            imageset.label_task = infer_label_task_from_dir(labels_dir, default=imageset.label_task)

        if prepared.class_names:
            write_class_metadata(imageset_dir, prepared.class_names, imageset.label_task)

        unmatched = sorted(set(prepared.label_payload.keys()) - matched_stems)
        return AppliedDatasetUpload(
            labels_imported=imported_labels,
            labels_unmatched=unmatched,
            duplicate_label_name_count=len(prepared.duplicate_label_stems),
            class_names_imported=len(prepared.class_names),
            class_name_sources=prepared.class_name_sources[:20],
            label_task=imageset.label_task,
        )
