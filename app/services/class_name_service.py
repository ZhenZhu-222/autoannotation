# ============================================================
# 类别名称解析服务
# 职责：从 classes.txt / data.yaml / 历史任务中汇总图片集的类别 ID → 名称映射
# 思路：多源合并，真实名称优先于占位符 class_N
# ============================================================
from __future__ import annotations

import json
from pathlib import Path

import yaml

from app.core.config import OUTPUTS_DIR
from app.core.state import AppState
from app.services.task_manager import TaskManager


# ---------- 工具函数 ----------
# 将 list 或 dict 格式的类别数据统一转换为 {id: name} 字典
def _coerce_name_map(payload) -> dict[int, str]:
    if isinstance(payload, list):
        out: dict[int, str] = {}
        for idx, raw_name in enumerate(payload):
            name = str(raw_name or "").strip()
            if name:
                out[idx] = name
        return out
    if isinstance(payload, dict):
        out = {}
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


# 判断类别名是否是占位符（如 class_0），用于判断是否需要用更好的名称替换
def _is_placeholder_name(class_id: int, name: str) -> bool:
    return str(name or "").strip().lower() == f"class_{int(class_id)}"


# 将 source 合并入 target，真实名称优先覆盖占位符名
def _merge_name_map(target: dict[int, str], source: dict[int, str]) -> None:
    for class_id, name in source.items():
        clean = str(name or "").strip()
        if not clean:
            continue
        existing = target.get(class_id, "")
        if not existing:
            target[class_id] = clean
            continue
        if _is_placeholder_name(class_id, existing) and not _is_placeholder_name(class_id, clean):
            target[class_id] = clean


# 读取图片集目录中的 classes.txt，返回 {id: name} 字典
def _read_classes_txt_map(imageset_dir: Path) -> dict[int, str]:
    for candidate in [imageset_dir / "labels" / "classes.txt", imageset_dir / "classes.txt"]:
        if not candidate.exists():
            continue
        try:
            return _coerce_name_map(candidate.read_text(encoding="utf-8").splitlines())
        except Exception:  # noqa: BLE001
            continue
    return {}


# 读取图片集目录中的 data.yaml，返回 {id: name} 字典
def _read_data_yaml_map(imageset_dir: Path) -> dict[int, str]:
    candidate = imageset_dir / "data.yaml"
    if not candidate.exists():
        return {}
    try:
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        return {}
    names = payload.get("names", {})
    return _coerce_name_map(names)


# ---------- 扫描图片集实际使用的类别 ----------
# 扫描图片集所有标签文件，统计每个 class_id 出现次数，返回 {id: count}
def collect_used_class_stats(imageset_dir: Path) -> dict[int, int]:
    usage: dict[int, int] = {}
    labels_dir = imageset_dir / "labels"
    if not labels_dir.exists():
        return usage
    for label_file in labels_dir.glob("*.txt"):
        if label_file.name == "classes.txt":
            continue
        try:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    usage[class_id] = usage.get(class_id, 0) + 1
        except Exception:  # noqa: BLE001
            continue
    return usage


# 扫描图片集标签文件，返回实际使用的 class_id 集合
def _collect_used_class_ids(imageset_dir: Path) -> set[int]:
    return set(collect_used_class_stats(imageset_dir).keys())


# ---------- 读取图片集本地类别名 ----------
# 从 classes.txt 和 data.yaml 合并读取当前图片集的类别名映射
def read_current_imageset_class_names(imageset_dir: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    _merge_name_map(result, _read_classes_txt_map(imageset_dir))
    _merge_name_map(result, _read_data_yaml_map(imageset_dir))
    return result


# 只返回实际在标签文件中出现过的类别及其名称
def read_used_imageset_class_names(imageset_dir: Path) -> dict[int, str]:
    current = read_current_imageset_class_names(imageset_dir)
    usage = collect_used_class_stats(imageset_dir)
    result: dict[int, str] = {}
    for class_id in sorted(usage.keys()):
        result[class_id] = current.get(class_id, f"class_{class_id}")
    return result


# ---------- 从历史任务中提取类别名 ----------
# 读取指定任务的 run_meta.json（记录了打标时的类别映射等信息）
def _read_run_meta(job_id: str) -> dict:
    if not job_id:
        return {}
    candidate = OUTPUTS_DIR / job_id / "run_meta.json"
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# 读取指定任务的 label_map.json（Qwen 任务写入的 id→name 映射）
def _read_label_map(job_id: str) -> dict[int, str]:
    if not job_id:
        return {}
    candidate = OUTPUTS_DIR / job_id / "label_map.json"
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return _coerce_name_map(payload.get("id_to_name", {}))


# 从 YOLO 打标任务的历史记录+模型中提取类别名映射
def _build_model_history_name_map(summary: dict, job_id: str, state: AppState) -> dict[int, str]:
    result: dict[int, str] = {}
    meta = _read_run_meta(job_id)
    _merge_name_map(result, _coerce_name_map(meta.get("final_class_name_map", {})))
    _merge_name_map(result, _coerce_name_map(summary.get("final_class_name_map", {})))

    model_id = str(summary.get("model_id") or (meta.get("model_id") if meta else "") or "").strip()
    if model_id:
        model = state.get_model(model_id)
        if model and model.classes:
            _merge_name_map(result, _coerce_name_map(model.classes))
    return result


# 从 Qwen 打标任务的历史记录中提取类别名映射
def _build_qwen_history_name_map(summary: dict, job_id: str) -> dict[int, str]:
    result: dict[int, str] = {}
    meta = _read_run_meta(job_id)
    _merge_name_map(result, _read_label_map(job_id))
    _merge_name_map(result, _coerce_name_map(meta.get("final_class_name_map", {})))
    _merge_name_map(result, _coerce_name_map(meta.get("class_names", [])))
    _merge_name_map(result, _coerce_name_map(summary.get("class_names", [])))
    return result


# ---------- 主入口：多源合并解析图片集类别名 ----------
# 主入口：多源合并解析图片集类别名（classes.txt + data.yaml + 历史任务）
def resolve_imageset_class_names(
    imageset_id: str,
    imageset_dir: Path,
    state: AppState,
    tasks: TaskManager | None = None,
) -> dict[str, str]:
    used_ids = _collect_used_class_ids(imageset_dir)
    current_map = read_current_imageset_class_names(imageset_dir)
    if not used_ids:
        return {str(class_id): current_map[class_id] for class_id in sorted(current_map.keys())}

    result: dict[int, str] = {
        class_id: current_map.get(class_id, f"class_{class_id}")
        for class_id in sorted(used_ids)
    }

    history = tasks.list_history(limit=500) if tasks else []
    model_maps: list[dict[int, str]] = []
    qwen_maps: list[dict[int, str]] = []
    for entry in history:
        if entry.get("status") != "succeeded":
            continue
        summary = entry.get("result_summary", {}) or {}
        if summary.get("imageset_id") != imageset_id:
            continue
        job_id = str(entry.get("job_id") or "").strip()
        job_type = str(entry.get("job_type") or "").strip()
        if job_type == "annotate":
            model_maps.append(_build_model_history_name_map(summary, job_id, state))
        elif job_type == "qwen_annotate":
            qwen_maps.append(_build_qwen_history_name_map(summary, job_id))

    for name_map in model_maps:
        _merge_name_map(result, {class_id: name for class_id, name in name_map.items() if class_id in used_ids})
    for name_map in qwen_maps:
        _merge_name_map(result, {class_id: name for class_id, name in name_map.items() if class_id in used_ids})

    for class_id in sorted(used_ids):
        if class_id not in result:
            result[class_id] = f"class_{class_id}"

    return {str(class_id): result[class_id] for class_id in sorted(result.keys())}
