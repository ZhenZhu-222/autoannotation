# ============================================================
# 类别 ID 重映射服务
# 职责：校验映射规则 / 重写标签文件 / 更新 classes.txt
# ============================================================
from __future__ import annotations

from pathlib import Path

from app.core.state import AppState
from app.services.annotation_service import rebuild_classes_txt
from app.services.class_name_service import collect_used_class_stats, resolve_imageset_class_names
from app.services.label_format import sort_label_lines


class RemapService:

    @staticmethod
    def get_class_mapping(imageset_id: str, state: AppState) -> dict:
        """返回图片集当前的 class_id → name 映射及使用统计。"""
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise KeyError("imageset 不存在")
        imageset_dir = Path(imageset.dir_path)
        class_names = resolve_imageset_class_names(
            imageset_id=imageset_id, imageset_dir=imageset_dir, state=state,
        )
        usage = collect_used_class_stats(imageset_dir)
        if not usage:
            return {"imageset_id": imageset_id, "entries": []}

        entries = []
        for cid in sorted(usage.keys()):
            entries.append({
                "class_id": cid,
                "name": class_names.get(str(cid), f"class_{cid}"),
                "count": usage[cid],
            })
        return {"imageset_id": imageset_id, "entries": entries}

    @staticmethod
    def remap_classes(imageset_id: str, id_mapping: dict[str, int], state: AppState) -> dict:
        """重映射类别 ID：校验 → 重写标签 → 更新 classes.txt。
        参数错误抛 ValueError，imageset 不存在抛 KeyError。
        """
        imageset = state.get_imageset(imageset_id)
        if not imageset:
            raise KeyError("imageset 不存在")
        imageset_dir = Path(imageset.dir_path)
        labels_dir = imageset_dir / "labels"
        if not labels_dir.exists():
            raise ValueError("labels 目录不存在")

        # ---------- 解析并校验映射 ----------
        mapping: dict[int, int] = {}
        for old_str, new_id in id_mapping.items():
            try:
                old_id = int(old_str)
            except (TypeError, ValueError):
                raise ValueError(f"无效的旧 ID: {old_str}") from None
            if new_id < 0:
                raise ValueError(f"新 ID 不能为负数: {new_id}")
            mapping[old_id] = new_id

        if not mapping:
            raise ValueError("映射为空")

        # ---------- 读取当前类别信息 ----------
        class_names = resolve_imageset_class_names(
            imageset_id=imageset_id, imageset_dir=imageset_dir, state=state,
        )
        usage = collect_used_class_stats(imageset_dir)
        used_class_names = {
            class_id: class_names.get(str(class_id), f"class_{class_id}")
            for class_id in sorted(usage.keys())
        }

        # ---------- 冲突检测 ----------
        target_to_names: dict[int, set[str]] = {}
        for old_id, new_id in mapping.items():
            if old_id not in used_class_names:
                continue
            name = used_class_names[old_id]
            target_to_names.setdefault(new_id, set()).add(name)
        for new_id, names in target_to_names.items():
            if len(names) > 1:
                raise ValueError(
                    f"目标 ID {new_id} 对应多个不同类名: {', '.join(sorted(names))}，只有同名类别才能合并"
                )

        # ---------- 构建最终类别名映射 ----------
        final_used_class_names: dict[int, str] = {}
        for old_id, new_id in mapping.items():
            if old_id not in used_class_names:
                continue
            final_used_class_names[new_id] = used_class_names[old_id]
        for cid, name in used_class_names.items():
            if cid not in mapping:
                if cid in final_used_class_names and final_used_class_names[cid] != name:
                    raise ValueError(
                        f"ID {cid} 未参与映射但与映射目标冲突（原名 {name} vs 映射名 {final_used_class_names[cid]}）"
                    )
                final_used_class_names[cid] = name

        # ---------- 重写标签文件 ----------
        files_modified = 0
        lines_modified = 0
        for lf in sorted(labels_dir.glob("*.txt")):
            if lf.name == "classes.txt":
                continue
            try:
                original = lf.read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                continue
            new_lines = []
            changed = False
            for line in original.splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(float(parts[0]))
                    except (TypeError, ValueError):
                        new_lines.append(line)
                        continue
                    if cid in mapping and mapping[cid] != cid:
                        parts[0] = str(mapping[cid])
                        new_lines.append(" ".join(parts))
                        changed = True
                        lines_modified += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            if changed:
                new_lines = sort_label_lines(new_lines, getattr(imageset, "label_task", "detect"))
                lf.write_text("\n".join(new_lines), encoding="utf-8")
                files_modified += 1

        rebuild_classes_txt(
            imageset_labels_dir=labels_dir,
            imageset_dir=imageset_dir,
            new_class_names=final_used_class_names,
        )

        return {
            "imageset_id": imageset_id,
            "files_modified": files_modified,
            "lines_modified": lines_modified,
            "new_class_count": len(final_used_class_names),
        }
