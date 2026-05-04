from __future__ import annotations

from pathlib import Path

import pytest

from app.services.dataset_import_service import DatasetImportService, dataset_match_key, parse_class_metadata


def test_dataset_match_key_uses_relative_path_after_images_and_labels() -> None:
    assert dataset_match_key("dataset/images/a/sample.jpg", "image") == "a/sample"
    assert dataset_match_key("dataset/labels/a/sample.txt", "label") == "a/sample"


def test_parse_class_metadata_supports_classes_txt() -> None:
    names, task = parse_class_metadata("labels/classes.txt", "person\ncar\n")
    assert names == {0: "person", 1: "car"}
    assert task == ""


def test_parse_class_metadata_supports_data_yaml_names_and_task() -> None:
    names, task = parse_class_metadata("data.yaml", "task: segment\nnames:\n  0: person\n  2: road\n")
    assert names == {0: "person", 2: "road"}
    assert task == "segment"


def test_parse_class_metadata_rejects_empty_classes_txt() -> None:
    with pytest.raises(ValueError, match="类别名文件为空"):
        parse_class_metadata("classes.txt", "\n\n")


def test_prepare_upload_keeps_duplicate_basenames_distinct(tmp_path: Path) -> None:
    img_a = tmp_path / "img_a.bin"
    img_b = tmp_path / "img_b.bin"
    label_a = tmp_path / "label_a.bin"
    label_b = tmp_path / "label_b.bin"
    img_a.write_bytes(b"a")
    img_b.write_bytes(b"b")
    label_a.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    label_b.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    prepared = DatasetImportService.prepare_upload(
        [
            ("dataset/images/a/sample.jpg", img_a),
            ("dataset/images/b/sample.jpg", img_b),
            ("dataset/labels/a/sample.txt", label_a),
            ("dataset/labels/b/sample.txt", label_b),
        ],
        tmp_path,
    )

    assert len(prepared.image_files) == 2
    assert set(prepared.image_match_keys.values()) == {"a/sample", "b/sample"}
    assert set(prepared.label_payload.keys()) == {"a/sample", "b/sample"}
    assert not prepared.duplicate_label_stems
