from __future__ import annotations

import csv
import io
import json
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from app.services.label_format import parse_label_line
from app.services.task_manager import TaskManager


def _data_url_path(tmp_path: Path, data_url: str) -> Path:
    assert data_url.startswith("/data/"), data_url
    return tmp_path / "data" / data_url.removeprefix("/data/")


def wait_job(client, path: str, timeout_s: float = 15.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(path)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"Job timeout: {path}")


def _make_video(path: Path, frames: int = 20) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(frames):
        frame = np.full((48, 64, 3), i * 5 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _make_image(path: Path) -> None:
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[:, :] = (120, 30, 200)
    cv2.imwrite(str(path), img)


class _FakePredictor:
    def __init__(self, _model_path: str) -> None:
        self.class_names = ["target"]

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        return [
            type("D", (), {
                "cls_id": 0,
                "cls_name": "target",
                "conf": 0.88,
                "x1": 10.0,
                "y1": 10.0,
                "x2": 50.0,
                "y2": 50.0,
            })()
        ]


class _SlowPredictor:
    def __init__(self, _model_path: str) -> None:
        self.class_names = ["target"]

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        time.sleep(1.0)
        return [
            type("D", (), {
                "cls_id": 0,
                "cls_name": "target",
                "conf": 0.90,
                "x1": 5.0,
                "y1": 5.0,
                "x2": 40.0,
                "y2": 40.0,
            })()
        ]


class _FakePredictorCarBus:
    def __init__(self, _model_path: str) -> None:
        self.class_names = ["car", "bus"]

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        return [
            type("D", (), {
                "cls_id": 0,
                "cls_name": "car",
                "conf": 0.88,
                "x1": 10.0,
                "y1": 10.0,
                "x2": 48.0,
                "y2": 48.0,
            })(),
            type("D", (), {
                "cls_id": 1,
                "cls_name": "bus",
                "conf": 0.86,
                "x1": 55.0,
                "y1": 18.0,
                "x2": 96.0,
                "y2": 60.0,
            })(),
        ]


COCO80_TEST_CLASSES = [f"class_{i}" for i in range(80)]
COCO80_TEST_CLASSES[0] = "person"
COCO80_TEST_CLASSES[1] = "bicycle"
COCO80_TEST_CLASSES[2] = "car"
COCO80_TEST_CLASSES[67] = "cell phone"
COCO80_TEST_CLASSES[68] = "microwave"


class _FakePredictorCellPhone:
    def __init__(self, _model_path: str) -> None:
        self.class_names = COCO80_TEST_CLASSES

    def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
        _ = (conf, iou, device)
        return [
            type("D", (), {
                "cls_id": 67,
                "cls_name": "cell phone",
                "conf": 0.91,
                "x1": 12.0,
                "y1": 14.0,
                "x2": 52.0,
                "y2": 58.0,
            })()
        ]


def test_upload_extract_list_delete(app_client):
    client, _, tmp_path = app_client
    video_path = tmp_path / "demo.mp4"
    _make_video(video_path)

    with video_path.open("rb") as f:
        r = client.post("/api/videos/upload", files={"file": ("demo.mp4", f, "video/mp4")})
    assert r.status_code == 200, r.text
    video_id = r.json()["video_id"]

    r = client.post("/api/extract/jobs", json={"video_id": video_id, "sample_every_seconds": 0.2})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/extract/jobs/{job_id}")
    assert final["status"] == "succeeded"
    imageset_id = final["result"]["imageset_id"]

    r = client.get(f"/api/imagesets/{imageset_id}/images")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) > 0

    r = client.delete(f"/api/images/{items[0]['image_id']}")
    assert r.status_code == 200


def test_onnx_without_classes_rejected(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, _ = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: [])

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("demo.onnx", b"onnx", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "ONNX 未解析到类别" in r.text


def test_upload_model_and_annotate(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img1.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[
                ("files", ("img1.jpg", f.read(), "image/jpeg")),
                ("files", ("img1.txt", b"0 0.100000 0.100000 0.100000 0.100000\n", "text/plain")),
            ],
            data={"imageset_name": "upload_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]
    assert r.json()["labels_imported"] == 1

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    assert images_resp.status_code == 200, images_resp.text
    images_items = images_resp.json()["items"]
    assert len(images_items) == 1
    assert images_items[0]["label_exists"] is True

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("demo.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
            "label_mode": "append",
            "update_imageset_labels": True,
            "round_tag": "round-2",
            "operator": "alice",
            "class_id_overrides": {"0": 5},
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    r = client.get(f"/api/annotate/jobs/{job_id}/artifacts")
    assert r.status_code == 200, r.text
    payload = r.json()
    artifacts = payload["artifacts"]
    summary = payload["summary"]
    manifest = artifacts["manifest_csv"]

    manifest_resp = client.get(manifest)
    assert manifest_resp.status_code == 200
    rows = list(csv.DictReader(io.StringIO(manifest_resp.text)))
    assert len(rows) == 1
    row = rows[0]
    assert row["existing_boxes"] == "1"
    assert row["new_boxes"] == "1"
    assert row["final_boxes"] == "2"
    assert row["label_mode"] == "append"
    label_lines = [x.strip() for x in Path(row["label_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(label_lines) == 2
    assert any(x.startswith("0 ") for x in label_lines)
    assert any(x.startswith("5 ") for x in label_lines)
    assert summary["total_existing_boxes"] == 1
    assert summary["total_new_boxes"] == 1
    assert summary["total_final_boxes"] == 2
    assert summary["round_tag"] == "round-2"
    assert summary["label_mode"] == "append"
    assert summary["update_imageset_labels"] is True
    assert summary["operator"] == "root"
    assert summary["class_id_overrides"] == {"0": 5}

    run_meta_resp = client.get(artifacts["run_meta_json"])
    assert run_meta_resp.status_code == 200
    assert run_meta_resp.json()["round_tag"] == "round-2"

    cvat11_zip_resp = client.get(artifacts["cvat_yolo11_zip"])
    assert cvat11_zip_resp.status_code == 200
    cvat_ultra_zip_resp = client.get(artifacts["cvat_ultralytics_zip"])
    assert cvat_ultra_zip_resp.status_code == 200
    assert artifacts["preview_page_url"].startswith("/front/preview.html?job_id=")

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    preview_payload = preview_resp.json()
    assert preview_payload["job_id"] == job_id
    assert preview_payload["total"] >= 1
    assert preview_payload["items"]
    item0 = preview_payload["items"][0]
    assert "source_url" in item0
    assert "overlay_url" in item0
    assert "label_url" in item0
    assert item0["source_url"].startswith("/data/")
    assert item0["overlay_url"].startswith("/data/")
    assert item0["label_url"].startswith("/data/")
    assert client.get(item0["source_url"]).status_code == 200
    assert client.get(item0["overlay_url"]).status_code == 200

    preview_with_boxes = client.get(f"/api/annotate/jobs/{job_id}/preview?only_with_boxes=true")
    assert preview_with_boxes.status_code == 200


def test_preview_recovers_stale_manifest_paths_after_move_or_remap(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_preview_recover.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_preview_recover.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "preview_recover_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    model_resp = client.post(
        "/api/models/upload",
        files={"model_file": ("preview_recover.pt", b"pt-model", "application/octet-stream")},
    )
    assert model_resp.status_code == 200, model_resp.text
    model_id = model_resp.json()["model_id"]

    create_resp = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    job_id = create_resp.json()["id"]
    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    artifacts = final["result"].get("artifacts", {})
    manifest_path = _data_url_path(tmp_path, artifacts["manifest_csv"])
    rows = list(csv.DictReader(io.StringIO(manifest_path.read_text(encoding="utf-8"))))
    assert rows
    row = rows[0]
    filename = Path(row["source_image"]).name
    label_name = Path(row["label_file"]).name

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    (imageset_dir / "labels" / label_name).write_text("", encoding="utf-8")

    stale_root = Path("/definitely/stale/autoannotation")
    row["source_image"] = str(stale_root / "data" / "imagesets" / imageset_id / "images" / filename)
    row["label_file"] = str(stale_root / "data" / "outputs" / job_id / "labels" / label_name)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    item = preview_resp.json()["items"][0]
    assert item["source_url"].startswith("/data/")
    assert item["label_url"].startswith("/data/")
    assert client.get(item["source_url"]).status_code == 200
    label_resp = client.get(item["label_url"])
    assert label_resp.status_code == 200
    assert label_resp.text.strip()
    assert label_resp.text == _data_url_path(tmp_path, item["label_url"]).read_text(encoding="utf-8")


def test_preview_prefers_current_imageset_labels_when_requested(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_preview_prefer_current.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_preview_prefer_current.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "preview_prefer_current_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    model_resp = client.post(
        "/api/models/upload",
        files={"model_file": ("preview_prefer_current.pt", b"pt-model", "application/octet-stream")},
    )
    assert model_resp.status_code == 200, model_resp.text
    model_id = model_resp.json()["model_id"]

    create_resp = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    job_id = create_resp.json()["id"]
    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    stem = Path(imageset.images[0].filename).stem
    label_path = Path(imageset.dir_path) / "labels" / f"{stem}.txt"
    label_path.write_text("5 0.500000 0.500000 0.300000 0.300000", encoding="utf-8")
    (Path(imageset.dir_path) / "labels" / "classes.txt").write_text("person\nclass_1\nclass_2\nclass_3\nclass_4\n窗户", encoding="utf-8")

    default_preview = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert default_preview.status_code == 200, default_preview.text
    default_item = default_preview.json()["items"][0]
    assert default_item["label_url"].startswith(f"/data/outputs/{job_id}/labels/")

    current_preview = client.get(f"/api/annotate/jobs/{job_id}/preview?prefer_current_labels=true")
    assert current_preview.status_code == 200, current_preview.text
    current_item = current_preview.json()["items"][0]
    assert current_item["label_url"].startswith(f"/data/imagesets/{imageset_id}/labels/")
    current_label = client.get(current_item["label_url"])
    assert current_label.status_code == 200
    assert current_label.text.strip().startswith("5 ")


def test_images_endpoint_class_names_ignore_stale_history_ids(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    image_path = tmp_path / "img_class_names_filter.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_class_names_filter.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "class_names_filter_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]
    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None

    labels_dir = Path(imageset.dir_path) / "labels"
    (labels_dir / "img_class_names_filter.txt").write_text(
        "0 0.200000 0.200000 0.100000 0.100000\n5 0.600000 0.600000 0.150000 0.150000",
        encoding="utf-8",
    )
    (labels_dir / "classes.txt").write_text("person\nclass_1\nclass_2\nclass_3\nclass_4\n窗户", encoding="utf-8")

    monkeypatch.setattr(
        app.state.task_manager,
        "list_history",
        lambda limit=500: [
            {
                "status": "succeeded",
                "job_type": "qwen_annotate",
                "job_id": "job_stale_history",
                "result_summary": {
                    "imageset_id": imageset_id,
                    "class_names": ["person", "bicycle", "car", "motorcycle", "airplane", "窗户"],
                },
            }
        ],
    )

    resp = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=5")
    assert resp.status_code == 200, resp.text
    class_names = resp.json()["class_names"]
    assert class_names == {"0": "person", "5": "窗户"}


def test_annotate_requires_mapping_confirmed(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_need_map.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_need_map.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "need_map_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("need_map.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "conf": 0.25,
            "iou": 0.45,
            "label_mode": "append",
        },
    )
    assert r.status_code == 400
    assert "确认映射" in r.text


def test_many_to_one_mapping_with_target_classes(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["car", "bus"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictorCarBus(p))

    image_path = tmp_path / "img_vehicle.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_vehicle.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "vehicle_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("vehicle_map.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0, 1],
            "mapping_confirmed": True,
            "target_classes": ["vehicle"],
            "class_id_overrides": {"0": 0, "1": 0},
            "save_overlays": True,
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    artifacts = client.get(f"/api/annotate/jobs/{job_id}/artifacts").json()["artifacts"]
    manifest_resp = client.get(artifacts["manifest_csv"])
    rows = list(csv.DictReader(io.StringIO(manifest_resp.text)))
    assert len(rows) == 1
    label_lines = [x.strip() for x in Path(rows[0]["label_file"]).read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(label_lines) == 2
    assert all(line.startswith("0 ") for line in label_lines)

    map_resp = client.get(artifacts["label_id_map"])
    assert map_resp.status_code == 200
    class_names = map_resp.json()["class_names"]
    assert class_names["0"] == "vehicle"


def test_concurrent_extract_and_annotate(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    img = tmp_path / "img2.jpg"
    _make_image(img)
    with img.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img2.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "set_for_concurrent"},
        )
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("demo2.pt", b"pt-model", "application/octet-stream")},
    )
    model_id = r.json()["model_id"]

    v1 = tmp_path / "v1.mp4"
    v2 = tmp_path / "v2.mp4"
    _make_video(v1, 15)
    _make_video(v2, 18)

    with v1.open("rb") as f:
        rv1 = client.post("/api/videos/upload", files={"file": ("v1.mp4", f, "video/mp4")})
    with v2.open("rb") as f:
        rv2 = client.post("/api/videos/upload", files={"file": ("v2.mp4", f, "video/mp4")})

    j1 = client.post("/api/extract/jobs", json={"video_id": rv1.json()["video_id"], "sample_every_seconds": 0.2}).json()["id"]
    j2 = client.post("/api/extract/jobs", json={"video_id": rv2.json()["video_id"], "sample_every_seconds": 0.2}).json()["id"]
    j3 = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    ).json()["id"]

    r1 = wait_job(client, f"/api/extract/jobs/{j1}")
    r2 = wait_job(client, f"/api/extract/jobs/{j2}")
    r3 = wait_job(client, f"/api/annotate/jobs/{j3}")

    assert r1["status"] == "succeeded"
    assert r2["status"] == "succeeded"
    assert r3["status"] == "succeeded"


def test_annotate_rollback_restores_previous_labels(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_rollback.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[
                ("files", ("img_rollback.jpg", f.read(), "image/jpeg")),
                ("files", ("img_rollback.txt", b"0 0.100000 0.100000 0.100000 0.100000\n", "text/plain")),
            ],
            data={"imageset_name": "set_rollback"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("rollback.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
            "label_mode": "append",
            "update_imageset_labels": True,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    filename = imageset.images[0].filename
    label_path = Path(imageset.dir_path) / "labels" / f"{Path(filename).stem}.txt"
    current_lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(current_lines) == 2

    rollback_resp = client.post(f"/api/annotate/jobs/{job_id}/rollback")
    assert rollback_resp.status_code == 200, rollback_resp.text
    rollback_data = rollback_resp.json()
    assert rollback_data["status"] == "rolled_back"
    assert rollback_data["restored_files"] >= 1

    restored_lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert restored_lines == ["0 0.100000 0.100000 0.100000 0.100000"]


def test_delete_job_artifacts_cleanup_endpoint(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_cleanup.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_cleanup.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "set_cleanup"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("cleanup.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    preview_before = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_before.status_code == 200

    cleanup_resp = client.delete(f"/api/jobs/{job_id}/artifacts")
    assert cleanup_resp.status_code == 200, cleanup_resp.text
    cleanup_data = cleanup_resp.json()
    assert cleanup_data["removed_dirs"] >= 1

    preview_after = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_after.status_code == 404


def test_delete_history_entry_endpoint(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_history_delete.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_history_delete.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "set_history_delete"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("history_delete.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    before = client.get("/api/jobs/history?limit=50")
    assert before.status_code == 200
    assert any(item["job_id"] == job_id for item in before.json()["items"])
    assert app.state.app_state.get_imageset(imageset_id) is not None

    delete_resp = client.delete(f"/api/jobs/history/{job_id}?with_artifacts=true&with_imageset=true")
    assert delete_resp.status_code == 200, delete_resp.text
    payload = delete_resp.json()
    assert payload["removed_history_rows"] >= 1
    assert payload["imageset_removed"]["imageset_id"] == imageset_id
    assert payload["imageset_removed"]["imageset_deleted"] is True
    assert payload["artifacts_removed"]["removed_dirs"] >= 1
    assert app.state.app_state.get_imageset(imageset_id) is None

    after = client.get("/api/jobs/history?limit=50")
    assert after.status_code == 200
    assert not any(item["job_id"] == job_id for item in after.json()["items"])
    preview_after = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_after.status_code == 404


def test_preview_returns_409_before_job_finish(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _SlowPredictor(p))

    image_path = tmp_path / "img_preview_pending.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_preview_pending.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "set_preview_pending"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("preview_pending.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 409

    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"


def test_preview_returns_409_for_failed_job(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])

    def _raise_predictor(_):
        raise RuntimeError("predictor init failed")

    monkeypatch.setattr("app.services.annotation_service.create_predictor", _raise_predictor)

    image_path = tmp_path / "img_preview_failed.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_preview_failed.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "set_preview_failed"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("preview_failed.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "failed"

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 409


def test_preview_available_from_history_after_task_manager_restart(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictor(p))

    image_path = tmp_path / "img_preview_history.jpg"
    _make_image(image_path)
    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("img_preview_history.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "set_preview_history"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("preview_history.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    r = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "conf": 0.25,
            "iou": 0.45,
            "device": "",
            "save_overlays": True,
        },
    )
    assert r.status_code == 200
    job_id = r.json()["id"]
    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    history_file = tmp_path / "data" / "history" / "jobs_history.jsonl"
    app.state.task_manager = TaskManager(max_workers=2, history_file=history_file)

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    payload = preview_resp.json()
    assert payload["job_id"] == job_id
    assert payload["items"]


def test_qwen_suggest_classes_endpoint(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, _ = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["person", "car", "bike"])

    r = client.post(
        "/api/models/upload",
        files={"model_file": ("demo_qwen.pt", b"pt-model", "application/octet-stream")},
    )
    assert r.status_code == 200
    model_id = r.json()["model_id"]

    async def _fake_suggest(description, class_names, images, qwen_model="", api_key=""):
        _ = (description, class_names, images, qwen_model, api_key)
        return {
            "selected_class_ids": [0, 2],
            "mapping_confirmed": True,
            "selected_classes": [{"id": 0, "name": "person"}, {"id": 2, "name": "bike"}],
            "reason": "匹配到行人和自行车",
            "provider_model": "qwen-vl-max-latest",
            "image_count": 1,
            "raw_text": '{"class_ids":[0,2],"reason":"匹配到行人和自行车"}',
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.suggest_classes", _fake_suggest)
    r = client.post(
        "/api/ai/qwen/suggest-classes",
        data={"model_id": model_id, "description": "只标注人和自行车"},
        files=[("files", ("ref.jpg", b"fake-image", "image/jpeg"))],
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["selected_class_ids"] == [0, 2]
    assert data["model_id"] == model_id
    assert data["class_count"] == 3

    r2 = client.post("/api/ai/qwen/suggest-classes", data={"model_id": model_id})
    assert r2.status_code == 400


def test_parse_target_classes_endpoint(app_client):
    client, _, _ = app_client
    resp = client.post(
        "/api/classes/parse",
        files={"classes_file": ("target.txt", "vehicle\nperson\n".encode("utf-8"), "text/plain")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["count"] == 2
    assert data["classes"][0]["name"] == "vehicle"
    assert data["classes"][1]["name"] == "person"


def test_qwen_zero_shot_annotate_and_preview(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    image_path = tmp_path / "qwen_img.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[
                ("files", ("qwen_img.jpg", f.read(), "image/jpeg")),
                ("files", ("qwen_img.txt", b"0 0.100000 0.100000 0.100000 0.100000\n", "text/plain")),
            ],
            data={"imageset_name": "qwen_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    async def _fake_detect(**kwargs):
        w = int(kwargs.get("image_width") or 120)
        h = int(kwargs.get("image_height") or 80)
        return {
            "objects": [],
            "svg_overlay": (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
                '<rect x="10" y="12" width="60" height="48" fill="none" stroke="#00FF00" stroke-width="3"/>'
                "</svg>"
            ),
            "label": "zebra_crossing",
            "note": "ok",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [w, h],
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect)

    with image_path.open("rb") as ref_fp:
        r = client.post(
            "/api/qwen/annotate/jobs",
            data={
                "imageset_id": imageset_id,
                "description": "请把斑马线全部标注出来",
                "label_mode": "append",
                "update_imageset_labels": "true",
                "round_tag": "qwen-round-1",
                "operator": "bob",
                "strict_size_check": "true",
                "size_retry": "1",
                "precision_mode": "fast",
            },
            files=[("files", ("ref.jpg", ref_fp.read(), "image/jpeg"))],
        )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    assert r.json()["job_type"] == "qwen_annotate"

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    artifacts_resp = client.get(f"/api/annotate/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200, artifacts_resp.text
    artifacts_payload = artifacts_resp.json()
    artifacts = artifacts_payload["artifacts"]
    summary = artifacts_payload["summary"]
    assert summary["pipeline"] == "qwen_zero_shot"
    assert summary["label_mode"] == "append"
    assert summary["operator"] == "root"
    assert summary["precision_mode"] == "fast"
    assert artifacts["preview_page_url"].endswith(job_id)

    manifest_resp = client.get(artifacts["manifest_csv"])
    assert manifest_resp.status_code == 200
    rows = list(csv.DictReader(io.StringIO(manifest_resp.text)))
    assert len(rows) == 1
    assert rows[0]["existing_boxes"] == "1"
    assert rows[0]["new_boxes"] == "1"
    assert rows[0]["final_boxes"] == "2"
    assert rows[0]["size_check_passed"] == "1"
    assert rows[0]["qwen_note"] == "ok"

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    preview_payload = preview_resp.json()
    assert preview_payload["job_type"] == "qwen_annotate"
    assert preview_payload["items"]
    item0 = preview_payload["items"][0]
    assert item0["size_check_passed"] == 1
    assert item0["qwen_note"] == "ok"
    assert "quality_score" in item0
    assert "consensus_rate" in item0
    assert "calibration_error_px" in item0
    assert "svg_overlay_url" in item0
    assert "mask_url" in item0
    src_resp = client.get(item0["source_url"])
    ov_resp = client.get(item0["overlay_url"])
    svg_resp = client.get(item0["svg_overlay_url"])
    mask_resp = client.get(item0["mask_url"])
    assert src_resp.status_code == 200
    assert ov_resp.status_code == 200
    assert svg_resp.status_code == 200
    assert mask_resp.status_code == 200
    assert client.get(item0["label_url"]).status_code == 200

    src_img = cv2.imdecode(np.frombuffer(src_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    ov_img = cv2.imdecode(np.frombuffer(ov_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert src_img is not None and ov_img is not None
    # fake bbox [10,12,70,60] should leave visible drawing at border pixel
    assert np.abs(src_img[12, 10].astype(np.int16) - ov_img[12, 10].astype(np.int16)).sum() > 20


def test_qwen_strict_size_check_marks_image_error(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    image_path = tmp_path / "qwen_size_fail.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_size_fail.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_size_fail_set"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    async def _fake_detect_fail(**kwargs):
        _ = kwargs
        return {
            "objects": [],
            "svg_overlay": "",
            "label": "obj",
            "note": "size-mismatch",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": False,
            "size_echo": [999, 999],
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_fail)

    r = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "标注目标",
            "strict_size_check": "true",
            "size_retry": "0",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    artifacts_resp = client.get(f"/api/annotate/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200, artifacts_resp.text
    summary = artifacts_resp.json()["summary"]
    assert summary["size_check_failed_images"] == 1
    assert summary["strict_size_check"] is True

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    item0 = preview_resp.json()["items"][0]
    assert item0["status"] == "error"
    assert item0["size_check_passed"] == 0
    assert "尺寸校验失败" in item0["error"]


def test_qwen_reference_image_validation(app_client):
    client, _, tmp_path = app_client
    image_path = tmp_path / "qwen_ref_validate.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_ref_validate.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_ref_validate_set"},
        )
    assert r.status_code == 200
    imageset_id = r.json()["imageset_id"]

    bad = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "",
        },
        files=[("files", ("bad_ref.jpg", b"not-image", "image/jpeg"))],
    )
    assert bad.status_code == 400
    assert "参考图片解析失败" in bad.text


def test_qwen_strict_precision_mode_outputs_quality_fields(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    image_path = tmp_path / "qwen_strict_precision.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_strict_precision.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_strict_precision_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    async def _fake_detect_strict(**kwargs):
        w = int(kwargs.get("image_width") or 120)
        h = int(kwargs.get("image_height") or 80)
        return {
            "objects": [],
            "svg_overlay": (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
                '<polygon points="24,16 84,16 84,60 24,60" fill="none" stroke="#00ff00" stroke-width="4"/>'
                "</svg>"
            ),
            "label": "ball",
            "note": "strict-ok",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [w, h],
            "calibration_points": [],
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_strict)

    r = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "标注球体",
            "strict_size_check": "true",
            "size_retry": "1",
            "precision_mode": "strict",
            "sample_count": "3",
            "max_calibration_error_px": "8",
            "min_consensus_rate": "0.67",
            "duplicate_iou": "0.98",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    artifacts_resp = client.get(f"/api/annotate/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200, artifacts_resp.text
    summary = artifacts_resp.json()["summary"]
    assert summary["pipeline"] == "qwen_zero_shot"
    assert summary["precision_mode"] == "strict"
    assert summary["sample_count"] == 3
    assert summary["quality_rejected_images"] == 0
    assert summary["avg_quality_score"] > 0

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    item0 = preview_resp.json()["items"][0]
    assert item0["status"] == "ok"
    assert item0["final_boxes"] > 0
    assert item0["quality_score"] > 0
    assert item0["consensus_rate"] >= 0.67
    assert item0["extract_quality_score"] > 0
    assert item0["svg_parse_ok"] == 1
    assert item0["reject_reason"] == ""
    assert item0["refine_rounds"] >= 1


def test_qwen_segment_affine_calibration_corrects_polygon_points(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    image_path = tmp_path / "qwen_segment_affine.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_segment_affine.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_segment_affine_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    async def _fake_detect_segment(**kwargs):
        w = int(kwargs.get("image_width") or 120)
        h = int(kwargs.get("image_height") or 80)
        model_points = [[12.0, 8.0], [42.0, 9.0], [41.0, 33.0], [10.0, 32.0]]
        return {
            "objects": [
                {
                    "label": "road_area",
                    "bbox": [10.0, 8.0, 42.0, 33.0],
                    "points": model_points,
                    "shape_type": "segment",
                    "confidence": 0.92,
                }
            ],
            "label": "road_area",
            "note": "segment-affine",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [w, h],
            "raw_object_count": 1,
            "calibration_points": [
                {"id": "tl", "x": 2.0, "y": 3.0},
                {"id": "tr", "x": (w - 1.0 + 4.0) / 2.0, "y": 3.0},
                {"id": "bl", "x": 2.0, "y": (h - 1.0 + 6.0) / 2.0},
                {"id": "br", "x": (w - 1.0 + 4.0) / 2.0, "y": (h - 1.0 + 6.0) / 2.0},
                {"id": "center", "x": ((w - 1.0) / 2.0 + 4.0) / 2.0, "y": ((h - 1.0) / 2.0 + 6.0) / 2.0},
            ],
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_segment)

    job_resp = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "把道路区域按多边形标出来",
            "label_task": "segment",
            "label_mode": "replace",
            "update_imageset_labels": "true",
            "precision_mode": "fast",
            "sample_count": "1",
            "max_calibration_error_px": "8",
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_id = job_resp.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    assert imageset.label_task == "segment"

    label_path = Path(imageset.dir_path) / "labels" / f"{Path(imageset.images[0].filename).stem}.txt"
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = parse_label_line(lines[0], "segment")
    assert parsed is not None
    points = parsed.points_pixels(120, 80)
    assert len(points) == 4
    assert abs(points[0][0] - 20.0) < 2.5
    assert abs(points[0][1] - 10.0) < 2.5
    assert abs(points[2][0] - 78.0) < 2.5
    assert abs(points[2][1] - 60.0) < 2.5

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    preview_item = preview_resp.json()["items"][0]
    assert preview_item["final_boxes"] == 1
    assert preview_item["calibration_error_px"] >= 0.0


def test_qwen_obb_affine_calibration_corrects_four_points(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    image_path = tmp_path / "qwen_obb_affine.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_obb_affine.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_obb_affine_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    async def _fake_detect_obb(**kwargs):
        w = int(kwargs.get("image_width") or 120)
        h = int(kwargs.get("image_height") or 80)
        model_points = [[12.0, 8.0], [42.0, 11.0], [40.0, 33.0], [10.0, 30.0]]
        return {
            "objects": [
                {
                    "label": "helmet",
                    "bbox": [10.0, 8.0, 42.0, 33.0],
                    "points": model_points,
                    "shape_type": "obb",
                    "confidence": 0.92,
                }
            ],
            "label": "helmet",
            "note": "obb-affine",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [w, h],
            "raw_object_count": 1,
            "calibration_points": [
                {"id": "tl", "x": 2.0, "y": 3.0},
                {"id": "tr", "x": (w - 1.0 + 4.0) / 2.0, "y": 3.0},
                {"id": "bl", "x": 2.0, "y": (h - 1.0 + 6.0) / 2.0},
                {"id": "br", "x": (w - 1.0 + 4.0) / 2.0, "y": (h - 1.0 + 6.0) / 2.0},
                {"id": "center", "x": ((w - 1.0) / 2.0 + 4.0) / 2.0, "y": ((h - 1.0) / 2.0 + 6.0) / 2.0},
            ],
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_obb)

    job_resp = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "把安全帽按旋转框四点标出来",
            "label_task": "obb",
            "label_mode": "replace",
            "update_imageset_labels": "true",
            "precision_mode": "fast",
            "sample_count": "1",
            "max_calibration_error_px": "8",
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_id = job_resp.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    assert imageset.label_task == "obb"

    label_path = Path(imageset.dir_path) / "labels" / f"{Path(imageset.images[0].filename).stem}.txt"
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = parse_label_line(lines[0], "obb")
    assert parsed is not None
    points = parsed.points_pixels(120, 80)
    assert len(points) == 4
    assert abs(points[0][0] - 20.0) < 2.5
    assert abs(points[0][1] - 10.0) < 2.5
    assert abs(points[1][0] - 80.0) < 2.5
    assert abs(points[1][1] - 16.0) < 2.5
    assert abs(points[2][0] - 76.0) < 2.5
    assert abs(points[2][1] - 60.0) < 2.5

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    preview_item = preview_resp.json()["items"][0]
    assert preview_item["final_boxes"] == 1
    assert preview_item["calibration_error_px"] >= 0.0


def test_qwen_svg_overlay_invalid_color_rejected(app_client, monkeypatch: pytest.MonkeyPatch):
    client, _, tmp_path = app_client

    image_path = tmp_path / "qwen_svg_invalid.jpg"
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    img[:, :] = (140, 140, 140)
    cv2.imwrite(str(image_path), img)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_svg_invalid.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_svg_invalid_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]

    async def _fake_detect_reverse(**kwargs):
        w = int(kwargs.get("image_width") or 120)
        h = int(kwargs.get("image_height") or 80)
        return {
            "objects": [],
            "svg_overlay": (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
                '<rect x="20" y="10" width="60" height="40" fill="none" stroke="#ff0000" stroke-width="3"/>'
                "</svg>"
            ),
            "label": "person",
            "note": "wrong-color",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [w, h],
            "calibration_points": [],
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_reverse)

    r = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "请标注红框目标",
            "strict_size_check": "true",
            "size_retry": "1",
            "precision_mode": "strict",
            "sample_count": "1",
            "reject_on_invalid_svg": "true",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    artifacts_resp = client.get(f"/api/annotate/jobs/{job_id}/artifacts")
    assert artifacts_resp.status_code == 200, artifacts_resp.text
    artifacts = artifacts_resp.json()["artifacts"]
    manifest_resp = client.get(artifacts["manifest_csv"])
    assert manifest_resp.status_code == 200
    rows = list(csv.DictReader(io.StringIO(manifest_resp.text)))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["svg_parse_ok"] == "0"
    assert rows[0]["new_boxes"] == "0"
    assert "SVG颜色不合规" in rows[0]["reject_reason"]


def test_qwen_append_preserves_existing_sparse_class_names(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    image_path = tmp_path / "qwen_sparse_names.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        r = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_sparse_names.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_sparse_names_set"},
        )
    assert r.status_code == 200, r.text
    imageset_id = r.json()["imageset_id"]
    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None

    labels_dir = Path(imageset.dir_path) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "qwen_sparse_names.txt").write_text(
        "\n".join(
            [
                "0 0.200000 0.200000 0.100000 0.100000",
                "71 0.600000 0.600000 0.150000 0.150000",
            ]
        ),
        encoding="utf-8",
    )
    cls_lines = [""] * 72
    cls_lines[0] = "car"
    cls_lines[71] = "phone"
    (labels_dir / "classes.txt").write_text("\n".join(cls_lines), encoding="utf-8")

    async def _fake_detect_append(**kwargs):
        _ = kwargs
        return {
            "objects": [
                {"label": "grass", "bbox": [20, 18, 70, 50], "confidence": 0.93},
            ],
            "label": "grass",
            "note": "append-ok",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [120, 80],
            "raw_object_count": 1,
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_append)

    r = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "补充草地",
            "label_mode": "append",
            "update_imageset_labels": "true",
            "precision_mode": "fast",
        },
    )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    assert images_resp.status_code == 200, images_resp.text
    class_names = images_resp.json()["class_names"]
    assert class_names["0"] == "car"
    assert class_names["71"] == "phone"
    assert class_names["72"] == "grass"
    assert class_names["0"] != "class_0"
    assert class_names["71"] != "class_71"

    classes_txt = (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt[0] == "car"
    assert classes_txt[71] == "phone"
    assert classes_txt[72] == "grass"

    label_map = client.get(f"/data/outputs/{job_id}/label_map.json")
    assert label_map.status_code == 200, label_map.text
    payload = label_map.json()["id_to_name"]
    assert payload["0"] == "car"
    assert payload["71"] == "phone"
    assert payload["72"] == "grass"

    preview_resp = client.get(f"/api/annotate/jobs/{job_id}/preview")
    assert preview_resp.status_code == 200, preview_resp.text
    preview_item = preview_resp.json()["items"][0]
    assert "/overlays_resolved/" in preview_item["overlay_url"]
    assert client.get(preview_item["overlay_url"]).status_code == 200


def test_annotate_only_persists_used_class_names(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: COCO80_TEST_CLASSES)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictorCellPhone(p))

    image_path = tmp_path / "annotate_sparse_used.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("annotate_sparse_used.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "annotate_sparse_used"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    model_resp = client.post(
        "/api/models/upload",
        files={"model_file": ("coco80.pt", b"pt-model", "application/octet-stream")},
    )
    assert model_resp.status_code == 200, model_resp.text
    model_id = model_resp.json()["model_id"]

    job_resp = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": list(range(80)),
            "mapping_confirmed": True,
            "label_mode": "replace",
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_id = job_resp.json()["id"]

    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    classes_txt = (Path(imageset.dir_path) / "labels" / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt[0] == "class_0"
    assert classes_txt[1] == "class_1"
    assert classes_txt[67] == "cell phone"
    assert "person" not in classes_txt[:67]

    artifacts = client.get(f"/api/annotate/jobs/{job_id}/artifacts").json()["artifacts"]
    run_meta = client.get(artifacts["run_meta_json"]).json()
    assert run_meta["final_class_name_map"] == {"67": "cell phone"}


def test_annotate_override_high_ids_do_not_preserve_old_semantic_names(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: COCO80_TEST_CLASSES)
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _FakePredictorCellPhone(p))

    image_path = tmp_path / "annotate_override_high.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("annotate_override_high.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "annotate_override_high"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    model_resp = client.post(
        "/api/models/upload",
        files={"model_file": ("coco80_override.pt", b"pt-model", "application/octet-stream")},
    )
    assert model_resp.status_code == 200, model_resp.text
    model_id = model_resp.json()["model_id"]

    overrides = {str(idx): idx + 140 for idx in range(80)}
    job_resp = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": list(range(80)),
            "mapping_confirmed": True,
            "label_mode": "replace",
            "class_id_overrides": overrides,
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_id = job_resp.json()["id"]

    final = wait_job(client, f"/api/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    classes_txt = (imageset_dir / "labels" / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt[0] == "class_0"
    assert classes_txt[140] == "class_140"
    assert classes_txt[207] == "cell phone"
    assert "person" not in classes_txt[:207]

    data_yaml = yaml.safe_load((imageset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == {207: "cell phone"}

    artifacts = client.get(f"/api/annotate/jobs/{job_id}/artifacts").json()["artifacts"]
    run_meta = client.get(artifacts["run_meta_json"]).json()
    assert run_meta["final_class_name_map"] == {"207": "cell phone"}


def test_qwen_append_cleans_unused_residual_class_names(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, tmp_path = app_client

    image_path = tmp_path / "qwen_cleanup.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("qwen_cleanup.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "qwen_cleanup_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]
    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None

    labels_dir = Path(imageset.dir_path) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "qwen_cleanup.txt").write_text(
        "207 0.200000 0.200000 0.100000 0.100000",
        encoding="utf-8",
    )
    cls_lines = [""] * 208
    cls_lines[0] = "person"
    cls_lines[207] = "cell phone"
    (labels_dir / "classes.txt").write_text("\n".join(cls_lines), encoding="utf-8")

    async def _fake_detect_cleanup(**kwargs):
        _ = kwargs
        return {
            "objects": [
                {"label": "grass", "bbox": [22, 20, 74, 56], "confidence": 0.95},
            ],
            "label": "grass",
            "note": "cleanup-ok",
            "provider_model": "qwen-vl-max-latest",
            "raw_text": "{}",
            "size_check_passed": True,
            "size_echo": [120, 80],
            "raw_object_count": 1,
        }

    monkeypatch.setattr("app.services.qwen_service.QwenService.detect_objects_on_image", _fake_detect_cleanup)

    job_resp = client.post(
        "/api/qwen/annotate/jobs",
        data={
            "imageset_id": imageset_id,
            "description": "补充草地",
            "label_mode": "append",
            "update_imageset_labels": "true",
            "precision_mode": "fast",
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_id = job_resp.json()["id"]

    final = wait_job(client, f"/api/qwen/annotate/jobs/{job_id}")
    assert final["status"] == "succeeded"

    classes_txt = (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt[0] == "class_0"
    assert classes_txt[207] == "cell phone"
    assert classes_txt[208] == "grass"

    label_map = client.get(f"/data/outputs/{job_id}/label_map.json")
    assert label_map.status_code == 200, label_map.text
    payload = label_map.json()["id_to_name"]
    assert payload == {"207": "cell phone", "208": "grass"}


def test_remap_ignores_unused_residual_name_conflicts(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "remap_cleanup.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("remap_cleanup.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "remap_cleanup_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]
    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None

    labels_dir = Path(imageset.dir_path) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "remap_cleanup.txt").write_text(
        "208 0.200000 0.200000 0.100000 0.100000",
        encoding="utf-8",
    )
    cls_lines = [""] * 209
    cls_lines[0] = "person"
    cls_lines[208] = "草地"
    (labels_dir / "classes.txt").write_text("\n".join(cls_lines), encoding="utf-8")

    remap_resp = client.post(
        f"/api/imagesets/{imageset_id}/remap-classes",
        json={"id_mapping": {"208": 0}},
    )
    assert remap_resp.status_code == 200, remap_resp.text

    remapped_lines = (labels_dir / "remap_cleanup.txt").read_text(encoding="utf-8").splitlines()
    assert remapped_lines == ["0 0.200000 0.200000 0.100000 0.100000"]
    classes_txt = (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt == ["草地"]


def test_remap_rejects_real_used_name_conflicts(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "remap_real_conflict.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("remap_real_conflict.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "remap_real_conflict_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]
    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None

    labels_dir = Path(imageset.dir_path) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "remap_real_conflict.txt").write_text(
        "\n".join(
            [
                "0 0.200000 0.200000 0.100000 0.100000",
                "208 0.600000 0.600000 0.150000 0.150000",
            ]
        ),
        encoding="utf-8",
    )
    cls_lines = [""] * 209
    cls_lines[0] = "person"
    cls_lines[208] = "草地"
    (labels_dir / "classes.txt").write_text("\n".join(cls_lines), encoding="utf-8")

    remap_resp = client.post(
        f"/api/imagesets/{imageset_id}/remap-classes",
        json={"id_mapping": {"208": 0}},
    )
    assert remap_resp.status_code == 400
    assert "冲突" in remap_resp.text


def test_remap_allows_same_name_merge_for_used_classes(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "remap_same_name.jpg"
    _make_image(image_path)

    with image_path.open("rb") as f:
        upload_resp = client.post(
            "/api/imagesets/upload-folder",
            files=[("files", ("remap_same_name.jpg", f.read(), "image/jpeg"))],
            data={"imageset_name": "remap_same_name_set"},
        )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]
    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None

    labels_dir = Path(imageset.dir_path) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "remap_same_name.txt").write_text(
        "\n".join(
            [
                "1 0.200000 0.200000 0.100000 0.100000",
                "2 0.600000 0.600000 0.150000 0.150000",
            ]
        ),
        encoding="utf-8",
    )
    cls_lines = ["", "car", "car"]
    (labels_dir / "classes.txt").write_text("\n".join(cls_lines), encoding="utf-8")

    remap_resp = client.post(
        f"/api/imagesets/{imageset_id}/remap-classes",
        json={"id_mapping": {"2": 1}},
    )
    assert remap_resp.status_code == 200, remap_resp.text

    remapped_lines = (labels_dir / "remap_same_name.txt").read_text(encoding="utf-8").splitlines()
    assert remapped_lines == [
        "1 0.200000 0.200000 0.100000 0.100000",
        "1 0.600000 0.600000 0.150000 0.150000",
    ]
    classes_txt = (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes_txt[0] == "class_0"
    assert classes_txt[1] == "car"


def test_upload_folder_supports_images_labels_subfolders(app_client, tmp_path: Path):
    client, _, _ = app_client
    image_path = tmp_path / "sample.jpg"
    _make_image(image_path)

    payload_files = [
        ("files", ("images/sample.jpg", image_path.read_bytes(), "image/jpeg")),
        ("files", ("labels/sample.txt", b"0 0.5 0.5 0.2 0.2\n", "text/plain")),
        ("files", ("labels/unmatched.txt", b"0 0.1 0.1 0.1 0.1\n", "text/plain")),
    ]
    resp = client.post(
        "/api/imagesets/upload-folder",
        files=payload_files,
        data={"imageset_name": "nested_set"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["image_count"] == 1
    assert data["labels_imported"] == 1
    assert data["labels_unmatched"] == 1

    imageset_id = data["imageset_id"]
    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    assert images_resp.status_code == 200
    items = images_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["label_exists"] is True


def test_upload_folder_accepts_more_than_default_multipart_file_limit(app_client):
    client, _, _ = app_client
    payload_files = [
        ("files", (f"images/sample_{idx:04d}.jpg", b"x", "image/jpeg"))
        for idx in range(1001)
    ]

    resp = client.post(
        "/api/imagesets/upload-folder",
        files=payload_files,
        data={"imageset_name": "large_folder_set"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["image_count"] == 1001


def test_folder_upload_session_uploads_files_incrementally(app_client, tmp_path: Path):
    client, _, _ = app_client
    image_path = tmp_path / "chunked.jpg"
    _make_image(image_path)

    start = client.post(
        "/api/imagesets/upload-folder/session",
        json={"imageset_name": "session_dataset", "total_files": 2, "total_bytes": image_path.stat().st_size + 22},
    )
    assert start.status_code == 200, start.text
    session_id = start.json()["session_id"]

    image_upload = client.post(
        f"/api/imagesets/upload-folder/session/{session_id}/file",
        files={"file": ("chunked.jpg", image_path.read_bytes(), "image/jpeg")},
        data={"relative_path": "dataset/images/chunked.jpg"},
    )
    assert image_upload.status_code == 200, image_upload.text
    assert image_upload.json()["uploaded_files"] == 1

    label_upload = client.post(
        f"/api/imagesets/upload-folder/session/{session_id}/file",
        files={"file": ("chunked.txt", b"0 0.5 0.5 0.2 0.2\n", "text/plain")},
        data={"relative_path": "dataset/labels/chunked.txt"},
    )
    assert label_upload.status_code == 200, label_upload.text
    assert label_upload.json()["uploaded_files"] == 2

    finish = client.post(f"/api/imagesets/upload-folder/session/{session_id}/finish")
    assert finish.status_code == 200, finish.text
    data = finish.json()
    assert data["name"] == "session_dataset"
    assert data["image_count"] == 1
    assert data["labels_imported"] == 1


def test_upload_folder_imports_class_names_from_classes_txt(app_client, tmp_path: Path):
    client, app, _ = app_client
    image_path = tmp_path / "named.jpg"
    _make_image(image_path)

    resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("images/named.jpg", image_path.read_bytes(), "image/jpeg")),
            ("files", ("labels/named.txt", b"1 0.5 0.5 0.2 0.2\n", "text/plain")),
            ("files", ("labels/classes.txt", "person\ncar\n".encode("utf-8"), "text/plain")),
        ],
        data={"imageset_name": "named_set"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["labels_imported"] == 1
    assert data["labels_unmatched"] == 0
    assert data["class_names_imported"] == 2

    imageset = app.state.app_state.get_imageset(data["imageset_id"])
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    assert (imageset_dir / "labels" / "classes.txt").read_text(encoding="utf-8").splitlines() == ["person", "car"]

    images_resp = client.get(f"/api/imagesets/{data['imageset_id']}/images")
    assert images_resp.status_code == 200, images_resp.text
    assert images_resp.json()["class_names"] == {"1": "car"}


def test_upload_folder_rejects_empty_class_names_file(app_client, tmp_path: Path):
    client, _, _ = app_client
    image_path = tmp_path / "empty_classes.jpg"
    _make_image(image_path)

    resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("images/empty_classes.jpg", image_path.read_bytes(), "image/jpeg")),
            ("files", ("labels/classes.txt", b"\n\n", "text/plain")),
        ],
        data={"imageset_name": "empty_classes_set"},
    )
    assert resp.status_code == 400, resp.text
    assert "类别名文件为空" in resp.json()["detail"]


def test_upload_folder_imports_class_names_from_data_yaml(app_client, tmp_path: Path):
    client, app, _ = app_client
    image_path = tmp_path / "seg.jpg"
    _make_image(image_path)
    data_yaml = yaml.safe_dump(
        {"path": ".", "train": "images", "val": "images", "task": "segment", "names": {0: "person", 2: "road"}},
        allow_unicode=True,
        sort_keys=False,
    ).encode("utf-8")

    resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("dataset/images/seg.jpg", image_path.read_bytes(), "image/jpeg")),
            ("files", ("dataset/labels/seg.txt", b"2 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", "text/plain")),
            ("files", ("dataset/data.yaml", data_yaml, "application/x-yaml")),
        ],
        data={"imageset_name": "yaml_named_set"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["class_names_imported"] == 2
    assert data["label_task"] == "segment"

    imageset = app.state.app_state.get_imageset(data["imageset_id"])
    assert imageset is not None
    assert imageset.label_task == "segment"
    imageset_dir = Path(imageset.dir_path)
    assert (imageset_dir / "labels" / "classes.txt").read_text(encoding="utf-8").splitlines() == [
        "person",
        "class_1",
        "road",
    ]
    stored_yaml = yaml.safe_load((imageset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert stored_yaml["path"] == "."
    assert stored_yaml["task"] == "segment"
    assert stored_yaml["names"][2] == "road"

    images_resp = client.get(f"/api/imagesets/{data['imageset_id']}/images")
    assert images_resp.status_code == 200, images_resp.text
    assert images_resp.json()["class_names"] == {"2": "road"}


def test_upload_folder_matches_duplicate_basenames_by_relative_path(app_client, tmp_path: Path):
    client, app, _ = app_client
    image_a = tmp_path / "a_sample.jpg"
    image_b = tmp_path / "b_sample.jpg"
    _make_image(image_a)
    _make_image(image_b)

    resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("dataset/images/a/sample.jpg", image_a.read_bytes(), "image/jpeg")),
            ("files", ("dataset/images/b/sample.jpg", image_b.read_bytes(), "image/jpeg")),
            ("files", ("dataset/labels/a/sample.txt", b"0 0.5 0.5 0.2 0.2\n", "text/plain")),
            ("files", ("dataset/labels/b/sample.txt", b"1 0.5 0.5 0.2 0.2\n", "text/plain")),
            ("files", ("dataset/classes.txt", b"cat\ndog\n", "text/plain")),
        ],
        data={"imageset_name": "duplicate_basename_set"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["image_count"] == 2
    assert data["labels_imported"] == 2
    assert data["duplicate_label_name_count"] == 0

    imageset = app.state.app_state.get_imageset(data["imageset_id"])
    assert imageset is not None
    labels_dir = Path(imageset.dir_path) / "labels"
    label_texts = sorted(
        path.read_text(encoding="utf-8").strip()
        for path in labels_dir.glob("*.txt")
        if path.name != "classes.txt"
    )
    assert label_texts == [
        "0 0.5 0.5 0.2 0.2",
        "1 0.5 0.5 0.2 0.2",
    ]


def test_imageset_images_support_pagination_and_filters(app_client, tmp_path: Path):
    client, _, _ = app_client

    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img3 = tmp_path / "c.jpg"
    _make_image(img1)
    _make_image(img2)
    _make_image(img3)

    payload_files = [
        ("files", ("images/a.jpg", img1.read_bytes(), "image/jpeg")),
        ("files", ("images/b.jpg", img2.read_bytes(), "image/jpeg")),
        ("files", ("images/c.jpg", img3.read_bytes(), "image/jpeg")),
        ("files", ("labels/a.txt", b"0 0.5 0.5 0.2 0.2\n", "text/plain")),
    ]
    resp = client.post("/api/imagesets/upload-folder", files=payload_files, data={"imageset_name": "paged_set"})
    assert resp.status_code == 200, resp.text
    imageset_id = resp.json()["imageset_id"]

    page1 = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=2")
    assert page1.status_code == 200, page1.text
    payload1 = page1.json()
    assert payload1["total"] == 3
    assert payload1["page"] == 1
    assert payload1["has_next"] is True
    assert len(payload1["items"]) == 2

    page2 = client.get(f"/api/imagesets/{imageset_id}/images?page=2&page_size=2")
    assert page2.status_code == 200, page2.text
    payload2 = page2.json()
    assert payload2["page"] == 2
    assert payload2["has_next"] is False
    assert len(payload2["items"]) == 1

    only_labeled = client.get(f"/api/imagesets/{imageset_id}/images?has_label=true")
    assert only_labeled.status_code == 200
    labeled_items = only_labeled.json()["items"]
    assert labeled_items
    assert all(bool(x["label_exists"]) for x in labeled_items)


def test_image_review_workflow_update_and_summary(app_client, tmp_path: Path):
    client, _, _ = app_client
    img = tmp_path / "qa.jpg"
    _make_image(img)
    resp = client.post(
        "/api/imagesets/upload-folder",
        files=[("files", ("qa.jpg", img.read_bytes(), "image/jpeg"))],
        data={"imageset_name": "review_set"},
    )
    assert resp.status_code == 200, resp.text
    imageset_id = resp.json()["imageset_id"]

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images")
    assert images_resp.status_code == 200
    image_id = images_resp.json()["items"][0]["image_id"]

    update_resp = client.patch(
        f"/api/images/{image_id}/review",
        json={"review_status": "accepted", "reviewer": "qa_alice", "review_note": "ok"},
    )
    assert update_resp.status_code == 200, update_resp.text
    assert update_resp.json()["review_status"] == "accepted"

    after_resp = client.get(f"/api/imagesets/{imageset_id}/images?review_status=accepted")
    assert after_resp.status_code == 200
    items = after_resp.json()["items"]
    assert len(items) == 1
    assert items[0]["review_status"] == "accepted"
    assert items[0]["reviewer"] == "root"

    summary_resp = client.get(f"/api/imagesets/{imageset_id}/review-summary")
    assert summary_resp.status_code == 200, summary_resp.text
    counts = summary_resp.json()["counts"]
    assert counts["accepted"] == 1


def test_cancel_running_annotate_job(app_client, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    client, _, _ = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])
    monkeypatch.setattr("app.services.annotation_service.create_predictor", lambda p: _SlowPredictor(p))

    img = tmp_path / "cancel.jpg"
    _make_image(img)
    set_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[("files", ("cancel.jpg", img.read_bytes(), "image/jpeg"))],
        data={"imageset_name": "cancel_set"},
    )
    assert set_resp.status_code == 200
    imageset_id = set_resp.json()["imageset_id"]

    model_resp = client.post(
        "/api/models/upload",
        files={"model_file": ("cancel.pt", b"pt-model", "application/octet-stream")},
    )
    assert model_resp.status_code == 200
    model_id = model_resp.json()["model_id"]

    job_resp = client.post(
        "/api/annotate/jobs",
        json={
            "model_id": model_id,
            "imageset_id": imageset_id,
            "selected_class_ids": [0],
            "mapping_confirmed": True,
            "save_overlays": True,
        },
    )
    assert job_resp.status_code == 200
    job_id = job_resp.json()["id"]

    cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200, cancel_resp.text
    assert cancel_resp.json()["cancel_requested"] is True

    deadline = time.time() + 5
    final = None
    while time.time() < deadline:
        res = client.get(f"/api/annotate/jobs/{job_id}")
        assert res.status_code == 200
        final = res.json()
        if final["status"] in {"cancelled", "succeeded", "failed"}:
            break
        time.sleep(0.1)
    assert final is not None
    assert final["status"] == "cancelled"


def test_train_job_rejects_sparse_class_ids(app_client, tmp_path: Path):
    client, app, _ = app_client

    outputs_dir = tmp_path / "data" / "outputs"
    before_dirs = sorted(p.name for p in outputs_dir.iterdir())

    img_a = tmp_path / "train_sparse_a.jpg"
    img_b = tmp_path / "train_sparse_b.jpg"
    _make_image(img_a)
    _make_image(img_b)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("train_sparse_a.jpg", img_a.read_bytes(), "image/jpeg")),
            ("files", ("train_sparse_b.jpg", img_b.read_bytes(), "image/jpeg")),
        ],
        data={"imageset_name": "train_sparse_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    image_files = sorted((imageset_dir / "images").glob("*.jpg"))
    assert len(image_files) == 2

    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{image_files[0].stem}.txt").write_text("0 0.200000 0.200000 0.100000 0.100000\n", encoding="utf-8")
    (labels_dir / f"{image_files[1].stem}.txt").write_text("2 0.400000 0.400000 0.200000 0.200000\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("cell phone\nclass_1\n草地", encoding="utf-8")
    (imageset_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(imageset_dir),
                "train": "images",
                "val": "images",
                "names": {0: "cell phone", 2: "草地"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    job_resp = client.post(
        "/api/train/jobs",
        json={
            "imageset_id": imageset_id,
            "epochs": 1,
            "batch_size": 2,
            "img_size": 64,
            "base_model": "yolo11n.pt",
            "save_to_system": False,
        },
    )
    assert job_resp.status_code == 400, job_resp.text
    detail = job_resp.json()["detail"]
    assert "YOLO 训练要求类 ID 连续" in detail
    assert "当前实际使用的 IDs: 0, 2" in detail
    assert "训练应为连续编号 0..1" in detail
    assert "类 ID 调整" in detail

    after_dirs = sorted(p.name for p in outputs_dir.iterdir())
    assert after_dirs == before_dirs


def test_train_job_allows_contiguous_used_ids(app_client, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    client, app, _ = app_client

    import app.services.training_service as training_service_module

    outputs_dir = tmp_path / "data" / "outputs"
    monkeypatch.setattr(training_service_module, "OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(training_service_module, "resolve_device", lambda _: "cpu")
    monkeypatch.setattr(training_service_module, "extract_classes_from_model_file", lambda _: ["cell phone", "草地"])

    predict_calls: list[float] = []

    class _FakeTrainPredictor:
        def predict(self, _image_path: str, conf: float, iou: float, device: str = ""):
            _ = (iou, device)
            predict_calls.append(conf)
            if conf > 0.01:
                return []
            return [
                types.SimpleNamespace(
                    cls_id=0,
                    cls_name="cell phone",
                    conf=0.012,
                    x1=8.0,
                    y1=9.0,
                    x2=40.0,
                    y2=42.0,
                )
            ]

    monkeypatch.setattr(training_service_module, "create_predictor", lambda _: _FakeTrainPredictor())

    class _FakeYOLOTrain:
        def __init__(self, model_path: str) -> None:
            self.model_path = model_path
            self.callbacks = {}

        def add_callback(self, name: str, callback) -> None:
            self.callbacks[name] = callback

        def train(self, **kwargs):
            data_path = Path(kwargs["data"])
            payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            assert payload["names"] == {0: "cell phone", 1: "草地"}
            assert payload["path"] == str(imageset_dir.resolve())
            assert payload["train"] == str((imageset_dir / "images").resolve())
            assert payload["val"] == str((imageset_dir / "images").resolve())
            assert data_path.parent == outputs_dir / data_path.parent.name

            label_ids = set()
            for label_path in sorted((Path(payload["path"]) / "labels").glob("*.txt")):
                if label_path.name == "classes.txt":
                    continue
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if parts:
                        label_ids.add(int(parts[0]))
            assert label_ids == {0, 1}

            callback = self.callbacks.get("on_train_epoch_end")
            if callback is not None:
                callback(types.SimpleNamespace(epoch=kwargs["epochs"] - 1))

            weights_dir = Path(kwargs["project"]) / kwargs["name"] / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            (weights_dir / "best.pt").write_bytes(b"trained")
            return {"status": "ok"}

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=_FakeYOLOTrain))

    img_a = tmp_path / "train_ok_a.jpg"
    img_b = tmp_path / "train_ok_b.jpg"
    _make_image(img_a)
    _make_image(img_b)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("train_ok_a.jpg", img_a.read_bytes(), "image/jpeg")),
            ("files", ("train_ok_b.jpg", img_b.read_bytes(), "image/jpeg")),
        ],
        data={"imageset_name": "train_ok_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    image_files = sorted((imageset_dir / "images").glob("*.jpg"))
    assert len(image_files) == 2

    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{image_files[0].stem}.txt").write_text("0 0.200000 0.200000 0.100000 0.100000\n", encoding="utf-8")
    (labels_dir / f"{image_files[1].stem}.txt").write_text("1 0.400000 0.400000 0.200000 0.200000\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("cell phone\n草地\nold_residual", encoding="utf-8")
    (imageset_dir / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "/stale/docker/or/host/path",
                "train": "/bad/images",
                "val": "/bad/images",
                "names": {0: "cell phone", 1: "草地"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    job_resp = client.post(
        "/api/train/jobs",
        json={
            "imageset_id": imageset_id,
            "epochs": 1,
            "batch_size": 2,
            "img_size": 64,
            "base_model": "yolo11n.pt",
            "save_to_system": False,
        },
    )
    assert job_resp.status_code == 200, job_resp.text
    job_id = job_resp.json()["job_id"]

    final = wait_job(client, f"/api/train/jobs/{job_id}")
    assert final["status"] == "succeeded", final
    preview_items = final["result"].get("prediction_preview") or []
    assert len(preview_items) == 2
    assert preview_items[0]["overlay_url"].startswith(f"/data/outputs/{job_id}/prediction_preview/overlays/")
    assert preview_items[0]["preview_conf"] == 0.01
    assert preview_items[0]["prediction_text"].startswith("预览阈值 conf=0.010")
    assert "0 cell phone" in preview_items[0]["prediction_text"]
    assert (outputs_dir / job_id / "prediction_preview" / "overlays" / Path(preview_items[0]["filename"]).name).exists()
    assert 0.25 in predict_calls and 0.01 in predict_calls

    run_meta = yaml.safe_load((outputs_dir / job_id / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["trained_classes"] == ["cell phone", "草地"]
    assert len(run_meta.get("prediction_preview") or []) == 2
    assert run_meta["prediction_preview"][0]["preview_conf"] == 0.01
    assert "train_dataset_dir" not in run_meta
    assert "original_to_train_class_id_map" not in run_meta

    source_data_yaml = yaml.safe_load((imageset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert source_data_yaml["path"] == "/stale/docker/or/host/path"
    assert source_data_yaml["train"] == "/bad/images"


def test_refine_save_new_class_updates_labels_and_supports_rollback(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "refine_editor.jpg"
    _make_image(image_path)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("refine_editor.jpg", image_path.read_bytes(), "image/jpeg")),
            ("files", ("refine_editor.txt", b"0 0.250000 0.250000 0.200000 0.200000\n", "text/plain")),
        ],
        data={"imageset_name": "refine_editor_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    labels_dir = imageset_dir / "labels"
    label_path = labels_dir / f"{Path(imageset.images[0].filename).stem}.txt"
    (labels_dir / "classes.txt").write_text("person\n", encoding="utf-8")
    (imageset_dir / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images", "val": "images", "names": {0: "person"}}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    image_id = imageset.images[0].id
    refine_resp = client.get(f"/api/images/{image_id}/refine")
    assert refine_resp.status_code == 200, refine_resp.text
    refine_payload = refine_resp.json()
    assert refine_payload["image_id"] == image_id
    assert refine_payload["class_options"] == [{"class_id": 0, "name": "person"}]
    assert len(refine_payload["boxes"]) == 1
    assert refine_payload["boxes"][0]["class_name"] == "person"

    save_resp = client.post(
        f"/api/images/{image_id}/refine",
        json={
            "boxes": [
                {
                    "box_id": refine_payload["boxes"][0]["box_id"],
                    "class_id": 0,
                    "class_name": "person",
                    "x1": 8,
                    "y1": 9,
                    "x2": 44,
                    "y2": 46,
                },
                {
                    "box_id": "box_new_window",
                    "class_id": None,
                    "class_name": "窗户",
                    "x1": 60,
                    "y1": 12,
                    "x2": 96,
                    "y2": 48,
                },
            ],
            "new_classes": [{"name": "窗户"}],
            "operator": "alice",
        },
    )
    assert save_resp.status_code == 200, save_resp.text
    saved_payload = save_resp.json()
    assert saved_payload["saved"] is True
    assert saved_payload["can_rollback"] is True
    assert {item["name"] for item in saved_payload["class_options"]} == {"person", "窗户"}
    assert sorted(box["class_id"] for box in saved_payload["boxes"]) == [0, 1]

    label_lines = label_path.read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 2
    assert label_lines[0].startswith("0 ")
    assert label_lines[1].startswith("1 ")
    assert (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["person", "窗户"]

    data_yaml = yaml.safe_load((imageset_dir / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml["names"] == {0: "person", 1: "窗户"}

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=10")
    assert images_resp.status_code == 200, images_resp.text
    assert images_resp.json()["class_names"] == {"0": "person", "1": "窗户"}

    rollback_resp = client.post(f"/api/images/{image_id}/refine/rollback")
    assert rollback_resp.status_code == 200, rollback_resp.text
    rollback_payload = rollback_resp.json()
    assert rollback_payload["rolled_back"] is True
    assert len(rollback_payload["boxes"]) == 1
    assert rollback_payload["boxes"][0]["class_id"] == 0
    assert rollback_payload["boxes"][0]["class_name"] == "person"
    assert label_path.read_text(encoding="utf-8").splitlines() == [
        "0 0.250000 0.250000 0.200000 0.200000",
    ]
    assert (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["person"]

    rollback_images_resp = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=10")
    assert rollback_images_resp.status_code == 200, rollback_images_resp.text
    assert rollback_images_resp.json()["class_names"] == {"0": "person"}


def test_refine_segment_imageset_coerces_legacy_detect_rows(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "refine_segment_legacy.jpg"
    _make_image(image_path)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[("files", ("refine_segment_legacy.jpg", image_path.read_bytes(), "image/jpeg"))],
        data={"imageset_name": "refine_segment_legacy_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset.label_task = "segment"
    app.state.app_state._save_imageset(imageset)

    imageset_dir = Path(imageset.dir_path)
    labels_dir = imageset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_path = labels_dir / f"{Path(imageset.images[0].filename).stem}.txt"
    label_path.write_text("0 0.500000 0.500000 0.250000 0.300000\n", encoding="utf-8")
    (labels_dir / "classes.txt").write_text("object\n", encoding="utf-8")

    image_id = imageset.images[0].id
    refine_resp = client.get(f"/api/images/{image_id}/refine")
    assert refine_resp.status_code == 200, refine_resp.text
    payload = refine_resp.json()
    assert payload["label_task"] == "segment"
    assert len(payload["boxes"]) == 1
    assert payload["boxes"][0]["shape_type"] == "segment"
    assert len(payload["boxes"][0]["points"]) == 4

    save_resp = client.post(
        f"/api/images/{image_id}/refine",
        json={"boxes": payload["boxes"], "new_classes": [], "operator": "alice"},
    )
    assert save_resp.status_code == 200, save_resp.text
    assert len(label_path.read_text(encoding="utf-8").split()) == 9


def test_refine_save_empty_boxes_removes_label_and_rollback_restores_it(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "refine_empty.jpg"
    _make_image(image_path)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[
            ("files", ("refine_empty.jpg", image_path.read_bytes(), "image/jpeg")),
            ("files", ("refine_empty.txt", b"0 0.250000 0.250000 0.200000 0.200000\n", "text/plain")),
        ],
        data={"imageset_name": "refine_empty_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    imageset_dir = Path(imageset.dir_path)
    labels_dir = imageset_dir / "labels"
    label_path = labels_dir / f"{Path(imageset.images[0].filename).stem}.txt"
    (labels_dir / "classes.txt").write_text("person\n", encoding="utf-8")
    (imageset_dir / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "images", "val": "images", "names": {0: "person"}}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    image_id = imageset.images[0].id
    save_resp = client.post(
        f"/api/images/{image_id}/refine",
        json={
            "boxes": [],
            "new_classes": [],
            "operator": "bob",
        },
    )
    assert save_resp.status_code == 200, save_resp.text
    empty_payload = save_resp.json()
    assert empty_payload["saved"] is True
    assert empty_payload["boxes"] == []
    assert empty_payload["label_exists"] is False
    assert empty_payload["label_url"] == ""
    assert empty_payload["can_rollback"] is True
    assert not label_path.exists()
    assert not (labels_dir / "classes.txt").exists()
    assert not (imageset_dir / "data.yaml").exists()

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=10")
    assert images_resp.status_code == 200, images_resp.text
    assert images_resp.json()["items"][0]["label_exists"] is False
    assert images_resp.json()["class_names"] == {}

    rollback_resp = client.post(f"/api/images/{image_id}/refine/rollback")
    assert rollback_resp.status_code == 200, rollback_resp.text
    restored_payload = rollback_resp.json()
    assert restored_payload["rolled_back"] is True
    assert restored_payload["label_exists"] is True
    assert len(restored_payload["boxes"]) == 1
    assert label_path.read_text(encoding="utf-8").splitlines() == [
        "0 0.250000 0.250000 0.200000 0.200000",
    ]
    assert (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["person"]


def test_imageset_reload_prefers_current_data_dir_over_stale_absolute_path(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "imageset_move_fix.jpg"
    _make_image(image_path)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[("files", ("imageset_move_fix.jpg", image_path.read_bytes(), "image/jpeg"))],
        data={"imageset_name": "imageset_move_fix_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    current_dir = Path(imageset.dir_path)
    metadata_path = current_dir / "metadata.json"

    stale_root = tmp_path / "old_repo_copy" / "data" / "imagesets" / imageset_id
    stale_root.mkdir(parents=True, exist_ok=True)
    (stale_root / "images").mkdir(exist_ok=True)
    (stale_root / "labels").mkdir(exist_ok=True)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["dir_path"] = str(stale_root)
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import app.core.state as state_module

    app.state.app_state = state_module.AppState()
    reloaded = app.state.app_state.get_imageset(imageset_id)
    assert reloaded is not None
    assert Path(reloaded.dir_path).resolve() == current_dir.resolve()

    images_resp = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=10")
    assert images_resp.status_code == 200, images_resp.text
    image_item = images_resp.json()["items"][0]
    assert image_item["url"].startswith(f"/data/imagesets/{imageset_id}/images/")


def test_images_endpoint_accepts_data_path_aliases(app_client):
    client, app, tmp_path = app_client

    image_path = tmp_path / "imageset_alias_fix.jpg"
    _make_image(image_path)

    upload_resp = client.post(
        "/api/imagesets/upload-folder",
        files=[("files", ("imageset_alias_fix.jpg", image_path.read_bytes(), "image/jpeg"))],
        data={"imageset_name": "imageset_alias_fix_set"},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    imageset_id = upload_resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    current_dir = str(Path(imageset.dir_path))
    alias_dir = ""
    if current_dir.startswith("/private/"):
        alias_dir = current_dir.removeprefix("/private")
    elif current_dir.startswith("/var/"):
        alias_dir = "/private" + current_dir
    if not alias_dir:
        pytest.skip("当前平台没有可用的 /private 与 /var 路径别名")

    imageset.dir_path = alias_dir
    images_resp = client.get(f"/api/imagesets/{imageset_id}/images?page=1&page_size=10")
    assert images_resp.status_code == 200, images_resp.text
    item = images_resp.json()["items"][0]
    assert item["url"].startswith(f"/data/imagesets/{imageset_id}/images/")
