from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_uploaded_video_metadata_uses_relative_path(app_client):
    client, app, _ = app_client

    resp = client.post(
        "/api/videos/upload",
        files={"file": ("demo.mp4", b"fake-video", "video/mp4")},
    )
    assert resp.status_code == 200, resp.text

    index_path = app.state.app_state.videos_index_file
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["videos"]
    stored = payload["videos"][0]["path"]
    assert not Path(stored).is_absolute()
    assert stored.startswith("uploads/videos/")


def test_imageset_metadata_uses_relative_dir_path(app_client, tmp_path: Path):
    client, app, _ = app_client
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image")

    resp = client.post(
        "/api/imagesets/upload-folder",
        files=[("files", ("sample.jpg", image_path.read_bytes(), "image/jpeg"))],
        data={"imageset_name": "relative_set"},
    )
    assert resp.status_code == 200, resp.text
    imageset_id = resp.json()["imageset_id"]

    imageset = app.state.app_state.get_imageset(imageset_id)
    assert imageset is not None
    meta_path = Path(imageset.dir_path) / "metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = payload["dir_path"]
    assert not Path(stored).is_absolute()
    assert stored == f"imagesets/{imageset_id}"


def test_model_metadata_uses_relative_model_path(app_client, monkeypatch: pytest.MonkeyPatch):
    client, app, _ = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])

    resp = client.post(
        "/api/models/upload",
        files={"model_file": ("demo.pt", b"pt-model", "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.text
    model_id = resp.json()["model_id"]

    model = app.state.app_state.get_model(model_id)
    assert model is not None
    meta_path = Path(model.model_path).parent / "metadata.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stored = payload["model_path"]
    assert not Path(stored).is_absolute()
    assert stored == f"models/{model_id}/demo.pt"
