from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    front_dir = tmp_path / "front"
    upload_videos = data_dir / "uploads" / "videos"
    qwen_refs = data_dir / "uploads" / "qwen_refs"
    imagesets = data_dir / "imagesets"
    models = data_dir / "models"
    outputs = data_dir / "outputs"
    history_dir = data_dir / "history"
    jobs_history_file = history_dir / "jobs_history.jsonl"

    front_dir.mkdir(parents=True, exist_ok=True)
    (front_dir / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")

    import app.core.config as config
    import app.core.state as state_module
    import app.core.auth as auth_module
    import app.main as main_module
    import app.services.annotate_preview_service as annotate_preview_service_module
    import app.services.annotation_service as annotation_service_module
    import app.services.class_name_service as class_name_service_module
    import app.services.imageset_service as imageset_service_module
    import app.services.job_cleanup_service as job_cleanup_service_module
    import app.services.model_service as model_service_module
    import app.services.qwen_annotation_service as qwen_annotation_service_module
    import app.services.qwen_upload_service as qwen_upload_service_module
    import app.services.task_manager as task_manager_module
    import app.api.routes_videos as routes_videos_module

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "FRONT_DIR", front_dir)
    monkeypatch.setattr(config, "UPLOAD_VIDEOS_DIR", upload_videos)
    monkeypatch.setattr(config, "QWEN_REFS_DIR", qwen_refs)
    monkeypatch.setattr(config, "IMAGESETS_DIR", imagesets)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(config, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(config, "JOBS_HISTORY_FILE", jobs_history_file)
    monkeypatch.setattr(config, "SYSTEM_DIR", data_dir / "system")
    monkeypatch.setattr(config, "LICENSE_FILE", data_dir / "system" / "license.dat")
    monkeypatch.setattr(config, "TRIAL_STATE_FILE", data_dir / "system" / "trial_state.dat")
    monkeypatch.setattr(config, "USERS_DIR", data_dir / "users")
    monkeypatch.setattr(config, "USERS_INDEX_FILE", data_dir / "users" / "index.json")
    monkeypatch.setattr(config, "LOGIN_REQUIRED", False)
    monkeypatch.setattr(config, "ADMIN_USERNAME", "root")
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "herefly")

    monkeypatch.setattr(auth_module, "LOGIN_REQUIRED", False)
    monkeypatch.setattr(auth_module, "ADMIN_USERNAME", "root")
    monkeypatch.setattr(auth_module, "ADMIN_PASSWORD", "herefly")

    monkeypatch.setattr(main_module, "LOGIN_REQUIRED", False)
    monkeypatch.setattr(main_module, "SESSION_COOKIE_NAME", "autoannotation_session")

    monkeypatch.setattr(state_module, "UPLOAD_VIDEOS_DIR", upload_videos)
    monkeypatch.setattr(state_module, "IMAGESETS_DIR", imagesets)
    monkeypatch.setattr(state_module, "MODELS_DIR", models)

    monkeypatch.setattr(model_service_module, "MODELS_DIR", models)
    monkeypatch.setattr(class_name_service_module, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(imageset_service_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(job_cleanup_service_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(annotate_preview_service_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(annotation_service_module, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(qwen_annotation_service_module, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(qwen_upload_service_module, "QWEN_REFS_DIR", qwen_refs)
    monkeypatch.setattr(task_manager_module, "JOBS_HISTORY_FILE", jobs_history_file)

    monkeypatch.setattr(routes_videos_module, "UPLOAD_VIDEOS_DIR", upload_videos)

    monkeypatch.setattr(main_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(main_module, "FRONT_DIR", front_dir)

    config.ensure_dirs()

    app = main_module.create_app()
    client = TestClient(app)
    try:
        yield client, app, tmp_path
    finally:
        client.close()


@pytest.fixture()
def app_state(app_client):
    _, app, _ = app_client
    return app.state.app_state


def wait_job(client: TestClient, path: str, timeout_s: float = 15.0) -> dict:
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(path)
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        if payload["status"] in {"succeeded", "failed"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"Job timeout: {path}")
