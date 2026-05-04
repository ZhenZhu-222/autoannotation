from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


def _build_secured_client(tmp_path: Path, monkeypatch):
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
    (front_dir / "index.html").write_text("<html><body>secured</body></html>", encoding="utf-8")
    (front_dir / "auth.html").write_text("auth", encoding="utf-8")
    (front_dir / "license.html").write_text("license", encoding="utf-8")
    (front_dir / "style.css").write_text("body{}", encoding="utf-8")
    (front_dir / "app.js").write_text("console.log('ok')", encoding="utf-8")
    (front_dir / "auth.js").write_text("console.log('ok')", encoding="utf-8")
    (front_dir / "license.js").write_text("console.log('ok')", encoding="utf-8")

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
    monkeypatch.setattr(config, "LOGIN_REQUIRED", True)
    monkeypatch.setattr(config, "ADMIN_USERNAME", "root")
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "herefly")
    monkeypatch.setattr(config, "SESSION_COOKIE_NAME", "autoannotation_session")

    monkeypatch.setattr(auth_module, "LOGIN_REQUIRED", True)
    monkeypatch.setattr(auth_module, "ADMIN_USERNAME", "root")
    monkeypatch.setattr(auth_module, "ADMIN_PASSWORD", "herefly")

    monkeypatch.setattr(main_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(main_module, "FRONT_DIR", front_dir)
    monkeypatch.setattr(main_module, "LOGIN_REQUIRED", True)
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

    config.ensure_dirs()
    app = main_module.create_app()
    return TestClient(app)


def test_root_requires_login(tmp_path: Path, monkeypatch):
    client = _build_secured_client(tmp_path, monkeypatch)
    try:
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/front/auth.html"

        bad = client.post("/api/system/login", json={"username": "root", "password": "wrong"})
        assert bad.status_code == 401

        ok = client.post("/api/system/login", json={"username": "root", "password": "herefly"})
        assert ok.status_code == 200, ok.text
        assert ok.json()["ok"] is True

        home = client.get("/")
        assert home.status_code == 200
        assert "secured" in home.text
    finally:
        client.close()


def test_trial_allows_login_and_legacy_unlock_is_gone(tmp_path: Path, monkeypatch):
    client = _build_secured_client(tmp_path, monkeypatch)
    try:
        login = client.post("/api/system/login", json={"username": "root", "password": "herefly"})
        assert login.status_code == 200, login.text
        assert login.json()["license"]["mode"] == "trial"
        assert login.json()["license"]["locked"] is False

        jobs = client.get("/api/jobs")
        assert jobs.status_code == 200, jobs.text

        legacy = client.post("/api/system/license/unlock", json={"key": "nanguijian"})
        assert legacy.status_code == 410
    finally:
        client.close()
