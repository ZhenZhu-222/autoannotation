from __future__ import annotations


def test_system_readiness_reports_operational_state(app_client) -> None:
    client, _, tmp_path = app_client
    front_dir = tmp_path / "front"
    for name in [
        "style.css",
        "ui_utils.js",
        "api.js",
        "dataset_upload.js",
        "imageset_preview.js",
        "gallery.js",
        "class_mapping.js",
        "annotate.js",
        "qwen_annotate.js",
        "jobs_history.js",
        "refine.js",
        "train.js",
        "app.js",
        "preview.html",
        "preview.js",
    ]:
        (front_dir / name).write_text("ok", encoding="utf-8")

    resp = client.get("/api/system/readiness")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["operational"] is True
    assert data["status"] in {"ok", "needs_attention"}
    assert data["acceptance"]["minimum_test_command"] == "PYTHONPATH=. pytest -q"
    assert all(item["ok"] for item in data["checks"]["writable_dirs"])
    front_names = {item["name"]: item["ok"] for item in data["checks"]["front_files"]}
    assert front_names["index.html"] is True
