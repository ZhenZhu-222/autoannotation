from __future__ import annotations

from urllib.parse import unquote

import pytest


def test_download_model_appends_weight_extension_after_rename(
    app_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, _ = app_client
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: ["target"])

    upload_resp = client.post(
        "/api/models/upload",
        files={"model_file": ("source.pt", b"pt-model", "application/octet-stream")},
    )
    assert upload_resp.status_code == 200, upload_resp.text
    model_id = upload_resp.json()["model_id"]

    rename_resp = client.patch(f"/api/models/{model_id}/name", json={"name": "无证摊贩"})
    assert rename_resp.status_code == 200, rename_resp.text

    download_resp = client.get(f"/api/models/{model_id}/download")
    assert download_resp.status_code == 200, download_resp.text
    content_disposition = unquote(download_resp.headers["content-disposition"])

    assert "无证摊贩.pt" in content_disposition
    assert download_resp.content == b"pt-model"
