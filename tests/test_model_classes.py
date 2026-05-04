from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.services.model_service import ModelService


def test_parse_classes_file_variants() -> None:
    txt = ModelService.parse_classes_file("classes.txt", b"cat\ndog\n")
    assert txt == ["cat", "dog"]

    js = ModelService.parse_classes_file("classes.json", b"{\"names\": [\"a\", \"b\"]}")
    assert js == ["a", "b"]

    yml = ModelService.parse_classes_file("classes.yaml", b"names:\n  0: car\n  1: person\n")
    assert yml == ["car", "person"]


def test_onnx_requires_classes_if_model_has_none(app_state, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.model_service.extract_classes_from_model_file", lambda _: [])

    with pytest.raises(HTTPException) as exc:
        ModelService.save_model_upload(
            state=app_state,
            model_filename="demo.onnx",
            model_bytes=b"fake_onnx",
            classes_filename=None,
            classes_bytes=None,
        )

    assert exc.value.status_code == 400
    assert "ONNX 未解析到类别" in str(exc.value.detail)
