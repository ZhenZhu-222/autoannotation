from __future__ import annotations

from app.services.inference import Detection, Predictor, create_predictor, get_predictor_factory, register_predictor_factory


class _FakePredictor:
    class_names = ["fake"]
    task = "detect"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, image_path: str, conf: float, iou: float, device: str = "") -> list[Detection]:
        _ = (image_path, conf, iou, device)
        return []


def test_predictor_factory_can_register_extension(monkeypatch) -> None:
    from app.services import inference

    factories_before = dict(inference._PREDICTOR_FACTORIES)
    monkeypatch.setattr(inference, "_PREDICTOR_FACTORIES", factories_before)

    register_predictor_factory("fake", _FakePredictor)

    assert get_predictor_factory("demo.fake") is _FakePredictor
    predictor = create_predictor("demo.fake")
    assert isinstance(predictor, Predictor)
    assert predictor.class_names == ["fake"]
    assert predictor.task == "detect"


def test_predictor_factory_defaults_unknown_suffix_to_yolo() -> None:
    from app.services.inference import YoloPredictor

    assert get_predictor_factory("demo.unknown") is YoloPredictor
