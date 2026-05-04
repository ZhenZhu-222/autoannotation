from __future__ import annotations

from pathlib import Path

from app.services import imageset_service
from app.services.imageset_service import ImageSetService


class _HistoryTasks:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def list_history(self, limit: int = 500) -> list[dict]:
        return self._rows[:limit]


def test_find_latest_job_skips_missing_gallery_artifacts(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    good_manifest = data_dir / "outputs" / "good-job" / "manifest.csv"
    good_manifest.parent.mkdir(parents=True)
    good_manifest.write_text("image_id,output_image\n", encoding="utf-8")
    monkeypatch.setattr(imageset_service, "DATA_DIR", data_dir)

    tasks = _HistoryTasks(
        [
            {
                "job_id": "missing-job",
                "job_type": "annotate",
                "status": "succeeded",
                "result_summary": {
                    "imageset_id": "set-1",
                    "artifacts": {"manifest_csv": "/data/outputs/missing-job/manifest.csv"},
                },
            },
            {
                "job_id": "good-job",
                "job_type": "qwen_annotate",
                "status": "succeeded",
                "result_summary": {
                    "imageset_id": "set-1",
                    "artifacts": {"manifest_csv": "/data/outputs/good-job/manifest.csv"},
                },
            },
        ]
    )

    assert ImageSetService._find_latest_job("set-1", tasks) == "good-job"
