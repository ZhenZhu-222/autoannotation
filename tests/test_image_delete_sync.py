from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def test_delete_image_removes_linked_label(app_state, tmp_path: Path) -> None:
    seed = tmp_path / "seed.jpg"
    cv2.imwrite(str(seed), np.zeros((32, 32, 3), dtype=np.uint8))

    imageset = app_state.create_imageset(name="set1", source="test", image_files=[seed])
    assert len(imageset.images) == 1

    image = imageset.images[0]
    image_path = Path(imageset.dir_path) / image.rel_path
    label_path = Path(imageset.dir_path) / "labels" / f"{image_path.stem}.txt"
    label_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    result = app_state.delete_image(image.id)

    assert result["deleted"]["image"] is True
    assert result["deleted"]["label"] is True
    assert not image_path.exists()
    assert not label_path.exists()
