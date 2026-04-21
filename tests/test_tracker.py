from pathlib import Path

import numpy as np
import pytest

from src.tracking.tracker import ByteTrackerWrapper


def test_tracker_init_missing_model(tmp_path) -> None:
    cfg = tmp_path / "bytetrack.yaml"
    cfg.write_text("tracker_type: bytetrack\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        ByteTrackerWrapper(
            model_path=str(tmp_path / "missing.pt"),
            bytetrack_cfg=str(cfg),
        )


def test_tracker_rejects_empty_frame(tmp_path) -> None:
    model = tmp_path / "model.pt"
    cfg = tmp_path / "bytetrack.yaml"
    model.write_text("fake", encoding="utf-8")
    cfg.write_text("tracker_type: bytetrack\n", encoding="utf-8")

    class _Dummy:
        def __call__(self, *args, **kwargs):
            return None

    tracker = object.__new__(ByteTrackerWrapper)
    tracker.model_path = model
    tracker.bytetrack_cfg = cfg
    tracker.device = "cpu"
    tracker.conf = 0.25
    tracker.iou = 0.7
    tracker.homography_matrix = None
    tracker.model = _Dummy()

    with pytest.raises(ValueError):
        tracker.update(np.array([]))
