import numpy as np

from src.geometry.dynamic_calibration import (
    histogram_similarity,
    detect_shot_boundaries,
    DynamicHomography,
    reprojection_drift,
)


def _solid(color):
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:] = color
    return img


def _textured(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)


def test_histogram_similarity_identical_is_high():
    a = _textured(1)
    assert histogram_similarity(a, a) > 0.99


def test_detect_shot_boundaries_flags_cut():
    frames = [_solid((20, 120, 40)), _solid((20, 120, 40)), _solid((200, 30, 30))]
    bounds = detect_shot_boundaries(frames, similarity_threshold=0.6)
    assert 0 in bounds and 2 in bounds


def test_dynamic_homography_identity_translation():
    # Reference homography = identity; shift the textured image by a few pixels.
    base = _textured(7)
    H = np.eye(3, dtype=np.float64)
    dh = DynamicHomography(H, base, min_tracked_points=10)
    shifted = np.roll(base, shift=2, axis=1)
    H_t = dh.update(shifted)
    assert H_t is not None
    assert H_t.shape == (3, 3)


def test_reprojection_drift_keys():
    H = np.eye(3, dtype=np.float64)
    pts = np.array([[52.5, 34.0], [0.0, 0.0], [105.0, 68.0]], dtype=np.float32)
    out = reprojection_drift(pts, [H, H, H], H)
    assert {"mean_correction_px", "max_correction_px", "n_frames", "n_points"} <= set(out)
    assert out["mean_correction_px"] < 1e-3  # static == dynamic here
