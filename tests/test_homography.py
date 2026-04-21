import numpy as np

from src.geometry.homography import estimate_homography, project_points


def test_identity_homography_projection() -> None:
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], dtype=np.float32)
    dst = src.copy()
    h = estimate_homography(src, dst)

    projected = project_points(np.array([[100.0, 50.0]], dtype=np.float32), h)
    assert abs(float(projected[0, 0]) - 100.0) < 1e-3
    assert abs(float(projected[0, 1]) - 50.0) < 1e-3
