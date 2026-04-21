"""
Module : homography.py
But    : Calibration et projection image -> BEV
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def estimate_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
	"""Estime la matrice d'homographie 3x3.

	Args:
		src_points: Points source (N,2) dans l'image caméra.
		dst_points: Points destination (N,2) dans le plan terrain.

	Returns:
		Matrice d'homographie (3,3).
	"""
	if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
		raise ValueError("Au moins 4 points sont nécessaires pour estimer l'homographie.")

	h, mask = cv2.findHomography(src_points.astype(np.float32), dst_points.astype(np.float32), method=cv2.RANSAC)
	if h is None:
		raise RuntimeError("Échec d'estimation de l'homographie.")

	inliers = int(mask.sum()) if mask is not None else 0
	LOGGER.info("Homographie estimée avec %d inliers.", inliers)
	return h


def project_points(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
	"""Projette des points 2D via homographie.

	Args:
		points_xy: Tableau (N,2).
		homography: Matrice 3x3.

	Returns:
		Tableau (N,2) projeté.
	"""
	if points_xy.ndim != 2 or points_xy.shape[1] != 2:
		raise ValueError("`points_xy` doit avoir la forme (N,2).")

	src = points_xy.reshape(-1, 1, 2).astype(np.float32)
	dst = cv2.perspectiveTransform(src, homography)
	return dst.reshape(-1, 2)


def project_bbox_footpoint(bbox_xyxy: Tuple[float, float, float, float], homography: np.ndarray) -> Tuple[float, float]:
	"""Projette le pied d'une bbox (milieu-bas) vers le plan BEV."""
	x1, y1, x2, y2 = bbox_xyxy
	point = np.array([[[(x1 + x2) / 2.0, y2]]], dtype=np.float32)
	dst = cv2.perspectiveTransform(point, homography)
	return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def save_homography(homography: np.ndarray, file_path: str) -> str:
	"""Sauvegarde une matrice d'homographie (.npy)."""
	target = Path(file_path).expanduser().resolve()
	target.parent.mkdir(parents=True, exist_ok=True)
	np.save(str(target), homography)
	return str(target)


def load_homography(file_path: str) -> np.ndarray:
	"""Charge une matrice d'homographie depuis un `.npy`."""
	source = Path(file_path).expanduser().resolve()
	if not source.exists():
		raise FileNotFoundError(f"Fichier homographie introuvable: {source}")
	return np.load(str(source))


__all__ = [
	"estimate_homography",
	"project_points",
	"project_bbox_footpoint",
	"save_homography",
	"load_homography",
]

