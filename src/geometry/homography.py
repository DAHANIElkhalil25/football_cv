"""
Module : homography.py
But    : Calibration et projection image -> BEV
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026

Supporte les annotations de calibration SoccerNet (lignes de terrain)
pour estimer automatiquement l'homographie caméra → terrain.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# ================================================================
# Coordonnées réelles (mètres) des repères standard d'un terrain
# FIFA : 105m x 68m, surface de réparation 40.32m x 16.5m
# ================================================================
PITCH_LANDMARKS: Dict[str, Tuple[float, float]] = {
	"center": (52.5, 34.0),
	"top_left_corner": (0.0, 0.0),
	"top_right_corner": (105.0, 0.0),
	"bottom_left_corner": (0.0, 68.0),
	"bottom_right_corner": (105.0, 68.0),
	"left_penalty_spot": (11.0, 34.0),
	"right_penalty_spot": (94.0, 34.0),
	"left_goal_top": (0.0, 30.34),
	"left_goal_bottom": (0.0, 37.66),
	"right_goal_top": (105.0, 30.34),
	"right_goal_bottom": (105.0, 37.66),
	"left_box_top_right": (16.5, 13.84),
	"left_box_bottom_right": (16.5, 54.16),
	"right_box_top_left": (88.5, 13.84),
	"right_box_bottom_left": (88.5, 54.16),
	"halfway_top": (52.5, 0.0),
	"halfway_bottom": (52.5, 68.0),
}


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



def parse_soccernet_calibration(json_path: str) -> List[Dict]:
	"""Parse un fichier JSON d'annotations de calibration SoccerNet.

	Le format SoccerNet contient des segments de lignes annotés sur le terrain.
	Chaque ligne a une classe (ex: sideline, goal_line) et des coordonnées.

	Returns:
		Liste de dicts avec 'class', 'points_image' (pixels).
	"""
	with open(json_path) as f:
		data = json.load(f)

	lines = []
	for line_key, line_data in data.items():
		if not isinstance(line_data, dict):
			continue
		# Format possible : {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
		# ou liste de points
		points = []
		if "points" in line_data:
			coords = line_data["points"]
			for i in range(0, len(coords), 2):
				points.append((float(coords[i]), float(coords[i + 1])))
		elif "x" in line_data and "y" in line_data:
			# Points individuels
			points.append((float(line_data["x"]), float(line_data["y"])))

		if points:
			lines.append({
				"class": line_key,
				"points_image": points,
			})

	LOGGER.info("SoccerNet calibration: %d lignes parsées depuis %s", len(lines), json_path)
	return lines


def compute_reprojection_error(
	src_points: np.ndarray,
	dst_points: np.ndarray,
	homography: np.ndarray,
) -> Dict[str, float]:
	"""Calcule l'erreur de reprojection pour valider l'homographie.

	Projecte src_points via H et mesure la distance aux dst_points attendus.

	Returns:
		Dict avec mean_error, max_error, std_error (en mètres).
	"""
	projected = project_points(src_points, homography)
	errors = np.linalg.norm(projected - dst_points, axis=1)
	return {
		"mean_error": float(errors.mean()),
		"max_error": float(errors.max()),
		"std_error": float(errors.std()),
		"n_points": len(errors),
	}


__all__ = [
	"estimate_homography",
	"project_points",
	"project_bbox_footpoint",
	"save_homography",
	"load_homography",
	"parse_soccernet_calibration",
	"compute_reprojection_error",
	"PITCH_LANDMARKS",
]
