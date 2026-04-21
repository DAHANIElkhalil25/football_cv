"""
Module : compactness.py
But    : Mesurer la compacité défensive d'une équipe
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial import ConvexHull


def team_centroid(team_xy: np.ndarray) -> np.ndarray:
	"""Calcule le centroïde d'une équipe."""
	if team_xy.size == 0:
		return np.array([0.0, 0.0], dtype=float)
	return team_xy.mean(axis=0)


def compactness_metrics(team_xy: np.ndarray) -> Dict[str, float]:
	"""Calcule plusieurs métriques de compacité.

	Returns:
		- mean_distance_to_centroid
		- max_pairwise_distance
		- convex_hull_area
	"""
	if team_xy.ndim != 2 or team_xy.shape[1] != 2:
		raise ValueError("`team_xy` doit avoir la forme (N,2).")
	if team_xy.shape[0] < 2:
		return {
			"mean_distance_to_centroid": 0.0,
			"max_pairwise_distance": 0.0,
			"convex_hull_area": 0.0,
		}

	center = team_centroid(team_xy)
	mean_dist = float(np.linalg.norm(team_xy - center[None, :], axis=1).mean())

	diffs = team_xy[:, None, :] - team_xy[None, :, :]
	max_pair = float(np.linalg.norm(diffs, axis=2).max())

	if team_xy.shape[0] >= 3:
		hull = ConvexHull(team_xy)
		hull_area = float(hull.volume)
	else:
		hull_area = 0.0

	return {
		"mean_distance_to_centroid": mean_dist,
		"max_pairwise_distance": max_pair,
		"convex_hull_area": hull_area,
	}


__all__ = ["team_centroid", "compactness_metrics"]

