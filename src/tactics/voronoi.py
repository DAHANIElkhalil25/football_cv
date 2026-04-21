"""
Module : voronoi.py
But    : Estimer les zones de contrôle d'espace par équipe
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_control_map(
	team_a_xy: np.ndarray,
	team_b_xy: np.ndarray,
	pitch_width: float = 105.0,
	pitch_height: float = 68.0,
	grid_x: int = 80,
	grid_y: int = 52,
) -> Dict[str, np.ndarray | float]:
	"""Calcule une carte de contrôle simplifiée (plus proche joueur).

	Returns:
		Dictionnaire avec grille équipe dominante et ratios de surface.
	"""
	gx = np.linspace(0.0, pitch_width, grid_x)
	gy = np.linspace(0.0, pitch_height, grid_y)
	xx, yy = np.meshgrid(gx, gy)
	grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

	if len(team_a_xy) == 0 or len(team_b_xy) == 0:
		dominant = np.full((grid_y, grid_x), -1, dtype=int)
		return {"dominant_team": dominant, "team_a_ratio": 0.0, "team_b_ratio": 0.0}

	dist_a = np.linalg.norm(grid[:, None, :] - team_a_xy[None, :, :], axis=2).min(axis=1)
	dist_b = np.linalg.norm(grid[:, None, :] - team_b_xy[None, :, :], axis=2).min(axis=1)
	dominant_flat = (dist_b < dist_a).astype(int)
	dominant = dominant_flat.reshape(grid_y, grid_x)

	team_b_ratio = float(dominant_flat.mean())
	team_a_ratio = 1.0 - team_b_ratio
	return {
		"dominant_team": dominant,
		"team_a_ratio": team_a_ratio,
		"team_b_ratio": team_b_ratio,
	}


__all__ = ["compute_control_map"]

