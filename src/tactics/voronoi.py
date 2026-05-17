"""
Module : voronoi.py
But    : Estimer les zones de contrôle d'espace par équipe
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026

Amélioration v2 :
- Ajout de compute_control_map_clipped() qui limite le Voronoi
  à la zone réellement visible par la caméra (au lieu du terrain complet).
  Cela évite les pourcentages artificiels quand les joueurs sont
  concentrés dans une seule zone du terrain.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def compute_control_map_clipped(
	team_a_xy: np.ndarray,
	team_b_xy: np.ndarray,
	visible_bounds: Optional[Tuple[float, float, float, float]] = None,
	grid_x: int = 80,
	grid_y: int = 52,
	margin_m: float = 5.0,
) -> Dict[str, object]:
	"""Voronoi clippé à la zone réellement visible par la caméra.

	Problème résolu : quand la caméra ne filme qu'une partie du terrain
	(ex: zone de penalty), le Voronoi sur 105×68m donne des pourcentages
	artificiels (76/24) car l'espace vide est attribué au joueur le plus
	proche, même à 50m de distance. Le clipping résout ça.

	Args:
		team_a_xy: Positions équipe A (N, 2) en mètres.
		team_b_xy: Positions équipe B (M, 2) en mètres.
		visible_bounds: (x_min, y_min, x_max, y_max) en mètres.
			Si None, calculé automatiquement depuis les positions des joueurs.
		grid_x: Résolution horizontale de la grille.
		grid_y: Résolution verticale de la grille.
		margin_m: Marge en mètres autour de la zone occupée.

	Returns:
		Dict avec dominant_team, team_a_ratio, team_b_ratio, bounds.
	"""
	all_xy = np.concatenate([
		xy for xy in [team_a_xy, team_b_xy]
		if len(xy) > 0
	])

	if len(all_xy) == 0:
		return {
			"dominant_team": np.full((grid_y, grid_x), -1, dtype=int),
			"team_a_ratio": 0.0,
			"team_b_ratio": 0.0,
			"bounds": (0, 0, 105, 68),
		}

	# Calculer les bornes automatiquement si non fournies
	if visible_bounds is None:
		x_min = max(0, float(all_xy[:, 0].min()) - margin_m)
		x_max = min(105, float(all_xy[:, 0].max()) + margin_m)
		y_min = max(0, float(all_xy[:, 1].min()) - margin_m)
		y_max = min(68, float(all_xy[:, 1].max()) + margin_m)
	else:
		x_min, y_min, x_max, y_max = visible_bounds

	# Grille limitée à la zone visible
	gx = np.linspace(x_min, x_max, grid_x)
	gy = np.linspace(y_min, y_max, grid_y)
	xx, yy = np.meshgrid(gx, gy)
	grid = np.stack([xx.ravel(), yy.ravel()], axis=1)

	if len(team_a_xy) == 0 or len(team_b_xy) == 0:
		dominant = np.full((grid_y, grid_x), -1, dtype=int)
		ratio = 0.0
		return {
			"dominant_team": dominant,
			"team_a_ratio": 0.0 if len(team_a_xy) == 0 else 1.0,
			"team_b_ratio": 0.0 if len(team_b_xy) == 0 else 1.0,
			"bounds": (x_min, y_min, x_max, y_max),
		}

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
		"bounds": (x_min, y_min, x_max, y_max),
	}


__all__ = ["compute_control_map", "compute_control_map_clipped"]
