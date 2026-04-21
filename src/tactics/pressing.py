"""
Module : pressing.py
But    : Estimer un indicateur de pressing type PPDA
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def estimate_ppda(
	opponent_passes: int,
	defensive_actions: int,
	epsilon: float = 1e-6,
) -> float:
	"""Calcule le PPDA simple: passes adverses / actions défensives.

	Plus la valeur est faible, plus le pressing est intense.
	"""
	if opponent_passes < 0 or defensive_actions < 0:
		raise ValueError("Les compteurs doivent être positifs.")
	return float(opponent_passes) / float(defensive_actions + epsilon)


def estimate_defensive_actions_from_tracking(
	defender_xy: np.ndarray,
	ball_xy: np.ndarray,
	pressure_radius_m: float = 4.0,
) -> int:
	"""Approxime les actions défensives par proximité balle-défenseur.

	Args:
		defender_xy: Tableau (N,2) des défenseurs.
		ball_xy: Tableau (M,2) de la position ballon par frame.
		pressure_radius_m: Rayon de pression en mètres.

	Returns:
		Nombre d'événements de pression estimés.
	"""
	if len(defender_xy) == 0 or len(ball_xy) == 0:
		return 0

	actions = 0
	for b in ball_xy:
		distances = np.linalg.norm(defender_xy - b[None, :], axis=1)
		if float(distances.min()) <= pressure_radius_m:
			actions += 1
	return actions


def pressing_summary(opponent_passes: int, defensive_actions: int) -> Dict[str, float | str]:
	"""Retourne un résumé interprétable du pressing."""
	ppda = estimate_ppda(opponent_passes=opponent_passes, defensive_actions=defensive_actions)
	if ppda < 8.0:
		level = "high"
	elif ppda < 12.0:
		level = "medium"
	else:
		level = "low"
	return {"ppda": ppda, "pressing_level": level}


__all__ = ["estimate_ppda", "estimate_defensive_actions_from_tracking", "pressing_summary"]

