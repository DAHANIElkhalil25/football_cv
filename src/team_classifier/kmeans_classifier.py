"""
Module : kmeans_classifier.py
But    : Séparer les équipes par couleur de maillot (K-Means)
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamAssignment:
	"""Affectation d'un joueur à une équipe couleur."""

	track_id: int
	team_id: int
	color_rgb: Tuple[int, int, int]


class TeamColorKMeansClassifier:
	"""Classifieur KMeans basé sur la couleur torse des joueurs."""

	def __init__(self, n_clusters: int = 2, random_state: int = 42) -> None:
		self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
		self.is_fitted = False

	@staticmethod
	def extract_jersey_patch(frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
		"""Extrait un patch torse (partie haute de la bbox)."""
		x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(frame_bgr.shape[1], x2)
		y2 = min(frame_bgr.shape[0], y2)
		if x2 <= x1 or y2 <= y1:
			return np.empty((0, 3), dtype=np.uint8)

		h = y2 - y1
		torso_y2 = y1 + max(1, int(0.55 * h))
		patch = frame_bgr[y1:torso_y2, x1:x2]
		if patch.size == 0:
			return np.empty((0, 3), dtype=np.uint8)

		patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
		return patch_rgb.reshape(-1, 3)

	def fit_from_samples(self, samples_rgb: np.ndarray) -> None:
		"""Entraîne le KMeans à partir de pixels RGB concaténés."""
		if samples_rgb.ndim != 2 or samples_rgb.shape[1] != 3 or len(samples_rgb) < 10:
			raise ValueError("Échantillons RGB invalides pour KMeans.")
		self.model.fit(samples_rgb)
		self.is_fitted = True

	def predict_team(self, sample_rgb: np.ndarray) -> int:
		"""Prédit l'équipe d'un patch RGB (pixels)."""
		if not self.is_fitted:
			raise RuntimeError("Le classifieur doit être entraîné avant la prédiction.")
		center = sample_rgb.mean(axis=0, keepdims=True)
		return int(self.model.predict(center)[0])

	def assign_tracks(
		self,
		frame_bgr: np.ndarray,
		tracked_objects: List[Dict[str, object]],
	) -> List[TeamAssignment]:
		"""Affecte les objets trackés aux équipes selon la couleur.

		Accepte les clés 'bbox_xyxy' ou 'bbox' pour compatibilité.
		Filtre uniquement les joueurs (class_id == 0) par défaut.
		"""
		assignments: List[TeamAssignment] = []
		for obj in tracked_objects:
			bbox = obj.get("bbox_xyxy") or obj.get("bbox")
			track_id = int(obj.get("track_id", -1))
			class_id = int(obj.get("class_id", 0))
			if bbox is None or class_id != 0:
				continue
			patch = self.extract_jersey_patch(frame_bgr, tuple(bbox))
			if patch.size == 0:
				continue
			team_id = self.predict_team(patch)
			color = patch.mean(axis=0).astype(int)
			assignments.append(
				TeamAssignment(
					track_id=track_id,
					team_id=team_id,
					color_rgb=(int(color[0]), int(color[1]), int(color[2])),
				)
			)
		return assignments


__all__ = ["TeamAssignment", "TeamColorKMeansClassifier"]

