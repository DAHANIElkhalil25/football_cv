"""
Module : kmeans_classifier.py
But    : Séparer les équipes par couleur de maillot (K-Means en HSV)
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026

Choix technique : HSV au lieu de RGB
- H (Hue / Teinte) : invariant aux changements d'éclairage
- S (Saturation) : distingue maillots colorés vs blancs
- V (Value / Luminosité) : IGNORÉ car varie avec l'ombre et l'éclairage
→ Le clustering sur (H, S) est robuste aux conditions de match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


@dataclass
class TeamAssignment:
	"""Affectation d'un joueur à une équipe/cluster."""

	track_id: int
	team_id: int
	color_hsv: Tuple[int, int, int]


class TeamColorKMeansClassifier:
	"""Classifieur KMeans basé sur la couleur torse en espace HSV.

	K=3 par défaut : Équipe A, Équipe B, Arbitres.
	Utilise uniquement les canaux H et S (ignore V = luminosité).
	"""

	def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
		self.n_clusters = n_clusters
		self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
		self.is_fitted = False

	@staticmethod
	def extract_jersey_patch(frame_bgr: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
		"""Extrait un patch torse (55% supérieur de la bbox) en HSV.

		Returns:
			Tableau (N, 2) contenant les valeurs (H, S) de chaque pixel.
			Retourne un tableau vide si la bbox est invalide.
		"""
		x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(frame_bgr.shape[1], x2)
		y2 = min(frame_bgr.shape[0], y2)
		if x2 <= x1 or y2 <= y1:
			return np.empty((0, 2), dtype=np.float32)

		h = y2 - y1
		torso_y2 = y1 + max(1, int(0.55 * h))
		patch = frame_bgr[y1:torso_y2, x1:x2]
		if patch.size == 0:
			return np.empty((0, 2), dtype=np.float32)

		# Convertir BGR → HSV, garder seulement H et S
		patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
		hs = patch_hsv[:, :, :2].reshape(-1, 2).astype(np.float32)
		return hs

	def fit_from_samples(self, samples_hs: np.ndarray) -> None:
		"""Entraîne le KMeans à partir de pixels (H, S) concaténés."""
		if samples_hs.ndim != 2 or samples_hs.shape[1] != 2 or len(samples_hs) < 10:
			raise ValueError("Échantillons (H, S) invalides pour KMeans.")
		self.model.fit(samples_hs)
		self.is_fitted = True

	def predict_team(self, sample_hs: np.ndarray) -> int:
		"""Prédit le cluster d'un patch (H, S).

		Calcule la moyenne (H, S) du patch puis assigne au cluster le plus proche.
		"""
		if not self.is_fitted:
			raise RuntimeError("Le classifieur doit être entraîné avant la prédiction.")
		center = sample_hs.mean(axis=0, keepdims=True)
		return int(self.model.predict(center)[0])

	def get_cluster_colors_hsv(self) -> np.ndarray:
		"""Retourne les centres des clusters en HSV (H, S, V=200 par défaut)."""
		if not self.is_fitted:
			raise RuntimeError("Le classifieur doit être entraîné avant.")
		centers_hs = self.model.cluster_centers_.astype(int)
		# Ajouter V=200 pour la visualisation
		centers_hsv = np.column_stack([centers_hs, np.full(self.n_clusters, 200)])
		return centers_hsv

	def get_cluster_colors_rgb(self) -> np.ndarray:
		"""Retourne les centres des clusters convertis en RGB (pour affichage)."""
		centers_hsv = self.get_cluster_colors_hsv()
		# Convertir chaque centre HSV → RGB via OpenCV
		colors_rgb = []
		for hsv in centers_hsv:
			pixel = np.array([[hsv]], dtype=np.uint8)
			rgb = cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0]
			colors_rgb.append(rgb)
		return np.array(colors_rgb)

	def assign_tracks(
		self,
		frame_bgr: np.ndarray,
		tracked_objects: List[Dict[str, object]],
	) -> List[TeamAssignment]:
		"""Affecte les objets trackés aux clusters selon la couleur HSV.

		Accepte les clés 'bbox_xyxy' ou 'bbox' pour compatibilité.
		Filtre uniquement les joueurs (class_id == 0).
		"""
		assignments: List[TeamAssignment] = []
		for obj in tracked_objects:
			bbox = obj.get("bbox_xyxy") or obj.get("bbox")
			track_id = int(obj.get("track_id", -1))
			class_id = int(obj.get("class_id", 0))
			if bbox is None or class_id != 0:
				continue
			patch_hs = self.extract_jersey_patch(frame_bgr, tuple(bbox))
			if patch_hs.size == 0:
				continue
			team_id = self.predict_team(patch_hs)
			mean_hs = patch_hs.mean(axis=0).astype(int)
			assignments.append(
				TeamAssignment(
					track_id=track_id,
					team_id=team_id,
					color_hsv=(int(mean_hs[0]), int(mean_hs[1]), 200),
				)
			)
		return assignments


__all__ = ["TeamAssignment", "TeamColorKMeansClassifier"]
