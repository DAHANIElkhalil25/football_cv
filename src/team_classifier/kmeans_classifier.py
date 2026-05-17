"""
Module : kmeans_classifier.py
But    : Séparer les équipes par couleur de maillot (K-Means en HSV)
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026

Améliorations v2 :
- Patch torse précis : skip la tête (top 20%) et les shorts (bottom 40%)
  → zone 20%-60% de la bbox = uniquement le maillot
- Entraînement sur les MOYENNES (H, S) par joueur, pas les pixels bruts
  → K-Means reçoit ~N points propres au lieu de millions de pixels bruités
- HSV : H (teinte) invariant à l'éclairage, S (saturation) sépare blanc/coloré
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
	"""Classifieur KMeans basé sur la couleur maillot en espace HSV.

	K=3 par défaut : Équipe A, Équipe B, Arbitres.
	Utilise uniquement les canaux H et S (ignore V = luminosité).
	"""

	def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
		self.n_clusters = n_clusters
		self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
		self.is_fitted = False

	@staticmethod
	def extract_jersey_patch(
		frame_bgr: np.ndarray,
		bbox_xyxy: Tuple[float, float, float, float],
	) -> np.ndarray:
		"""Extrait les pixels (H, S) du MAILLOT uniquement.

		Zone extraite : 20%-60% vertical de la bbox
		- Skip le top 20% (tête/cheveux → même couleur pour tous)
		- Skip le bottom 40% (shorts/jambes → couleur différente du maillot)
		- Garde seulement la zone torse = le maillot

		Returns:
			Tableau (N, 2) de valeurs (H, S). Vide si bbox invalide.
		"""
		x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(frame_bgr.shape[1], x2)
		y2 = min(frame_bgr.shape[0], y2)
		if x2 <= x1 or y2 <= y1:
			return np.empty((0, 2), dtype=np.float32)

		h = y2 - y1
		# Zone maillot : de 20% à 60% de la hauteur de la bbox
		jersey_y1 = y1 + max(1, int(0.20 * h))
		jersey_y2 = y1 + max(2, int(0.60 * h))

		# Réduire aussi horizontalement (10% de marge) pour éviter les bras/fond
		w = x2 - x1
		jersey_x1 = x1 + max(0, int(0.10 * w))
		jersey_x2 = x2 - max(0, int(0.10 * w))

		if jersey_x2 <= jersey_x1 or jersey_y2 <= jersey_y1:
			return np.empty((0, 2), dtype=np.float32)

		patch = frame_bgr[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
		if patch.size == 0:
			return np.empty((0, 2), dtype=np.float32)

		patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
		hs = patch_hsv[:, :, :2].reshape(-1, 2).astype(np.float32)
		return hs

	@staticmethod
	def _patch_mean(patch_hs: np.ndarray) -> Optional[np.ndarray]:
		"""Calcule la couleur moyenne (H, S) d'un patch."""
		if patch_hs.size == 0:
			return None
		return patch_hs.mean(axis=0)

	def collect_mean_colors(
		self,
		frames_data: List[Dict],
		frame_reader,
	) -> np.ndarray:
		"""Collecte les couleurs MOYENNES (H, S) de chaque joueur détecté.

		Args:
			frames_data: Liste de dicts avec 'frame_path' et 'tracks'.
			frame_reader: Fonction qui lit une frame (ex: cv2.imread).

		Returns:
			Tableau (M, 2) avec une couleur moyenne par détection joueur.
		"""
		mean_colors = []
		for ft in frames_data:
			frame = frame_reader(ft["frame_path"])
			for t in ft["tracks"]:
				if int(t.get("class_id", -1)) != 0:
					continue
				bbox = t.get("bbox_xyxy") or t.get("bbox")
				if bbox is None:
					continue
				patch_hs = self.extract_jersey_patch(frame, tuple(bbox))
				mean = self._patch_mean(patch_hs)
				if mean is not None:
					mean_colors.append(mean)
		return np.array(mean_colors, dtype=np.float32)

	def fit_from_samples(self, samples_hs: np.ndarray) -> None:
		"""Entraîne le KMeans à partir de couleurs moyennes (H, S)."""
		if samples_hs.ndim != 2 or samples_hs.shape[1] != 2 or len(samples_hs) < 10:
			raise ValueError("Échantillons (H, S) invalides pour KMeans.")
		self.model.fit(samples_hs)
		self.is_fitted = True

	def predict_team(self, patch_hs: np.ndarray) -> int:
		"""Prédit le cluster d'un patch via sa couleur moyenne (H, S)."""
		if not self.is_fitted:
			raise RuntimeError("Le classifieur doit être entraîné avant la prédiction.")
		mean = patch_hs.mean(axis=0, keepdims=True)
		return int(self.model.predict(mean)[0])

	def get_cluster_colors_hsv(self) -> np.ndarray:
		"""Retourne les centres des clusters en HSV complet (H, S, V=200)."""
		if not self.is_fitted:
			raise RuntimeError("Le classifieur doit être entraîné avant.")
		centers_hs = self.model.cluster_centers_.astype(int)
		centers_hsv = np.column_stack([centers_hs, np.full(self.n_clusters, 200)])
		return centers_hsv

	def get_cluster_colors_rgb(self) -> np.ndarray:
		"""Retourne les centres des clusters convertis en RGB (pour affichage)."""
		centers_hsv = self.get_cluster_colors_hsv()
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
		"""Affecte les joueurs (class_id=0) aux clusters par couleur HSV."""
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

	def build_temporal_voting(
		self,
		all_tracks: List[Dict],
		frame_reader,
	) -> Dict[int, int]:
		"""Vote temporel : assigne chaque track_id à l'équipe la plus fréquente.

		Problème résolu : sans vote temporel, un même joueur peut changer
		d'équipe entre les frames (flickering) à cause du bruit sur la couleur.
		Solution : on collecte TOUTES les prédictions pour chaque track_id
		sur toute la séquence, puis on prend le vote majoritaire.

		Args:
			all_tracks: Liste complète [{frame_idx, frame_path, tracks}, ...].
			frame_reader: Fonction de lecture (ex: cv2.imread).

		Returns:
			Dict {track_id: team_id} stable pour toute la séquence.
		"""
		from collections import Counter

		votes: Dict[int, list] = {}
		for ft in all_tracks:
			frame = frame_reader(ft["frame_path"])
			assignments = self.assign_tracks(frame, ft["tracks"])
			for a in assignments:
				if a.track_id not in votes:
					votes[a.track_id] = []
				votes[a.track_id].append(a.team_id)

		# Vote majoritaire par track_id
		stable_map: Dict[int, int] = {}
		for tid, team_votes in votes.items():
			counter = Counter(team_votes)
			stable_map[tid] = counter.most_common(1)[0][0]

		return stable_map


__all__ = ["TeamAssignment", "TeamColorKMeansClassifier"]
