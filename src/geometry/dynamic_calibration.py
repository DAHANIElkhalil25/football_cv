"""
Module : dynamic_calibration.py
But    : Calibration caméra DYNAMIQUE pour flux broadcast (sans modèle appris)
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026

Problème résolu
---------------
Le pipeline initial calcule la matrice d'homographie H une seule fois sur une
frame de référence, puis l'applique à toutes les frames. Or une caméra broadcast
effectue des travellings (gauche/droite) et des zooms : H devient obsolète et les
joueurs « glissent » sur le terrain BEV.

Ce module maintient une homographie image -> terrain À JOUR à chaque frame, en
n'utilisant QUE des outils OpenCV (aucun GPU, aucun jeu de données, aucun
entraînement) :

1. Détection de coupures de plan (corrélation d'histogrammes HSV) ;
2. Suivi de points de repère du terrain par flux optique de Lucas-Kanade ;
3. Ré-estimation de l'homographie image_courante -> image_référence puis
   composition avec référence -> terrain ;
4. Ré-ancrage automatique quand trop de points sont perdus ;
5. Lissage temporel et garde-fou sur l'erreur de reprojection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


# ================================================================
# Détection de coupures de plan (shot-cut detection)
# ================================================================
def histogram_similarity(frame_a_bgr: np.ndarray, frame_b_bgr: np.ndarray) -> float:
    """Similarité [0,1] entre deux frames via corrélation d'histogrammes HSV.

    Une valeur proche de 1 = même plan ; une chute brutale = coupure de caméra.
    """
    a = cv2.cvtColor(frame_a_bgr, cv2.COLOR_BGR2HSV)
    b = cv2.cvtColor(frame_b_bgr, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, score)))


def detect_shot_boundaries(
    frames_bgr: List[np.ndarray],
    similarity_threshold: float = 0.6,
) -> List[int]:
    """Renvoie les indices de frames marquant le début d'un NOUVEAU plan.

    Args:
        frames_bgr: Séquence de frames BGR.
        similarity_threshold: En dessous, on déclare une coupure.

    Returns:
        Liste d'indices (la frame 0 est toujours un début de plan).
    """
    boundaries = [0]
    for i in range(1, len(frames_bgr)):
        if histogram_similarity(frames_bgr[i - 1], frames_bgr[i]) < similarity_threshold:
            boundaries.append(i)
    return boundaries


# ================================================================
# Homographie dynamique par flux optique
# ================================================================
@dataclass
class DynamicHomographyState:
    """État interne du suiveur d'homographie."""

    current_H: np.ndarray
    anchor_H: np.ndarray                  # référence -> terrain
    anchor_pts: np.ndarray               # points dans l'image d'ancrage (N,1,2)
    live_pts: np.ndarray                 # mêmes points dans l'image courante
    prev_gray: np.ndarray
    valid: bool = True
    reanchor_count: int = 0
    history: List[np.ndarray] = field(default_factory=list)


class DynamicHomography:
    """Maintient une homographie image -> terrain à jour le long d'un plan.

    Utilisation
    -----------
    >>> dh = DynamicHomography(H_ref, ref_frame_bgr)
    >>> for frame in frames[1:]:
    ...     H_t = dh.update(frame)          # H valable pour CETTE frame
    ...     pts_bev = cv2.perspectiveTransform(pts_img, H_t)

    Lors d'une coupure de plan, appeler `reset(new_H, new_frame_bgr)` avec une
    nouvelle homographie de référence (recalculée pour le nouveau plan).
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(
        self,
        reference_homography: np.ndarray,
        reference_frame_bgr: np.ndarray,
        max_corners: int = 400,
        quality_level: float = 0.01,
        min_distance: int = 8,
        min_tracked_points: int = 25,
        ransac_thresh: float = 3.0,
        smooth_window: int = 5,
    ) -> None:
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_tracked_points = min_tracked_points
        self.ransac_thresh = ransac_thresh
        self.smooth_window = max(1, smooth_window)
        self.reset(reference_homography, reference_frame_bgr)

    # --------------------------------------------------------------
    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return pts.astype(np.float32)

    def reset(self, reference_homography: np.ndarray, reference_frame_bgr: np.ndarray) -> None:
        """(Ré)ancre le suiveur sur un nouveau plan."""
        H = np.asarray(reference_homography, dtype=np.float64)
        gray = cv2.cvtColor(reference_frame_bgr, cv2.COLOR_BGR2GRAY)
        pts = self._detect_features(gray)
        self.state = DynamicHomographyState(
            current_H=H.copy(),
            anchor_H=H.copy(),
            anchor_pts=pts,
            live_pts=pts.copy(),
            prev_gray=gray,
            valid=pts.shape[0] >= 4,
            history=[H.copy()],
        )

    # --------------------------------------------------------------
    def _smooth(self, H: np.ndarray) -> np.ndarray:
        """Lissage temporel des paramètres de H (moyenne glissante normalisée)."""
        self.state.history.append(H.copy())
        if len(self.state.history) > self.smooth_window:
            self.state.history.pop(0)
        stack = np.stack([h / h[2, 2] for h in self.state.history], axis=0)
        return stack.mean(axis=0)

    def _reanchor(self, gray: np.ndarray) -> None:
        """Repart de la frame courante comme nouvelle ancre (sans coupure)."""
        self.state.anchor_H = self.state.current_H.copy()
        self.state.anchor_pts = self._detect_features(gray)
        self.state.live_pts = self.state.anchor_pts.copy()
        self.state.reanchor_count += 1

    def update(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Met à jour et renvoie l'homographie image -> terrain pour cette frame.

        Returns:
            Matrice (3,3) lissée, ou None si l'estimation est impossible
            (le plan a probablement changé : appeler `reset`).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        st = self.state

        if st.live_pts.shape[0] < 4:
            self._reanchor(gray)
            st.prev_gray = gray
            return st.current_H

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            st.prev_gray, gray, st.live_pts, None, **self.LK_PARAMS
        )
        if new_pts is None or status is None:
            st.valid = False
            st.prev_gray = gray
            return None

        good = status.reshape(-1) == 1
        anchor_good = st.anchor_pts[good]
        live_good = new_pts[good]

        if live_good.shape[0] < max(4, self.min_tracked_points // 4):
            # Trop de points perdus : on tente un ré-ancrage
            self._reanchor(gray)
            st.prev_gray = gray
            st.valid = st.anchor_pts.shape[0] >= 4
            return st.current_H

        # Homographie image_courante -> image_ancre
        H_cur2anchor, mask = cv2.findHomography(
            live_good.reshape(-1, 2),
            anchor_good.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_thresh,
        )
        if H_cur2anchor is None:
            st.valid = False
            st.prev_gray = gray
            return None

        # Composition : image_courante -> terrain
        H_cur2pitch = st.anchor_H @ H_cur2anchor
        H_cur2pitch = H_cur2pitch / H_cur2pitch[2, 2]

        st.current_H = self._smooth(H_cur2pitch)
        st.live_pts = live_good.reshape(-1, 1, 2).astype(np.float32)
        st.anchor_pts = anchor_good.reshape(-1, 1, 2).astype(np.float32)
        st.prev_gray = gray
        st.valid = True

        # Ré-ancrage préventif si la population de points fond
        if st.live_pts.shape[0] < self.min_tracked_points:
            self._reanchor(gray)

        return st.current_H

    @property
    def is_valid(self) -> bool:
        return self.state.valid


# ================================================================
# Mesure de dérive (validation quantitative)
# ================================================================
def reprojection_drift(
    pitch_points: np.ndarray,
    homography_per_frame: List[np.ndarray],
    homography_static: np.ndarray,
) -> dict:
    """Compare la stabilité d'une H statique vs une H dynamique.

    On projette des points TERRAIN connus (mètres) vers l'image avec l'inverse de
    chaque homographie, puis on mesure de combien la position image d'un même point
    terrain bouge d'une frame à l'autre. Une bonne calibration dynamique réduit
    fortement cette variation pour les vrais points fixes du terrain.

    Args:
        pitch_points: (N,2) points terrain en mètres (ex: ronds, coins).
        homography_per_frame: liste de H (image->terrain) dynamiques.
        homography_static: H (image->terrain) statique de référence.

    Returns:
        Dict avec la dérive moyenne (px) statique et dynamique.
    """
    def _img_positions(Hs):
        seq = []
        for H in Hs:
            Hinv = np.linalg.inv(H)
            src = pitch_points.reshape(-1, 1, 2).astype(np.float32)
            seq.append(cv2.perspectiveTransform(src, Hinv).reshape(-1, 2))
        return np.stack(seq, axis=0)  # (T, N, 2)

    n = len(homography_per_frame)
    static_seq = _img_positions([homography_static] * n)
    dyn_seq = _img_positions(homography_per_frame)

    # La référence statique ne bouge jamais (dérive 0 par construction) -> on mesure
    # plutôt l'écart entre la position dynamique et la position statique initiale,
    # qui reflète la correction appliquée par le suivi.
    correction = np.linalg.norm(dyn_seq - static_seq[0:1], axis=2)  # (T, N)
    return {
        "mean_correction_px": float(correction.mean()),
        "max_correction_px": float(correction.max()),
        "n_frames": n,
        "n_points": int(pitch_points.shape[0]),
    }


__all__ = [
    "histogram_similarity",
    "detect_shot_boundaries",
    "DynamicHomography",
    "DynamicHomographyState",
    "reprojection_drift",
]
