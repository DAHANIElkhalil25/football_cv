"""
Module : tracker.py
But    : Tracking multi-objets via ByteTrack (intégré Ultralytics)
Output : Liste de TrackedObject avec id, bbox, classe, position BEV
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Représentation d'un objet tracké sur une frame."""

    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    class_id: int
    confidence: float
    bev_position: Optional[Tuple[float, float]] = None


class ByteTrackerWrapper:
    """Wrapper haut niveau de ByteTrack via l'API `YOLO.track` d'Ultralytics."""

    def __init__(
        self,
        model_path: str,
        bytetrack_cfg: str,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 1280,
        homography_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise le tracker.

        Args:
            model_path: Chemin du modèle YOLO (pt/onnx).
            bytetrack_cfg: Chemin du yaml ByteTrack.
            device: Device d'inférence (`cpu`, `cuda`, `0`, etc.).
            conf: Seuil de confiance détection.
            iou: Seuil IoU NMS.
            imgsz: Taille d'image pour l'inférence (doit correspondre à l'entraînement).
            homography_matrix: Matrice 3x3 pour projection BEV optionnelle.
        """
        self._configure_logging()

        self.model_path = Path(model_path).expanduser().resolve()
        self.bytetrack_cfg = Path(bytetrack_cfg).expanduser().resolve()
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.homography_matrix = homography_matrix

        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")
        if not self.bytetrack_cfg.exists():
            raise FileNotFoundError(f"Config ByteTrack introuvable: {self.bytetrack_cfg}")

        self.model = YOLO(str(self.model_path))
        LOGGER.info("ByteTrackerWrapper initialisé: model=%s tracker=%s", self.model_path, self.bytetrack_cfg)

    @staticmethod
    def _configure_logging() -> None:
        """Configure le logging standard si besoin."""
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )

    def _to_bev(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """Projette un point image vers le plan BEV via homographie.

        Args:
            x: Abscisse pixel image.
            y: Ordonnée pixel image.

        Returns:
            Coordonnées BEV (x, y) si homographie disponible, sinon None.
        """
        if self.homography_matrix is None:
            return None

        try:
            src = np.array([[[x, y]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self.homography_matrix)
            return float(dst[0, 0, 0]), float(dst[0, 0, 1])
        except Exception as exc:
            LOGGER.warning("Projection BEV échouée: %s", exc)
            return None

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """Exécute détection+tracking sur une frame.

        Args:
            frame: Image BGR OpenCV.

        Returns:
            Liste d'objets trackés avec ID persistants.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame invalide fournie à update().")

        try:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=str(self.bytetrack_cfg),
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Erreur pendant le tracking ByteTrack: {exc}") from exc

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=float)

        if boxes.id is None:
            track_ids = np.arange(len(xyxy), dtype=int)
        else:
            track_ids = boxes.id.cpu().numpy().astype(int)

        tracked: List[TrackedObject] = []
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            bev = self._to_bev(foot_x, foot_y)

            tracked.append(
                TrackedObject(
                    track_id=int(track_ids[idx]),
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_id=int(cls[idx]),
                    confidence=float(confs[idx]),
                    bev_position=bev,
                )
            )

        return tracked


def interpolate_missing_tracks(
    all_tracks: List[Dict],
    max_gap: int = 5,
) -> List[Dict]:
    """Interpole les détections manquantes pour chaque track_id.

    Problème résolu : ByteTrack perd parfois un joueur pendant 1-3 frames
    (occlusion, flou). Au lieu de créer un trou dans la trajectoire,
    on interpole linéairement la bbox entre la dernière et la prochaine
    détection connue. Cela augmente le recall sans modifier le tracker.

    Args:
        all_tracks: Liste [{frame_idx, frame_path, tracks: [...]}, ...].
        max_gap: Gap maximum (en frames) à interpoler. Au-delà, on
            considère que le joueur a vraiment disparu.

    Returns:
        all_tracks enrichi avec les tracks interpolés (marqués interpolated=True).
    """
    from collections import defaultdict

    # Indexer toutes les apparitions par track_id
    track_history: Dict[int, List[Tuple[int, Dict]]] = defaultdict(list)
    for frame_i, ft in enumerate(all_tracks):
        for t in ft["tracks"]:
            track_history[t["track_id"]].append((frame_i, t))

    # Pour chaque track_id, trouver les trous et interpoler
    for tid, appearances in track_history.items():
        appearances.sort(key=lambda x: x[0])

        for i in range(len(appearances) - 1):
            fi_start, t_start = appearances[i]
            fi_end, t_end = appearances[i + 1]
            gap = fi_end - fi_start - 1

            if gap <= 0 or gap > max_gap:
                continue

            # Interpolation linéaire des bboxes
            bbox_start = np.array(t_start["bbox"])
            bbox_end = np.array(t_end["bbox"])

            for g in range(1, gap + 1):
                alpha = g / (gap + 1)
                bbox_interp = (1 - alpha) * bbox_start + alpha * bbox_end
                interp_track = {
                    "track_id": tid,
                    "bbox": bbox_interp.tolist(),
                    "class_id": t_start["class_id"],
                    "conf": min(t_start.get("conf", 0.5), t_end.get("conf", 0.5)) * 0.8,
                    "interpolated": True,
                }
                all_tracks[fi_start + g]["tracks"].append(interp_track)

    return all_tracks


__all__ = ["TrackedObject", "ByteTrackerWrapper", "interpolate_missing_tracks"]
