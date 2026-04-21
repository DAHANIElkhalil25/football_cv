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
        homography_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise le tracker.

        Args:
            model_path: Chemin du modèle YOLO (pt/onnx).
            bytetrack_cfg: Chemin du yaml ByteTrack.
            device: Device d'inférence (`cpu`, `cuda`, `0`, etc.).
            conf: Seuil de confiance détection.
            iou: Seuil IoU NMS.
            homography_matrix: Matrice 3x3 pour projection BEV optionnelle.
        """
        self._configure_logging()

        self.model_path = Path(model_path).expanduser().resolve()
        self.bytetrack_cfg = Path(bytetrack_cfg).expanduser().resolve()
        self.device = device
        self.conf = conf
        self.iou = iou
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


__all__ = ["TrackedObject", "ByteTrackerWrapper"]
