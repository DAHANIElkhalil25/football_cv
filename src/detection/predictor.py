"""
Module : predictor.py
But    : Inférence YOLOv8 sur image/vidéo pour détection football
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


class YOLOPredictor:
	"""Wrapper simple pour l'inférence YOLOv8 sur frames vidéo."""

	def __init__(self, model_path: str, device: str = "cpu", conf: float = 0.25, iou: float = 0.7) -> None:
		self._configure_logging()
		self.model_path = Path(model_path).expanduser().resolve()
		if not self.model_path.exists():
			raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")

		self.device = device
		self.conf = conf
		self.iou = iou
		self.model = YOLO(str(self.model_path))

	@staticmethod
	def _configure_logging() -> None:
		if not logging.getLogger().handlers:
			logging.basicConfig(
				level=logging.INFO,
				format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
			)

	def predict_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
		"""Inférence sur une image BGR OpenCV.

		Args:
			frame: Image BGR.

		Returns:
			Liste de détections sérialisées.
		"""
		if frame is None or frame.size == 0:
			raise ValueError("Frame invalide fournie à predict_frame().")

		results = self.model.predict(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
		if not results:
			return []

		res = results[0]
		boxes = getattr(res, "boxes", None)
		if boxes is None or boxes.xyxy is None:
			return []

		xyxy = boxes.xyxy.cpu().numpy()
		cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
		confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy), dtype=float)

		out: List[Dict[str, Any]] = []
		for idx, box in enumerate(xyxy):
			x1, y1, x2, y2 = [float(v) for v in box.tolist()]
			out.append(
				{
					"bbox_xyxy": [x1, y1, x2, y2],
					"class_id": int(cls[idx]),
					"confidence": float(confs[idx]),
				}
			)
		return out

	def predict_video(self, video_path: str, output_path: Optional[str] = None, max_frames: int = 0) -> Dict[str, Any]:
		"""Inférence image par image sur vidéo, avec export optionnel annoté.

		Args:
			video_path: Chemin vidéo source.
			output_path: Chemin de sortie vidéo annotée.
			max_frames: Limite de frames (0 = tout).

		Returns:
			Résumé global des détections.
		"""
		src = Path(video_path).expanduser().resolve()
		if not src.exists():
			raise FileNotFoundError(f"Vidéo introuvable: {src}")

		cap = cv2.VideoCapture(str(src))
		if not cap.isOpened():
			raise RuntimeError(f"Impossible d'ouvrir la vidéo: {src}")

		writer = None
		total_frames = 0
		total_detections = 0

		try:
			if output_path:
				w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
				target = Path(output_path).expanduser().resolve()
				target.parent.mkdir(parents=True, exist_ok=True)
				writer = cv2.VideoWriter(
					str(target),
					cv2.VideoWriter_fourcc(*"mp4v"),
					fps,
					(w, h),
				)

			while True:
				ok, frame = cap.read()
				if not ok:
					break

				detections = self.predict_frame(frame)
				total_frames += 1
				total_detections += len(detections)

				if writer is not None:
					for det in detections:
						x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
						cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
					writer.write(frame)

				if max_frames > 0 and total_frames >= max_frames:
					break
		finally:
			cap.release()
			if writer is not None:
				writer.release()

		summary = {
			"video_path": str(src),
			"frames_processed": total_frames,
			"detections_total": total_detections,
			"detections_per_frame": (total_detections / total_frames) if total_frames > 0 else 0.0,
		}
		LOGGER.info("Inférence terminée: %s", summary)
		return summary


__all__ = ["YOLOPredictor"]

