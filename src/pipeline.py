"""
Module : pipeline.py
But    : Pipeline bout-en-bout (download/conversion/train/tracking)
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from src.data.converter import convert_soccernet_to_yolo
from src.detection.trainer import TrainConfig, YOLOTrainer
from src.tracking.tracker import ByteTrackerWrapper

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
	if not logging.getLogger().handlers:
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		)


def run_pipeline(
	soccernet_dir: str,
	output_dir: str,
	run_training: bool = True,
	model_name: str = "yolov8m.pt",
	run_name: str = "yolov8m_soccernet",
	device: str = "auto",
	sample_video_path: Optional[str] = None,
	bytetrack_cfg: str = "./config/bytetrack.yaml",
) -> Dict[str, Any]:
	"""Exécute la conversion, puis entraînement, puis tracking optionnel.

	Returns:
		Dictionnaire de résumés par étape.
	"""
	_configure_logging()
	out_root = Path(output_dir).expanduser().resolve()
	out_root.mkdir(parents=True, exist_ok=True)

	result: Dict[str, Any] = {}

	LOGGER.info("Étape 1/3: conversion SoccerNet -> YOLO")
	conversion = convert_soccernet_to_yolo(soccernet_dir=soccernet_dir, output_dir=str(out_root))
	result["conversion"] = conversion

	if run_training:
		LOGGER.info("Étape 2/3: entraînement YOLO")
		trainer = YOLOTrainer(
			TrainConfig(
				data_yaml=str(out_root / "soccernet.yaml"),
				output_dir=str(out_root / "runs"),
				model_name=model_name,
				run_name=run_name,
				device=device,
			)
		)
		train_summary = trainer.train()
		eval_summary = trainer.evaluate(split="val")
		export_summary = trainer.export(format_name="onnx")
		result["train"] = train_summary
		result["eval"] = eval_summary
		result["export"] = export_summary

		if sample_video_path:
			LOGGER.info("Étape 3/3: tracking sample vidéo")
			model_path = train_summary.get("weights_best", "")
			tracker = ByteTrackerWrapper(model_path=model_path, bytetrack_cfg=bytetrack_cfg, device=device)
			cap = cv2.VideoCapture(str(Path(sample_video_path).expanduser().resolve()))
			ok, frame = cap.read()
			cap.release()
			if not ok:
				raise RuntimeError("Impossible de lire la vidéo de test pour tracking.")
			tracks = tracker.update(frame)
			result["tracking"] = {
				"sample_video": sample_video_path,
				"tracks_detected_first_frame": len(tracks),
			}

	LOGGER.info("Pipeline terminé.")
	return result


__all__ = ["run_pipeline"]

