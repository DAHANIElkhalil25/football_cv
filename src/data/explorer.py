"""
Module : explorer.py
But    : Explorer SoccerNet et les datasets YOLO générés
Auteur : Elkhalil DAHANI — INSEA PFE 2025-2026
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
	"""Configure le logging racine si nécessaire."""
	if not logging.getLogger().handlers:
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		)


def summarize_soccernet_tracking(root_dir: str) -> Dict[str, Any]:
	"""Résume rapidement un dossier SoccerNet-Tracking.

	Args:
		root_dir: Dossier contenant les splits `train` et/ou `test`.

	Returns:
		Dictionnaire de statistiques globales et par split.
	"""
	_configure_logging()

	root = Path(root_dir).expanduser().resolve()
	if not root.exists():
		raise FileNotFoundError(f"Dossier introuvable: {root}")

	report: Dict[str, Any] = {
		"root": str(root),
		"splits": {},
		"totals": {
			"sequences": 0,
			"frames": 0,
			"gt_files": 0,
		},
	}

	for split in ["train", "test", "val"]:
		split_root = root / split
		if not split_root.exists():
			continue

		sequences = 0
		frames = 0
		gt_files = 0

		for seq_dir in split_root.rglob("*"):
			if not seq_dir.is_dir():
				continue
			gt_file = seq_dir / "gt" / "gt.txt"
			img1_dir = seq_dir / "img1"
			if gt_file.exists() and img1_dir.exists():
				sequences += 1
				gt_files += 1
				frames += len(list(img1_dir.glob("*.jpg"))) + len(list(img1_dir.glob("*.png")))

		report["splits"][split] = {
			"sequences": sequences,
			"frames": frames,
			"gt_files": gt_files,
		}
		report["totals"]["sequences"] += sequences
		report["totals"]["frames"] += frames
		report["totals"]["gt_files"] += gt_files

	LOGGER.info("SoccerNet résumé: %s", report["totals"])
	return report


def summarize_yolo_dataset(dataset_root: str) -> Dict[str, Any]:
	"""Résume un dataset YOLO structuré en `images/` et `labels/`.

	Args:
		dataset_root: Dossier racine du dataset YOLO.

	Returns:
		Statistiques par split (train/val/test).
	"""
	_configure_logging()

	root = Path(dataset_root).expanduser().resolve()
	if not root.exists():
		raise FileNotFoundError(f"Dossier introuvable: {root}")

	report: Dict[str, Any] = {
		"root": str(root),
		"splits": {},
		"totals": {
			"images": 0,
			"labels": 0,
		},
	}

	for split in ["train", "val", "test"]:
		images_dir = root / "images" / split
		labels_dir = root / "labels" / split
		images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
		labels = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

		report["splits"][split] = {"images": images, "labels": labels}
		report["totals"]["images"] += images
		report["totals"]["labels"] += labels

	LOGGER.info("YOLO résumé: %s", report["totals"])
	return report


def write_report_json(report: Dict[str, Any], output_file: str) -> str:
	"""Écrit un rapport JSON sur disque.

	Args:
		report: Dictionnaire à sérialiser.
		output_file: Chemin du fichier JSON cible.

	Returns:
		Chemin absolu du rapport écrit.
	"""
	target = Path(output_file).expanduser().resolve()
	target.parent.mkdir(parents=True, exist_ok=True)
	with target.open("w", encoding="utf-8") as handle:
		json.dump(report, handle, indent=2, ensure_ascii=False)
	return str(target)


__all__ = [
	"summarize_soccernet_tracking",
	"summarize_yolo_dataset",
	"write_report_json",
]

