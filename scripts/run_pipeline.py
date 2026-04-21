"""Script pipeline bout-en-bout pour Football CV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.converter import convert_soccernet_to_yolo
from src.detection.trainer import TrainConfig, YOLOTrainer


def parse_args() -> argparse.Namespace:
	"""Parse les arguments de pipeline."""
	parser = argparse.ArgumentParser(description="Pipeline Football CV")
	parser.add_argument("--soccernet_dir", type=str, default="./data/raw/soccernet")
	parser.add_argument("--output_dir", type=str, default="./data/processed")
	parser.add_argument("--train", action="store_true", help="Lancer aussi le fine-tuning")
	return parser.parse_args()


def main() -> None:
	"""Exécute conversion et optionnellement entraînement."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	args = parse_args()
	logger = logging.getLogger("run_pipeline")

	stats = convert_soccernet_to_yolo(
		soccernet_dir=args.soccernet_dir,
		output_dir=args.output_dir,
		splits_mapping={"train": "train", "test": "val"},
	)
	logger.info("Conversion terminée: %s", stats["totals"])

	if args.train:
		trainer = YOLOTrainer(
			TrainConfig(
				data_yaml=str(Path(args.output_dir).resolve() / "soccernet.yaml"),
				output_dir=str(Path(args.output_dir).resolve() / "runs"),
			)
		)
		trainer.train()
		trainer.evaluate("val")
		trainer.export("onnx")
		logger.info("Entraînement + export terminés.")


if __name__ == "__main__":
	main()
