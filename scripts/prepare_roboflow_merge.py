"""Prépare un dataset fusionné SoccerNet + Roboflow pour entraînement YOLO."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.roboflow_merge import download_roboflow_yolov8, merge_roboflow_with_soccernet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Télécharger Roboflow et fusionner avec SoccerNet")

    parser.add_argument("--soccernet_processed_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./data/processed_merged")

    parser.add_argument("--roboflow_dataset_dir", type=str, default="", help="Dataset Roboflow déjà exporté (contient data.yaml)")
    parser.add_argument("--download_dir", type=str, default="./data/raw/roboflow")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--workspace", type=str, default="")
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--version", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("prepare_roboflow_merge")
    args = parse_args()

    if args.roboflow_dataset_dir:
        roboflow_dataset_dir = Path(args.roboflow_dataset_dir).expanduser().resolve()
        if not roboflow_dataset_dir.exists():
            raise FileNotFoundError(f"Dataset Roboflow introuvable: {roboflow_dataset_dir}")
    else:
        missing = [
            key
            for key, value in {
                "api_key": args.api_key,
                "workspace": args.workspace,
                "project": args.project,
                "version": args.version,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(
                "Paramètres manquants pour télécharger Roboflow: " + ", ".join(missing)
            )

        logger.info("Téléchargement Roboflow en cours...")
        roboflow_dataset_dir = Path(
            download_roboflow_yolov8(
                api_key=args.api_key,
                workspace=args.workspace,
                project=args.project,
                version=args.version,
                location=args.download_dir,
            )
        )

    logger.info("Fusion avec SoccerNet en cours...")
    stats = merge_roboflow_with_soccernet(
        soccernet_processed_dir=args.soccernet_processed_dir,
        roboflow_dataset_dir=str(roboflow_dataset_dir),
        output_dir=args.output_dir,
        include_roboflow_splits=["train"],
    )

    logger.info("Dataset fusionné prêt: %s", stats["output_dir"])
    logger.info("YAML entraînement: %s", stats["dataset_yaml"])
    logger.info("Résumé: %s", stats["totals"])


if __name__ == "__main__":
    main()
