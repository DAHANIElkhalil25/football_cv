"""CLI de fine-tuning YOLOv8 SoccerNet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.converter import convert_soccernet_to_yolo
from src.data.roboflow_merge import merge_roboflow_with_soccernet
from src.detection.trainer import TrainConfig, YOLOTrainer


def _configure_logging() -> None:
    """Configure le logging CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse les arguments de ligne de commande.

    Returns:
        Namespace argparse.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 sur SoccerNet-Tracking")
    parser.add_argument("--soccernet_dir", type=str, required=True, help="Dossier SoccerNet brut")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sortie dataset+runs")
    parser.add_argument("--model_size", type=str, default="m", choices=["n", "s", "m", "l", "x"], help="Taille YOLOv8")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Taille image")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device: auto/cpu/cuda/0")
    parser.add_argument("--sampling_step", type=int, default=3, help="Sous-échantillonnage frames")
    parser.add_argument(
        "--use_roboflow",
        action="store_true",
        help="Fusionner des données Roboflow dans le split train avant entraînement.",
    )
    parser.add_argument(
        "--roboflow_dataset_dir",
        type=str,
        default="",
        help="Chemin du dataset Roboflow exporté en YOLOv8 (doit contenir data.yaml).",
    )
    parser.add_argument(
        "--merged_output_dir",
        type=str,
        default="./data/processed_merged",
        help="Dossier de sortie du dataset fusionné SoccerNet+Roboflow.",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée CLI pour conversion puis fine-tuning."""
    _configure_logging()
    args = parse_args()
    logger = logging.getLogger("run_finetune")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Étape 1/3: conversion SoccerNet -> YOLO")
    stats = convert_soccernet_to_yolo(
        soccernet_dir=args.soccernet_dir,
        output_dir=str(output_dir),
        splits_mapping={"train": "train", "test": "test"},
        img_w=1920,
        img_h=1080,
        sampling_step=args.sampling_step,
        min_visibility=0.2,
        min_bbox_px=10,
    )

    totals = stats.get("totals", {})
    images_copied = int(totals.get("images_copied", 0) or 0)
    labels_written = int(totals.get("labels_written", 0) or 0)
    if images_copied <= 0 or labels_written <= 0:
        raise RuntimeError(
            "Conversion vide: aucune image/annotation exploitable trouvée. "
            "Vérifiez que le dossier SoccerNet contient bien `train|test/<sequence>/img1/*` et `gt/gt.txt`. "
            "Si le download SoccerNet échoue, utilisez le mode `--zip_path` ou `--zip_url` du script de téléchargement."
        )

    data_yaml = stats.get("dataset_yaml", str(output_dir / "soccernet.yaml"))
    train_data_root = output_dir

    if args.use_roboflow:
        roboflow_dataset_dir = Path(args.roboflow_dataset_dir).expanduser().resolve()
        if not args.roboflow_dataset_dir or not roboflow_dataset_dir.exists():
            raise FileNotFoundError(
                "`--use_roboflow` activé mais `--roboflow_dataset_dir` est manquant/invalide."
            )

        merged_output_dir = Path(args.merged_output_dir).expanduser().resolve()
        logger.info("Fusion SoccerNet(train)+Roboflow(train) -> %s", merged_output_dir)
        merge_stats = merge_roboflow_with_soccernet(
            soccernet_processed_dir=str(output_dir),
            roboflow_dataset_dir=str(roboflow_dataset_dir),
            output_dir=str(merged_output_dir),
            include_roboflow_splits=["train"],
        )
        logger.info("Résumé fusion: %s", merge_stats.get("totals", {}))
        data_yaml = merge_stats.get("dataset_yaml", str(merged_output_dir / "soccernet_roboflow.yaml"))
        train_data_root = merged_output_dir

    model_name = f"yolov8{args.model_size}.pt"

    logger.info("Étape 2/3: entraînement modèle %s", model_name)
    trainer = YOLOTrainer(
        TrainConfig(
            data_yaml=str(data_yaml),
            output_dir=str(train_data_root / "runs"),
            model_name=model_name,
            run_name=f"yolov8{args.model_size}_soccernet",
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    )

    train_summary = trainer.train()

    logger.info("Étape 3/3: évaluation + export ONNX")
    eval_summary = trainer.evaluate(split="val")
    export_summary = trainer.export(format_name="onnx")

    logger.info("Terminé. best=%s", train_summary.get("weights_best"))
    logger.info("Eval=%s", eval_summary)
    logger.info("Export=%s", export_summary)


if __name__ == "__main__":
    main()
