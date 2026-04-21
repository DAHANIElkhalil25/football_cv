"""
Module : trainer.py
But    : Fine-tuning YOLOv8m sur SoccerNet-Tracking
Modèle : yolov8m.pt  yolov8m_soccernet.pt
Cibles : mAP50 > 0.75 (joueurs), mAP50 > 0.70 (ballon), Recall > 0.80
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Configuration d'entraînement YOLOv8."""

    data_yaml: str
    output_dir: str
    model_name: str = "yolov8m.pt"
    run_name: str = "yolov8m_soccernet"
    epochs: int = 50
    imgsz: int = 1280
    batch: int = 8
    lr0: float = 0.003
    lrf: float = 0.01
    weight_decay: float = 5e-4
    workers: int = 8
    patience: int = 15
    pretrained: bool = True
    device: str = "auto"
    augment_hsv_h: float = 0.015
    augment_hsv_s: float = 0.7
    augment_hsv_v: float = 0.4
    augment_scale: float = 0.5
    augment_fliplr: float = 0.5
    augment_mosaic: float = 1.0
    augment_mixup: float = 0.1


class YOLOTrainer:
    """Classe utilitaire pour fine-tuning, évaluation et export YOLOv8."""

    def __init__(self, config: TrainConfig) -> None:
        """Initialise le trainer.

        Args:
            config: Paramètres d'entraînement.
        """
        self._configure_logging()
        self.config = config
        self.output_dir = Path(config.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(config.device)
        self.model = YOLO(config.model_name)
        self.last_train_results: Optional[Any] = None
        self.last_eval_results: Optional[Any] = None

        self._register_callbacks()
        LOGGER.info("YOLOTrainer initialisé sur device=%s", self.device)

    @staticmethod
    def _configure_logging() -> None:
        """Configure le logging global si besoin."""
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Résout automatiquement le device CPU/GPU.

        Args:
            requested: Device demandé (`auto`, `cpu`, `cuda`, `0`, ...).

        Returns:
            Device utilisable par Ultralytics.
        """
        req = requested.lower().strip()
        if req == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if req == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA demandé mais indisponible, fallback CPU.")
            return "cpu"
        return requested

    def _register_callbacks(self) -> None:
        """Ajoute des callbacks de log epoch par epoch."""

        def on_fit_epoch_end(trainer: Any) -> None:
            epoch = int(getattr(trainer, "epoch", -1))
            metrics = getattr(trainer, "metrics", {}) or {}
            precision = metrics.get("metrics/precision(B)", None)
            recall = metrics.get("metrics/recall(B)", None)
            map50 = metrics.get("metrics/mAP50(B)", None)
            map5095 = metrics.get("metrics/mAP50-95(B)", None)
            LOGGER.info(
                "Epoch %03d | precision=%s recall=%s mAP50=%s mAP50-95=%s",
                epoch,
                f"{precision:.4f}" if isinstance(precision, (int, float)) else "n/a",
                f"{recall:.4f}" if isinstance(recall, (int, float)) else "n/a",
                f"{map50:.4f}" if isinstance(map50, (int, float)) else "n/a",
                f"{map5095:.4f}" if isinstance(map5095, (int, float)) else "n/a",
            )

        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    def train(self) -> Dict[str, Any]:
        """Lance l'entraînement YOLOv8 sur SoccerNet.

        Returns:
            Dictionnaire de résumé d'entraînement.
        """
        LOGGER.info("Démarrage entraînement run=%s", self.config.run_name)
        self.last_train_results = self.model.train(
            data=self.config.data_yaml,
            project=str(self.output_dir),
            name=self.config.run_name,
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            device=self.device,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            weight_decay=self.config.weight_decay,
            workers=self.config.workers,
            patience=self.config.patience,
            pretrained=self.config.pretrained,
            hsv_h=self.config.augment_hsv_h,
            hsv_s=self.config.augment_hsv_s,
            hsv_v=self.config.augment_hsv_v,
            scale=self.config.augment_scale,
            fliplr=self.config.augment_fliplr,
            mosaic=self.config.augment_mosaic,
            mixup=self.config.augment_mixup,
            plots=True,
            verbose=True,
        )

        result_dict = {
            "run_dir": str(self.output_dir / self.config.run_name),
            "weights_best": str(self.output_dir / self.config.run_name / "weights" / "best.pt"),
            "weights_last": str(self.output_dir / self.config.run_name / "weights" / "last.pt"),
            "device": self.device,
            "config": asdict(self.config),
        }
        self._write_summary_json("train_summary.json", result_dict)
        return result_dict

    def evaluate(self, split: str = "val") -> Dict[str, Any]:
        """Évalue le modèle entraîné et compare aux objectifs.

        Args:
            split: Split d'évaluation (`val` recommandé).

        Returns:
            Dictionnaire de métriques + statut des objectifs.
        """
        LOGGER.info("Évaluation du modèle sur split=%s", split)
        self.last_eval_results = self.model.val(data=self.config.data_yaml, split=split, device=self.device)

        map50 = float(getattr(self.last_eval_results.box, "map50", 0.0))
        recall = float(getattr(self.last_eval_results.box, "mr", 0.0))

        per_class_map50 = {}
        class_targets = {
            "player": 0.75,
            "ball": 0.70,
        }

        names = getattr(self.last_eval_results, "names", {}) or {}
        map50_per_class = getattr(self.last_eval_results.box, "maps", [])
        for idx, score in enumerate(map50_per_class):
            class_name = str(names.get(idx, idx))
            per_class_map50[class_name] = float(score)

        targets_status = {
            "player_map50": per_class_map50.get("player", 0.0) >= class_targets["player"],
            "ball_map50": per_class_map50.get("ball", 0.0) >= class_targets["ball"],
            "recall": recall >= 0.80,
        }

        summary = {
            "map50_global": map50,
            "recall_global": recall,
            "map50_per_class": per_class_map50,
            "targets": {
                "player_map50_target": class_targets["player"],
                "ball_map50_target": class_targets["ball"],
                "recall_target": 0.80,
                "status": targets_status,
            },
        }

        LOGGER.info(
            "Résultats: mAP50=%.4f recall=%.4f | targets=%s",
            map50,
            recall,
            targets_status,
        )
        self._write_summary_json("eval_summary.json", summary)
        return summary

    def export(self, format_name: str = "onnx") -> Dict[str, Any]:
        """Exporte le modèle au format demandé (ONNX par défaut).

        Args:
            format_name: Format d'export Ultralytics.

        Returns:
            Dictionnaire contenant le chemin exporté.
        """
        LOGGER.info("Export du modèle au format=%s", format_name)
        exported_path = self.model.export(format=format_name, imgsz=self.config.imgsz, device=self.device)
        info = {
            "format": format_name,
            "exported_path": str(exported_path),
        }
        self._write_summary_json("export_summary.json", info)
        return info

    def _write_summary_json(self, file_name: str, payload: Dict[str, Any]) -> None:
        """Écrit un résumé JSON dans le dossier de run.

        Args:
            file_name: Nom du fichier JSON.
            payload: Contenu à sérialiser.
        """
        run_dir = self.output_dir / self.config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / file_name
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)


__all__ = ["TrainConfig", "YOLOTrainer"]
