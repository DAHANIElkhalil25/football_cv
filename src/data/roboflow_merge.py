"""Utilitaires Roboflow: téléchargement YOLOv8 et fusion avec SoccerNet."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

LOGGER = logging.getLogger(__name__)

TARGET_CLASS_NAMES = ["player", "goalkeeper", "referee", "ball"]

_CLASS_ALIASES = {
    "player": "player",
    "person": "player",
    "players": "player",
    "goalkeeper": "goalkeeper",
    "goalkeepers": "goalkeeper",
    "keeper": "goalkeeper",
    "gk": "goalkeeper",
    "goalie": "goalkeeper",
    "referee": "referee",
    "ref": "referee",
    "arbitre": "referee",
    "ball": "ball",
    "football": "ball",
    "soccer ball": "ball",
}


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().replace("_", " ").replace("-", " ").split())


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def download_roboflow_yolov8(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    location: str,
) -> str:
    """Télécharge un dataset Roboflow au format YOLOv8.

    Returns:
        Chemin du dossier contenant `data.yaml`.
    """
    _configure_logging()

    try:
        from roboflow import Roboflow
    except Exception as exc:
        raise ImportError(
            "La librairie `roboflow` est requise. Installez via `pip install roboflow`."
        ) from exc

    target_dir = Path(location).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download("yolov8", location=str(target_dir))

    candidate = Path(getattr(dataset, "location", target_dir)).resolve()
    yaml_file = _find_data_yaml(candidate)
    if yaml_file is None:
        yaml_file = _find_data_yaml(target_dir)

    if yaml_file is None:
        raise FileNotFoundError("Téléchargement Roboflow terminé mais `data.yaml` introuvable.")

    LOGGER.info("Dataset Roboflow prêt: %s", yaml_file.parent)
    return str(yaml_file.parent)


def _find_data_yaml(root: Path) -> Path | None:
    direct = root / "data.yaml"
    if direct.exists():
        return direct

    for candidate in root.rglob("data.yaml"):
        return candidate
    return None


def _load_names_from_yaml(dataset_root: Path) -> Dict[int, str]:
    yaml_file = _find_data_yaml(dataset_root)
    if yaml_file is None:
        raise FileNotFoundError(f"`data.yaml` introuvable dans {dataset_root}")

    payload = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})

    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        normalized: Dict[int, str] = {}
        for key, value in names.items():
            normalized[int(key)] = str(value)
        return dict(sorted(normalized.items()))

    raise ValueError("Format `names` non supporté dans data.yaml Roboflow.")


def build_class_id_remap(source_names: Dict[int, str]) -> Dict[int, int]:
    """Construit le mapping classes source -> classes cibles SoccerNet."""
    target_index = {name: idx for idx, name in enumerate(TARGET_CLASS_NAMES)}
    remap: Dict[int, int] = {}

    for source_idx, source_name in source_names.items():
        alias = _CLASS_ALIASES.get(_normalize_name(source_name))
        if alias is None:
            continue
        remap[source_idx] = target_index[alias]

    return remap


def _resolve_rf_split_dirs(dataset_root: Path, split: str) -> tuple[Path, Path]:
    split_key = split.lower().strip()
    split_aliases = {
        "train": ["train"],
        "val": ["valid", "val", "validation"],
        "test": ["test"],
    }

    for candidate in split_aliases.get(split_key, [split_key]):
        images_dir = dataset_root / candidate / "images"
        labels_dir = dataset_root / candidate / "labels"
        if images_dir.exists() and labels_dir.exists():
            return images_dir, labels_dir

    raise FileNotFoundError(f"Split Roboflow '{split}' introuvable dans {dataset_root}")


def _find_image_for_label(images_dir: Path, stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _remap_label_content(lines: Iterable[str], class_remap: Dict[int, int]) -> List[str]:
    mapped: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue

        try:
            source_class = int(parts[0])
        except ValueError:
            continue

        if source_class not in class_remap:
            continue

        target_class = class_remap[source_class]
        mapped.append(f"{target_class} {' '.join(parts[1:])}")

    return mapped


def _write_merged_yaml(output_root: Path) -> str:
    payload = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(TARGET_CLASS_NAMES)},
        "nc": len(TARGET_CLASS_NAMES),
    }

    yaml_path = output_root / "soccernet_roboflow.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return str(yaml_path)


def merge_roboflow_with_soccernet(
    soccernet_processed_dir: str,
    roboflow_dataset_dir: str,
    output_dir: str,
    include_roboflow_splits: List[str] | None = None,
) -> Dict[str, Any]:
    """Fusionne SoccerNet (base) avec Roboflow (train additionnel)."""
    _configure_logging()

    include_roboflow_splits = include_roboflow_splits or ["train"]

    soccernet_root = Path(soccernet_processed_dir).expanduser().resolve()
    roboflow_root = Path(roboflow_dataset_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()

    if not soccernet_root.exists():
        raise FileNotFoundError(f"Dataset SoccerNet introuvable: {soccernet_root}")
    if not roboflow_root.exists():
        raise FileNotFoundError(f"Dataset Roboflow introuvable: {roboflow_root}")

    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(soccernet_root, output_root)

    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    source_names = _load_names_from_yaml(roboflow_root)
    class_remap = build_class_id_remap(source_names)
    if not class_remap:
        raise ValueError(
            "Aucune classe Roboflow compatible trouvée. Vérifiez les noms de classes dans data.yaml."
        )

    stats: Dict[str, Any] = {
        "soccernet_processed_dir": str(soccernet_root),
        "roboflow_dataset_dir": str(roboflow_root),
        "output_dir": str(output_root),
        "include_roboflow_splits": include_roboflow_splits,
        "class_remap": class_remap,
        "totals": {
            "roboflow_images_added": 0,
            "roboflow_labels_added": 0,
            "roboflow_objects_added": 0,
            "roboflow_samples_skipped": 0,
        },
    }

    out_train_images = output_root / "images" / "train"
    out_train_labels = output_root / "labels" / "train"

    for split in include_roboflow_splits:
        images_dir, labels_dir = _resolve_rf_split_dirs(roboflow_root, split)
        for label_file in sorted(labels_dir.glob("*.txt")):
            image_file = _find_image_for_label(images_dir, label_file.stem)
            if image_file is None:
                stats["totals"]["roboflow_samples_skipped"] += 1
                continue

            remapped_lines = _remap_label_content(
                label_file.read_text(encoding="utf-8").splitlines(),
                class_remap=class_remap,
            )
            if not remapped_lines:
                stats["totals"]["roboflow_samples_skipped"] += 1
                continue

            target_stem = f"rf_{split}_{label_file.stem}"
            dst_image = out_train_images / f"{target_stem}{image_file.suffix.lower()}"
            dst_label = out_train_labels / f"{target_stem}.txt"

            shutil.copy2(image_file, dst_image)
            dst_label.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

            stats["totals"]["roboflow_images_added"] += 1
            stats["totals"]["roboflow_labels_added"] += 1
            stats["totals"]["roboflow_objects_added"] += len(remapped_lines)

    stats["dataset_yaml"] = _write_merged_yaml(output_root)
    LOGGER.info("Fusion Roboflow terminée: %s", stats["totals"])
    return stats


__all__ = [
    "TARGET_CLASS_NAMES",
    "build_class_id_remap",
    "download_roboflow_yolov8",
    "merge_roboflow_with_soccernet",
]
