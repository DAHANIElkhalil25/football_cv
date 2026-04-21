"""
Module : converter.py
But    : Convertir les annotations SoccerNet MOTChallenge  format YOLO
Format MOT  : frame, track_id, x, y, w, h, conf, class, visibility, _
Format YOLO : class_id cx_norm cy_norm w_norm h_norm
Classes     : 0=joueur, 1=gardien, 2=arbitre, 3=ballon
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

import yaml
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

MOT_TO_YOLO_CLASS = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
}

TRACKLET_ROLE_TO_YOLO = {
    "player": 0,
    "goalkeeper": 1,
    "referee": 2,
    "ball": 3,
}


def _configure_logging() -> None:
    """Configure un logger racine si nécessaire."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    """Clamp une valeur dans un intervalle.

    Args:
        value: Valeur à contraindre.
        lower: Borne inférieure.
        upper: Borne supérieure.

    Returns:
        Valeur contrainte.
    """
    return max(lower, min(value, upper))


def _mot_row_to_yolo(
    row: List[float],
    img_w: int,
    img_h: int,
    class_id_override: int | None = None,
) -> Tuple[int, float, float, float, float] | None:
    """Convertit une ligne MOT en annotation YOLO normalisée.

    Args:
        row: Ligne MOT décodée en floats.
        img_w: Largeur image en pixels.
        img_h: Hauteur image en pixels.

    Returns:
        Tuple YOLO (class_id, cx, cy, w, h) ou None si classe invalide.
    """
    x, y, w, h = row[2], row[3], row[4], row[5]
    mot_class = int(row[7])

    if class_id_override is not None:
        yolo_class = class_id_override
    else:
        if mot_class not in MOT_TO_YOLO_CLASS:
            return None
        yolo_class = MOT_TO_YOLO_CLASS[mot_class]

    x1 = _clamp(x, 0.0, float(img_w - 1))
    y1 = _clamp(y, 0.0, float(img_h - 1))
    x2 = _clamp(x + w, 0.0, float(img_w - 1))
    y2 = _clamp(y + h, 0.0, float(img_h - 1))

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return (
        yolo_class,
        cx / img_w,
        cy / img_h,
        bw / img_w,
        bh / img_h,
    )


def _normalize_tracklet_role(role: str) -> str | None:
    role = role.strip().lower()
    if "ball" in role:
        return "ball"
    if "referee" in role:
        return "referee"
    if "goalkeeper" in role or "goalkeepers" in role:
        return "goalkeeper"
    if "player" in role:
        return "player"
    return None


def _read_tracklet_class_map(gameinfo_file: Path) -> Dict[int, int]:
    """Lit gameinfo.ini et mappe track_id -> class_id YOLO.

    Args:
        gameinfo_file: Chemin vers `gameinfo.ini`.

    Returns:
        Mapping track_id -> class_id YOLO.
    """
    if not gameinfo_file.exists():
        return {}

    track_map: Dict[int, int] = {}
    pattern = re.compile(r"^trackletID_(\d+)\s*=\s*(.+)$")

    with gameinfo_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line.strip())
            if not match:
                continue

            track_id = int(match.group(1))
            role_spec = match.group(2)
            role = role_spec.split(";", maxsplit=1)[0]
            normalized = _normalize_tracklet_role(role)
            if normalized is None:
                continue
            track_map[track_id] = TRACKLET_ROLE_TO_YOLO[normalized]

    return track_map


def _read_gt_file(gt_file: Path) -> List[List[float]]:
    """Lit un fichier gt.txt MOTChallenge.

    Args:
        gt_file: Chemin vers le fichier gt.txt.

    Returns:
        Liste de lignes décodées.
    """
    rows: List[List[float]] = []
    if not gt_file.exists():
        return rows

    with gt_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            try:
                rows.append([float(v) for v in parts[:10]])
            except ValueError:
                continue
    return rows


def _ensure_yolo_dirs(output_dir: Path, yolo_split: str) -> Tuple[Path, Path]:
    """Crée les sous-dossiers images/labels d'un split.

    Args:
        output_dir: Dossier de sortie principal.
        yolo_split: Nom du split YOLO.

    Returns:
        Tuple (images_dir, labels_dir).
    """
    images_dir = output_dir / "images" / yolo_split
    labels_dir = output_dir / "labels" / yolo_split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def _find_sequences(split_root: Path) -> List[Path]:
    """Trouve les séquences d'un split SoccerNet.

    Args:
        split_root: Dossier racine du split.

    Returns:
        Liste des dossiers séquences détectés.
    """
    if not split_root.exists():
        return []

    sequences: List[Path] = []
    for candidate in split_root.rglob("*"):
        if not candidate.is_dir():
            continue
        if (candidate / "gt" / "gt.txt").exists() and (candidate / "img1").exists():
            sequences.append(candidate)
    return sorted(set(sequences))


def generate_soccernet_yaml(
    output_dir: str,
    yaml_path: str | None = None,
    val_split: str = "val",
) -> str:
    """Génère le fichier YAML dataset pour Ultralytics YOLO.

    Args:
        output_dir: Racine contenant `images/` et `labels/`.
        yaml_path: Chemin explicite du yaml; par défaut `<output_dir>/soccernet.yaml`.

    Returns:
        Chemin du fichier YAML généré.
    """
    out_root = Path(output_dir).resolve()
    target = Path(yaml_path).resolve() if yaml_path else (out_root / "soccernet.yaml")

    payload = {
        "path": str(out_root),
        "train": "images/train",
        "val": f"images/{val_split}",
        "test": "images/test",
        "names": {
            0: "player",
            1: "goalkeeper",
            2: "referee",
            3: "ball",
        },
        "nc": 4,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    return str(target)


def validate_yolo_dataset(output_dir: str, splits: List[str] | None = None) -> Dict[str, Any]:
    """Vérifie l'intégrité du dataset YOLO généré.

    Args:
        output_dir: Répertoire de sortie YOLO.
        splits: Splits à valider.

    Returns:
        Dictionnaire contenant les résultats d'intégrité.
    """
    root = Path(output_dir).resolve()
    splits = splits or ["train", "val", "test"]

    report: Dict[str, Any] = {
        "root": str(root),
        "splits": {},
        "is_valid": True,
        "errors": [],
    }

    for split in splits:
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        labels = sorted(labels_dir.glob("*.txt"))

        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        split_ok = len(missing_labels) == 0 and len(missing_images) == 0
        if not split_ok:
            report["is_valid"] = False
            if missing_labels:
                report["errors"].append(f"{split}: images sans labels ({len(missing_labels)})")
            if missing_images:
                report["errors"].append(f"{split}: labels sans images ({len(missing_images)})")

        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "ok": split_ok,
        }

    return report


def convert_soccernet_to_yolo(
    soccernet_dir: str,
    output_dir: str,
    splits_mapping: Dict[str, str] = {"train": "train", "test": "test"},
    img_w: int = 1920,
    img_h: int = 1080,
    sampling_step: int = 3,
    min_visibility: float = 0.2,
    min_bbox_px: int = 10,
) -> Dict[str, Any]:
    """Convertit SoccerNet Tracking du format MOTChallenge vers YOLO.

    Args:
        soccernet_dir: Dossier SoccerNet contenant les splits (`train`, `test`).
        output_dir: Dossier de sortie YOLO.
        splits_mapping: Mapping split source -> split destination YOLO.
        img_w: Largeur d'image utilisée pour normalisation.
        img_h: Hauteur d'image utilisée pour normalisation.
        sampling_step: Pas d'échantillonnage (1=toutes les frames).
        min_visibility: Seuil minimum de visibilité MOT.
        min_bbox_px: Taille minimum de bbox (w,h).

    Returns:
        Dictionnaire de statistiques de conversion.

    Raises:
        ValueError: Si paramètres invalides.
        FileNotFoundError: Si le dossier source est absent.
    """
    _configure_logging()

    if sampling_step < 1:
        raise ValueError("`sampling_step` doit être >= 1.")

    src_root = Path(soccernet_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Dossier SoccerNet introuvable: {src_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    class_counter: Counter[int] = Counter()
    global_stats: Dict[str, Any] = {
        "source_dir": str(src_root),
        "output_dir": str(out_root),
        "splits_mapping": splits_mapping,
        "params": {
            "img_w": img_w,
            "img_h": img_h,
            "sampling_step": sampling_step,
            "min_visibility": min_visibility,
            "min_bbox_px": min_bbox_px,
        },
        "splits": {},
        "totals": {
            "sequences": 0,
            "images_copied": 0,
            "labels_written": 0,
            "objects_written": 0,
            "objects_filtered": 0,
        },
    }

    for src_split, dst_split in splits_mapping.items():
        split_root = src_root / src_split
        images_dir, labels_dir = _ensure_yolo_dirs(out_root, dst_split)
        sequences = _find_sequences(split_root)

        split_stats = {
            "source_split": src_split,
            "target_split": dst_split,
            "sequences": len(sequences),
            "images_copied": 0,
            "labels_written": 0,
            "objects_written": 0,
            "objects_filtered": 0,
            "classes": defaultdict(int),
        }

        LOGGER.info("Split %s -> %s: %d séquences détectées", src_split, dst_split, len(sequences))

        for seq_dir in tqdm(sequences, desc=f"Conversion {src_split}", unit="seq"):
            gt_file = seq_dir / "gt" / "gt.txt"
            img_dir = seq_dir / "img1"
            gameinfo_file = seq_dir / "gameinfo.ini"
            seq_name = seq_dir.name
            rows = _read_gt_file(gt_file)
            tracklet_class_map = _read_tracklet_class_map(gameinfo_file)

            annotations_by_frame: DefaultDict[int, List[str]] = defaultdict(list)

            for row in rows:
                frame_id = int(row[0])
                track_id = int(row[1])
                conf = float(row[6])
                visibility = float(row[8])
                width, height = float(row[4]), float(row[5])

                if frame_id % sampling_step != 0:
                    continue

                if conf != 1 or width <= min_bbox_px or height <= min_bbox_px:
                    split_stats["objects_filtered"] += 1
                    continue

                if visibility >= 0 and visibility <= min_visibility:
                    split_stats["objects_filtered"] += 1
                    continue

                class_override = tracklet_class_map.get(track_id)
                yolo_box = _mot_row_to_yolo(
                    row,
                    img_w=img_w,
                    img_h=img_h,
                    class_id_override=class_override,
                )
                if yolo_box is None:
                    split_stats["objects_filtered"] += 1
                    continue

                class_id, cx, cy, bw, bh = yolo_box
                line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                annotations_by_frame[frame_id].append(line)
                split_stats["objects_written"] += 1
                split_stats["classes"][class_id] += 1
                class_counter[class_id] += 1

            for frame_id, lines in annotations_by_frame.items():
                src_image = img_dir / f"{frame_id:06d}.jpg"
                if not src_image.exists():
                    src_image = img_dir / f"{frame_id:06d}.png"
                if not src_image.exists():
                    split_stats["objects_filtered"] += len(lines)
                    split_stats["objects_written"] -= len(lines)
                    continue

                target_stem = f"{seq_name}_{frame_id:06d}"
                dst_image = images_dir / f"{target_stem}{src_image.suffix.lower()}"
                dst_label = labels_dir / f"{target_stem}.txt"

                shutil.copy2(src_image, dst_image)
                with dst_label.open("w", encoding="utf-8") as handle:
                    handle.write("\n".join(lines) + "\n")

                split_stats["images_copied"] += 1
                split_stats["labels_written"] += 1

        split_stats["classes"] = dict(sorted(split_stats["classes"].items()))
        global_stats["splits"][dst_split] = split_stats

        global_stats["totals"]["sequences"] += split_stats["sequences"]
        global_stats["totals"]["images_copied"] += split_stats["images_copied"]
        global_stats["totals"]["labels_written"] += split_stats["labels_written"]
        global_stats["totals"]["objects_written"] += split_stats["objects_written"]
        global_stats["totals"]["objects_filtered"] += split_stats["objects_filtered"]

    produced_splits = set(splits_mapping.values())
    if "val" in produced_splits:
        yolo_val_split = "val"
    elif "test" in produced_splits:
        yolo_val_split = "test"
    else:
        yolo_val_split = "train"

    yaml_path = generate_soccernet_yaml(
        str(out_root),
        yaml_path=str(out_root / "soccernet.yaml"),
        val_split=yolo_val_split,
    )
    global_stats["dataset_yaml"] = yaml_path
    global_stats["yolo_val_split"] = yolo_val_split
    global_stats["class_distribution"] = dict(sorted(class_counter.items()))

    integrity = validate_yolo_dataset(str(out_root), splits=list(set(splits_mapping.values())))
    global_stats["integrity"] = integrity

    LOGGER.info(
        "Conversion terminée: images=%d, labels=%d, objets=%d, filtrés=%d",
        global_stats["totals"]["images_copied"],
        global_stats["totals"]["labels_written"],
        global_stats["totals"]["objects_written"],
        global_stats["totals"]["objects_filtered"],
    )

    return global_stats


__all__ = [
    "convert_soccernet_to_yolo",
    "generate_soccernet_yaml",
    "validate_yolo_dataset",
]
