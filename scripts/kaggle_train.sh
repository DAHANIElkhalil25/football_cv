#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1

EPOCHS="${EPOCHS:-30}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
MODEL="${MODEL:-yolov8n.pt}"
RUN_NAME="${RUN_NAME:-yolov8n_soccernet}"

echo "[1/4] Installing Kaggle-safe dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements-kaggle.txt

echo "[2/4] Validating processed dataset..."
if [[ ! -f "data/processed/soccernet.yaml" ]]; then
  echo "ERROR: data/processed/soccernet.yaml not found."
  echo "Create/copy your processed dataset first into data/processed."
  exit 1
fi

echo "[3/4] Fixing soccernet.yaml paths for Kaggle..."
python - << 'PY'
from pathlib import Path
import yaml

project_root = Path.cwd().resolve()
yaml_path = project_root / "data" / "processed" / "soccernet.yaml"
payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

payload["path"] = str((project_root / "data" / "processed").resolve())

dataset_root = Path(payload["path"])
if not (dataset_root / "images" / "val").exists() and (dataset_root / "images" / "test").exists():
    payload["val"] = "images/test"

required = ["train", "val"]
for split in required:
    rel_path = payload.get(split)
    if not rel_path:
        raise RuntimeError(f"Missing '{split}' entry in soccernet.yaml")
    full = dataset_root / rel_path
    if not full.exists():
        raise RuntimeError(f"Missing dataset path for '{split}': {full}")

yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
print("Updated:", yaml_path)
print(payload)
PY

echo "[4/4] Starting training..."
yolo detect train \
  data=./data/processed/soccernet.yaml \
  model="${MODEL}" \
  epochs="${EPOCHS}" \
  imgsz="${IMGSZ}" \
  batch="${BATCH}" \
  device="${DEVICE}" \
  project=./data/processed/runs \
  name="${RUN_NAME}"

echo "Training completed. Outputs in ./data/processed/runs"