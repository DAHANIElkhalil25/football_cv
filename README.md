# Football CV Tactical Analysis

Projet PFE INSEA (2025-2026) pour l'analyse tactique du football via Computer Vision.

## Structure
- `src/data`: téléchargement SoccerNet et conversion MOT->YOLO
- `src/detection`: fine-tuning YOLOv8
- `src/tracking`: tracking ByteTrack

## Installation
```bash
pip install -r requirements.txt
```

## Télécharger / préparer les données

### Méthode 1 (avec compte SoccerNet)
```bash
python scripts/download_soccernet.py --local_dir ./data/raw/soccernet
```

### Méthode 2 (sans compte): importer un ZIP local/public
```bash
python scripts/download_soccernet.py --local_dir ./data/raw/soccernet --zip_path ./data/raw/soccernet.zip
# ou
python scripts/download_soccernet.py --local_dir ./data/raw/soccernet --zip_url "https://example.com/soccernet.zip"
```

### Méthode 3 (sans compte): initialiser l'arborescence puis copier manuellement
```bash
python scripts/download_soccernet.py --local_dir ./data/raw/soccernet --init_only
```

## Option: augmenter `train` avec Roboflow

Protocole recommandé:
- `train`: SoccerNet + Roboflow
- `val/test`: SoccerNet uniquement

Téléchargement + fusion:
```bash
python scripts/prepare_roboflow_merge.py --soccernet_processed_dir ./data/processed --output_dir ./data/processed_merged --api_key YOUR_ROBOFLOW_API_KEY --workspace YOUR_WORKSPACE --project YOUR_PROJECT --version 1
```

Ou si le dataset Roboflow est déjà exporté en YOLOv8:
```bash
python scripts/prepare_roboflow_merge.py --soccernet_processed_dir ./data/processed --output_dir ./data/processed_merged --roboflow_dataset_dir ./data/raw/roboflow/my_project-1
```

## Lancer le fine-tuning
```bash
python scripts/run_finetune.py --soccernet_dir ./data/raw/soccernet --output_dir ./data/processed --model_size m --epochs 50 --imgsz 1280 --batch 8 --device cuda --sampling_step 3
```

Fine-tuning avec Roboflow fusionné:
```bash
python scripts/run_finetune.py --soccernet_dir ./data/raw/soccernet --output_dir ./data/processed --use_roboflow --roboflow_dataset_dir ./data/raw/roboflow/my_project-1 --merged_output_dir ./data/processed_merged --model_size m --epochs 50 --imgsz 1280 --batch 8 --device cuda --sampling_step 3
```
