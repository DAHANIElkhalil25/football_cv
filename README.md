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

## Exécution Kaggle (recommandé pour GPU limité Colab)

### 1) Redémarrer la session Kaggle (important)
Après des conflits pip, redémarrez le kernel Kaggle avant toute nouvelle installation.

### 2) Installer puis entraîner en une commande
```bash
%cd /kaggle/working/football_cv
bash ./scripts/kaggle_train.sh
```

Le script:
- installe `requirements-kaggle.txt` (set minimal stable),
- corrige `data/processed/soccernet.yaml` pour les chemins Kaggle,
- vérifie que `train` et `val` existent,
- lance l'entraînement YOLO.

### 3) Paramètres optionnels
```bash
%cd /kaggle/working/football_cv
EPOCHS=40 IMGSZ=960 BATCH=16 DEVICE=0 MODEL=yolov8n.pt RUN_NAME=yolov8n_soccernet_kaggle bash ./scripts/kaggle_train.sh
```
