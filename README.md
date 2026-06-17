# Football CV : Analyse Tactique Automatique ⚽🤖

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)

Projet de Fin d'Études (PFE) INSEA 2025-2026 portant sur **l'analyse tactique automatisée de matchs de football à partir de vidéos broadcast (caméra unique)** en utilisant des techniques de vision par ordinateur (Computer Vision) et de Machine Learning.

## 🎯 Objectif du Projet

Ce projet vise à extraire automatiquement des données tactiques complexes à partir de simples vidéos de matchs de football (plans larges). Il permet aux analystes vidéo et aux entraîneurs d'obtenir des indicateurs avancés sans avoir besoin de systèmes de caméras multi-angles coûteux.

## ✨ Fonctionnalités Clés

Le pipeline d'analyse est modulaire et se décompose en 4 blocs principaux :

1. **Détection d'Objets (YOLOv8m)** : Fine-tuning sur le dataset SoccerNet Tracking pour détecter les joueurs, les arbitres et le ballon avec une grande précision.
2. **Tracking Multi-Objets (ByteTrack)** : Suivi temporel robuste des joueurs pour maintenir leur identité tout au long de la séquence, incluant une interpolation linéaire pour gérer les occlusions courtes.
3. **Classification des Équipes (K-Means + HSV)** : Clustering non supervisé basé sur la couleur des maillots (espace colorimétrique HSV) avec vote temporel majoritaire pour éliminer le "flickering" (changements intempestifs d'équipe).
4. **Géométrie & Homographie Dynamique** : Transformation de la vue caméra (2D) en vue de dessus (Bird's Eye View - BEV) sur un terrain en 2D (mètres). Inclut une **calibration dynamique par flux optique (Lucas-Kanade)** pour gérer les panoramiques et les zooms de la caméra sans ré-ancrage manuel.
5. **Analyse Tactique** :
   - **Contrôle Territorial** : Diagrammes de Voronoï clippés aux limites du terrain visible.
   - **Compacité Défensive** : Enveloppes convexes, centres de gravité et dispersion des blocs d'équipe.
   - **Pressing (PPDA v2)** : Calcul événementiel de la pression défensive, intégrant un filtre de zone, un debounce temporel et calibré sur les données de référence StatsBomb 360.

---

## 📂 Structure du Projet

```text
football_cv/
├── config/              # Fichiers de configuration (YAML) du pipeline
├── data/                # Données brutes et prétraitées (SoccerNet, Roboflow)
├── models/              # Poids des modèles entraînés (ex: best.pt)
├── notebooks/           # Jupyter notebooks pour l'EDA et les expérimentations
├── scripts/             # Scripts d'utilité (téléchargement, entraînement Kaggle)
└── src/                 # Code source principal
    ├── data/            # Téléchargement et conversion (MOT -> YOLO)
    ├── detection/       # Fine-tuning et inférence YOLOv8
    ├── tracking/        # Logique ByteTrack et interpolation
    ├── geometry/        # Homographie statique et dynamique (Flux Optique)
    ├── tactics/         # Indicateurs tactiques (PPDA, Voronoï, Compacité)
    └── pipeline.py      # Orchestrateur liant tous les blocs
```

---

## 🚀 Installation

Il est recommandé d'utiliser un environnement virtuel (venv ou conda).

```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/football_cv.git
cd football_cv

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📊 Préparation des Données

Le modèle de détection utilise le dataset **SoccerNet Tracking**. 

### 1. Télécharger SoccerNet

**Avec un compte SoccerNet :**
```bash
python scripts/download_soccernet.py --local_dir ./data/raw/soccernet
```

**Sans compte (via un zip local) :**
```bash
python scripts/download_soccernet.py --local_dir ./data/raw/soccernet --zip_path ./data/raw/soccernet.zip
```

### 2. (Optionnel) Fusionner avec Roboflow
Pour améliorer la détection (notamment du ballon), vous pouvez fusionner SoccerNet avec un dataset externe annoté via Roboflow.
```bash
python scripts/prepare_roboflow_merge.py \
    --soccernet_processed_dir ./data/processed \
    --output_dir ./data/processed_merged \
    --roboflow_dataset_dir ./data/raw/roboflow/my_project-1
```

---

## 🏋️ Entraînement du Modèle (Fine-Tuning YOLO)

L'entraînement peut se faire en local si vous disposez d'un GPU puissant, ou sur Kaggle/Colab.

**En local :**
```bash
python scripts/run_finetune.py \
    --soccernet_dir ./data/raw/soccernet \
    --output_dir ./data/processed \
    --model_size m \
    --epochs 50 \
    --imgsz 1280 \
    --batch 8 \
    --device cuda
```

**Sur Kaggle (Recommandé) :**
Un script d'exécution optimisé pour Kaggle est fourni.
```bash
%cd /kaggle/working/football_cv
bash ./scripts/kaggle_train.sh
```

---

## 🏃 Exécution du Pipeline Complet

Pour exécuter le pipeline d'inférence complet (Détection → Tracking → Classification → Homographie → Tactique) sur une vidéo :

*(Remarque : la CLI d'exécution principale sera documentée ici une fois finalisée)*

```python
# Exemple d'utilisation via le code
from src.pipeline import FootballPipeline
from config.config import load_config

config = load_config("config/config.yaml")
pipeline = FootballPipeline(config)

pipeline.process_video(
    input_video="data/test/match_clip.mp4",
    output_video="runs/output/tactical_view.mp4"
)
```

---

## 📈 Résultats et Performances

- **Détection (mAP@50)** : `0.770` (Joueurs : `0.972`, Ballon : `0.520`)
- **Tracking (MOTA)** : `0.772`
- **Erreur de Reprojection Moyenne** : `1.67` mètres
- **Précision de la Règle de Pressing (F1-Score)** : `0.573` (calibré contre StatsBomb 360 avec un rayon optimal de 2 mètres).

---

## 🎓 Auteur
**Elkhalil DAHANI** — INSEA (Institut National de Statistique et d'Économie Appliquée)
Projet de Fin d'Études 2025-2026.
