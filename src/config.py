from __future__ import annotations

from pathlib import Path

RANDOM_SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 45
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.02
USE_CLASS_WEIGHTS = False
CLASS_NAMES = ["baixa", "media", "alta"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_COLUMN = "visual_density_class"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROJECT_ROOT = Path(
    r"c:\repo\Fiap\global\space_urban_density_ml\space_urban_density_ml"
)
SOURCE_METADATA_PATH = SOURCE_PROJECT_ROOT / "data" / "urban_density_dataset.csv"
SOURCE_IMAGES_DIR = SOURCE_PROJECT_ROOT / "data" / "images"

DATA_DIR = ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
SPLITS_DIR = DATA_DIR / "splits"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"
APP_DIR = ROOT / "app"
CLASS_DISTRIBUTION_PATH = REPORTS_DIR / "class_distribution.csv"

CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}
