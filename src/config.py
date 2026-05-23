from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_DIR = RAW_DATA_DIR / "CD368"

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
TRAIN_LABEL_DIR = DATASET_DIR / "train" / "labels"
VALID_IMAGE_DIR = DATASET_DIR / "valid" / "images"
VALID_LABEL_DIR = DATASET_DIR / "valid" / "labels"
TEST_IMAGE_DIR = DATASET_DIR / "test" / "images"
TEST_LABEL_DIR = DATASET_DIR / "test" / "labels"