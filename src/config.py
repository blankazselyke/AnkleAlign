import os
from pathlib import Path

DATA_DIR = Path("/app/data")
OUTPUT_DIR = Path("/app/output")
PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
CSV_PATH = PROCESSED_DATA_DIR / "dataset.csv"
CSV_PATH_TRAIN = PROCESSED_DATA_DIR / "dataset_train.csv"
CSV_PATH_VAL = PROCESSED_DATA_DIR / "dataset_val.csv"
CSV_PATH_TEST = PROCESSED_DATA_DIR / "dataset_test.csv"
MODEL_SAVE_PATH = OUTPUT_DIR / "best_model.pth"
BASELINE_SAVE_PATH = OUTPUT_DIR / "baseline_model.pth"

EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5

EPOCHS_BASE = 10
BATCH_SIZE_BASE = 16
LEARNING_RATE_BASE = 1e-3
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_SPLIT= 0.8
SEED = 42

LABEL_MAP = {
    "1_Pronacio": 0, "2_Neutralis": 1, "3_Szupinacio": 2,
    "pronation": 0, "Pronation": 0, "Pronacio": 0,
    "neutral": 1, "Neutral": 1, "Neutralis": 1,
    "supination": 2, "Supination": 2, "Szupinacio": 2
}
CLASS_NAMES = ["Pronacio", "Neutralis", "Szupinacio"]