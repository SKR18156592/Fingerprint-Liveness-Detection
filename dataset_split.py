from pathlib import Path
import shutil
import random

# ==========================
# Configuration
# ==========================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("dataset_split")


CLASSES = ["live", "spoof"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ==========================
# Create Output Folders
# ==========================
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        folder = OUTPUT_DIR / split / cls
        folder.mkdir(parents=True, exist_ok=True)

print("Folder structure created successfully!")