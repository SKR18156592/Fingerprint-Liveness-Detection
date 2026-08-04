from pathlib import Path
import shutil
import random

# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("dataset_split")

CLASSES = ["live", "spoof"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def copy_images(image_list, split_name, class_name):
    destination = OUTPUT_DIR / split_name / class_name

    for image in image_list:
        shutil.copy(image, destination / image.name)


# Create folders
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        folder=OUTPUT_DIR / split / cls
        folder.mkdir(parents=True, exist_ok=True)


for cls in CLASSES:

    source_dir = DATA_DIR / cls

    images = list(source_dir.glob("*"))

    random.shuffle(images)

    num_images = len(images)

    train_size = int(num_images * TRAIN_RATIO)
    val_size = int(num_images * VAL_RATIO)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    copy_images(train_images, "train", cls)
    copy_images(val_images, "val", cls)
    copy_images(test_images, "test", cls)

    print(f"\n{cls.upper()} SUMMARY")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")

print("\nDataset split completed successfully!")