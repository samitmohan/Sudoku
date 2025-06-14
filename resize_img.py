import os
import cv2
import random
from pathlib import Path
from shutil import rmtree

# Paths
RAW_DIR = Path("raw_digits")
OUT_DIR = Path("printed_digits")
TRAIN_DIR = OUT_DIR / "train"
TEST_DIR = OUT_DIR / "test"

# Reset output dirs
for d in [TRAIN_DIR, TEST_DIR]:
    if d.exists():
        rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# Settings
IMG_SIZE = (28, 28)
SPLIT_RATIO = 0.8  # 80% train, 20% test

# Loop through each digit class
for digit_dir in RAW_DIR.iterdir():
    if not digit_dir.is_dir():
        continue
    label = digit_dir.name
    images = list(digit_dir.glob("*"))

    # Shuffle for random split
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for phase, img_list in zip(["train", "test"], [train_imgs, test_imgs]):
        out_path = OUT_DIR / phase / label
        out_path.mkdir(parents=True, exist_ok=True)
        for i, img_path in enumerate(img_list):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_path / f"{i:05}.png"), resized)

print("✅ Dataset prepared in 'printed_digits/train' and 'printed_digits/test'")
