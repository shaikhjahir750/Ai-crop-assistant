import os
import shutil
import random

# ===========================================
# Source and Target Paths
# ===========================================
SOURCE_DIR = "dataset/plantvillage"     # original folder
TARGET_DIR = "dataset_split"             # output folder
TRAIN_DIR = os.path.join(TARGET_DIR, "train")
VAL_DIR = os.path.join(TARGET_DIR, "val")

# Train/validation ratio
SPLIT_RATIO = 0.8  # 80% train, 20% val

# ===========================================
# Create folders
# ===========================================
for path in [TRAIN_DIR, VAL_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# ===========================================
# Split each class
# ===========================================
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # List all images in this class
    images = os.listdir(class_path)
    random.shuffle(images)

    # Calculate split point
    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class folders in train/val
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

    # Copy train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(TRAIN_DIR, class_name, img)
        shutil.copy2(src, dst)

    # Copy val images
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(VAL_DIR, class_name, img)
        shutil.copy2(src, dst)

    print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")

print("\n🎯 Dataset splitting completed successfully!")
