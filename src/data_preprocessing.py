"""
Data preprocessing, harmonization, undersampling, and augmentation utilities
for lip texture condition dataset.
"""

import os
import glob
import shutil
import random
from typing import Dict, Tuple, List
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
CLASS_LABELS = ['kurang_sehat', 'sehat', 'tidak_sehat']


def is_valid_image(file_path: str) -> bool:
    """Check whether a given image file can be decoded without corruption."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False
    except Exception:
        return False


def clean_corrupted_images(directory: str) -> int:
    """
    Scan directory recursively and remove corrupted or non-decodable images.
    """
    removed_count = 0
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_files: List[str] = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))

    for path in all_files:
        if not is_valid_image(path):
            try:
                os.remove(path)
                removed_count += 1
            except OSError:
                pass
    return removed_count


def sanitize_and_rename_dataset(src_root: str, dst_root: str) -> None:
    """
    Normalize image filenames per class into sequential numeric identifiers (e.g. 1.png, 2.png).
    """
    for split in ['train', 'val', 'test']:
        src_split = os.path.join(src_root, split)
        dst_split = os.path.join(dst_root, split)

        if not os.path.isdir(src_split):
            continue
        os.makedirs(dst_split, exist_ok=True)

        for class_name in os.listdir(src_split):
            src_class_dir = os.path.join(src_split, class_name)
            dst_class_dir = os.path.join(dst_split, class_name)

            if not os.path.isdir(src_class_dir):
                continue
            os.makedirs(dst_class_dir, exist_ok=True)

            images = [
                f for f in os.listdir(src_class_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]

            for i, fname in enumerate(sorted(images), start=1):
                ext = os.path.splitext(fname)[1].lower()
                new_name = f"{i}{ext}"
                shutil.copy2(
                    os.path.join(src_class_dir, fname),
                    os.path.join(dst_class_dir, new_name)
                )


def balance_dataset(
    src_root: str,
    dst_root: str,
    targets: Dict[str, int] = None,
    seed: int = 42
) -> None:
    """
    Undersample classes to create a perfectly balanced dataset distribution.
    Default targets: train=230, val=100, test=100 per class.
    """
    if targets is None:
        targets = {'train': 230, 'val': 100, 'test': 100}

    random.seed(seed)

    for split, target_count in targets.items():
        for label in CLASS_LABELS:
            src_dir = os.path.join(src_root, split, label)
            dst_dir = os.path.join(dst_root, split, label)
            os.makedirs(dst_dir, exist_ok=True)

            if not os.path.isdir(src_dir):
                continue

            all_images = [
                f for f in os.listdir(src_dir)
                if f.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(src_dir, f))
            ]

            if len(all_images) >= target_count:
                selected = random.sample(all_images, target_count)
            else:
                selected = all_images

            for fname in selected:
                shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))


def create_data_generators(
    dataset_dir: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    color_mode: str = 'grayscale'
) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """
    Construct augmented training and normalized validation/test data generators.
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    # Standard normalization for validation and test
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=True,
        classes=CLASS_LABELS,
        interpolation='nearest'
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False,
        classes=CLASS_LABELS,
        interpolation='nearest'
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode=color_mode,
        shuffle=False,
        classes=CLASS_LABELS,
        interpolation='nearest'
    )

    return train_gen, val_gen, test_gen
