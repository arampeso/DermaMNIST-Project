
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from medmnist import DermaMNIST
from sklearn.model_selection import train_test_split, StratifiedKFold


SEED = 59
IMAGE_SIZE = 128  # Alternatively: 28, 128, or 224

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "images"
SPLIT_DIR = DATA_DIR / "splits"


def prepare_directories():
    """Create the required directories and optionally clean old outputs."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for file in IMAGE_DIR.glob("*.png"):
        file.unlink()

    for file in SPLIT_DIR.glob("*.csv"):
        file.unlink()


def load_dataset():
    """Download and combine the official dataset splits."""
    datasets = [
        DermaMNIST(split=split, download=True, size=IMAGE_SIZE)
        for split in ("train", "val", "test")
    ]

    images = np.concatenate(
        [dataset.imgs for dataset in datasets],
        axis=0,
    )

    labels = np.concatenate(
        [dataset.labels for dataset in datasets],
        axis=0,
    ).reshape(-1).astype(int)

    return images, labels


def save_images(images):
    """Save all images as consecutively numbered PNG files."""
    names = []

    for index, image_array in enumerate(images, start=1):
        name = f"{index:05d}.png"
        image_path = IMAGE_DIR / name

        Image.fromarray(image_array).save(image_path)
        names.append(name)

    return names


def write_csv(filename, indices, names, labels):
    """
    Write a CSV file with the columns:
    path, name, label
    """
    csv_path = SPLIT_DIR / filename

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["path", "name", "label"])

        for index in indices:
            name = names[index]

            writer.writerow([
                f"data/images/{name}",
                name,
                int(labels[index]),
            ])

    print(f"Created: {csv_path} ({len(indices)} images)")


def main():
    prepare_directories()

    print("Downloading DermaMNIST...")
    images, labels = load_dataset()

    print(f"Total images: {len(images)}")
    print(f"Classes: {np.unique(labels).tolist()}")

    names = save_images(images)

    # Split the dataset into 80% development and 20% test.
    all_indices = np.arange(len(labels))

    dev_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )

    # Save the independent test set.
    write_csv(
        "test.csv",
        test_indices,
        names,
        labels,
    )

    # Generate five stratified folds within the development set.
    dev_labels = labels[dev_indices]

    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    for split_id, (train_pos, val_pos) in enumerate(
        cross_validator.split(dev_indices, dev_labels)
    ):
        train_indices = dev_indices[train_pos]
        val_indices = dev_indices[val_pos]

        write_csv(
            f"split_{split_id}_train.csv",
            train_indices,
            names,
            labels,
        )

        write_csv(
            f"split_{split_id}_val.csv",
            val_indices,
            names,
            labels,
        )

    print("\nDataset preparation complete!")
    print(f"Total images: {len(images)}")
    print(f"Development images: {len(dev_indices)}")
    print(f"Test images: {len(test_indices)}")
    print(f"Image directory: {IMAGE_DIR.resolve()}")
    print(f"Split directory: {SPLIT_DIR.resolve()}")
    print()


if __name__ == "__main__":
    main()