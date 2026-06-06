import torch
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ==========================================================
# CONFIG
# ==========================================================

CLASS_NAMES = [
    "No_DR",
    "Moderate_NPDR",
    "Severe_NPDR",
    "PDR"
]

IMAGE_SIZE = 512


# ==========================================================
# TRANSFORMS
# ==========================================================

def get_transforms(split):

    if split == "train":

        return transforms.Compose([

            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=15
            ),

            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ==========================================================
# DATA LOADERS
# ==========================================================

def get_dataloaders(
    dataset_path,
    batch_size=8,
    num_workers=8,
    random_state=42
):

    # --------------------------------------
    # Base dataset for labels
    # --------------------------------------

    base_dataset = ImageFolder(
        root=dataset_path
    )

    labels = [
        label
        for _, label in base_dataset.samples
    ]

    indices = np.arange(
        len(labels)
    )

    # --------------------------------------
    # Train split
    # --------------------------------------

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=random_state
    )

    temp_labels = [
        labels[i]
        for i in temp_idx
    ]

    # --------------------------------------
    # Validation + Test split
    # --------------------------------------

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=random_state
    )

    # --------------------------------------
    # Datasets
    # --------------------------------------

    train_dataset = ImageFolder(
        root=dataset_path,
        transform=get_transforms("train")
    )

    val_dataset = ImageFolder(
        root=dataset_path,
        transform=get_transforms("val")
    )

    test_dataset = ImageFolder(
        root=dataset_path,
        transform=get_transforms("test")
    )

    train_dataset = Subset(
        train_dataset,
        train_idx
    )

    val_dataset = Subset(
        val_dataset,
        val_idx
    )

    test_dataset = Subset(
        test_dataset,
        test_idx
    )

    # --------------------------------------
    # Class weights
    # --------------------------------------

    train_labels = [
        labels[i]
        for i in train_idx
    ]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    # --------------------------------------
    # DataLoaders
    # --------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # --------------------------------------
    # Summary
    # --------------------------------------

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Train Samples : {len(train_dataset)}"
    )

    print(
        f"Val Samples   : {len(val_dataset)}"
    )

    print(
        f"Test Samples  : {len(test_dataset)}"
    )

    print("\nClass Mapping")

    for k, v in base_dataset.class_to_idx.items():

        print(
            f"{k:<20} -> {v}"
        )

    print("\nClass Weights")

    for idx, weight in enumerate(class_weights):

        print(
            f"{CLASS_NAMES[idx]:<20} : "
            f"{weight:.4f}"
        )

    print("=" * 60)

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    DATASET_PATH = (
        r"F:\Work\Diabetic-Retinopathy"
        r"\Approach_1\MergedDataset"
    )

    train_loader, val_loader, test_loader, class_weights = (
        get_dataloaders(
            DATASET_PATH,
            batch_size=8
        )
    )

    images, labels = next(
        iter(train_loader)
    )

    print(
        "\nBatch Shape:",
        images.shape
    )

    print(
        "Labels:",
        labels[:10]
    )