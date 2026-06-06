import os
import time
import json
import shutil

import torch
import pandas as pd

from tqdm import tqdm

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from dataset import get_dataloaders
from models import get_model
from evaluate import evaluate_model
from utils import (
    FocalLoss,
    EarlyStopping
)

# ==========================================================
# CONFIG
# ==========================================================

DATASET_PATH = r"Approach_1\MergedDataset"

MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "Moderate_NPDR",
    "No_DR",
    "PDR",
    "Severe_NPDR"
]

MODELS_TO_TRAIN = [
    "custom_cnn",
    "densenet121",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t"
]


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0.0
    running_correct = 0

    for inputs, labels in tqdm(
        dataloader,
        desc="Training",
        leave=False
    ):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() *
            inputs.size(0)
        )

        _, preds = torch.max(
            outputs,
            1
        )

        running_correct += torch.sum(
            preds == labels
        )

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    epoch_acc = (
        running_correct.double() /
        len(dataloader.dataset)
    )

    return (
        epoch_loss,
        epoch_acc.item()
    )


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(
    model_name,
    train_loader,
    val_loader,
    class_weights,
    device
):

    print("\n" + "=" * 70)
    print(f"TRAINING {model_name}")
    print("=" * 70)

    model = get_model(
        model_name=model_name,
        num_classes=4,
        pretrained=True
    )

    model = model.to(device)

    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=2.0
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
    )

    save_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_best.pth"
    )

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        verbose=True,
        path=save_path
    )

    scaler = GradScaler()

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    start_time = time.time()

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device
            )
        )

        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            model_name=model_name,
            class_names=CLASS_NAMES,
            plots_dir=PLOTS_DIR
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        scheduler.step(val_loss)

        early_stopping(
            val_loss,
            model
        )

        if early_stopping.early_stop:

            print(
                "\nEarly stopping triggered."
            )

            break

    training_time = (
        time.time() - start_time
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return (
        training_time,
        num_params
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    os.makedirs(
        LOGS_DIR,
        exist_ok=True
    )

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing Device: {device}"
    )

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights
    ) = get_dataloaders(
        DATASET_PATH,
        batch_size=BATCH_SIZE
    )

    print("\nClass Weights")
    print(class_weights)

    results = []

    criterion = FocalLoss(
        alpha=class_weights.to(device),
        gamma=2.0
    )

    for model_name in MODELS_TO_TRAIN:

        training_time, num_params = (
            train_model(
                model_name,
                train_loader,
                val_loader,
                class_weights,
                device
            )
        )

        model = get_model(
            model_name=model_name,
            num_classes=4,
            pretrained=False
        )

        model.load_state_dict(
            torch.load(
                os.path.join(
                    MODELS_DIR,
                    f"{model_name}_best.pth"
                ),
                map_location=device
            )
        )

        model = model.to(device)

        print(
            f"\nTesting {model_name}"
        )

        test_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            model_name=f"{model_name}_test",
            class_names=CLASS_NAMES,
            plots_dir=PLOTS_DIR
        )

        results.append({

            "Model Name":
                model_name,

            "Accuracy":
                test_metrics["accuracy"],

            "Precision":
                test_metrics["precision"],

            "Recall":
                test_metrics["recall"],

            "F1 Score":
                test_metrics["f1"],

            "Balanced Accuracy":
                test_metrics["balanced_accuracy"],

            "Cohen Kappa":
                test_metrics["cohen_kappa"],

            "ROC-AUC":
                test_metrics["roc_auc"],

            "Training Time":
                training_time,

            "Parameters":
                num_params
        })

        with open(
            os.path.join(
                LOGS_DIR,
                f"{model_name}_metrics.json"
            ),
            "w"
        ) as f:

            json.dump(
                test_metrics,
                f,
                indent=4
            )

    leaderboard = pd.DataFrame(
        results
    )

    leaderboard = leaderboard.sort_values(
        by="F1 Score",
        ascending=False
    )

    leaderboard.to_csv(
        os.path.join(
            RESULTS_DIR,
            "leaderboard.csv"
        ),
        index=False
    )

    print("\n")
    print("=" * 70)
    print("FINAL LEADERBOARD")
    print("=" * 70)

    print(
        leaderboard[
            [
                "Model Name",
                "F1 Score",
                "Accuracy",
                "Balanced Accuracy",
                "ROC-AUC"
            ]
        ]
    )

    best_model = leaderboard.iloc[0][
        "Model Name"
    ]

    shutil.copy(
        os.path.join(
            MODELS_DIR,
            f"{best_model}_best.pth"
        ),
        os.path.join(
            MODELS_DIR,
            "best_model.pth"
        )
    )

    with open(
        os.path.join(
            RESULTS_DIR,
            "best_config.json"
        ),
        "w"
    ) as f:

        json.dump(
            {
                "best_model":
                    best_model
            },
            f,
            indent=4
        )

    print(
        f"\nBest Model: {best_model}"
    )


if __name__ == "__main__":
    main()