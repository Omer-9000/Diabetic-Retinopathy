"""
=============================================================================
  Diabetic Retinopathy - Model Training & Evaluation Superscript
  Approach_1 | MergedDataset (DIP-enhanced fundus images)
=============================================================================

  Classes   : No_DR | Moderate_NPDR | Severe_NPDR | PDR
  Models    : CustomCNN | DenseNet121 | EfficientNet-B3 |
              EfficientNet-V2-S | ConvNeXt-Tiny | Swin-T
  Outputs   : Approach_1/models/   — best .pth per model
              Approach_1/plots/    — learning curves, confusion matrices
              Approach_1/results/  — leaderboard.csv, per-model JSON logs
=============================================================================
"""

import os
import sys
import time
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.amp import GradScaler
from torch.utils.data import DataLoader, Subset

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    densenet121,         DenseNet121_Weights,
    efficientnet_b3,     EfficientNet_B3_Weights,
    efficientnet_v2_s,   EfficientNet_V2_S_Weights,
    convnext_tiny,       ConvNeXt_Tiny_Weights,
    swin_t,              Swin_T_Weights,
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)

from tqdm import tqdm


# =============================================================================
# ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
# ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
# ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
# ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# =============================================================================

# ---------------------------------------------------------------------------
# Paths  —  everything lives inside Approach_1
# ---------------------------------------------------------------------------

BASE_DIR    = r"F:\Work\Diabetic-Retinopathy\Approach_1"
DATASET_DIR = os.path.join(BASE_DIR, "MergedDataset")

MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ---------------------------------------------------------------------------
# Class names  — MUST match subfolder names exactly (alphabetical by default)
# ImageFolder sorts folders alphabetically:
#   Moderate_NPDR=0  |  No_DR=1  |  PDR=2  |  Severe_NPDR=3
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "Moderate_NPDR",
    "No_DR",
    "PDR",
    "Severe_NPDR",
]

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

IMAGE_SIZE    = 512
BATCH_SIZE    = 8
NUM_WORKERS   = 4       # lower if you get DataLoader memory errors on Windows
EPOCHS        = 50
PATIENCE      = 10      # early stopping
LEARNING_RATE = 3e-4    # slightly higher start; scheduler will decay it
WEIGHT_DECAY  = 1e-4
FOCAL_GAMMA   = 2.0
RANDOM_STATE  = 42

# ---------------------------------------------------------------------------
# Models to benchmark (comment out any you want to skip)
# ---------------------------------------------------------------------------

MODELS_TO_TRAIN = [
    "custom_cnn",
    "densenet121",
    "efficientnet_b3",
    "efficientnet_v2_s",
    "convnext_tiny",
    "swin_t",
]


# =============================================================================
# DATASET
# =============================================================================

def get_transforms(split: str):
    """
    Fundus-specific augmentations:
      - Vertical flip added (fundus images have no fixed orientation)
      - ColorJitter carefully tuned (fundus colour carries diagnostic info)
      - No aggressive perspective/affine that would destroy vessel patterns
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.90, 1.10),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.01,           # tiny — hue shift ruins fundus colour
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_dataloaders(dataset_path, batch_size=8, num_workers=4, random_state=42):
    """
    Stratified 70 / 15 / 15 split.
    Returns (train_loader, val_loader, test_loader, class_weights, class_names)
    class_names is taken from the actual folder structure so order is guaranteed.
    """

    # ---- labels from plain ImageFolder (no transform needed for indexing) ----
    base = ImageFolder(root=dataset_path)
    labels = [lbl for _, lbl in base.samples]
    indices = np.arange(len(labels))

    # ---- split ---------------------------------------------------------------
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )
    temp_labels = [labels[i] for i in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=random_state,
    )

    # ---- datasets with correct transforms ------------------------------------
    def make_ds(split, idx):
        ds = ImageFolder(root=dataset_path, transform=get_transforms(split))
        return Subset(ds, idx)

    train_ds = make_ds("train", train_idx)
    val_ds   = make_ds("val",   val_idx)
    test_ds  = make_ds("test",  test_idx)

    # ---- class weights (computed on train set only) --------------------------
    train_labels = [labels[i] for i in train_idx]
    raw_weights  = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(raw_weights, dtype=torch.float32)

    # ---- loaders -------------------------------------------------------------
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ---- summary -------------------------------------------------------------
    actual_class_names = [c for c, _ in sorted(base.class_to_idx.items(), key=lambda x: x[1])]

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total samples : {len(base)}")
    print(f"  Train         : {len(train_ds)}")
    print(f"  Val           : {len(val_ds)}")
    print(f"  Test          : {len(test_ds)}")
    print(f"\n  Class mapping (alphabetical = ImageFolder order):")
    for cls, idx in sorted(base.class_to_idx.items(), key=lambda x: x[1]):
        count = labels.count(idx)
        print(f"    [{idx}] {cls:<20}  {count:>5} samples")
    print(f"\n  Class weights (balanced, train split):")
    for i, (name, w) in enumerate(zip(actual_class_names, class_weights)):
        print(f"    [{i}] {name:<20}  {w:.4f}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, class_weights, actual_class_names


# =============================================================================
# MODELS
# =============================================================================

class CustomDRCNN(nn.Module):
    """
    Deeper custom CNN with residual-style skip connection in block 4,
    SE attention, and stronger regularisation.
    """

    def __init__(self, num_classes=4):
        super().__init__()

        self.block1 = self._conv_block(3,    32)
        self.block2 = self._conv_block(32,   64)
        self.block3 = self._conv_block(64,  128)
        self.block4 = self._conv_block(128, 256)
        self.block5 = self._conv_block(256, 512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.se = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)                          # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)                # (B, 512)
        se_w = self.se(x)
        x = x * se_w                              # channel attention
        x = x.unsqueeze(-1).unsqueeze(-1)         # restore shape for classifier
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(model_name: str, num_classes: int = 4, pretrained: bool = True):

    W = "DEFAULT" if pretrained else None

    if model_name == "custom_cnn":
        return CustomDRCNN(num_classes)

    elif model_name == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m

    elif model_name == "efficientnet_b3":
        m = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    elif model_name == "efficientnet_v2_s":
        m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    elif model_name == "convnext_tiny":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m

    elif model_name == "swin_t":
        m = swin_t(weights=Swin_T_Weights.DEFAULT if pretrained else None)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss with class-weight alpha.
    alpha : tensor of shape (num_classes,) — per-class weights
    gamma : focusing parameter (2.0 is standard)
    """

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:

    def __init__(self, patience=10, path="checkpoint.pth", verbose=True):
        self.patience  = patience
        self.path      = path
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            if self.verbose:
                print(f"  ✓ Val loss improved ({self.best_loss:.5f} → {val_loss:.5f})  Saving …")
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  · No improvement  [{self.counter}/{self.patience}]")
            if self.counter >= self.patience:
                self.early_stop = True


# =============================================================================
# TRAINING (one epoch)
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct = 0.0, 0

    for inputs, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_correct / n


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Eval ", leave=False, ncols=80):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            probs       = torch.softmax(outputs, dim=1)
            preds       = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    n           = len(loader.dataset)
    epoch_loss  = total_loss / n
    all_preds   = np.array(all_preds)
    all_labels  = np.array(all_labels)
    all_probs   = np.array(all_probs)

    accuracy  = accuracy_score(all_labels, all_preds)
    bal_acc   = balanced_accuracy_score(all_labels, all_preds)
    kappa     = cohen_kappa_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    return {
        "loss":              float(epoch_loss),
        "accuracy":          float(accuracy),
        "balanced_accuracy": float(bal_acc),
        "precision":         float(prec),
        "recall":            float(rec),
        "f1":                float(f1),
        "cohen_kappa":       float(kappa),
        "roc_auc":           float(roc_auc),
        "per_class_precision": pc_prec.tolist(),
        "per_class_recall":    pc_rec.tolist(),
        "per_class_f1":        pc_f1.tolist(),
        "_preds":   all_preds,
        "_labels":  all_labels,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def save_learning_curves(history: dict, model_name: str, save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Learning Curves", fontsize=14)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_confusion_matrix(
    labels, preds, class_names: list, model_name: str, save_dir: str
):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_per_class_bar(metrics: dict, class_names: list, model_name: str, save_dir: str):
    x    = np.arange(len(class_names))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   metrics["per_class_precision"], w, label="Precision")
    ax.bar(x,       metrics["per_class_recall"],    w, label="Recall")
    ax.bar(x + w,   metrics["per_class_f1"],        w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name} — Per-Class Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_per_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PRINT METRICS
# =============================================================================

def print_metrics(metrics: dict, class_names: list, tag: str = ""):
    header = f"{'─'*60}"
    print(f"\n{header}")
    if tag:
        print(f"  {tag}")
    print(header)
    print(f"  Loss              : {metrics['loss']:.4f}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision (w)     : {metrics['precision']:.4f}")
    print(f"  Recall    (w)     : {metrics['recall']:.4f}")
    print(f"  F1        (w)     : {metrics['f1']:.4f}")
    print(f"  Cohen Kappa       : {metrics['cohen_kappa']:.4f}")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"\n  Per-class F1:")
    for name, f1 in zip(class_names, metrics["per_class_f1"]):
        bar = "█" * int(f1 * 20)
        print(f"    {name:<20} {f1:.4f}  {bar}")
    print(header)


# =============================================================================
# TRAIN ONE MODEL
# =============================================================================

def train_model(
    model_name: str,
    train_loader,
    val_loader,
    class_weights,
    class_names: list,
    device,
):
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  TRAINING  :  {model_name.upper()}")
    print(f"{sep}")

    model = get_model(model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    ckpt_path      = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    early_stopping = EarlyStopping(patience=PATIENCE, path=ckpt_path, verbose=True)
    scaler         = GradScaler("cuda")

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch:>2}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss    = val_metrics["loss"]
        val_acc     = val_metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}"
            f"    Val  loss={val_loss:.4f}  acc={val_acc:.4f}"
            f"    LR={lr_now:.2e}"
        )

        scheduler.step()
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"\n  ⚑ Early stopping at epoch {epoch}")
            break

    training_time = time.time() - t0
    n_params      = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- save learning curves -----------------------------------------------
    save_learning_curves(history, model_name, PLOTS_DIR)

    print(f"\n  Done.  Time={training_time/60:.1f} min  Params={n_params/1e6:.2f}M")
    return training_time, n_params


# =============================================================================
# TEST ONE MODEL  (load best weights → evaluate on test set)
# =============================================================================

def test_model(model_name, test_loader, class_weights, class_names, device):
    model = get_model(model_name, num_classes=len(class_names), pretrained=False)
    ckpt  = os.path.join(MODELS_DIR, f"{model_name}_best.pth")

    if not os.path.exists(ckpt):
        print(f"  [!] Checkpoint not found: {ckpt}  Skipping test.")
        return None

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
    metrics   = evaluate(model, test_loader, criterion, device)

    print_metrics(metrics, class_names, tag=f"{model_name.upper()}  ·  TEST SET")

    save_confusion_matrix(
        metrics.pop("_labels"), metrics.pop("_preds"),
        class_names, f"{model_name}_test", PLOTS_DIR,
    )
    save_per_class_bar(metrics, class_names, f"{model_name}_test", PLOTS_DIR)

    # save JSON log
    log_path = os.path.join(RESULTS_DIR, f"{model_name}_test_metrics.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved: {log_path}")

    return metrics


# =============================================================================
# LEADERBOARD
# =============================================================================

def build_leaderboard(all_results: list):
    df = pd.DataFrame(all_results)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    path = os.path.join(RESULTS_DIR, "leaderboard.csv")
    df.to_csv(path, index=False)

    sep = "═" * 80
    print(f"\n{sep}")
    print("  FINAL LEADERBOARD")
    print(sep)
    print(
        df[[
            "Rank", "Model Name", "F1 Score", "Accuracy",
            "Balanced Accuracy", "ROC-AUC", "Cohen Kappa",
            "Params (M)", "Time (min)",
        ]].to_string(index=False)
    )
    print(sep)
    print(f"\n  Leaderboard saved → {path}")
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():

    # ---- directories ---------------------------------------------------------
    for d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    # ---- device --------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- data ----------------------------------------------------------------
    (
        train_loader, val_loader, test_loader,
        class_weights, class_names,
    ) = get_dataloaders(
        DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        random_state=RANDOM_STATE,
    )

    # ---- sanity check: print one batch shape --------------------------------
    imgs, lbls = next(iter(train_loader))
    print(f"\n  Batch check  →  images {tuple(imgs.shape)}  labels {tuple(lbls.shape)}")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    all_results = []

    for model_name in MODELS_TO_TRAIN:

        # train
        try:
            training_time, n_params = train_model(
                model_name, train_loader, val_loader,
                class_weights, class_names, device,
            )
        except Exception as e:
            print(f"\n  [ERROR] Training {model_name} failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # test
        try:
            test_metrics = test_model(
                model_name, test_loader,
                class_weights, class_names, device,
            )
        except Exception as e:
            print(f"\n  [ERROR] Testing {model_name} failed: {e}")
            import traceback; traceback.print_exc()
            continue

        if test_metrics is None:
            continue

        all_results.append({
            "Model Name":       model_name,
            "F1 Score":         test_metrics["f1"],
            "Accuracy":         test_metrics["accuracy"],
            "Balanced Accuracy":test_metrics["balanced_accuracy"],
            "Precision":        test_metrics["precision"],
            "Recall":           test_metrics["recall"],
            "Cohen Kappa":      test_metrics["cohen_kappa"],
            "ROC-AUC":          test_metrics["roc_auc"],
            "Params (M)":       round(n_params / 1e6, 2),
            "Time (min)":       round(training_time / 60, 1),
        })

    # =========================================================================
    # LEADERBOARD + BEST MODEL COPY
    # =========================================================================

    if not all_results:
        print("\n  No results to report.")
        return

    leaderboard = build_leaderboard(all_results)
    best_name   = leaderboard.iloc[0]["Model Name"]

    src = os.path.join(MODELS_DIR, f"{best_name}_best.pth")
    dst = os.path.join(MODELS_DIR, "best_model.pth")
    shutil.copy(src, dst)

    best_cfg = {
        "best_model":    best_name,
        "f1_score":      leaderboard.iloc[0]["F1 Score"],
        "roc_auc":       leaderboard.iloc[0]["ROC-AUC"],
        "class_names":   class_names,
    }
    with open(os.path.join(RESULTS_DIR, "best_config.json"), "w") as f:
        json.dump(best_cfg, f, indent=4)

    print(f"\n  🏆  Best model  : {best_name}")
    print(f"      F1 Score    : {leaderboard.iloc[0]['F1 Score']:.4f}")
    print(f"      Checkpoint  : {dst}")
    print("\n  Next step: modify the winning architecture and re-run on it alone.\n")


# =============================================================================

if __name__ == "__main__":
    main()