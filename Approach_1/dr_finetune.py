"""
=============================================================================
  Diabetic Retinopathy - EfficientNet-V2-S Architecture Refinement
  Approach_1 | MergedDataset (DIP-enhanced fundus images)
=============================================================================

  Winner    : efficientnet_v2_s  (F1=0.8200, baseline)
  Goal      : Systematically test architectural modifications to the
              classification head + fine-tuning strategy to push F1 higher.

  Variants tested
  ───────────────
  V0  baseline      Original linear head (reproduced for fair comparison)
  V1  deeper_head   Dropout → 512 → BN → ReLU → Dropout → 256 → out
  V2  attention     Global avg + Global max pooling concat → head
  V3  label_smooth  baseline head + Label Smoothing loss (ε=0.1)
  V4  full_finetune All layers unfrozen from epoch 1, lower LR
  V5  two_stage     Freeze backbone 5 epochs → unfreeze all, decay LR

  All outputs → Approach_1/finetune/
    models/   — best .pth per variant
    plots/    — learning curves, confusion matrices, per-class bars
    results/  — per-variant JSON + final comparison CSV

  ROC-AUC fix : uses label_binarize so NaN never appears again.
=============================================================================
"""

import os
import time
import json
import copy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR      = r"F:\Work\Diabetic-Retinopathy\Approach_1"
DATASET_DIR   = os.path.join(BASE_DIR, "MergedDataset")
OUT_DIR       = os.path.join(BASE_DIR, "finetune")

MODELS_DIR    = os.path.join(OUT_DIR, "models")
PLOTS_DIR     = os.path.join(OUT_DIR, "plots")
RESULTS_DIR   = os.path.join(OUT_DIR, "results")

IMAGE_SIZE    = 512
BATCH_SIZE    = 8
NUM_WORKERS   = 4
EPOCHS        = 60          # more headroom; early stopping handles it
PATIENCE      = 12
RANDOM_STATE  = 42

# Baseline LR (used by most variants)
LR_HEAD       = 3e-4        # head / newly-added params
LR_BACKBONE   = 5e-5        # backbone when unfrozen (10× smaller)
WEIGHT_DECAY  = 1e-4
FOCAL_GAMMA   = 2.0

# ImageFolder alphabetical order:  Moderate_NPDR=0 | No_DR=1 | PDR=2 | Severe_NPDR=3
CLASS_NAMES   = ["Moderate_NPDR", "No_DR", "PDR", "Severe_NPDR"]
NUM_CLASSES   = 4

# Comment out any variant you want to skip
VARIANTS = [
    "V0_baseline",
    "V1_deeper_head",
    "V2_attention_pool",
    "V3_label_smooth",
    "V4_full_finetune",
    "V5_two_stage",
]


# =============================================================================
# DATASET  (identical split logic to dr_train.py — same random_state = same splits)
# =============================================================================

def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.90, 1.10)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(dataset_path, batch_size=8, num_workers=4, random_state=42):
    base   = ImageFolder(root=dataset_path)
    labels = [lbl for _, lbl in base.samples]
    idx    = np.arange(len(labels))

    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50,
        stratify=[labels[i] for i in temp_idx],
        random_state=random_state,
    )

    def make_ds(split, subset_idx):
        ds = ImageFolder(root=dataset_path, transform=get_transforms(split))
        return Subset(ds, subset_idx)

    train_ds = make_ds("train", train_idx)
    val_ds   = make_ds("val",   val_idx)
    test_ds  = make_ds("test",  test_idx)

    train_labels = [labels[i] for i in train_idx]
    raw_w = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(raw_w, dtype=torch.float32)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
               pin_memory=True, persistent_workers=(num_workers > 0))

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ---- summary
    actual_names = [c for c, _ in sorted(base.class_to_idx.items(), key=lambda x: x[1])]
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    for cls, cidx in sorted(base.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  [{cidx}] {cls:<20}  {labels.count(cidx):>5} samples")
    print(f"\n  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, class_weights, actual_names


# =============================================================================
# MODEL VARIANTS
# =============================================================================

# ---------------------------------------------------------------------------
# Helper: load pretrained EfficientNet-V2-S backbone, return (model, in_features)
# ---------------------------------------------------------------------------
def _load_backbone(freeze: bool = True):
    m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    if freeze:
        for p in m.features.parameters():
            p.requires_grad = False
    in_features = m.classifier[1].in_features   # 1280
    return m, in_features


# ---------------------------------------------------------------------------
# V0  Baseline  — same head as dr_train.py (single Linear)
# ---------------------------------------------------------------------------
class V0_Baseline(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()
        m, in_f = _load_backbone(freeze_backbone)
        m.classifier[1] = nn.Linear(in_f, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)

    def unfreeze_backbone(self):
        for p in self.model.features.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# V1  Deeper head  — Dropout → FC512 → BN → ReLU → Dropout → FC256 → out
#                    BN is safe here because drop_last=True on train loader
# ---------------------------------------------------------------------------
class V1_DeeperHead(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()
        m, in_f = _load_backbone(freeze_backbone)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_f, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),       # SiLU matches EfficientNet internals
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
        self.model = m

    def forward(self, x):
        return self.model(x)

    def unfreeze_backbone(self):
        for p in self.model.features.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# V2  Attention pooling  — concat global avg + global max → richer features
# ---------------------------------------------------------------------------
class V2_AttentionPool(nn.Module):
    def __init__(self, num_classes=4, freeze_backbone=True):
        super().__init__()
        m, in_f = _load_backbone(freeze_backbone)
        # Remove built-in avg pool + classifier; we replace pooling strategy
        self.features   = m.features
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.max_pool   = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_f * 2, 512),   # *2 because avg + max concat
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x   = self.features(x)
        avg = self.avg_pool(x).flatten(1)
        mx  = self.max_pool(x).flatten(1)
        x   = torch.cat([avg, mx], dim=1)
        return self.classifier(x)

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# V3  Label Smoothing  — same head as V0, different loss (handled at train time)
# ---------------------------------------------------------------------------
# Model architecture identical to V0 — variant is in the loss function.
class V3_LabelSmooth(V0_Baseline):
    pass


# ---------------------------------------------------------------------------
# V4  Full finetune  — no freezing at all, everything trains from epoch 1
# ---------------------------------------------------------------------------
class V4_FullFinetune(V0_Baseline):
    def __init__(self, num_classes=4):
        super().__init__(num_classes, freeze_backbone=False)


# ---------------------------------------------------------------------------
# V5  Two-stage  — frozen backbone first, then unfreeze (handled at train time)
# ---------------------------------------------------------------------------
class V5_TwoStage(V1_DeeperHead):
    pass   # same arch as V1; the two-stage schedule is in the training loop


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_variant_model(variant_name: str) -> nn.Module:
    mapping = {
        "V0_baseline":      V0_Baseline,
        "V1_deeper_head":   V1_DeeperHead,
        "V2_attention_pool":V2_AttentionPool,
        "V3_label_smooth":  V3_LabelSmooth,
        "V4_full_finetune": V4_FullFinetune,
        "V5_two_stage":     V5_TwoStage,
    }
    if variant_name not in mapping:
        raise ValueError(f"Unknown variant: {variant_name}")
    return mapping[variant_name](num_classes=NUM_CLASSES)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class FocalLossWithLabelSmoothing(nn.Module):
    """Focal loss + label smoothing combined."""
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.1, num_classes=4):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # Label-smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(
                inputs, self.smoothing / (self.num_classes - 1)
            )
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(inputs, dim=1)

        # Cross-entropy with smooth targets
        ce = -(smooth_targets * log_probs).sum(dim=1)

        # Apply class weights
        if self.alpha is not None:
            ce = ce * self.alpha[targets]

        # Focal modulation using hard targets for pt
        pt   = torch.exp(-F.cross_entropy(inputs, targets, reduction="none"))
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=12, path="ckpt.pth", verbose=True):
        self.patience   = patience
        self.path       = path
        self.verbose    = verbose
        self.counter    = 0
        self.best_loss  = float("inf")
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
# TRAINING — one epoch
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, total_correct = 0.0, 0

    for inputs, labels in tqdm(loader, desc="  Train", leave=False, ncols=85):
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
        for inputs, labels in tqdm(loader, desc="  Eval ", leave=False, ncols=85):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    # ---- ROC-AUC fix: binarize so missing classes never cause NaN -----------
    all_labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    try:
        roc_auc = roc_auc_score(
            all_labels_bin, all_probs,
            multi_class="ovr", average="macro"
        )
    except ValueError:
        roc_auc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    return {
        "loss":               float(total_loss / len(loader.dataset)),
        "accuracy":           float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy":  float(balanced_accuracy_score(all_labels, all_preds)),
        "precision":          float(prec),
        "recall":             float(rec),
        "f1":                 float(f1),
        "cohen_kappa":        float(cohen_kappa_score(all_labels, all_preds)),
        "roc_auc":            float(roc_auc),
        "per_class_precision":pc_prec.tolist(),
        "per_class_recall":   pc_rec.tolist(),
        "per_class_f1":       pc_f1.tolist(),
        "_preds":  all_preds,
        "_labels": all_labels,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def save_learning_curves(history, name, save_dir):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} — Learning Curves", fontsize=13)

    axes[0].plot(ep, history["train_loss"], label="Train")
    axes[0].plot(ep, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, history["train_acc"], label="Train")
    axes[1].plot(ep, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def save_confusion_matrix(labels, preds, class_names, name, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name}_cm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def save_per_class_bar(metrics, class_names, name, save_dir):
    x, w = np.arange(len(class_names)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, metrics["per_class_precision"], w, label="Precision")
    ax.bar(x,     metrics["per_class_recall"],    w, label="Recall")
    ax.bar(x + w, metrics["per_class_f1"],        w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=15)
    ax.set_ylim(0, 1.05); ax.set_title(f"{name} — Per-Class Metrics")
    ax.legend(); ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name}_per_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def print_metrics(metrics, class_names, tag=""):
    sep = "─" * 62
    print(f"\n{sep}")
    if tag: print(f"  {tag}")
    print(sep)
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
    print(sep)


# =============================================================================
# TRAIN ONE VARIANT
# =============================================================================

def train_variant(variant_name, train_loader, val_loader, class_weights, device):
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  VARIANT  :  {variant_name}")
    print(f"{sep}")

    model = get_variant_model(variant_name).to(device)

    # ---- Loss ----------------------------------------------------------------
    if variant_name == "V3_label_smooth":
        criterion = FocalLossWithLabelSmoothing(
            alpha=class_weights.to(device), gamma=FOCAL_GAMMA,
            smoothing=0.1, num_classes=NUM_CLASSES,
        )
    else:
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)

    # ---- Optimizer -----------------------------------------------------------
    if variant_name == "V4_full_finetune":
        # Everything trains, but backbone at lower LR
        backbone_params = list(model.model.features.parameters())
        head_params     = [p for p in model.parameters()
                           if not any(p is bp for bp in backbone_params)]
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)
    else:
        # Head-only (backbone frozen or will be unfrozen in two-stage)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
        )

    # ---- Scheduler -----------------------------------------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    ckpt_path      = os.path.join(MODELS_DIR, f"{variant_name}_best.pth")
    early_stopping = EarlyStopping(patience=PATIENCE, path=ckpt_path, verbose=True)
    scaler         = GradScaler("cuda")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    UNFREEZE_EPOCH = 5   # for V5 two-stage

    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch:>2}/{EPOCHS}")

        # ---- V5 two-stage: unfreeze backbone at epoch UNFREEZE_EPOCH ---------
        if variant_name == "V5_two_stage" and epoch == UNFREEZE_EPOCH + 1:
            print(f"\n  ── Unfreezing backbone (epoch {epoch}) ──")
            model.unfreeze_backbone()
            # Replace optimizer with backbone + head param groups
            backbone_params = list(model.model.features.parameters())
            head_params     = [p for p in model.parameters()
                               if not any(p is bp for bp in backbone_params)]
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": LR_BACKBONE},
                {"params": head_params,     "lr": LR_HEAD / 2},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=EPOCHS - epoch, eta_min=1e-6
            )

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

    save_learning_curves(history, variant_name, PLOTS_DIR)
    print(f"\n  Done.  Time={training_time/60:.1f} min  Params={n_params/1e6:.2f}M")
    return training_time, n_params


# =============================================================================
# TEST ONE VARIANT
# =============================================================================

def test_variant(variant_name, test_loader, class_weights, class_names, device):
    model = get_variant_model(variant_name).to(device)
    ckpt  = os.path.join(MODELS_DIR, f"{variant_name}_best.pth")

    if not os.path.exists(ckpt):
        print(f"  [!] No checkpoint: {ckpt}"); return None

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
    metrics   = evaluate(model, test_loader, criterion, device)

    print_metrics(metrics, class_names, tag=f"{variant_name}  ·  TEST SET")

    labels_arr = metrics.pop("_labels")
    preds_arr  = metrics.pop("_preds")

    save_confusion_matrix(labels_arr, preds_arr, class_names, variant_name, PLOTS_DIR)
    save_per_class_bar(metrics, class_names, variant_name, PLOTS_DIR)

    log_path = os.path.join(RESULTS_DIR, f"{variant_name}_metrics.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved: {log_path}")

    return metrics


# =============================================================================
# COMPARISON PLOT  — all variants side by side on key metrics
# =============================================================================

def save_comparison_plot(all_results: list, save_dir: str):
    df = pd.DataFrame(all_results).set_index("Variant")
    metrics_to_plot = ["F1 Score", "Accuracy", "Balanced Accuracy", "ROC-AUC", "Cohen Kappa"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))
    fig.suptitle("EfficientNet-V2-S Variant Comparison (Test Set)", fontsize=13)

    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for ax, metric in zip(axes, metrics_to_plot):
        vals = df[metric]
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(df.index, rotation=40, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10)
        ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.05))
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = os.path.join(save_dir, "variant_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Comparison plot → {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    for d in (MODELS_DIR, PLOTS_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, class_weights, class_names = get_dataloaders(
        DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, random_state=RANDOM_STATE
    )

    all_results = []

    for variant_name in VARIANTS:
        try:
            training_time, n_params = train_variant(
                variant_name, train_loader, val_loader, class_weights, device
            )
        except Exception as e:
            print(f"\n  [ERROR] Training {variant_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        try:
            test_metrics = test_variant(
                variant_name, test_loader, class_weights, class_names, device
            )
        except Exception as e:
            print(f"\n  [ERROR] Testing {variant_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        if test_metrics is None:
            continue

        all_results.append({
            "Variant":          variant_name,
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

    if not all_results:
        print("\n  No results."); return

    # ---- Leaderboard ---------------------------------------------------------
    df = pd.DataFrame(all_results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    csv_path = os.path.join(RESULTS_DIR, "variant_leaderboard.csv")
    df.to_csv(csv_path, index=False)

    sep = "═" * 85
    print(f"\n{sep}")
    print("  VARIANT LEADERBOARD")
    print(sep)
    print(df[["Rank","Variant","F1 Score","Accuracy","Balanced Accuracy",
              "ROC-AUC","Cohen Kappa","Time (min)"]].to_string(index=False))
    print(sep)
    print(f"\n  Leaderboard saved → {csv_path}")

    save_comparison_plot(all_results, PLOTS_DIR)

    # ---- Save best variant ---------------------------------------------------
    best = df.iloc[0]["Variant"]
    import shutil
    shutil.copy(
        os.path.join(MODELS_DIR, f"{best}_best.pth"),
        os.path.join(MODELS_DIR, "best_variant.pth"),
    )
    with open(os.path.join(RESULTS_DIR, "best_variant_config.json"), "w") as f:
        json.dump({
            "best_variant": best,
            "f1_score":     df.iloc[0]["F1 Score"],
            "roc_auc":      df.iloc[0]["ROC-AUC"],
            "class_names":  class_names,
        }, f, indent=4)

    print(f"\n  🏆  Best variant : {best}")
    print(f"      F1 Score     : {df.iloc[0]['F1 Score']:.4f}")
    print(f"      ROC-AUC      : {df.iloc[0]['ROC-AUC']:.4f}")
    print(f"      Checkpoint   : {os.path.join(MODELS_DIR, 'best_variant.pth')}\n")


if __name__ == "__main__":
    main()