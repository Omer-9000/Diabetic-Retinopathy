"""
=============================================================================
  Diabetic Retinopathy - Advanced Refinement of V4 Full Finetune
  Approach_1 | MergedDataset
=============================================================================

  Baseline  : V4_full_finetune  F1=0.7903  ROC-AUC=0.9483
  Goal      : Push F1 higher via techniques that don't need more data.

  Techniques
  ──────────
  Stage 1 — Post-hoc (no retraining, uses existing V4 checkpoint)
    T1  Threshold optimisation  : per-class thresholds on val set → best F1
    T2  Test-time augmentation  : 8-flip/rotate passes averaged → better probs

  Stage 2 — Retrain with improved training recipe
    R1  Differential LR         : 5 backbone stage groups, each 2× lower than next
    R2  MixUp                   : α=0.2 fundus-safe mixing
    R3  Cosine warmup           : 3-epoch linear warmup before cosine decay
    R1+R2 combined              : differential LR AND MixUp together

  Outputs → Approach_1/advanced/
    models/   — checkpoints
    plots/    — curves, confusion matrices, per-class bars, calibration curves
    results/  — JSON logs, final leaderboard CSV

  FIXES
  ─────
  1. load_v4()          : strips "model." prefix from Lightning-wrapped checkpoints.
  2. get_tta_transforms(): replaced unpicklable local lambdas with module-level
                          rotate callable classes (Python 3.13 spawn fix).
  3. evaluate_tta()     : uses num_workers=0 to avoid multiprocessing pickle errors
                          with lambda-based transforms on Windows/Python 3.13+.
  4. T1+T2 guard        : uses explicit variables instead of fragile dir() check.
=============================================================================
"""

import os
import time
import json
import math

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
    f1_score,
)
from sklearn.preprocessing import label_binarize

from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR    = r"F:\Work\Diabetic-Retinopathy\Approach_1"
DATASET_DIR = os.path.join(BASE_DIR, "MergedDataset")
V4_CKPT     = os.path.join(BASE_DIR, "finetune", "models", "V4_full_finetune_best.pth")

OUT_DIR     = os.path.join(BASE_DIR, "advanced")
MODELS_DIR  = os.path.join(OUT_DIR, "models")
PLOTS_DIR   = os.path.join(OUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUT_DIR, "results")

IMAGE_SIZE   = 512
BATCH_SIZE   = 8
NUM_WORKERS  = 4
EPOCHS       = 60
PATIENCE     = 12
RANDOM_STATE = 42
NUM_CLASSES  = 4
FOCAL_GAMMA  = 2.0
WEIGHT_DECAY = 1e-4

CLASS_NAMES = ["Moderate_NPDR", "No_DR", "PDR", "Severe_NPDR"]

POST_HOC_TECHNIQUES = [
    "T1_threshold_opt",
    "T2_tta",
    "T1+T2_combined",
]

RETRAIN_VARIANTS = [
    "R1_differential_lr",
    "R2_mixup",
    "R3_warmup",
    "R1+R2_combined",
]


# =============================================================================
# PICKLABLE ROTATE TRANSFORMS
# FIX: Python 3.13 uses 'spawn' multiprocessing on Windows, which requires all
# objects passed to worker processes to be picklable. Local lambdas are NOT
# picklable. We use module-level callable classes instead.
# =============================================================================

class Rotate90:
    def __call__(self, x):
        return transforms.functional.rotate(x, 90)

class Rotate180:
    def __call__(self, x):
        return transforms.functional.rotate(x, 180)

class Rotate270:
    def __call__(self, x):
        return transforms.functional.rotate(x, 270)


# =============================================================================
# DATASET
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


def get_tta_transforms():
    """
    8 deterministic TTA passes using picklable module-level rotate classes.
    FIX: replaced local lambdas (unpicklable under Python 3.13 spawn) with
    Rotate90 / Rotate180 / Rotate270 classes defined at module level.
    """
    norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def make(extra):
        return transforms.Compose(
            [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))] + extra + norm
        )

    return [
        make([]),                                                          # original
        make([transforms.RandomHorizontalFlip(p=1.0)]),                   # H-flip
        make([transforms.RandomVerticalFlip(p=1.0)]),                     # V-flip
        make([transforms.RandomHorizontalFlip(p=1.0),
              transforms.RandomVerticalFlip(p=1.0)]),                     # HV-flip
        make([Rotate90()]),                                                # 90°
        make([Rotate180()]),                                               # 180°
        make([Rotate270()]),                                               # 270°
        make([transforms.RandomHorizontalFlip(p=1.0), Rotate90()]),       # H-flip + 90°
    ]


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

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)
    for cls, cidx in sorted(base.class_to_idx.items(), key=lambda x: x[1]):
        print(f"  [{cidx}] {cls:<20}  {labels.count(cidx):>5}")
    print(f"  Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")
    print("=" * 60)

    return train_loader, val_loader, test_loader, class_weights, train_idx, val_idx, test_idx


# =============================================================================
# MODEL
# =============================================================================

def build_model(pretrained=True, freeze_backbone=False):
    m = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for p in m.features.parameters():
            p.requires_grad = False
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m


def load_v4(device):
    """
    Load V4 checkpoint, stripping the 'model.' prefix added by wrapper classes
    (e.g. LightningModule) when the checkpoint was saved.
    FIX: key remapping so bare EfficientNet state_dict matches.
    """
    model = build_model(pretrained=False)
    state_dict = torch.load(V4_CKPT, map_location=device, weights_only=True)

    new_state_dict = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict)
    return model.to(device)


# =============================================================================
# LOSS
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class FocalLossSoft(nn.Module):
    """Focal loss that accepts soft (MixUp) targets."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, soft_targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs     = torch.softmax(inputs, dim=1)

        ce = -(soft_targets * log_probs).sum(dim=1)

        if self.alpha is not None:
            w  = (soft_targets * self.alpha.to(inputs.device)).sum(dim=1)
            ce = ce * w

        hard_class = soft_targets.argmax(dim=1)
        pt = probs.gather(1, hard_class.unsqueeze(1)).squeeze(1).detach()
        return (((1 - pt) ** self.gamma) * ce).mean()


# =============================================================================
# MIXUP
# =============================================================================

def mixup_batch(inputs, targets, alpha=0.2, num_classes=4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam = max(lam, 1 - lam)

    batch_size = inputs.size(0)
    idx        = torch.randperm(batch_size, device=inputs.device)

    mixed_inputs       = lam * inputs + (1 - lam) * inputs[idx]
    targets_onehot     = F.one_hot(targets, num_classes).float()
    mixed_targets      = lam * targets_onehot + (1 - lam) * targets_onehot[idx]

    return mixed_inputs, mixed_targets


# =============================================================================
# SCHEDULER
# =============================================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= self.warmup_epochs:
                pg["lr"] = base_lr * e / self.warmup_epochs
            else:
                progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                pg["lr"] = self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                    1 + math.cos(math.pi * progress)
                )

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


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
                print(f"  ✓ {self.best_loss:.5f} → {val_loss:.5f}  Saving …")
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  · No improvement [{self.counter}/{self.patience}]")
            if self.counter >= self.patience:
                self.early_stop = True


# =============================================================================
# EVALUATION
# =============================================================================

def compute_metrics(all_labels, all_preds, all_probs, total_loss, n):
    all_labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    try:
        roc_auc = roc_auc_score(all_labels_bin, all_probs,
                                multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    pc_prec, pc_rec, pc_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    return {
        "loss":                float(total_loss / n),
        "accuracy":            float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy":   float(balanced_accuracy_score(all_labels, all_preds)),
        "precision":           float(prec),
        "recall":              float(rec),
        "f1":                  float(f1),
        "cohen_kappa":         float(cohen_kappa_score(all_labels, all_preds)),
        "roc_auc":             float(roc_auc),
        "per_class_precision": pc_prec.tolist(),
        "per_class_recall":    pc_rec.tolist(),
        "per_class_f1":        pc_f1.tolist(),
        "_preds":  np.array(all_preds),
        "_labels": np.array(all_labels),
        "_probs":  np.array(all_probs),
    }


def evaluate_standard(model, loader, criterion, device):
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

    return compute_metrics(
        np.array(all_labels), np.array(all_preds),
        np.array(all_probs), total_loss, len(loader.dataset)
    )


# =============================================================================
# T2  TEST-TIME AUGMENTATION
# FIX: num_workers=0 inside TTA to avoid pickling issues with custom transforms
# on Windows / Python 3.13+. TTA is already doing 8 sequential passes so
# worker processes give no real benefit here.
# =============================================================================

def evaluate_tta(model, dataset_path, subset_idx, device, batch_size=8):
    model.eval()
    tta_tfms = get_tta_transforms()
    n        = len(subset_idx)

    prob_sum    = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    true_labels = None

    for t_idx, tfm in enumerate(tta_tfms):
        ds     = ImageFolder(root=dataset_path, transform=tfm)
        subset = Subset(ds, subset_idx)
        # num_workers=0: avoids pickle errors with custom transform objects
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

        batch_probs, batch_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(loader,
                                       desc=f"  TTA {t_idx+1}/{len(tta_tfms)}",
                                       leave=False, ncols=85):
                inputs = inputs.to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                batch_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                batch_labels.extend(labels.numpy())

        prob_sum += np.array(batch_probs)
        if true_labels is None:
            true_labels = np.array(batch_labels)

    avg_probs  = prob_sum / len(tta_tfms)
    final_pred = avg_probs.argmax(axis=1)
    return compute_metrics(true_labels, final_pred, avg_probs, 0.0, n)


# =============================================================================
# T1  THRESHOLD OPTIMISATION
# =============================================================================

def optimise_thresholds(probs, true_labels, n_steps=20):
    best_thresholds = [0.5] * NUM_CLASSES
    best_f1         = 0.0
    threshold_grid  = np.linspace(0.1, 0.9, n_steps)

    for cls in range(NUM_CLASSES):
        best_t      = 0.5
        best_cls_f1 = 0.0
        for t in threshold_grid:
            thresholds      = best_thresholds.copy()
            thresholds[cls] = t
            preds = apply_thresholds(probs, thresholds)
            score = f1_score(true_labels, preds, average="weighted", zero_division=0)
            if score > best_cls_f1:
                best_cls_f1 = score
                best_t      = t
        best_thresholds[cls] = best_t

    for _ in range(3):
        for cls in range(NUM_CLASSES):
            for t in threshold_grid:
                candidate      = best_thresholds.copy()
                candidate[cls] = t
                preds = apply_thresholds(probs, candidate)
                score = f1_score(true_labels, preds, average="weighted", zero_division=0)
                if score > best_f1:
                    best_f1         = score
                    best_thresholds = candidate

    return best_thresholds, best_f1


def apply_thresholds(probs, thresholds):
    adjusted = probs - np.array(thresholds)
    return adjusted.argmax(axis=1)


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    use_mixup=False, alpha=0.2):
    model.train()
    total_loss, total_correct = 0.0, 0

    for inputs, labels in tqdm(loader, desc="  Train", leave=False, ncols=85):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            inputs, soft_targets = mixup_batch(inputs, labels, alpha=alpha)
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss    = criterion(outputs, soft_targets)
            total_correct += (outputs.argmax(1) == labels).sum().item()
        else:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
            total_correct += (outputs.argmax(1) == labels).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * inputs.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_correct / n


def get_optimizer_for_variant(variant_name, model):
    if "R1" in variant_name:
        feat_blocks = list(model.features.children())
        splits = [
            feat_blocks[:2],
            feat_blocks[2:4],
            feat_blocks[4:5],
            feat_blocks[5:7],
            feat_blocks[7:],
        ]
        base_lr = 3e-4
        decay   = 0.5
        groups  = []
        for i, group in enumerate(reversed(splits)):
            params = [p for m in group for p in m.parameters() if p.requires_grad]
            lr     = base_lr * (decay ** i)
            if params:
                groups.append({"params": params, "lr": lr})
                print(f"    Backbone group {5-i}  LR={lr:.2e}")
        head_params = list(model.classifier.parameters())
        groups.append({"params": head_params, "lr": base_lr})
        print(f"    Classifier head  LR={base_lr:.2e}")
        return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    else:
        return optim.AdamW(model.parameters(), lr=3e-4, weight_decay=WEIGHT_DECAY)


def retrain_variant(variant_name, train_loader, val_loader, class_weights, device):
    sep = "═" * 70
    print(f"\n{sep}\n  RETRAIN  :  {variant_name}\n{sep}")

    use_mixup  = "R2" in variant_name
    use_warmup = "R3" in variant_name or "R1+R2" in variant_name

    model = load_v4(device)

    if use_mixup:
        criterion      = FocalLossSoft(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
        criterion_eval = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
    else:
        criterion      = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
        criterion_eval = criterion

    optimizer = get_optimizer_for_variant(variant_name, model)

    warmup_epochs = 3 if use_warmup else 0
    if warmup_epochs > 0:
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, EPOCHS, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    ckpt_path      = os.path.join(MODELS_DIR, f"{variant_name}_best.pth")
    early_stopping = EarlyStopping(patience=PATIENCE, path=ckpt_path, verbose=True)
    scaler         = GradScaler("cuda")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch:>2}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_mixup=use_mixup, alpha=0.2
        )
        val_metrics = evaluate_standard(model, val_loader, criterion_eval, device)
        val_loss    = val_metrics["loss"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])

        if isinstance(scheduler, WarmupCosineScheduler):
            scheduler.step()
            lr_now = scheduler.get_lr()[0]
        else:
            scheduler.step()
            lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}"
            f"    Val  loss={val_loss:.4f}  acc={val_metrics['accuracy']:.4f}"
            f"    LR={lr_now:.2e}"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\n  ⚑ Early stopping at epoch {epoch}")
            break

    training_time = time.time() - t0
    save_learning_curves(history, variant_name, PLOTS_DIR)
    print(f"\n  Done.  Time={training_time/60:.1f} min")
    return training_time


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


def save_calibration_plot(probs, labels, name, save_dir):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(16, 4))
    fig.suptitle(f"{name} — Calibration (Reliability Diagram)", fontsize=12)
    for c, ax in enumerate(axes):
        cls_probs = probs[:, c]
        cls_true  = (labels == c).astype(int)
        bins      = np.linspace(0, 1, 11)
        bin_acc, bin_conf = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (cls_probs >= lo) & (cls_probs < hi)
            if mask.sum() > 0:
                bin_acc.append(cls_true[mask].mean())
                bin_conf.append(cls_probs[mask].mean())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(bin_conf, bin_acc, "bo-", markersize=5, label="Model")
        ax.set_title(CLASS_NAMES[c], fontsize=9)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{name}_calibration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def save_final_comparison(all_results, save_dir):
    df          = pd.DataFrame(all_results)
    metrics_cols = ["F1 Score", "Balanced Accuracy", "ROC-AUC", "Cohen Kappa"]
    df_sorted   = df.sort_values("F1 Score", ascending=False)
    colors      = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df_sorted)))

    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(20, 5))
    fig.suptitle("All Techniques vs V4 Baseline", fontsize=13)

    for ax, metric in zip(axes, metrics_cols):
        vals  = df_sorted[metric].values
        names = df_sorted["Technique"].values
        bars  = ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=10)
        ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.05))
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    path = os.path.join(save_dir, "final_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Final comparison plot → {path}")


def print_metrics(metrics, tag=""):
    sep = "─" * 62
    print(f"\n{sep}")
    if tag:
        print(f"  {tag}")
    print(sep)
    for k, v in [
        ("Loss",              metrics["loss"]),
        ("Accuracy",          metrics["accuracy"]),
        ("Balanced Accuracy", metrics["balanced_accuracy"]),
        ("F1 (weighted)",     metrics["f1"]),
        ("Cohen Kappa",       metrics["cohen_kappa"]),
        ("ROC-AUC",           metrics["roc_auc"]),
    ]:
        print(f"  {k:<20} : {v:.4f}")
    print("\n  Per-class F1:")
    for name, f1v in zip(CLASS_NAMES, metrics["per_class_f1"]):
        bar = "█" * int(f1v * 20)
        print(f"    {name:<20} {f1v:.4f}  {bar}")
    print(sep)


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

    (train_loader, val_loader, test_loader,
     class_weights, train_idx, val_idx, test_idx) = get_dataloaders(
        DATASET_DIR, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, random_state=RANDOM_STATE
    )

    criterion   = FocalLoss(alpha=class_weights.to(device), gamma=FOCAL_GAMMA)
    all_results = []

    # =========================================================================
    # STAGE 1 — POST-HOC
    # =========================================================================

    print("\n" + "═" * 70)
    print("  STAGE 1 — POST-HOC TECHNIQUES  (V4 checkpoint, no retraining)")
    print("═" * 70)

    v4_model = load_v4(device)

    print("\n  Collecting V4 val probs for threshold optimisation …")
    val_metrics_v4 = evaluate_standard(v4_model, val_loader, criterion, device)

    print("\n  Collecting V4 test probs (standard) …")
    test_metrics_v4 = evaluate_standard(v4_model, test_loader, criterion, device)

    print_metrics(test_metrics_v4, tag="V4 Baseline (standard eval on test)")
    save_calibration_plot(test_metrics_v4["_probs"], test_metrics_v4["_labels"],
                          "V4_baseline", PLOTS_DIR)
    save_confusion_matrix(test_metrics_v4["_labels"], test_metrics_v4["_preds"],
                          CLASS_NAMES, "V4_baseline", PLOTS_DIR)

    def metrics_to_row(name, m, time_min=None):
        row = {
            "Technique":         name,
            "F1 Score":          m["f1"],
            "Accuracy":          m["accuracy"],
            "Balanced Accuracy": m["balanced_accuracy"],
            "Cohen Kappa":       m["cohen_kappa"],
            "ROC-AUC":           m["roc_auc"],
        }
        if time_min is not None:
            row["Time (min)"] = time_min
        return row

    all_results.append(metrics_to_row("V4_baseline", test_metrics_v4, 0))

    stored_thresholds = None
    tta_probs         = None
    tta_labels        = None

    for technique in POST_HOC_TECHNIQUES:
        print(f"\n  ── {technique} ──")
        metrics = None

        if technique == "T1_threshold_opt":
            print("  Optimising thresholds on val set …")
            best_thresholds, val_f1 = optimise_thresholds(
                val_metrics_v4["_probs"], val_metrics_v4["_labels"]
            )
            print(f"  Best val F1 with optimised thresholds: {val_f1:.4f}")
            print(f"  Thresholds: {[f'{t:.3f}' for t in best_thresholds]}")

            test_preds = apply_thresholds(test_metrics_v4["_probs"], best_thresholds)
            metrics = compute_metrics(
                test_metrics_v4["_labels"], test_preds,
                test_metrics_v4["_probs"], 0.0,
                len(test_metrics_v4["_labels"])
            )
            print_metrics(metrics, tag=f"{technique} TEST")
            save_confusion_matrix(metrics["_labels"], metrics["_preds"],
                                  CLASS_NAMES, technique, PLOTS_DIR)
            save_per_class_bar(metrics, CLASS_NAMES, technique, PLOTS_DIR)

            with open(os.path.join(RESULTS_DIR, "optimised_thresholds.json"), "w") as f:
                json.dump({"thresholds": best_thresholds,
                           "class_names": CLASS_NAMES,
                           "val_f1_with_thresholds": val_f1}, f, indent=4)

            all_results.append(metrics_to_row(technique, metrics, 0))
            stored_thresholds = best_thresholds

        elif technique == "T2_tta":
            print("  Running TTA on test set (8 passes) …")
            t0 = time.time()
            metrics  = evaluate_tta(v4_model, DATASET_DIR, test_idx, device, BATCH_SIZE)
            tta_time = (time.time() - t0) / 60
            print_metrics(metrics, tag=f"{technique} TEST")
            save_confusion_matrix(metrics["_labels"], metrics["_preds"],
                                  CLASS_NAMES, technique, PLOTS_DIR)
            save_per_class_bar(metrics, CLASS_NAMES, technique, PLOTS_DIR)
            save_calibration_plot(metrics["_probs"], metrics["_labels"],
                                  technique, PLOTS_DIR)
            all_results.append(metrics_to_row(technique, metrics, round(tta_time, 1)))
            tta_probs  = metrics["_probs"]
            tta_labels = metrics["_labels"]

        elif technique == "T1+T2_combined":
            if stored_thresholds is None or tta_probs is None:
                print("  [!] T1 and T2 must both run first — skipping.")
                continue
            print("  Applying optimised thresholds to TTA probs …")
            combined_preds = apply_thresholds(tta_probs, stored_thresholds)
            metrics = compute_metrics(tta_labels, combined_preds, tta_probs,
                                      0.0, len(tta_labels))
            print_metrics(metrics, tag=f"{technique} TEST")
            save_confusion_matrix(metrics["_labels"], metrics["_preds"],
                                  CLASS_NAMES, technique, PLOTS_DIR)
            save_per_class_bar(metrics, CLASS_NAMES, technique, PLOTS_DIR)
            all_results.append(metrics_to_row(technique, metrics, 0))

        if metrics is not None:
            clean_m = {k: v for k, v in metrics.items() if not k.startswith("_")}
            with open(os.path.join(RESULTS_DIR, f"{technique}_metrics.json"), "w") as f:
                json.dump(clean_m, f, indent=4)

    # =========================================================================
    # STAGE 2 — RETRAIN VARIANTS
    # =========================================================================

    print("\n" + "═" * 70)
    print("  STAGE 2 — RETRAIN VARIANTS  (starting from V4 checkpoint)")
    print("═" * 70)

    for variant_name in RETRAIN_VARIANTS:
        try:
            training_time = retrain_variant(
                variant_name, train_loader, val_loader, class_weights, device
            )
        except Exception as e:
            print(f"\n  [ERROR] Retraining {variant_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        try:
            ckpt = os.path.join(MODELS_DIR, f"{variant_name}_best.pth")
            if not os.path.exists(ckpt):
                continue
            model = load_v4(device)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            metrics = evaluate_standard(model, test_loader, criterion, device)
            print_metrics(metrics, tag=f"{variant_name} TEST")
            save_confusion_matrix(metrics["_labels"], metrics["_preds"],
                                  CLASS_NAMES, variant_name, PLOTS_DIR)
            save_per_class_bar(metrics, CLASS_NAMES, variant_name, PLOTS_DIR)
            save_calibration_plot(metrics["_probs"], metrics["_labels"],
                                  variant_name, PLOTS_DIR)
            clean_m = {k: v for k, v in metrics.items() if not k.startswith("_")}
            with open(os.path.join(RESULTS_DIR, f"{variant_name}_metrics.json"), "w") as f:
                json.dump(clean_m, f, indent=4)
            all_results.append(metrics_to_row(variant_name, metrics,
                                              round(training_time / 60, 1)))
        except Exception as e:
            print(f"\n  [ERROR] Testing {variant_name}: {e}")
            import traceback; traceback.print_exc()

    # =========================================================================
    # FINAL LEADERBOARD
    # =========================================================================

    if not all_results:
        print("\n  No results.")
        return

    df = (pd.DataFrame(all_results)
            .sort_values("F1 Score", ascending=False)
            .reset_index(drop=True))
    df["Rank"] = df.index + 1

    csv_path = os.path.join(RESULTS_DIR, "advanced_leaderboard.csv")
    df.to_csv(csv_path, index=False)

    sep = "═" * 88
    print(f"\n{sep}\n  ADVANCED TECHNIQUE LEADERBOARD\n{sep}")
    print(df[["Rank", "Technique", "F1 Score", "Accuracy",
              "Balanced Accuracy", "ROC-AUC", "Cohen Kappa"]].to_string(index=False))
    print(f"{sep}\n\n  Saved → {csv_path}")

    save_final_comparison(all_results, PLOTS_DIR)

    best = df.iloc[0]
    print(f"\n  🏆  Best technique : {best['Technique']}")
    print(f"      F1 Score       : {best['F1 Score']:.4f}")
    print(f"      ROC-AUC        : {best['ROC-AUC']:.4f}")
    print(f"      Cohen Kappa    : {best['Cohen Kappa']:.4f}\n")


if __name__ == "__main__":
    main()