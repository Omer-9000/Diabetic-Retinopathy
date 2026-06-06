import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


# ==========================================================
# FOCAL LOSS
# ==========================================================

class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean"
    ):

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs,
        targets
    ):

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            weight=self.alpha
        )

        pt = torch.exp(-ce_loss)

        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

        elif self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss


# ==========================================================
# EARLY STOPPING
# ==========================================================

class EarlyStopping:

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0.0,
        path="checkpoint.pth"
    ):

        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.inf

        self.path = path

    def __call__(
        self,
        val_loss,
        model
    ):

        score = -val_loss

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(
                val_loss,
                model
            )

        elif score < self.best_score + self.delta:

            self.counter += 1

            if self.verbose:

                print(
                    f"EarlyStopping "
                    f"{self.counter}/"
                    f"{self.patience}"
                )

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(
                val_loss,
                model
            )

            self.counter = 0

    def save_checkpoint(
        self,
        val_loss,
        model
    ):

        if self.verbose:

            print(
                f"Validation Loss Improved "
                f"({self.val_loss_min:.6f}"
                f" -> "
                f"{val_loss:.6f})"
            )

        torch.save(
            model.state_dict(),
            self.path
        )

        self.val_loss_min = val_loss


# ==========================================================
# LEARNING CURVES
# ==========================================================

def plot_curves(
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    model_name,
    save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    epochs = range(
        1,
        len(train_losses) + 1
    )

    plt.figure(
        figsize=(14, 6)
    )

    # --------------------------------------
    # LOSS
    # --------------------------------------

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )

    plt.title(
        f"{model_name} Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    # --------------------------------------
    # ACCURACY
    # --------------------------------------

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_accs,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accs,
        label="Validation Accuracy"
    )

    plt.title(
        f"{model_name} Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name}_learning_curves.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    model_name,
    save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(8, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        f"{model_name} Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"{model_name}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==========================================================
# PARAMETER COUNTER
# ==========================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "utils.py loaded successfully."
    )