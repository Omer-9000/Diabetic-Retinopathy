import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

from utils import plot_confusion_matrix


def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
    model_name,
    class_names,
    plots_dir
):

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for inputs, labels in dataloader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                inputs.size(0)
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )

            _, preds = torch.max(
                outputs,
                1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    # ==================================================
    # LOSS
    # ==================================================

    epoch_loss = (
        running_loss /
        len(dataloader.dataset)
    )

    # ==================================================
    # CORE METRICS
    # ==================================================

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_preds,
            average="weighted",
            zero_division=0
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            all_labels,
            all_preds
        )
    )

    kappa = cohen_kappa_score(
        all_labels,
        all_preds
    )

    # ==================================================
    # PER-CLASS METRICS
    # ==================================================

    (
        per_class_precision,
        per_class_recall,
        per_class_f1,
        _
    ) = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average=None,
        zero_division=0
    )

    # ==================================================
    # ROC-AUC
    # ==================================================

    try:

        roc_auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class="ovr"
        )

    except ValueError:

        roc_auc = float("nan")

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        model_name,
        plots_dir
    )

    # ==================================================
    # PRINT REPORT
    # ==================================================

    print("\n" + "=" * 60)
    print(f"{model_name} Evaluation")
    print("=" * 60)

    print(f"Loss              : {epoch_loss:.4f}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1 Score          : {f1:.4f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.4f}")
    print(f"Cohen Kappa       : {kappa:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")

    print("\nPer-Class F1")

    for idx, score in enumerate(
        per_class_f1
    ):

        print(
            f"{class_names[idx]:<20}"
            f"{score:.4f}"
        )

    print("=" * 60)

    # ==================================================
    # RETURN METRICS
    # ==================================================

    metrics = {

        "loss":
            float(epoch_loss),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "balanced_accuracy":
            float(balanced_accuracy),

        "cohen_kappa":
            float(kappa),

        "roc_auc":
            float(roc_auc),

        "per_class_precision":
            per_class_precision.tolist(),

        "per_class_recall":
            per_class_recall.tolist(),

        "per_class_f1":
            per_class_f1.tolist()
    }

    return metrics