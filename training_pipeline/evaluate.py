import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from utils import plot_confusion_matrix

def evaluate_model(model, dataloader, criterion, device, model_name, class_names, plots_dir):
    """
    Evaluates the model on the given dataloader and calculates metrics.
    """
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
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # ROC-AUC (multi-class one-vs-rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        roc_auc = float('nan') # Sometimes happens if a class is missing in the true labels

    # Generate Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, model_name, plots_dir)

    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'per_class_f1': per_class_f1.tolist()
    }

    return metrics
