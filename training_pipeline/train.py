import os
import time
import json
import torch
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from dataset import get_dataloaders
from models import get_model
from utils import FocalLoss, EarlyStopping, plot_curves, plot_class_distribution
from evaluate import evaluate_model

# Configuration
CSV_PATH = r"f:\Work\Diabetic-Retinopathy\Dataset\train.csv"
IMAGE_DIR = r"f:\Work\Diabetic-Retinopathy\Dataset\colored_images"
SPLITS_DIR = "splits"
MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"
RESULTS_DIR = "results"

BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4

MODELS_TO_TRAIN = [
    'resnet50',
    'efficientnet_b0',
    'efficientnet_b3',
    'densenet121',
    'mobilenet_v3_large',
    'vit_b_16'
]

CLASS_NAMES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    corrects = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()

def train_model(model_name, train_loader, val_loader, class_weights, device):
    print(f"\n{'='*50}\nStarting Training for {model_name}\n{'='*50}")
    
    model = get_model(model_name, num_classes=5, pretrained=True)
    model = model.to(device)

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    model_save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=model_save_path)
    scaler = GradScaler()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device, model_name, CLASS_NAMES, PLOTS_DIR)
        
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    training_time = time.time() - start_time
    print(f"Training completed in {training_time//60:.0f}m {training_time%60:.0f}s")
    
    # Save learning curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, model_name, PLOTS_DIR)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return training_time, num_params

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_weights = get_dataloaders(
        CSV_PATH, IMAGE_DIR, SPLITS_DIR, batch_size=BATCH_SIZE
    )
    
    # Plot data distributions
    plot_class_distribution(train_loader.dataset.dataframe['diagnosis'], CLASS_NAMES, 'Train', PLOTS_DIR)
    plot_class_distribution(val_loader.dataset.dataframe['diagnosis'], CLASS_NAMES, 'Val', PLOTS_DIR)
    plot_class_distribution(test_loader.dataset.dataframe['diagnosis'], CLASS_NAMES, 'Test', PLOTS_DIR)

    results = []
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)

    for model_name in MODELS_TO_TRAIN:
        # Train
        train_time, num_params = train_model(model_name, train_loader, val_loader, class_weights, device)
        
        # Load best weights for testing
        model = get_model(model_name, num_classes=5, pretrained=False)
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{model_name}_best.pth")))
        model = model.to(device)
        
        # Test Evaluation
        print(f"Evaluating {model_name} on Test Set...")
        test_metrics = evaluate_model(model, test_loader, criterion, device, model_name + "_test", CLASS_NAMES, PLOTS_DIR)
        
        results.append({
            'Model Name': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1 Score': test_metrics['f1'],
            'ROC-AUC': test_metrics['roc_auc'],
            'Training Time (s)': train_time,
            'Number of Parameters': num_params
        })
        
        # Save detailed logs
        with open(os.path.join(LOGS_DIR, f"{model_name}_test_metrics.json"), 'w') as f:
            json.dump(test_metrics, f, indent=4)

    # Generate Leaderboard
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='F1 Score', ascending=False)
    leaderboard_path = os.path.join(RESULTS_DIR, 'leaderboard.csv')
    df_results.to_csv(leaderboard_path, index=False)
    
    print("\nTraining Complete! Leaderboard:")
    print(df_results[['Model Name', 'F1 Score', 'Accuracy']])
    
    # Save best model globally
    best_model_name = df_results.iloc[0]['Model Name']
    print(f"\nBest Model selected based on F1 Score: {best_model_name}")
    
    best_weights_path = os.path.join(MODELS_DIR, f"{best_model_name}_best.pth")
    final_best_path = os.path.join(MODELS_DIR, "best_model.pth")
    
    import shutil
    shutil.copy(best_weights_path, final_best_path)
    
    # Save best config
    with open(os.path.join(RESULTS_DIR, "best_config.json"), 'w') as f:
        json.dump({'best_model': best_model_name}, f)

if __name__ == "__main__":
    main()
