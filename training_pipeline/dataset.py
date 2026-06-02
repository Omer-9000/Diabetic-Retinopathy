import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Mapping integer labels to folder names
CLASS_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

class DRDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.loc[idx, 'id_code']
        label = self.dataframe.loc[idx, 'diagnosis']
        
        # Determine the subfolder based on the label
        folder_name = CLASS_MAP[label]
        
        # Support both .png and potentially missing extensions in the CSV
        img_name = img_id if img_id.endswith('.png') else f"{img_id}.png"
        img_path = os.path.join(self.image_dir, folder_name, img_name)
        
        # Fallback if image is not in subfolders but directly in image_dir
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_path}")
            raise e

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

def prepare_data(csv_path, image_dir, splits_dir, random_seed=42):
    """
    Reads the main CSV, performs a 70/15/15 stratified split, saves them,
    and calculates class weights.
    """
    df = pd.read_csv(csv_path)
    
    # Check if splits already exist
    train_path = os.path.join(splits_dir, "train.csv")
    val_path = os.path.join(splits_dir, "val.csv")
    test_path = os.path.join(splits_dir, "test.csv")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Loading existing splits...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
    else:
        print("Creating new 70/15/15 stratified splits...")
        # 70% train, 30% temp (val + test)
        train_df, temp_df = train_test_split(
            df, test_size=0.30, random_state=random_seed, stratify=df['diagnosis']
        )
        # Split temp into 50% val, 50% test (which equals 15% / 15% of total)
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=random_seed, stratify=temp_df['diagnosis']
        )
        
        # Save splits
        os.makedirs(splits_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"Splits saved to {splits_dir}")

    # Compute class weights based on the training set
    labels = train_df['diagnosis'].values
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    
    return train_df, val_df, test_df, class_weights

def get_dataloaders(csv_path, image_dir, splits_dir, batch_size=32, num_workers=4):
    train_df, val_df, test_df, class_weights = prepare_data(csv_path, image_dir, splits_dir)
    
    train_dataset = DRDataset(train_df, image_dir, transform=get_transforms('train'))
    val_dataset = DRDataset(val_df, image_dir, transform=get_transforms('val'))
    test_dataset = DRDataset(test_df, image_dir, transform=get_transforms('test'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_weights
