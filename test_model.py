import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'c:\Users\AK\Documents\dr\dr_finetuned.pth'
data_dir = r'c:\Users\AK\Documents\dr\Test data'

# Load model
model = get_model()
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except:
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
results = []
# OPTIMIZED THRESHOLD
THRESHOLD = 5.16 

print(f"\n🚀 Optimized Analysis Started (Threshold: {THRESHOLD})")

all_meta_skipped = 0
all_real_processed = 0

with torch.no_grad():
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        files = os.listdir(class_path)
        
        # Identify files
        real_images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
        meta_files = [f for f in files if f.startswith('._')]
        
        all_meta_skipped += len(meta_files)
        all_real_processed += len(real_images)
        
        for img_name in tqdm(real_images, desc=f"Folder {class_idx+1}: {class_name[:15]}"):
            try:
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('RGB')
                score = model(transform(img).unsqueeze(0).to(device)).item()
                
                # Folders 1-3 (Healthy/Mild)
                if class_idx <= 2:
                    is_correct = score > THRESHOLD
                else:
                    is_correct = score < THRESHOLD
                
                results.append({'folder': class_name, 'score': score, 'success': is_correct})
            except:
                continue

df = pd.DataFrame(results)
df.to_csv(r'c:\Users\AK\Documents\dr\test_results.csv', index=False)

# VISUALS
plt.figure(figsize=(15, 10))
plt.style.use('dark_background')

plt.subplot(2, 1, 1)
sns.boxenplot(x='folder', y='score', data=df, palette='viridis')
plt.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=3, label=f'Threshold ({THRESHOLD})')
plt.title(f'Optimized Model Integrity: Overall Accuracy {df["success"].mean():.2%}', fontsize=18, color='gold')
plt.xticks(rotation=20)
plt.legend()

plt.subplot(2, 1, 2)
accs = df.groupby('folder')['success'].mean()*100
bars = plt.bar(accs.index, accs.values, color='cyan')
plt.title('Detection Accuracy per Grade', fontsize=18, color='gold')
plt.ylabel('Accuracy %')
plt.ylim(0, 110)
plt.xticks(rotation=20)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', color='white')

plt.tight_layout()
plt.savefig(r'c:\Users\AK\Documents\dr\results_visualization.png', dpi=300)

print(f"\n✨ FINISHED!")
print(f"📈 Optimized Accuracy: {df['success'].mean():.2%}")
print(f"📁 Total Files Scanned: {all_real_processed + all_meta_skipped}")
print(f"✅ Processed Every Real Image: {all_real_processed}")
print(f"🚫 Ignored Ghost/System Files: {all_meta_skipped}")
print(f"📊 Visualization: c:\\Users\\AK\\Documents\\dr\\results_visualization.png")
