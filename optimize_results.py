import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'c:\Users\AK\Documents\dr\test_results.csv')

# Healthy group: Folders 1, 2, 3
healthy_folders = ['1. No DR signs', '2. Mild (or early) NPDR', '3. Moderate NPDR']
df['is_healthy'] = df['actual_folder'].isin(healthy_folders)

def calculate_accuracy(threshold):
    # Healthy group: Success if score > threshold
    # Severe group: Success if score < threshold
    pred_healthy = df['score'] > threshold
    correct = (df['is_healthy'] & pred_healthy) | (~df['is_healthy'] & ~pred_healthy)
    return correct.mean()

# Test a range of thresholds
thresholds = np.linspace(df['score'].min(), df['score'].max(), 500)
accuracies = [calculate_accuracy(t) for t in thresholds]

best_idx = np.argmax(accuracies)
best_threshold = thresholds[best_idx]
best_accuracy = accuracies[best_idx]

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Maximized Accuracy: {best_accuracy:.2%}")

# Plot accuracy vs threshold
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
plt.plot(thresholds, accuracies, color='cyan', linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold {best_threshold:.2f}')
plt.title(f'Accuracy Optimization (Max: {best_accuracy:.2%})', fontsize=15)
plt.xlabel('Threshold Value')
plt.ylabel('Overall Accuracy')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(r'c:\Users\AK\Documents\dr\threshold_optimization.png')

# Output detailed stats for the best threshold
df['best_correct'] = (df['is_healthy'] & (df['score'] > best_threshold)) | (~df['is_healthy'] & (df['score'] <= best_threshold))
print("\nAccuracy per folder with optimal threshold:")
print(df.groupby('actual_folder')['best_correct'].mean() * 100)
