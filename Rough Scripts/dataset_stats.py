import os
from PIL import Image
from collections import Counter
import numpy as np

# ==========================
# CONFIG
# ==========================

DATASET_ROOT = r"New Dataset\Enhanced_Test_data"
EDGEMAPS_ROOT = r"New Dataset\Enhanced_Test_data\EdgeMaps"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ==========================
# ANALYSIS
# ==========================

total_images = 0
class_counts = {}

widths = []
heights = []
channels = Counter()
formats = Counter()

corrupt_files = []

print("\n" + "=" * 80)
print("DATASET INSPECTION")
print("=" * 80)

for class_name in sorted(os.listdir(DATASET_ROOT)):

    class_path = os.path.join(DATASET_ROOT, class_name)

    if not os.path.isdir(class_path):
        continue

    image_count = 0

    for root, _, files in os.walk(class_path):

        for file in files:

            if not file.lower().endswith(VALID_EXTS):
                continue

            image_count += 1
            total_images += 1

            img_path = os.path.join(root, file)

            try:
                img = Image.open(img_path)

                widths.append(img.width)
                heights.append(img.height)

                channels[img.mode] += 1

                ext = os.path.splitext(file)[1].lower()
                formats[ext] += 1

            except Exception:
                corrupt_files.append(img_path)

    class_counts[class_name] = image_count

# ==========================
# CLASS DISTRIBUTION
# ==========================

print("\nCLASS DISTRIBUTION")
print("-" * 80)

for cls, count in class_counts.items():

    pct = (count / total_images) * 100

    print(f"{cls:<35} {count:>6} images ({pct:.2f}%)")

# ==========================
# IMAGE STATS
# ==========================

print("\nIMAGE STATISTICS")
print("-" * 80)

print(f"Total Images : {total_images}")

if widths:

    print(f"Average Width  : {np.mean(widths):.1f}")
    print(f"Average Height : {np.mean(heights):.1f}")

    print(f"Min Resolution : {min(widths)} x {min(heights)}")
    print(f"Max Resolution : {max(widths)} x {max(heights)}")

# ==========================
# CHANNELS
# ==========================

print("\nCHANNEL DISTRIBUTION")
print("-" * 80)

for mode, count in channels.items():
    print(f"{mode:<10} {count}")

# ==========================
# FORMATS
# ==========================

print("\nFILE FORMATS")
print("-" * 80)

for ext, count in formats.items():
    print(f"{ext:<10} {count}")

# ==========================
# IMBALANCE
# ==========================

print("\nCLASS IMBALANCE")
print("-" * 80)

if class_counts:

    largest = max(class_counts.values())
    smallest = min(class_counts.values())

    print("Largest Class :", largest)
    print("Smallest Class:", smallest)

    if smallest > 0:
        print("Imbalance Ratio:", round(largest / smallest, 2), ":1")

# ==========================
# CORRUPT FILES
# ==========================

print("\nCORRUPT FILES")
print("-" * 80)

if corrupt_files:

    print(f"Found {len(corrupt_files)} corrupt files")

    for f in corrupt_files[:20]:
        print(f)

else:
    print("No corrupt files found")

# ==========================
# EDGE MAP CHECK
# ==========================

print("\nEDGE MAP VALIDATION")
print("-" * 80)

if os.path.exists(EDGEMAPS_ROOT):

    total_edge = 0

    for root, _, files in os.walk(EDGEMAPS_ROOT):

        for file in files:

            if file.lower().endswith(VALID_EXTS):
                total_edge += 1

    print("Edge Maps Found:", total_edge)

    if total_edge == total_images:
        print("✓ Edge maps match image count")
    else:
        print("⚠ Edge map count differs from image count")

else:
    print("EdgeMaps folder not found")

# ==========================
# TRAINING RECOMMENDATION
# ==========================

print("\nTRAINING RECOMMENDATION")
print("-" * 80)

if widths:

    avg_dim = (np.mean(widths) + np.mean(heights)) / 2

    if avg_dim > 1000:
        print("Recommended input size: 512x512")
    elif avg_dim > 500:
        print("Recommended input size: 384x384")
    else:
        print("Recommended input size: 224x224")

print("\nInspection Complete.")
print("=" * 80)