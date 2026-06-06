import os
import shutil

SOURCE = r"F:\Work\Diabetic-Retinopathy\New Dataset\Enhanced_Test_data"
DEST = r"F:\Work\Diabetic-Retinopathy\MergedDataset"

mapping = {
    "1. No DR signs": "No_DR",
    "2. Mild (or early) NPDR": "Moderate_NPDR",
    "3. Moderate NPDR": "Moderate_NPDR",
    "4. Severe NPDR": "Severe_NPDR",
    "5. Very Severe NPDR": "Severe_NPDR",
    "6. PDR": "PDR",
    "7. Advanced PDR": "PDR"
}

for old_class, new_class in mapping.items():

    src = os.path.join(SOURCE, old_class)
    dst = os.path.join(DEST, new_class)

    os.makedirs(dst, exist_ok=True)

    for file in os.listdir(src):
        shutil.copy2(
            os.path.join(src, file),
            os.path.join(dst, file)
        )

print("Dataset merged successfully.")