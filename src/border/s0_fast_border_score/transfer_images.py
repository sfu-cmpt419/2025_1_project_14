import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
ALL_IMAGES_DIR = os.path.join(DESKTOP_DIR, "SPRING 2025", "CMPT 419", "ISIC2018_Task1_Test_GroundTruth")
MERGED_CSV_PATH = os.path.join(BASE_DIR, "merged_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "input_images/mask")

allowed_images_df = pd.read_csv(MERGED_CSV_PATH)
allowed_images_df["Image"] = allowed_images_df["Image"].str.strip()
allowed_images_set = set(allowed_images_df["Image"].values)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(ALL_IMAGES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        basename = filename.replace(".jpg", "").replace(".png", "").replace("_segmentation", "")

        if basename in allowed_images_set:
            src_path = os.path.join(ALL_IMAGES_DIR, filename)
            dst_path = os.path.join(OUTPUT_DIR, filename)
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Skipped: {filename}")

print("Done transferring selected images.")


allowed_images_df = pd.read_csv(MERGED_CSV_PATH)
allowed_images_set = set(allowed_images_df["Image"].values)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(ALL_IMAGES_DIR):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        basename = filename.replace(".jpg", "").replace(".png", "")
        if basename in allowed_images_set:
            src_path = os.path.join(ALL_IMAGES_DIR, filename)
            dst_path = os.path.join(OUTPUT_DIR, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied {filename}")

print("Done transferring images.")