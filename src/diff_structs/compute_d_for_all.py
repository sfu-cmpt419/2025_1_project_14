#!/usr/bin/env python3
"""
compute_d_for_all.py

Single-pass pipeline that computes five dermoscopic attribute values for each lesion
and calculates the D value (D = 0.5 * [# of structures present]).

The five attributes are:
  1. globules
  2. streaks
  3. pigment_network
  4. dots
  5. structureless

Folder Assumptions:
  - seg_folder (e.g., ISIC2018_Task1_Training_GroundTruth) contains normal colored images 
    named like "ISIC_0000000.jpg".
  - attr_folder (e.g., ISIC2018_Task2_Training_Groundtruth_v3) contains corresponding attribute masks:
      "ISIC_0000000_attribute_globules.png", 
      "ISIC_0000000_attribute_streaks.png",
      "ISIC_0000000_attribute_pigment_network.png"
      
For dots and structureless, we run a naive detection directly on the same image.

The final CSV (output_csv) will have one row per lesion with columns:
  lesion_id, globules_present, streaks_present, pigment_network_present, 
  dots_present, structureless_present, num_structures, D_value

Usage:
  python compute_d_for_all.py <seg_folder> <attr_folder> <output_csv>

Example:
  python compute_d_for_all.py ISIC2018_Task1_Training_GroundTruth ISIC2018_Task2_Training_Groundtruth_v3 results.csv
"""

import os
import sys
import csv
import cv2
import numpy as np

# Define CSV field names
FIELDNAMES = [
    "lesion_id",
    "globules_present",
    "streaks_present",
    "pigment_network_present",
    "dots_present",
    "structureless_present",
    "num_structures",
    "D_value"
]

# -----------------------------
# Attribute detection functions (Phase 1)
# -----------------------------
def detect_globules(attribute_mask_path):
    """Return True if the globules attribute mask exists and contains any non-zero pixel."""
    if not os.path.isfile(attribute_mask_path):
        return False
    mask = cv2.imread(attribute_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return bool(np.any(mask > 0))

def detect_streaks(attribute_mask_path):
    """Return True if the streaks attribute mask exists and contains any non-zero pixel."""
    if not os.path.isfile(attribute_mask_path):
        return False
    mask = cv2.imread(attribute_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return bool(np.any(mask > 0))

def detect_pigment_network(attribute_mask_path):
    """Return True if the pigment network attribute mask exists and contains any non-zero pixel."""
    if not os.path.isfile(attribute_mask_path):
        return False
    mask = cv2.imread(attribute_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return bool(np.any(mask > 0))

# -----------------------------
# Naive detection functions for dots and structureless (Phase 2)
# -----------------------------
def detect_dots_naive(image_path):
    """
    Naively detect "dots" by finding small dark blobs in the grayscale image.
    """
    if not os.path.isfile(image_path):
        return False
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    # Threshold: consider pixels darker than 50 as candidates for "dots"
    _, dark_bin = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels_im = cv2.connectedComponents(dark_bin)

    dot_count = 0
    min_blob_size = 5
    max_blob_size = 100
    for label in range(1, num_labels):
        blob_size = np.sum(labels_im == label)
        if min_blob_size <= blob_size <= max_blob_size:
            dot_count += 1
    return bool(dot_count >= 1)

def detect_structureless_naive(image_path):
    """
    Naively detect "structureless" regions by computing the overall variance of the grayscale image.
    If the variance is below a set threshold, consider the lesion structureless.
    """
    if not os.path.isfile(image_path):
        return False
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    data = img.astype(np.float32)
    var_ = np.var(data)
    var_threshold = 500.0  # Threshold may be tuned based on image properties
    return bool(var_ < var_threshold)

# -----------------------------
# Utility function: Compute D score
# -----------------------------
def compute_d_score(count, multiplier=0.5):
    return multiplier * count

# -----------------------------
# Main pipeline: Process each image and write results as we go
# -----------------------------
def process_and_write(seg_folder, attr_folder, output_csv):
    # List all segmentation files in seg_folder that end with ".jpg"
    seg_files = sorted([
        f for f in os.listdir(seg_folder) if f.lower().endswith(".jpg")
    ])
    print(f"[INFO] Found {len(seg_files)} files in '{seg_folder}' ending with '.jpg'.")

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for idx, seg_filename in enumerate(seg_files, start=1):
            # Derive lesion_id by stripping ".jpg"
            lesion_id = seg_filename.replace(".jpg", "")
            print(f"\n[{idx}/{len(seg_files)}] Processing lesion: {lesion_id}")

            seg_path = os.path.join(seg_folder, seg_filename)

            # (A) Attribute-based detection using attribute masks in attr_folder:
            globules_path = os.path.join(attr_folder, f"{lesion_id}_attribute_globules.png")
            streaks_path  = os.path.join(attr_folder, f"{lesion_id}_attribute_streaks.png")
            pigment_path  = os.path.join(attr_folder, f"{lesion_id}_attribute_pigment_network.png")

            globules_val = detect_globules(globules_path)
            streaks_val = detect_streaks(streaks_path)
            pigment_val = detect_pigment_network(pigment_path)

            print(f"   => Attributes: globules={globules_val}, streaks={streaks_val}, pigment_network={pigment_val}")

            # (B) Naive detection for dots and structureless using the same image file.
            # Since we have no separate color image, we simply use the .jpg.
            dots_val = detect_dots_naive(seg_path)
            structureless_val = detect_structureless_naive(seg_path)
            print(f"   => Naive: dots={dots_val}, structureless={structureless_val}")

            # (C) Compute total number of structures present and D value.
            # The five attributes are: globules, streaks, pigment_network, dots, structureless.
            num_structs = sum([globules_val, streaks_val, pigment_val, dots_val, structureless_val])
            d_value = compute_d_score(num_structs, 0.5)
            print(f"   => Total structures = {num_structs}, D value = {d_value:.2f}")

            # (D) Write the data for this lesion directly to the CSV.
            row = {
                "lesion_id": lesion_id,
                "globules_present": int(globules_val),
                "streaks_present": int(streaks_val),
                "pigment_network_present": int(pigment_val),
                "dots_present": int(dots_val),
                "structureless_present": int(structureless_val),
                "num_structures": num_structs,
                "D_value": d_value
            }
            writer.writerow(row)
            print("   Row written.")

    print(f"\n[Done] CSV file '{output_csv}' written with {len(seg_files)} rows.")

def main():
    if len(sys.argv) < 4:
        print("Usage: python compute_d_for_all.py <seg_folder> <attr_folder> <output_csv>")
        sys.exit(1)
    seg_folder = sys.argv[1]
    attr_folder = sys.argv[2]
    output_csv = sys.argv[3]

    process_and_write(seg_folder, attr_folder, output_csv)
    print("[All Done]")

if __name__ == "__main__":
    main()
