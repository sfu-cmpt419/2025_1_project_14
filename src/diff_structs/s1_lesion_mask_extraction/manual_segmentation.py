"""
manual_segmentation.py

Implements a naive/manual segmentation approach if no official mask is available.
"""

import cv2
import numpy as np
import os

def manual_segmentation(rgb_path):
    """
    If no official seg mask is found, do a naive segmentation:
      1) Convert to grayscale
      2) Invert threshold => everything < 200 is "lesion"
      3) Morphological opening
      4) Keep largest connected component

    Returns a binary mask (uint8) with 255=lesion, 0=background, or None if error.
    """
    if not os.path.isfile(rgb_path):
        print(f"[manual_segmentation] Missing image: {rgb_path}")
        return None

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"[manual_segmentation] Could not load image: {rgb_path}")
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # Step 1: Invert threshold
    # Everything below 200 => lesion
    _, rough_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Morphological opening to remove small artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(rough_mask, cv2.MORPH_OPEN, kernel)

    # Step 3: Keep largest connected component
    num_labels, labels_im = cv2.connectedComponents(opened)
    if num_labels <= 1:
        # No foreground
        return opened

    max_area = 0
    max_label = 0
    for label_val in range(1, num_labels):
        size = np.sum(labels_im == label_val)
        if size > max_area:
            max_area = size
            max_label = label_val

    final_mask = np.zeros_like(opened)
    final_mask[labels_im == max_label] = 255
    return final_mask

if __name__ == "__main__":
    # Demo usage
    base_id = "ISIC_9999999"
    rgb_path = f"/path/to/ISIC2018_Task1-2_Validation_Input/{base_id}.jpg"
    mask = manual_segmentation(rgb_path)
    if mask is not None:
        print(f"[manual_segmentation] Completed. Non-zero pixels = {cv2.countNonZero(mask)}")
    else:
        print("[manual_segmentation] Failed.")
