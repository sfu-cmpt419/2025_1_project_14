"""
detect_dots.py

Naive detection of "dots" from an RGB lesion image + segmentation mask.
Looks for small dark blobs within the lesion.
"""

import cv2
import numpy as np
import os

def detect_dots(rgb_image_path, seg_mask_path,
                darkness_threshold=60,
                min_blob_size=5,
                max_blob_size=100,
                min_num_dots=1):
    """
    Returns True if we find at least min_num_dots blobs with 
    pixel count in [min_blob_size, max_blob_size].
    """

    if not os.path.isfile(rgb_image_path):
        print(f"[detect_dots] Missing color image: {rgb_image_path}")
        return False

    if not os.path.isfile(seg_mask_path):
        print(f"[detect_dots] Missing segmentation mask: {seg_mask_path}")
        return False

    rgb = cv2.imread(rgb_image_path)
    if rgb is None:
        print(f"[detect_dots] Could not load image {rgb_image_path}")
        return False

    mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[detect_dots] Could not load mask {seg_mask_path}")
        return False

    # Binary mask
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Zero out non-lesion area to 255 => ignoring background
    lesion_gray = np.where(bin_mask == 255, gray, 255).astype(np.uint8)

    # threshold for "dark"
    _, dark_regions = cv2.threshold(lesion_gray, darkness_threshold, 255, cv2.THRESH_BINARY_INV)
    # dark_regions is white where gray < darkness_threshold

    num_labels, labels_im = cv2.connectedComponents(dark_regions)
    dot_count = 0
    for label_val in range(1, num_labels):
        blob_size = np.sum(labels_im == label_val)
        if min_blob_size <= blob_size <= max_blob_size:
            dot_count += 1

    if dot_count >= min_num_dots:
        print(f"[detect_dots] True. Found {dot_count} dot(s).")
        return True
    else:
        print(f"[detect_dots] False. Found {dot_count} dot(s).")
        return False
