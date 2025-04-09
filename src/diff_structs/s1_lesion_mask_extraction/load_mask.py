"""
load_mask.py

Loads a segmentation mask from disk (PNG) into a NumPy array.
"""

import cv2
import numpy as np
import os

def load_mask(mask_path):
    """
    Loads a binary segmentation mask in grayscale.
    
    Args:
        mask_path (str): Path to the segmentation PNG file.

    Returns:
        np.ndarray or None: Grayscale mask (0-255). None if failed.
    """
    if not os.path.isfile(mask_path):
        print(f"[load_mask] File not found: {mask_path}")
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[load_mask] Unable to read mask from: {mask_path}")
        return None

    return mask

# Example usage:
if __name__ == "__main__":
    # Adjust the path to point to your actual segmentation file
    example_path = os.path.join("..", "ISIC2018_Task1_Training_GroundTruth", "ISIC_0000001_segmentation.png")
    mask = load_mask(example_path)
    if mask is not None:
        print(f"[load_mask] Loaded mask with shape={mask.shape}")
