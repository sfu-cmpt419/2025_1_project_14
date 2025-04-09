"""
preprocess_mask.py

Applies morphological operations (e.g. opening/closing) to remove noise in a mask.
"""

import cv2
import numpy as np

def preprocess_mask(mask, kernel_size=5, operation="open"):
    """
    Applies morphological opening/closing to remove small artifacts.
    """
    if mask is None:
        raise ValueError("[preprocess_mask] Input mask is None.")
    
    # Ensure a strict binary mask
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if operation.lower() == "open":
        cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    elif operation.lower() == "close":
        cleaned = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"[preprocess_mask] Unknown operation: {operation}")
    
    return cleaned


# Example usage:
if __name__ == "__main__":
    import os
    from load_mask import load_mask
    
    mask_path = os.path.join("..", "ISIC2018_Task1_Training_GroundTruth", "ISIC_0000002_segmentation.png")
    raw_mask = load_mask(mask_path)
    
    cleaned = preprocess_mask(raw_mask, kernel_size=5, operation="open")
    print(f"[preprocess_mask] Finished morphological operation on {mask_path}")
