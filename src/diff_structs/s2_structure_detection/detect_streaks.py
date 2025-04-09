"""
detect_streaks.py

Checks if an attribute mask for 'streaks' is non-empty.
"""

import cv2
import numpy as np
import os

def detect_streaks(attribute_mask_path):
    """
    Returns True if the streaks attribute mask has any non-zero pixels.
    """
    if not os.path.isfile(attribute_mask_path):
        return False
    
    mask = cv2.imread(attribute_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    return bool(np.any(mask > 0))

# Example usage:
if __name__ == "__main__":
    path = os.path.join("..", "ISIC2018_Task2_Training_GroundTruth_v3", "ISIC_0000000_attribute_streaks.png")
    presence = detect_streaks(path)
    print(f"[detect_streaks] Streaks present: {presence}")
