"""
remove_artifacts.py

Keeps only the largest connected component in the binary lesion mask.
"""

import cv2
import numpy as np

def remove_artifacts(mask):
    """
    Keeps only the largest connected component in a binary mask.
    """
    if mask is None:
        raise ValueError("[remove_artifacts] Input mask is None.")
    
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Connected components
    num_labels, labels_im = cv2.connectedComponents(bin_mask)

    if num_labels <= 1:
        # Nothing or just background
        return bin_mask

    max_area = 0
    max_label = 0

    for label_val in range(1, num_labels):
        comp_area = np.sum(labels_im == label_val)
        if comp_area > max_area:
            max_area = comp_area
            max_label = label_val

    cleaned_mask = np.zeros_like(bin_mask)
    cleaned_mask[labels_im == max_label] = 255
    return cleaned_mask

# Example usage:
if __name__ == "__main__":
    import os
    from load_mask import load_mask
    from preprocess_mask import preprocess_mask

    mask_path = os.path.join("..", "ISIC2018_Task1_Training_GroundTruth", "ISIC_0000002_segmentation.png")
    raw_mask = load_mask(mask_path)
    opened_mask = preprocess_mask(raw_mask, 5, "open")
    
    largest_comp_mask = remove_artifacts(opened_mask)
    print(f"[remove_artifacts] Largest component kept for {mask_path}")
