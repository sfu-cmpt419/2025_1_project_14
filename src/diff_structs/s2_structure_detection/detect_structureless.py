"""
detect_structureless.py

Improved detection of "structureless" areas from an RGB lesion image + segmentation mask
using LOCAL PATCH ANALYSIS. If files are missing or can't be read, returns False by default.

Algorithm Summary:
------------------
1) Divide the lesion bounding box into small patches (e.g., 32×32).
2) For each patch, gather only the pixels that fall within the lesion mask.
3) Compute color variance (in BGR channels). If it's < variance_threshold => patch is structureless.
4) The final fraction = (# of structureless patches) / (# of valid patches).
   - A patch is valid if it contains enough lesion pixels to avoid spurious partial coverage.
5) If fraction >= min_fraction => "structureless" => True. Else False.
"""

import os
import cv2
import numpy as np

def detect_structureless(rgb_image_path, seg_mask_path,
                         patch_size=32,
                         min_coverage_ratio=0.3,
                         variance_threshold=400.0,
                         min_fraction=0.2):
    """
    Returns True if there's a "structureless" region occupying >= min_fraction of the lesion area.

    Args:
        rgb_image_path (str): Path to the color lesion image (e.g., ISIC_0012255.jpg).
        seg_mask_path (str): Path to the binary segmentation mask (e.g., ISIC_0012255_segmentation.png).
        patch_size (int): Size of each local patch in pixels, e.g. 32 -> patches are 32x32.
        min_coverage_ratio (float): The fraction of patch pixels that must be inside the lesion
                                    for the patch to be considered valid.
        variance_threshold (float): If the (patch) variance < this threshold, we label the patch as "structureless."
        min_fraction (float): If the fraction of structureless patches >= min_fraction, we label the entire lesion structureless.

    Returns:
        bool: True if local patch analysis suggests the lesion is structureless, False otherwise.
    """

    # 1) Ensure files exist
    if not os.path.isfile(rgb_image_path):
        print(f"[detect_structureless] Missing color image: {rgb_image_path}")
        return False

    if not os.path.isfile(seg_mask_path):
        print(f"[detect_structureless] Missing segmentation mask: {seg_mask_path}")
        return False

    # 2) Load the color image and segmentation mask
    rgb = cv2.imread(rgb_image_path)
    if rgb is None:
        print(f"[detect_structureless] Could not load image: {rgb_image_path}")
        return False

    mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[detect_structureless] Could not load mask: {seg_mask_path}")
        return False

    # Make sure the mask is binary
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 3) Find bounding rectangle of the lesion to limit patch iteration
    #    This step can speed things up if the image is large.
    #    boundingRect returns (x,y,width,height)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[detect_structureless] No contour found in mask => Empty lesion.")
        return False

    # largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)

    if w_rect <= 0 or h_rect <= 0:
        print("[detect_structureless] Zero-size bounding box => Invalid.")
        return False

    # 4) Slide over local patches in that bounding box
    structureless_count = 0
    valid_patches = 0

    for patch_y in range(y_rect, y_rect + h_rect, patch_size):
        for patch_x in range(x_rect, x_rect + w_rect, patch_size):
            # Patch coordinates
            patch_y2 = min(patch_y + patch_size, y_rect + h_rect)
            patch_x2 = min(patch_x + patch_size, x_rect + w_rect)

            patch_mask = bin_mask[patch_y:patch_y2, patch_x:patch_x2]
            patch_rgb  = rgb[patch_y:patch_y2, patch_x:patch_x2, :]

            # 4a) Count how many pixels in this patch are inside the lesion
            inside_pixels = (patch_mask == 255)
            inside_count = np.sum(inside_pixels)
            total_count  = patch_mask.size
            coverage_ratio = inside_count / total_count

            if coverage_ratio < min_coverage_ratio:
                # Not enough lesion coverage => skip
                continue

            # 4b) Extract the actual lesion pixels from patch
            lesion_pixels = patch_rgb[inside_pixels].astype(np.float32)
            if lesion_pixels.size == 0:
                continue

            valid_patches += 1

            # 4c) Compute variance across these patch pixels in BGR channels
            #     shape: (N, 3)
            std_dev = np.std(lesion_pixels, axis=0)  # (3,) standard dev for B,G,R
            patch_variance = np.mean(std_dev**2)     # average across channels

            # If patch variance < threshold => patch is structureless
            if patch_variance < variance_threshold:
                structureless_count += 1

    # 5) fraction of structureless patches
    if valid_patches == 0:
        # This means the bounding box didn't yield any patch with enough coverage.
        print("[detect_structureless] No valid patches => Can't determine => returning False.")
        return False

    fraction_below_threshold = structureless_count / valid_patches

    # 6) Compare fraction to min_fraction
    if fraction_below_threshold >= min_fraction:
        print(f"[detect_structureless] True => {structureless_count}/{valid_patches} patches (frac={fraction_below_threshold:.2f}) < var_threshold={variance_threshold}, coverage>={min_coverage_ratio}")
        return True
    else:
        print(f"[detect_structureless] False => {structureless_count}/{valid_patches} patches (frac={fraction_below_threshold:.2f}) < var_threshold={variance_threshold}, coverage>={min_coverage_ratio}")
        return False

# Demo usage
if __name__ == "__main__":
    base_id = "ISIC_0012255"
    rgb_image_path = os.path.join("..", "ISIC2018_Task1-2_Validation_Input", f"{base_id}.jpg")
    seg_mask_path = os.path.join("..", "ISIC2018_Task1_Training_GroundTruth", f"{base_id}_segmentation.png")

    print("[Local Patch Analysis Demo] Structureless detection for:", base_id)
    result = detect_structureless(rgb_image_path, seg_mask_path)
    print(f"[Result] structureless={result}")
