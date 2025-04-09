# Dermoscopic "D" Pipeline

This directory includes:
- **ISIC2018_Task1_Training_GroundTruth/**: Segmentation masks (PNG) for each lesion.
- **ISIC2018_Task2_Training_GroundTruth_v3/**: Attribute masks (PNG) indicating presence of globules, streaks, etc.

## s1_lesion_mask_extraction
- **load_mask.py**: Loads a segmentation PNG into NumPy.
- **preprocess_mask.py**: Morphological cleaning (open/close).
- **remove_artifacts.py**: Keeps only the largest connected component.

## s2_structure_detection
- **detect_globules.py**, **detect_streaks.py**, **detect_pigment_network.py**, ...
  - Each script checks whether the corresponding attribute mask is non-empty (if you have these masks).
  - (Optionally implement custom detection from color images here.)

## s3_measure_d_value
- **presence_count.py**: Counts how many structures are present.
- **compute_d_score.py**: Calculates the final "D" (e.g. 0.5 * #structures).
- **zernike_moments.py** (optional): Advanced shape/texture analysis.

## Usage
1. Obtain and clean your lesion mask from `ISIC2018_Task1_Training_GroundTruth`.
2. Identify structures from `ISIC2018_Task2_Training_GroundTruth_v3` (or your own detection).
3. Use `presence_count.py` + `compute_d_score.py` to get the final dermoscopic "D".

## Dependencies
- Python >= 3.7
- OpenCV
- NumPy
(See `requirements.txt`)
