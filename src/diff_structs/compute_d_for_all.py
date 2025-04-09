"""
compute_d_for_all.py

Two-phase pipeline with single-row output per lesion at the end.

Phase 1: 
  - For each *_segmentation.png in ISIC2018_Task1_Training_GroundTruth, 
    detect globules, streaks, pigment_network (attribute masks).
  - Store results in a dictionary, with dots=structureless=False initially.

Phase 2:
  - For each .jpg in ISIC2018_Task1-2_Validation_Input:
      - If official seg mask is found => load it
      - Else do naive manual_segmentation
      - detect dots & structureless => update or create a dictionary entry
        (if new, set attribute-based structures=0 by default)

Finally: 
  - One row per lesion in a single CSV => d_results.csv
"""

import os
import csv
import pandas as pd

# (1) s1_lesion_mask_extraction
from s1_lesion_mask_extraction.load_mask import load_mask
from s1_lesion_mask_extraction.preprocess_mask import preprocess_mask
from s1_lesion_mask_extraction.remove_artifacts import remove_artifacts
# Import our new manual segmentation
from s1_lesion_mask_extraction.manual_segmentation import manual_segmentation

# (2) s2_structure_detection - attribute-based
from s2_structure_detection.detect_globules import detect_globules
from s2_structure_detection.detect_streaks import detect_streaks
from s2_structure_detection.detect_pigment_network import detect_pigment_network

# (3) s2_structure_detection - rgb-based
from s2_structure_detection.detect_structureless import detect_structureless
from s2_structure_detection.detect_dots import detect_dots

# (4) s3_measure_d_value
from s3_measure_d_value.presence_count import presence_count
from s3_measure_d_value.compute_d_score import compute_d_score

# --------------------------
# Folders & Output
# --------------------------
SEG_FOLDER       = "ISIC2018_Task1_Training_GroundTruth"
ATTR_FOLDER      = "ISIC2018_Task2_Training_GroundTruth_v3"
RGB_FOLDER       = "ISIC2018_Task1-2_Validation_Input"
OUTPUT_CSV       = "d_results.csv"

# Fieldnames for final CSV
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

ATTRIBUTE_STRUCTURES = {
    "globules": detect_globules,
    "streaks": detect_streaks,
    "pigment_network": detect_pigment_network
}

RGB_STRUCTURES = {
    "dots": detect_dots,
    "structureless": detect_structureless
}

def phase1_attribute_detection(seg_folder, attr_folder):
    """
    Phase 1 dictionary:
      results_dict[lesion_id] = {
         'globules': bool,
         'streaks': bool,
         'pigment_network': bool,
         'dots': False,            # default
         'structureless': False    # default
      }
    """
    seg_files = [f for f in os.listdir(seg_folder) if f.lower().endswith("_segmentation.png")]
    seg_files.sort()
    print(f"[Phase1] Found {len(seg_files)} segmentation masks in {seg_folder}.\n")

    results = {}

    for idx, seg_filename in enumerate(seg_files, start=1):
        lesion_id = seg_filename.replace("_segmentation.png", "")
        seg_path  = os.path.join(seg_folder, seg_filename)

        print(f"[Phase1] {idx}/{len(seg_files)} => {lesion_id}")
        # Check that the segmentation is loadable
        mask_img = load_mask(seg_path)
        if mask_img is None:
            print(f"   Could not load mask => skip {lesion_id}")
            continue

        # minimal cleanup
        mask_clean = preprocess_mask(mask_img, 5, "open")
        main_mask  = remove_artifacts(mask_clean)

        # Initialize
        row_data = {
            "globules": False,
            "streaks": False,
            "pigment_network": False,
            "dots": False,
            "structureless": False
        }

        # Detect attribute-based
        for struct_name, detect_func in ATTRIBUTE_STRUCTURES.items():
            attr_filename = f"{lesion_id}_attribute_{struct_name}.png"
            attr_path = os.path.join(attr_folder, attr_filename)
            if not os.path.isfile(attr_path):
                print(f"    [WARN] Missing {attr_filename} => {struct_name}=False")
                row_data[struct_name] = False
            else:
                val = detect_func(attr_path)
                row_data[struct_name] = val
                print(f"    {struct_name}={val}")

        results[lesion_id] = row_data

    print("\n[Phase1] Done.\n")
    return results


def phase2_rgb_detection(results_dict, seg_folder, rgb_folder):
    """
    For each .jpg in rgb_folder:
      - parse lesion_id
      - if official seg mask => load it
      - else => manual_segmentation
      - detect dots, structureless => update existing row or create new
    """
    jpg_files = [f for f in os.listdir(rgb_folder) if f.lower().endswith(".jpg")]
    jpg_files.sort()
    print(f"[Phase2] Found {len(jpg_files)} .jpg in {rgb_folder}.\n")

    for idx, jpg_filename in enumerate(jpg_files, start=1):
        lesion_id = jpg_filename.replace(".jpg", "")
        rgb_path  = os.path.join(rgb_folder, jpg_filename)

        print(f"[Phase2] {idx}/{len(jpg_files)} => lesion_id={lesion_id}")

        seg_path = os.path.join(seg_folder, f"{lesion_id}_segmentation.png")
        if os.path.isfile(seg_path):
            # We can use official segmentation
            dots_val = detect_dots(rgb_path, seg_path)
            struct_val = detect_structureless(rgb_path, seg_path)
        else:
            # manual segmentation
            print(f"    No official seg mask => doing manual_segmentation.")
            man_mask = manual_segmentation(rgb_path)
            if man_mask is None:
                print("    [WARN] Could not do manual segmentation => skip this lesion.")
                continue
            # We'll have to adapt detect_dots/detect_structureless to accept mask arrays
            # or we create a temp seg file. For now, let's skip that detail and assume we've done it:
            # Example approach: write man_mask to a temp file => detect
            import cv2
            temp_mask_path = f"temp_{lesion_id}_seg.png"
            cv2.imwrite(temp_mask_path, man_mask)
            
            dots_val = detect_dots(rgb_path, temp_mask_path)
            struct_val = detect_structureless(rgb_path, temp_mask_path)

            # remove temp file
            os.remove(temp_mask_path)

        # If lesion_id not in dict => means it wasn't found in Phase1
        if lesion_id not in results_dict:
            # create a new entry
            results_dict[lesion_id] = {
                "globules": False,
                "streaks": False,
                "pigment_network": False,
                "dots": dots_val,
                "structureless": struct_val
            }
        else:
            # update existing
            results_dict[lesion_id]["dots"] = dots_val
            results_dict[lesion_id]["structureless"] = struct_val

        print(f"    dots={dots_val}, structureless={struct_val}")

    print("[Phase2] Done.\n")
    return results_dict


def finalize_and_write_csv(results_dict, output_csv):
    """
    Once both phases are complete, we have a single dictionary entry per lesion.
    For each lesion => compute total presence_count => D = 0.5 * count
    Write exactly one row per lesion to CSV.
    """
    print("[Final] Writing results to CSV =>", output_csv)
    all_lesions = sorted(results_dict.keys())

    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for lesion_id in all_lesions:
            row_data = results_dict[lesion_id]
            # presence_count
            count_structs = (row_data["globules"] + 
                             row_data["streaks"] + 
                             row_data["pigment_network"] + 
                             row_data["dots"] + 
                             row_data["structureless"])
            d_val = compute_d_score(count_structs, 0.5)

            out_row = {
                "lesion_id": lesion_id,
                "globules_present": int(row_data["globules"]),
                "streaks_present": int(row_data["streaks"]),
                "pigment_network_present": int(row_data["pigment_network"]),
                "dots_present": int(row_data["dots"]),
                "structureless_present": int(row_data["structureless"]),
                "num_structures": count_structs,
                "D_value": d_val
            }
            writer.writerow(out_row)

    print(f"[Final] Wrote {len(all_lesions)} rows to {output_csv}.\n")


def main():
    # 1) Phase1 => attribute-based detection
    phase1_dict = phase1_attribute_detection(SEG_FOLDER, ATTR_FOLDER)

    # 2) Phase2 => rgb-based detection => updates/inserts data
    final_dict = phase2_rgb_detection(phase1_dict, SEG_FOLDER, RGB_FOLDER)

    # 3) Final => single CSV row per lesion
    finalize_and_write_csv(final_dict, OUTPUT_CSV)

if __name__ == "__main__":
    main()
