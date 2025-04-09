
# Border Subproject Pipeline Overview (PENDING TESTING)

## Introduction

Code is adopted from the paper https://pmc.ncbi.nlm.nih.gov/articles/PMC7924469/#_ad93_
This subdirectory (borders) basically provides a (still need to test it) end-to-end approach to handling border irregularities for the A(B)CD method.
This pipeline calculates a border score and also (as a bonus!) predicts whether a skin lesion's borders are "regular" or "irregular" (which is overkill for this step, but it might come in handy later.)

---

## Pipeline Steps

### **0. Faster B-Score**

**Directory:** `s0_fast_border_score/`
- **Purpose:** Fast border score calculations for images that already have segmentation masks.
- **Key Files:**
  - `b_score.m`: Gets the b-scores and optionally generates visualizations.
  - **Output:** b_scores.csv  `output/features/` and visuals are stored in `output/features/visual/`.

### **1. Image Segmentation**

**Directory:** `s1_skin_lesion_extraction/`

- **Purpose:** Extract skin lesions from the input images using a fuzzy clustering-based segmentation approach.
- **Key Files:**
  - `gradual_focusing.m`: Main segmentation script using fuzzy clustering.
  - `threshold.m`: Determines the optimum threshold for segmentation.
  - `ambig_pixels.m`: Identifies ambiguous pixels.
  - **Output:** Segmented images are stored in `output/segmented/`.

### **2. Edge Detection & Smoothing**

**Directory:** `s2_border_detection/`

- **Purpose:** Enhance the lesion boundaries and smooth the segmented images.
- **Key Files:**
  - `smoothing.m`: Applies Gaussian smoothing to the segmented images.
  - `canny_edge.m`: Detects lesion borders using Canny edge detection.
  - **Output:**
    - `output/smoothed/`: Contains smoothed images.
    - `output/borders/`: Stores edge-detected images.

### **3. Feature Extraction**

**Directory:** `s3_measure_border_irregularity/`

- **Purpose:** Compute shape-based features for lesion classification.
- **Key Files:**
  - `zernike_moments.py`: Computes Zernike moments for shape descriptors.
  - `fractal_dimension.m`: Estimates the fractal dimension for border irregularity.
  - **Output:** Feature vectors stored in `output/features/`.

### **4. Classification**

**Directory:** `s4_classification/`

- **Purpose:** Classify lesions as regular or irregular using CNN and Naive Bayes classifiers.
- **Key Files:**
  - `cnn_model.py`: Deep learning classifier using a convolutional neural network (CNN).
  - `naive_bayes.py`: A probabilistic classifier using Gaussian Naive Bayes.
  - **Output:**
    - `output/predictions/cnn_predictions.csv`: CNN predictions.
    - `output/predictions/nb_predictions.csv`: Naive Bayes predictions.

### **5. Final Decision**

**Directory:** `s5_decision/`

- **Purpose:** Use ensemble learning to combine predictions from both classifiers.
- **Key Files:**
  - `final_decision.py`: Merges CNN and Naive Bayes predictions to make a final classification.
  - **Output:**
    - `output/predictions/final_predictions.npy`: Final classification results.

### **6. Evaluation**

**Directory:** `s6_evaluation/`

- **Purpose:** Evaluate the performance of the classification models.
- **Key Files:**
  - `evaluate.py`: Computes accuracy, precision, recall, specificity, and F1-score.
  - **Output:**
    - `output/evaluation/metrics.txt`: Stores evaluation results.
    - `output/evaluation/confusion_matrix.png`: Confusion matrix visualization.

---

## Execution Flow

1. **Run the main driver script:** `driver.py`
2. The script executes each step in sequence:
   - **Segmentation** → **Edge Detection** → **Feature Extraction** → **Classification** → **Final Decision** → **Evaluation**
3. The results are stored in the respective `output/` directories.
4. The evaluation step generates performance metrics for the models.

---

```
borders/
│── border_driver.py               
│── requirements.txt               
│── input_images/                  # You may need to create this yourself (Input)
│   │── image1.jpg
│   │── image2.jpg
│── output/                        # You may need to create this yourself (Output)
│   │── segmented/                 # Segmented lesion images
│   │── smoothed/                  # Smoothed images
│   │── borders/                   # Edge-detected images
│   │── features/                  # Extracted features (fractal, zernike, border score)
│   │── predictions/               # Model predictions (CNN, Naive Bayes, Final decision)
│   │── evaluation/                # Evaluation metrics and visualizations
│── s1_skin_lesion_extraction/      # Segmentation step
│   │── gradual_focusing.m
│   │── threshold.m
│   │── ambig_pixels.m
│── s2_border_detection/           # Edge detection and smoothing step
│   │── smoothing.m
│   │── canny_edge.m
│── s3_measure_border_irregularity/ # Feature extraction step
│   │── zernike_moments.py
│   │── fractal_dimension.m
│   │── border_score.py
│── s4_classification/             # Classification step
│   │── cnn_model.py
│   │── naive_bayes.py
│── s5_decision/                   # Final decision step
│   │── final_decision.py
│── s6_evaluation/                 # Evaluation step
│   │── evaluate.py
```

---

## Dependencies

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

---

## Notes

- Input images should be placed in `input_images/` before running the pipeline.
- Modify file paths in `driver.py` if needed.
- Adjust hyperparameters in the CNN model for improved performance.

---

## Contact

For any issues, contact Sasha Vujisic (sva39.) README.md formatted with ChatGPT.

