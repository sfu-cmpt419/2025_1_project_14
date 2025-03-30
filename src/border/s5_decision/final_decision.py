# Suppl 6

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
CNN_PREDICTIONS = os.path.join(PROJECT_ROOT, "output", "predictions", "test_predictions_cnn.csv")
GAUSSIAN_PREDICTIONS = os.path.join(PROJECT_ROOT, "output", "predictions", "test_predictions_gaussian_naive_bayes.csv")
if not os.path.exists(CNN_PREDICTIONS):
    raise FileNotFoundError(f"Missing CNN predictions file: {CNN_PREDICTIONS}")
if not os.path.exists(GAUSSIAN_PREDICTIONS):
    raise FileNotFoundError(f"Missing Gaussian Naive Bayes predictions file: {GAUSSIAN_PREDICTIONS}")

print(f"Loading predictions from:\n - CNN: {CNN_PREDICTIONS}\n - Gaussian: {GAUSSIAN_PREDICTIONS}")

cnn_predictions_csv = pd.read_csv(CNN_PREDICTIONS)
gaussian_predictions_csv = pd.read_csv(GAUSSIAN_PREDICTIONS)

if not {'0', '1'}.issubset(cnn_predictions_csv.columns):
    raise ValueError("CNN predictions CSV must contain columns ['0', '1'] for class probabilities.")
if not {'0', '1'}.issubset(gaussian_predictions_csv.columns):
    raise ValueError("Gaussian Naive Bayes predictions CSV must contain columns ['0', '1'] for class probabilities.")

cnn_class_0_probabilities = cnn_predictions_csv['0'].values
cnn_class_1_probabilities = cnn_predictions_csv['1'].values
gaussian_class_0_probabilities = gaussian_predictions_csv['0'].values
gaussian_class_1_probabilities = gaussian_predictions_csv['1'].values

total_prediction_probability = (
    (cnn_class_0_probabilities * gaussian_class_0_probabilities) +
    (cnn_class_1_probabilities * gaussian_class_1_probabilities)
) / 2

threshold = (np.max(total_prediction_probability) + np.mean(total_prediction_probability)) / 2
print(f"Computed Threshold: {threshold:.4f}")

final_decisions = ["irregular" if p > threshold else "regular" for p in total_prediction_probability]

final_decision_csv = os.path.join(PROJECT_ROOT, "output", "predictions", "final_decision.csv")
pd.DataFrame(final_decisions, columns=["Decision"]).to_csv(final_decision_csv, index=False)

num_irregular = sum(1 for d in final_decisions if d == "irregular")
num_regular = sum(1 for d in final_decisions if d == "regular")

print(f"Final Decision Summary:")
print(f" - Irregular cases: {num_irregular}")
print(f" - Regular cases: {num_regular}")
print(f"Final decisions saved to: {final_decision_csv}")
