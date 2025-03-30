import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PREDICTION_FILE = os.path.join(PROJECT_ROOT, "output", "predictions", "final_decision.csv")
TESTING_CSV_FILE = os.path.join(PROJECT_ROOT, "s6_evaluation", "test.csv")  # From Supplemental Info 2
TRAINING_CSV_FILE = os.path.join(PROJECT_ROOT, "s6_evaluation", "train.csv")  # From Supplemental Info 1

def load_ground_truth():
    """
    Load ground truth labels from the testing CSV file.
    The CSV is expected to have:
    | Image_Name  | Label |
    |------------|-------|
    | image1.jpg |   0   |
    | image2.jpg |   1   |
    
    Returns:
        dict: { "image1.jpg": 0, "image2.jpg": 1, ... }
    """
    if not os.path.exists(TESTING_CSV_FILE):
        raise FileNotFoundError(f"Missing testing dataset: {TESTING_CSV_FILE}")

    df = pd.read_csv(TESTING_CSV_FILE)
    if "Image_Name" not in df.columns or "Label" not in df.columns:
        raise ValueError(f"Incorrect format in {TESTING_CSV_FILE}. Expected columns: ['Image_Name', 'Label']")
    
    return dict(zip(df["Image_Name"], df["Label"]))

def load_predictions():
    """
    Load model predictions from `final_decision.csv`.
    """
    if not os.path.exists(PREDICTION_FILE):
        raise FileNotFoundError(f"Missing prediction file: {PREDICTION_FILE}")

    df = pd.read_csv(PREDICTION_FILE)
    
    if "Decision" not in df.columns:
        raise ValueError(f"Missing 'Decision' column in {PREDICTION_FILE}")

    return df["Decision"].map({"regular": 0, "irregular": 1}).values  # Convert to numerical labels

def evaluate_model():
    print("Loading ground truth labels from CSV...")
    ground_truth = load_ground_truth()
    
    print("Loading model predictions...")
    predictions = load_predictions()

    image_names = sorted(ground_truth.keys())
    true_labels = np.array([ground_truth[img] for img in image_names])

    if len(true_labels) != len(predictions):
        raise ValueError(f"Mismatch: {len(true_labels)} ground truth labels but {len(predictions)} predictions.")

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)  # Sensitivity
    specificity = recall_score(true_labels, predictions, pos_label=0, zero_division=0)  # Specificity for class 0
    f1 = f1_score(true_labels, predictions, zero_division=0)

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Regular", "Irregular"], yticklabels=["Regular", "Irregular"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    confusion_matrix_path = os.path.join(PROJECT_ROOT, "output", "predictions", "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    print(f"📂 Confusion matrix saved to: {confusion_matrix_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
