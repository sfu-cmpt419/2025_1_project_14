#!/usr/bin/env python3
"""
evaluate_abcd_model.py

This script evaluates a trained ABCD regression model and creates visualizations that compare
the true ABCD scores to the predicted scores for the test set.

It expects:
  - A CSV file with columns: Image, A_score, C_score, D_value, B_score.
    'Image' is an identifier (without extension).
  - An image directory where images are stored (filenames: <Image>.jpg or <Image>.png).
  - A saved trained model file (e.g., abcd_model.keras).
  
During training the targets were normalized:
   A_norm = A_score / 2.6
   C_norm = (C_score - 2.0) / 4.0
   D_norm = D_value / 1.5
   B_norm = B_score / 8.0

This script rebuilds those normalized columns for the test set, uses an ImageDataGenerator to load data,
evaluates the model, and then un-normalizes predictions for visualization. It then creates scatter plots
for each of the four outputs.

Usage:
  python evaluate_abcd_model.py --csv_path data.csv --image_dir /path/to/images --model_path abcd_model.keras --batch_size 32 --img_height 224 --img_width 224
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import r2_score, explained_variance_score
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ABCD model and create visualizations of predictions.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the CSV file with columns: Image, A_score, C_score, D_value, B_score.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory where the test images are stored (filenames <Image>.jpg or <Image>.png).")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved trained model (e.g., abcd_model.keras).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation (default: 32).")
    parser.add_argument("--img_height", type=int, default=224, help="Target image height (default: 224).")
    parser.add_argument("--img_width", type=int, default=224, help="Target image width (default: 224).")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Filename to save predictions (default: predictions.csv).")
    args = parser.parse_args()
    return args

def get_image_filepath(image_id, image_dir):
    """Construct full file path using image_id. Try .jpg first, then .png."""
    jpg_path = os.path.join(image_dir, f"{image_id}.jpg")
    png_path = os.path.join(image_dir, f"{image_id}.png")
    if os.path.isfile(jpg_path):
        return jpg_path
    elif os.path.isfile(png_path):
        return png_path
    else:
        return None

def build_test_generator(df, img_height, img_width, batch_size):
    """Create an ImageDataGenerator for the test set with only rescaling."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filepath",
        y_col=["A_norm", "C_norm", "D_norm", "B_norm"],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False
    )
    return test_generator

def unnormalize_predictions(pred_norm):
    """
    Given predictions in normalized scale, convert back to original scale.
      A_score_orig = pred[0] * 2.6
      C_score_orig = pred[1] * 4.0 + 2.0
      D_value_orig = pred[2] * 1.5
      B_score_orig = pred[3] * 8.0
    """
    A_pred = pred_norm[:, 0] * 2.6
    C_pred = pred_norm[:, 1] * 4.0 + 2.0
    D_pred = pred_norm[:, 2] * 1.5
    B_pred = pred_norm[:, 3] * 8.0
    return np.stack([A_pred, C_pred, D_pred, B_pred], axis=1)

def main():
    args = parse_args()
    csv_path = args.csv_path
    image_dir = args.image_dir
    model_path = args.model_path
    batch_size = args.batch_size
    img_height = args.img_height
    img_width = args.img_width
    output_csv = args.output_csv

    # Load the CSV file.
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples from {csv_path}.")

    # Build image file paths.
    df['filepath'] = df['Image'].apply(lambda x: get_image_filepath(x, image_dir))
    missing = df['filepath'].isnull().sum()
    if missing > 0:
        print(f"[WARN] {missing} samples are missing image files; dropping them.")
        df = df.dropna(subset=['filepath']).reset_index(drop=True)
    print(f"[INFO] {len(df)} samples remain after file check.")

    # Create normalized target columns.
    # Provided ranges:
    #   A_score: [0, 2.6]
    #   C_score: [2.0, 6.0]
    #   D_value: [0, 1.5]
    #   B_score: [0, 8.0]
    df['A_norm'] = df['A_score'] / 2.6
    df['C_norm'] = (df['C_score'] - 2.0) / 4.0
    df['D_norm'] = df['D_value'] / 1.5
    df['B_norm'] = df['B_score'] / 8.0

    # For evaluation, we use the entire CSV as test data.
    print(f"[INFO] Using {len(df)} samples for evaluation.")
    test_generator = build_test_generator(df, img_height, img_width, batch_size)
    
    # Load the trained model.
    print(f"[INFO] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # Evaluate the model using model.evaluate (MSE and MAE).
    print("[INFO] Evaluating model on test set...")
    eval_results = model.evaluate(test_generator, steps=test_generator.samples // batch_size, verbose=1)
    print(f"[Test Evaluation] Loss (MSE): {eval_results[0]:.4f}, MAE: {eval_results[1]:.4f}")
    
    # Make predictions on the test set.
    print("[INFO] Generating predictions on test set...")
    pred_norm = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
    
    # Unnormalize predictions to obtain outputs in the original scale.
    predictions = unnormalize_predictions(pred_norm)
    # Also unnormalize true values:
    y_true = df[['A_score', 'C_score', 'D_value', 'B_score']].values
    
    # Save true and predicted values to a CSV for further analysis.
    df_results = pd.DataFrame({
        "Image": df["Image"],
        "True_A": y_true[:, 0],
        "Pred_A": predictions[:, 0],
        "True_C": y_true[:, 1],
        "Pred_C": predictions[:, 1],
        "True_D": y_true[:, 2],
        "Pred_D": predictions[:, 2],
        "True_B": y_true[:, 3],
        "Pred_B": predictions[:, 3]
    })
    results_csv = "predictions.csv"
    df_results.to_csv(results_csv, index=False)
    print(f"[INFO] Saved predictions to {results_csv}.")

    # Compute additional regression metrics.
    from sklearn.metrics import r2_score, explained_variance_score
    r2 = r2_score(y_true, predictions, multioutput='uniform_average')
    evs = explained_variance_score(y_true, predictions, multioutput='uniform_average')
    print(f"[Test Metrics] R² Score: {r2:.4f}")
    print(f"[Test Metrics] Explained Variance Score: {evs:.4f}")

    # Visualizations: scatter plots for each target.
    targets = ["A", "C", "D", "B"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, target in enumerate(targets):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = predictions[:, i]
        ax.scatter(true_vals, pred_vals, alpha=0.5, edgecolor='k')
        ax.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--', lw=2)
        ax.set_title(f"True vs Predicted {target}_score")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    
    plt.tight_layout()
    plt.savefig("predictions_scatter.png")
    plt.show()
    print("[INFO] Visualizations saved as 'predictions_scatter.png'.")

if __name__ == "__main__":
    main()
