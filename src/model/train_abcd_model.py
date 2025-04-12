#!/usr/bin/env python3
"""
train_abcd_model.py

Train a deep learning regression model to predict ABCD scores directly from RGB skin lesion images.
The model is built using transfer learning with EfficientNetB0 and a custom regression head.
The CSV file (e.g., clean_data.csv) must contain the following columns:
    Image, A_score, C_score, D_value, B_score
where 'Image' corresponds to the image identifier (used to build the image filename).

Improvement(s) in this version:
  • Target Normalization: Each score is normalized based on its min/max range:
        A_score_norm = A_score / 2.6
        C_score_norm = (C_score - 2) / 4.0
        D_value_norm = D_value / 1.5
        B_score_norm = B_score / 8.0
  • The model is trained to predict these normalized values.
  • Extensive data augmentation is applied to learn invariance.
  • The data is split into training, validation, and test sets.

Usage:
  python train_abcd_model.py --csv_path clean_data.csv --image_dir /path/to/images --output_model abcd_model.h5

Optional arguments allow you to configure image size, batch size, number of epochs, and test split.
"""

import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model to predict ABCD scores from skin lesion images.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the CSV file with columns: Image, A_score, C_score, D_value, B_score.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory where lesion images are stored (filenames are <Image>.jpg or <Image>.png).")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Filename to save the trained model (e.g., abcd_model.h5).")
    parser.add_argument("--img_height", type=int, default=224, help="Image height after resizing (default: 224).")
    parser.add_argument("--img_width", type=int, default=224, help="Image width after resizing (default: 224).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for initial training (default: 30).")
    parser.add_argument("--test_split", type=float, default=0.15, help="Fraction of data to use as test set (default: 0.15).")
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

def build_generators(df_train, df_val, img_height, img_width, batch_size):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col='filepath',
        # Use normalized target columns
        y_col=['A_norm', 'C_norm', 'D_norm', 'B_norm'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col='filepath',
        y_col=['A_norm', 'C_norm', 'D_norm', 'B_norm'],
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )
    
    return train_generator, val_generator

def build_model(img_height, img_width):
    """Build model using EfficientNetB0 as the base and a custom regression head for 4 outputs."""
    input_tensor = Input(shape=(img_height, img_width, 3))
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='linear')(x)  # Output: A_norm, C_norm, D_norm, B_norm
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def main():
    args = parse_args()
    csv_path = args.csv_path
    image_dir = args.image_dir
    output_model = args.output_model
    img_height = args.img_height
    img_width = args.img_width
    batch_size = args.batch_size
    epochs = args.epochs
    test_split = args.test_split

    # Load CSV file
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} samples from {csv_path}.")

    # Create file paths for images
    df['filepath'] = df['Image'].apply(lambda x: get_image_filepath(x, image_dir))
    missing_files = df['filepath'].isnull().sum()
    if missing_files > 0:
        print(f"[WARN] {missing_files} samples do not have a corresponding image file and will be dropped.")
        df = df.dropna(subset=['filepath']).reset_index(drop=True)
    print(f"[INFO] {len(df)} samples remain after file check.")

    # Normalize target values using the provided min/max ranges:
    # A_score: [0, 2.6], C_score: [2, 6] -> range = 4, D_value: [0, 1.5], B_score: [0, 8]
    df['A_norm'] = df['A_score'] / 2.6
    df['C_norm'] = (df['C_score'] - 2.0) / 4.0
    df['D_norm'] = df['D_value'] / 1.5
    df['B_norm'] = df['B_score'] / 8.0

    # Split data into training, validation, and test sets.
    df_train_val, df_test = train_test_split(df, test_size=test_split, random_state=42)
    # Use ~17.65% of the train_val data for validation to obtain roughly 70/15/15 split.
    df_train, df_val = train_test_split(df_train_val, test_size=0.1765, random_state=42)
    print(f"[INFO] Training: {len(df_train)} samples, Validation: {len(df_val)} samples, Test: {len(df_test)} samples.")

    # Build data generators using the normalized target columns.
    train_gen, val_gen = build_generators(df_train, df_val, img_height, img_width, batch_size)

    # Build model.
    model, base_model = build_model(img_height, img_width)
    # Freeze the base model layers initially.
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae'])
    model.summary()

    # Define callbacks.
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(output_model, monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    # Train initial model.
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=val_gen.samples // batch_size,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    # Fine-tuning: unfreeze the last 50 layers of the base model.
    fine_tune_at = len(base_model.layers) - 50
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
    fine_tune_epochs = 10
    history_ft = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_tune_epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=val_gen.samples // batch_size,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    # Save the final model.
    model.save("abcd_model.keras")
    print(f"[INFO] Final model saved as {output_model}.")

if __name__ == "__main__":
    main()
