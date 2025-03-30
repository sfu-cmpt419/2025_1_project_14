# Suppl 9

import os
import pandas as pd
import numpy as np
import cv2
import keras
import torch  # ✅ Import PyTorch
keras.config.set_backend("torch") 

from keras import optimizers
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TRAIN_EDGES_DIR = os.path.join(PROJECT_ROOT, "output", "borders", "train")
TEST_EDGES_DIR = os.path.join(PROJECT_ROOT, "output", "borders", "test")
TRAIN_DF_PATH = os.path.join(PROJECT_ROOT, "s6_evaluation", "train.csv")
TEST_DF_PATH = os.path.join(PROJECT_ROOT, "s6_evaluation", "test.csv")

train_df = pd.read_csv(TRAIN_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

border_irregularity_features = train_df[:, :12]
test_border_irregularity_features = test_df[:, :12]

def load_images(directory):
    images = []
    labels = []
    for file in sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0])):
        img = cv2.imread(os.path.join(directory, file), cv2.IMREAD_GRAYSCALE)
        images.append(img)
        labels.append(1 if 'r' in file else 0)
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    return images, labels

images_edge, y = load_images(TRAIN_EDGES_DIR)
test_images_edge, y_test = load_images(TEST_EDGES_DIR)

images_edge = torch.tensor(images_edge, dtype=torch.float32)
border_irregularity_features = torch.tensor(border_irregularity_features, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

test_images_edge = torch.tensor(test_images_edge, dtype=torch.float32)
test_border_irregularity_features = torch.tensor(test_border_irregularity_features, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

image_input = keras.layers.Input(shape=(512,512,1))
features_input = keras.layers.Input(shape=(12,))

conv1 = keras.layers.Conv2D(32, (5,5), activation='relu')(image_input)
conv2 = keras.layers.Conv2D(24, (5,5), activation='relu', strides=(2,2))(conv1)
conv3 = keras.layers.Conv2D(36, (5,5), activation='relu', strides=(2,2))(conv2)
conv4 = keras.layers.Conv2D(48, (5,5), activation='relu', strides=(2,2))(conv3)
conv5 = keras.layers.Conv2D(64, (5,5), activation='relu')(conv4)
conv6 = keras.layers.Conv2D(64, (3,3), activation='relu')(conv5)
flatten = keras.layers.Flatten()(conv6)

merged = keras.layers.concatenate([flatten, features_input])
dense1 = keras.layers.Dense(100, activation='relu')(merged)
output = keras.layers.Dense(1, activation='sigmoid')(dense1)

model = keras.models.Model(inputs=[image_input, features_input], outputs=[output])

optimizer = optimizers.Adam(learning_rate=0.0001)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit([images_edge, border_irregularity_features], y, epochs=10, batch_size=1, validation_split=0.33)

MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "predictions", "cnn_model.pth")
PREDICTION_CSV = os.path.join(PROJECT_ROOT, "output", "predictions", "cnn_predictions.csv")

torch.save(model.state_dict(), MODEL_PATH)

predictions = model.predict([images_edge, border_irregularity_features])
rounded_preds = [round(x[0]) for x in predictions]

pd.DataFrame(rounded_preds, columns=['predictions']).to_csv(PREDICTION_CSV, index=False)

print(f"CNN Model Saved: {MODEL_PATH}")
print(f"Predictions Saved: {PREDICTION_CSV}")
