# Suppl 13

import os
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
TRAIN_CSV = os.path.join(PROJECT_ROOT, "s6_evaluation", "train.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "s6_evaluation", "test.csv")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "output", "predictions", "bayes_predictions.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train = train_df.drop(columns=['label'])
Y_train = train_df['label']
X_test = test_df.copy()

gaussian = GaussianNB()
model = gaussian.fit(X_train, Y_train) 
Y_pred_probabilities = model.predict_proba(X_test)
Y_pred = model.predict(X_test)  

pd.DataFrame(Y_pred_probabilities[:,1], columns=['predictions']).to_csv(PREDICTIONS_CSV, index=False)

print(f"Naive Bayes Predictions Saved: {PREDICTIONS_CSV}")
