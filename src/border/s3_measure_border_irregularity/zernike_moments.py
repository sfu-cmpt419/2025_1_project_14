# Suppl 7

import os
import cv2
import numpy as np
from scipy.spatial.qhull import ConvexHull
from scipy.spatial.distance import euclidean
import csv
import mahotas
from natsort import natsorted

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "output", "segmented")
PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, "output", "features")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
ZERNLIKE_CSV = os.path.join(PREDICTIONS_DIR, "zernike.csv")

moments_list = []
i = 1

print(f"Extracting Zernike moments from: {SEGMENTED_DIR}")

for root, dirs, files in os.walk(SEGMENTED_DIR):
    sortedFiles = natsorted(files)
    for file in sortedFiles:
        input_path = os.path.join(root, file)
        print(f"🔍 Processing: {input_path}")

        src = cv2.imread(input_path)
        if src is None:
            print(f"Warning: Unable to read {file}, skipping...")
            continue

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"Warning: No contours found in {file}, skipping...")
            continue

        contours = np.squeeze(contours)

        moments = mahotas.features.zernike_moments(gray, 21)
        moments_list.append(moments)

        hull = ConvexHull(contours, qhull_options="QJn")
        vertices = hull.vertices.tolist() + [hull.vertices[0]]
        convex_hull_perimeter = np.sum([euclidean(x, y) for x, y in zip(contours[vertices], contours[vertices][1:])])

        lesion_perimeter = cv2.arcLength(contours, True)

        convexity_ratio = convex_hull_perimeter / lesion_perimeter
        print(f"Convexity Ratio for {file}: {convexity_ratio}")

        i += 1

with open(ZERNLIKE_CSV, "w", newline="") as f1:
    writer = csv.writer(f1, delimiter="\t", lineterminator="\n")
    for m in moments_list:
        writer.writerow(m)

print(f"Zernike Moments saved to: {ZERNLIKE_CSV}")

