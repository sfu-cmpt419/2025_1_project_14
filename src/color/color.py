import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans

def calculateColorScore(image: cv2.Mat, mask: cv2.Mat, max_clusters: int = 6, min_cluster_fraction: float = 0.05) -> int:
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lesion_pixels = hsv[mask > 0]

    if len(lesion_pixels) == 0:
        return 0

    kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init='auto')
    kmeans.fit(lesion_pixels)

    _, counts = np.unique(kmeans.labels_, return_counts=True)
    total_pixels = len(lesion_pixels)

    # Count clusters with significant number of pixels
    significant_clusters = sum(count > total_pixels * min_cluster_fraction for count in counts)

    return significant_clusters

def processColorResults(imageFolder, maskFolder, outputCSV):
    results = []

    for filename in os.listdir(imageFolder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            imagePath = os.path.join(imageFolder, filename)
            maskFilename = filename.replace(".jpg", "_segmentation.png").replace(".JPG", "_segmentation.png")
            maskPath = os.path.join(maskFolder, maskFilename)

            if not os.path.exists(maskPath):
                print(f"Mask not found for {filename}, skipping.")
                continue

            image = cv2.imread(imagePath)
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error reading {filename} or its mask, skipping.")
                continue

            score = calculateColorScore(image, mask)
            results.append([filename, score])
            print(f"Processed {filename} -> Color Score: {score}")

    df = pd.DataFrame(results, columns=["Image", "Color Score"])
    df.to_csv(outputCSV, index=False)
    print(f"Saved results to {outputCSV}")

"""
trainingImageFolder = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/ISIC2018_Task1-2_Training_Input"
trainingMaskFolder = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/ISIC2018_Task1_Training_GroundTruth"
trainingOutputCSV = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/color_training_results.csv"

validationImageFolder = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/ISIC2018_Task1-2_Validation_Input"
validationMaskFolder = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/ISIC2018_Task1_Validation_GroundTruth"
validationOutputCSV = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/color_validation_results.csv"

testImageFolder = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/ISIC2018_Task1-2_Test_Input"
testMaskFolder = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/ISIC2018_Task1_Test_GroundTruth"
testOutputCSV = r"/Users/andy/Documents/GitHub/2025_1_project_14/src/color/color_test_results.csv"
"""

# Batch run for all datasets
datasetFolders = [
    (trainingImageFolder, trainingMaskFolder, trainingOutputCSV),
    (validationImageFolder, validationMaskFolder, validationOutputCSV),
    (testImageFolder, testMaskFolder, testOutputCSV)
]

for imgFolder, maskFolder, outputPath in datasetFolders:
    print(f"\nProcessing: {imgFolder}")
    processColorResults(imgFolder, maskFolder, outputPath)
