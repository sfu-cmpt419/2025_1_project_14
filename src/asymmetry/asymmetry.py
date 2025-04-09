import cv2
import numpy as np
import pandas as pd
import os

def calculateVerticalAsymmetry(binaryMask: np.ndarray, threshold: float, x: int, y: int, width: int, height: int) -> tuple[int, float]:
    lesionArea: float = np.sum(binaryMask > 0)
    
    xMiddle: int = x + width // 2
    
    leftMaskHalf: np.ndarray = binaryMask[y : y + height:, x : xMiddle]
    rightMaskHalf: np.ndarray = binaryMask[y : y + height:, xMiddle : x + width]
    
    rightMaskHalfFlipped: np.ndarray = np.fliplr(rightMaskHalf)
    
    # Make sure both halves have the same width for comparison
    minWidth: int = min(leftMaskHalf.shape[1], rightMaskHalfFlipped.shape[1])
    leftMaskHalf = leftMaskHalf[:, :minWidth]
    rightMaskHalfFlipped = rightMaskHalfFlipped[:, :minWidth]
    
    # Find pixels that are different between the two halves
    # and calculate the ratio of different pixels to the total lesion area
    differentPixelsMask: np.ndarray = cv2.absdiff(leftMaskHalf, rightMaskHalfFlipped)
    numDifferentPixels: float = np.sum(differentPixelsMask > 0) / 255.0 # 255.0 represents the value of a lesion pixel
    verticalDifferenceRatio: float = numDifferentPixels / lesionArea

    isGreaterThanThreshold: bool = verticalDifferenceRatio > threshold
    if isGreaterThanThreshold:
        return 1, verticalDifferenceRatio
    else:
        return 0, verticalDifferenceRatio
    
def calculateHorizontalAsymmetry(binaryMask: np.ndarray, threshold: float, x: int, y: int, width: int, height: int) -> tuple[int, float]:
    totalLesionArea: float = np.sum(binaryMask > 0)

    yMiddle: int = y + height // 2
    
    topMaskHalf: np.ndarray = binaryMask[y : yMiddle, x : x + width]
    bottomMaskHalf: np.ndarray = binaryMask[yMiddle : y + height, x : x + width]
    
    bottomMaskHalfFlipped: np.ndarray = np.flipud(bottomMaskHalf)
    
    # Make sure both halves have the same height for comparison
    minHeight: int = min(topMaskHalf.shape[0], bottomMaskHalfFlipped.shape[0])
    topMaskHalf = topMaskHalf[:minHeight, :]
    bottomMaskHalfFlipped = bottomMaskHalfFlipped[:minHeight, :]
    
    # Find pixels that are different between the two halves
    # and calculate the ratio of different pixels to the total lesion area
    differentPixelsMask: np.ndarray = cv2.absdiff(topMaskHalf, bottomMaskHalfFlipped)
    numDifferentPixels: float = np.sum(differentPixelsMask > 0) / 255.0 # 255.0 represents the value of a lesion pixel
    horizontalDifferenceRatio: float = numDifferentPixels / totalLesionArea

    isGreaterThanThreshold: bool = horizontalDifferenceRatio > threshold
    if isGreaterThanThreshold:
        return 1, horizontalDifferenceRatio
    else:
        return 0, horizontalDifferenceRatio

def calculateAsymmetry(mask: cv2.Mat, threshold: float = 0.0005) -> tuple[float, float, float, float]:
    # Returns the top-left coordinate (x, y) and width and height (w, h) of the bounding rectangle
    x: int
    y: int
    width: int
    height: int
    x, y, width, height = cv2.boundingRect(mask)
    
    verticalAsymmetry: int
    verticalDifference: float
    verticalAsymmetry, verticalDifference = calculateVerticalAsymmetry(mask, threshold, x, y, width, height)

    horizontalAsymmetry: int
    horizontalDifference: float
    horizontalAsymmetry, horizontalDifference = calculateHorizontalAsymmetry(mask, threshold, x, y, width, height)
    
    asymmetryContribution: float = (verticalAsymmetry + horizontalAsymmetry) * 1.3

    return (asymmetryContribution, verticalDifference, horizontalDifference)

def processAsymmetryResult(maskFolder, csv):
    results = []
    
    for filename in os.listdir(maskFolder):
        if filename.endswith(".png"):
            maskPath = os.path.join(maskFolder, filename)
            mask: cv2.Mat = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

            (asymmetryContribution, verticalDifferenceRatio, horizontalDifferenceRatio) = calculateAsymmetry(mask)

            results.append([filename, asymmetryContribution, verticalDifferenceRatio, horizontalDifferenceRatio])

            print(f"Processed {len(results)} images out of {len(os.listdir(maskFolder))}")
    
    df = pd.DataFrame(results, columns=["Image", "Asymmetry Contribution", "Vertical Diff Ratio", "Horizontal Diff Ratio"])
    df.to_csv(csv, index=False)
    print(f"Results saved to {csv}")


# Enter the path to the folders containing the segmentation masks and the output CSV files
# trainingDataGroundTruthFolderPath = r"C:\Code\2025_1_project_14\src\asymmetry\ISIC2018_Task1_Training_GroundTruth"
# trainingDataGroundTruthOutputCSVPath = r"C:\Code\2025_1_project_14\src\asymmetry\asymmetry_training_groundtruth_results.csv"

# validationDataGroundTruthFolderPath = r"C:\Code\2025_1_project_14\src\asymmetry\ISIC2018_Task1_Validation_GroundTruth"
# validationDataGroundTruthOutputCSVPath = r"C:\Code\2025_1_project_14\src\asymmetry\asymmetry_validation_groundtruth_results.csv"

# testDataGroundTruthFolderPath = r"C:\Code\2025_1_project_14\src\asymmetry\ISIC2018_Task1_Test_GroundTruth"
# testDataGroundTruthOutputCSVPath = r"C:\Code\2025_1_project_14\src\asymmetry\asymmetry_test_groundtruth_results.csv"

groundTruthFolders = [
    (trainingDataGroundTruthFolderPath, trainingDataGroundTruthOutputCSVPath),
    (validationDataGroundTruthFolderPath, validationDataGroundTruthOutputCSVPath),
    (testDataGroundTruthFolderPath, testDataGroundTruthOutputCSVPath)
]

for groundTruthFolderPath, outputCSVPath in groundTruthFolders:
    print(f"Begin processing {groundTruthFolderPath} into {outputCSVPath}")
    processAsymmetryResult(groundTruthFolderPath, outputCSVPath)
    print(f"Processed {groundTruthFolderPath} and saved results to {outputCSVPath}")