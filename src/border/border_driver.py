import os
import subprocess

# Project Paths
PROJECT_ROOT = os.getcwd()
INPUT_DIR = os.path.join(PROJECT_ROOT, "src\\border\input_images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Define output subdirectories
SEGMENTED_DIR = os.path.join(OUTPUT_DIR, "segmented")
SMOOTHED_DIR = os.path.join(OUTPUT_DIR, "smoothed")
BORDERS_DIR = os.path.join(OUTPUT_DIR, "borders")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

# Ensure output directories exist
os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(SMOOTHED_DIR, exist_ok=True)
os.makedirs(BORDERS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# -------------- STEP 1: SEGMENTATION --------------
def segment_images():
    STEP1_FOLDER = os.path.join(PROJECT_ROOT, "s1_skin_lesion_extraction")
    for image in os.listdir(INPUT_DIR):
        image_path = os.path.join(INPUT_DIR, image)
        output_image_path = os.path.join(OUTPUT_DIR["segmented"], f"segmented_{image}")

        print(f"Running segmentation on: {image}")

        matlab_command = f"""
        matlab -nodisplay -nosplash -r "addpath('{STEP1_FOLDER}'); image_name='{image_path}'; run('gradual_focusing.m'); exit;"
        """
        subprocess.run(matlab_command, shell=True)

        if os.path.exists("segmentation.png"):
            os.rename("segmentation.png", output_image_path)
            print(f"Saved segmented image: {output_image_path}")
        else:
            print(f"Segmentation failed for {image}")

# -------------- STEP 2: BORDER DETECTION --------------
def detect_edges():
    print("Running Step 2: Smoothing (MATLAB)...")
    matlab_script = f"""
    addpath('s2_border_detection');
    disp('Processing images for smoothing...');
    input_folder = fullfile('{SEGMENTED_DIR}');
    output_folder = fullfile('{SMOOTHED_DIR}');

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    image_files = dir(fullfile(input_folder, '*.png'));

    for k = 1:numel(image_files)
        input_path = fullfile(input_folder, image_files(k).name);
        output_path = fullfile(output_folder, strcat('smoothed_', image_files(k).name));
        
        fprintf('Processing: %s\\n', input_path);
        I = imread(input_path);
        S = rog_smooth(I, 0.01, 1, 3, 1, 2.0, false);
        imwrite(S, output_path);
        fprintf('Saved: %s\\n', output_path);
    end

    disp('Smoothing Completed.');
    """
    subprocess.run(f"matlab -batch \"{matlab_script}\"", shell=True, check=True)
    print("Smoothing Completed.")
    print("Running Step 3: Canny Edge Detection (MATLAB)...")
    matlab_script = f"addpath('s2_border_detection'); canny_edge;"
    subprocess.run(f"matlab -batch \"{matlab_script}\"", shell=True, check=True)
    print("Edge Detection Completed.")

# -------------- STEP 3: FEATURE EXTRACTION -------------
def extract_features():
    print("Running Step 3: Extracting Features...")

    # --- Run Zernike Moments (Python) ---
    STEP3_FOLDER = os.path.join(PROJECT_ROOT, "s3_measure_border_irregularity")
    script_path = os.path.join(STEP3_FOLDER, "zernike_moments.py")

    print("Extracting Zernike Moments...")
    subprocess.run(["python3", script_path], check=True)
    print("Zernike Moments Extraction Completed.")

    # --- Run Fractal Dimension (MATLAB) ---
    print("Extracting Fractal Dimension...")
    matlab_script = f"""
    addpath('s3_measure_border_irregularity');
    input_folder = fullfile('{BORDERS_DIR}');
    output_folder = fullfile('{FEATURES_DIR}');

    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    image_files = dir(fullfile(input_folder, '*.png'));

    for k = 1:numel(image_files)
        input_path = fullfile(input_folder, image_files(k).name);
        output_path = fullfile(output_folder, strcat('fractal_', image_files(k).name, '.mat'));

        fprintf('Processing: %s\\n', input_path);
        img = imread(input_path);

        % Compute Fractal Dimension
        fractal_value = fractal_dimesion(img);
        save(output_path, 'fractal_value');
        fprintf('Saved: %s\\n', output_path);
    end

    disp('Fractal Dimension Extraction Completed.');
    """
    subprocess.run(f"matlab -batch \"{matlab_script}\"", shell=True, check=True)

    # --- Run Border Score Calculation (Python) ---
    print("Calculating Border Score...")
    border_score_script = os.path.join(STEP3_FOLDER, "border_score.py")
    subprocess.run(["python3", border_score_script], check=True)
    print("Border Score Calculation Completed.")
    print("Feature Extraction Completed.")

# -------------- STEP 4: CLASSIFICATION --------------
def run_classification():
    print("Running Step 4: Classification...")

    STEP4_FOLDER = os.path.join(PROJECT_ROOT, "s4_classification")

    cnn_script = os.path.join(STEP4_FOLDER, "cnn_model.py")
    bayes_script = os.path.join(STEP4_FOLDER, "naive_bayes.py")

    print("Running CNN classification...")
    subprocess.run(["python3", cnn_script], check=True)
    print("CNN Classification Completed.")

    print("Running Naive Bayes classification...")
    subprocess.run(["python3", bayes_script], check=True)
    print("Naive Bayes Classification Completed.")

# -------------- STEP 5: DECISION MAKING --------------
def make_final_decision():
    STEP5_FOLDER = os.path.join(PROJECT_ROOT, "s5_decision")
    decision_script = os.path.join(STEP5_FOLDER, "final_decision.py")

    print("Making final decisions using ensemble learning...")
    subprocess.run(["python3", decision_script])

# -------------- STEP 6: EVALUATION --------------
def evaluate_results():
    STEP6_FOLDER = os.path.join(PROJECT_ROOT, "s6_evaluation")
    eval_script = os.path.join(STEP6_FOLDER, "evaluate.py")

    print("Evaluating results...")
    subprocess.run(["python3", eval_script])

# ---------------------- MAIN EXECUTION ----------------------
if __name__ == "__main__":
    print("**Starting Pipeline**")
    
    print("\n- STEP 1: Segmentation")
    segment_images()

    print("\n- STEP 2: Border Detection")
    detect_edges()

    print("\n- STEP 3: Feature Extraction")
    extract_features()

    print("\n- STEP 4: Classification")
    run_classification()

    print("\- STEP 5: Decision Making")
    make_final_decision()

    print("\n- STEP 6: Evaluation")
    evaluate_results()

    print("\n**Pipeline Completed!**")
