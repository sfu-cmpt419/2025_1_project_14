import pandas as pd

### --- ASYMMETRY FILES --- ###
asymmetry_files = [
    "asymmetry_test_groundtruth_results.csv",
    "asymmetry_training_groundtruth_results.csv",
    "asymmetry_validation_groundtruth_results.csv"
]

asymmetry_dfs = []
for file in asymmetry_files:
    df = pd.read_csv(file)
    df['Image'] = df['Image'].str.replace('_segmentation.png', '', regex=False)
    df = df[['Image', 'Asymmetry Contribution']].rename(columns={'Asymmetry Contribution': 'A_score'})
    asymmetry_dfs.append(df)

asymmetry_combined = pd.concat(asymmetry_dfs, ignore_index=True)
asymmetry_combined = asymmetry_combined.drop_duplicates(subset='Image')

### --- COLOR FILES --- ###
color_files = [
    "color_test_results.csv",
    "color_training_results.csv",
    "color_validation_results.csv"
]

color_dfs = []
for file in color_files:
    df = pd.read_csv(file)
    df['Image'] = df['Image'].str.replace('.jpg', '', regex=False)
    df = df[['Image', 'Color Score']].rename(columns={'Color Score': 'C_score'})
    color_dfs.append(df)

color_combined = pd.concat(color_dfs, ignore_index=True)
color_combined = color_combined.drop_duplicates(subset='Image')

### --- D VALUE FILE --- ###
d_df = pd.read_csv("d_results.csv")
d_df = d_df[['lesion_id', 'D_value']].rename(columns={'lesion_id': 'Image'})
d_df = d_df.drop_duplicates(subset='Image')

### --- B VALUE FILE --- ###
b_df = pd.read_csv("b_values.csv")
b_df = b_df.rename(columns={"Image_Name": "Image", "Border_Score": "B_score"})
b_df['Image'] = b_df['Image'].str.replace('.jpg', '', regex=False)
b_df = b_df.drop_duplicates(subset='Image')


### --- MERGE ALL DATAFRAMES --- ###
merged_df = pd.merge(asymmetry_combined, color_combined, on='Image', how='outer')
merged_df = pd.merge(merged_df, d_df, on='Image', how='outer')
merged_df = pd.merge(merged_df, b_df, on='Image', how='outer')

# Drop rows with missing values in any column
merged_df = merged_df.dropna()

# Sort and save
merged_df = merged_df.sort_values(by='Image').reset_index(drop=True)
merged_df.to_csv("clean_data.csv", index=False)

print("Saved fully merged and cleaned dataset to 'clean_data.csv'.")
