import pandas as pd

df = pd.read_csv("b_values_merged_data.csv")
df['Image_Name'] = df['Image_Name'].str.replace('.jpg', '', regex=False)
df.to_csv("merged_data_b_scores.csv", index=False)

