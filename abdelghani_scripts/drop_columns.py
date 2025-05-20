import pandas as pd

# Path to the original CSV file
input_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm/LARGE_CINGULATE./mask/Rskeleton_subject_test.csv"

# Load the CSV file
df = pd.read_csv(input_path)

# Keep only the 'Subject' column (make sure it exists)
if "Subject" in df.columns:
    df_subject_only = df[["Subject"]]
else:
    raise ValueError("The column 'Subject' does not exist in the CSV file.")

# Save the result to a new file (or overwrite if you prefer)

df_subject_only.to_csv(input_path, index=False)

print(f"Saved file with only 'Subject' column to: {input_path}")
