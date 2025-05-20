import pandas as pd

# Paths
participants_path = "/neurospin/dico/data/deep_folding/current/datasets/hcp/participants.csv"
volumes_path = "/neurospin/dico/data/human/hcp/derivatives/morphologist-2023/morphometry/brain_volumes.csv"

# Load participants (standard CSV)
participants_df = pd.read_csv(participants_path)
participants_df["Subject"] = participants_df["Subject"].astype(str)

# Load volumes (semicolon-separated file)
volumes_df = pd.read_csv(volumes_path, sep=";")

# Print columns to inspect if needed
# print(volumes_df.columns.tolist())

# Drop duplicate 'subject' column if needed
if "subject.1" in volumes_df.columns:
    volumes_df = volumes_df.drop(columns=["subject.1"])

# Rename 'subject' to match 'Subject'
volumes_df = volumes_df.rename(columns={"subject": "Subject"})

# Ensure type is string for merging
volumes_df["Subject"] = volumes_df["Subject"].astype(str)

# Merge on 'Subject'
merged_df = pd.merge(participants_df, volumes_df, on="Subject")

# Keep relevant columns
final_df = merged_df[["Subject", "Gender", "Age", "both.brain_volume"]]

# Save to CSV
output_path = "/neurospin/dico/data/deep_folding/current/datasets/hcp/hcp_subjects_gender_age_volume.csv"
final_df.to_csv(output_path, index=False)

print(f"File saved to: {output_path}")
