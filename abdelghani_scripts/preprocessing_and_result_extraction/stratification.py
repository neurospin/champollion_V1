import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import csv

# Input CSV path
csv_path = "/neurospin/dico/data/deep_folding/current/datasets/hcp/hcp_subjects_gender_age_volume_normalized.csv"

# Output directories
base_path = "/neurospin/dico/data/deep_folding/current/datasets/hcp/volume/"
fig_path = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/figure_folder/"

os.makedirs(base_path, exist_ok=True)
os.makedirs(fig_path, exist_ok=True)

# Load CSV
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File not found: {csv_path}")
df = pd.read_csv(csv_path)

# Drop missing values in key columns
df = df.dropna(subset=["Gender", "Age", "volume"])

# Bin volume for stratification
df["volume_bin"] = pd.cut(df["volume"], bins=np.arange(-3, 3.5, 0.5))

# Create stratification key
df["stratify_key"] = df["Gender"].astype(str) + "_" + df["Age"].astype(str) + "_" + df["volume_bin"].astype(str)

# Remove rare groups (less than 2 samples)
group_counts = df["stratify_key"].value_counts()
valid_keys = group_counts[group_counts >= 2].index
df = df[df["stratify_key"].isin(valid_keys)].reset_index(drop=True)

# Split 64% train, 36% temp
train_df, temp_df = train_test_split(df, test_size=0.36, stratify=df["stratify_key"], random_state=42)

# Split 36% temp into 16% val, 20% test
val_size = 16 / (16 + 20)

# Re-check stratify_key in temp to avoid rare groups
temp_group_counts = temp_df["stratify_key"].value_counts()
valid_temp_keys = temp_group_counts[temp_group_counts >= 2].index
temp_df = temp_df[temp_df["stratify_key"].isin(valid_temp_keys)].reset_index(drop=True)

# Safe stratified split
val_df, test_df = train_test_split(
    temp_df,
    test_size=1 - val_size,
    stratify=temp_df["stratify_key"],
    random_state=42
)

# Select only the desired columns for full metadata files
keep_cols = ["Subject", "Gender", "Age", "both.brain_volume", "volume"]
train_df = train_df[keep_cols]
val_df = val_df[keep_cols]
test_df = test_df[keep_cols]

# Save full metadata splits
train_df.to_csv(os.path.join(base_path, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(base_path, "val_split.csv"), index=False)
test_df.to_csv(os.path.join(base_path, "test_split.csv"), index=False)

# Merge train and val for CV
train_val_df = pd.concat([train_df, val_df], ignore_index=True)
train_val_df.to_csv(os.path.join(base_path, "train_val_split.csv"), index=False)

# Save 5-folds as Subject ID lists
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (_, val_idx) in enumerate(skf.split(train_val_df, train_val_df["Gender"].astype(str) + "_" + train_val_df["Age"].astype(str))):
    fold = train_val_df.iloc[val_idx]
    fold[["Subject"]].to_csv(
        os.path.join(base_path, f"train_val_split_{i+1}.csv"),
        index=False,
        header=False,
        quoting=csv.QUOTE_ALL
    )

# Plot volume distribution for Train, Val, Test
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    plt.figure(figsize=(8, 4))
    sns.histplot(split_df["volume"], bins=30, kde=True, color='orchid', edgecolor='black')
    plt.title(f"Z-scored Brain Volume Distribution - {split_name.capitalize()} Split")
    plt.xlabel("Z-scored Volume")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"{split_name}_volume_distribution.png"))
    plt.close()

# Plot volume distribution for each fold
for i in range(1, 6):
    fold_df = pd.read_csv(os.path.join(base_path, f"train_val_split_{i}.csv"), header=None, names=["Subject"])
    joined = pd.merge(fold_df, df[["Subject", "volume"]], on="Subject", how="left")

    plt.figure(figsize=(8, 4))
    sns.histplot(joined["volume"], bins=30, kde=True, edgecolor='black')
    plt.title(f"Z-scored Volume Distribution - Fold {i}")
    plt.xlabel("Z-scored Volume")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"fold{i}_volume_distribution.png"))
    plt.close()

print("Stratified split and plots done. Files saved:")
print(" - train_split.csv")
print(" - val_split.csv")
print(" - test_split.csv")
print(" - train_val_split.csv")
print(" - train_val_split_1.csv to train_val_split_5.csv (Subject IDs only)")
print(" - Volume distribution plots saved in:", fig_path)
