import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
mask_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm/LARGE_CINGULATE./mask/Rskeleton_subject.csv"
meta_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_sex_age_volumes_normalized.csv"
output_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm/LARGE_CINGULATE./mask/"

# Load data
mask_df = pd.read_csv(mask_path)
meta_df = pd.read_csv(meta_path)

# Rename participant_id to subject in metadata
meta_df = meta_df.rename(columns={"participant_id": "subject"})

# Rename subject column in mask_df if needed
if "subject" not in mask_df.columns:
    possible_keys = ["participant_id", "Subject", "ID"]
    for key in possible_keys:
        if key in mask_df.columns:
            mask_df = mask_df.rename(columns={key: "subject"})
            print(f"✅ Renamed column '{key}' to 'subject' in mask_df.")
            break
    else:
        raise ValueError("❌ No 'subject' column found in mask_df.")

# Convert subject columns to string
mask_df["subject"] = mask_df["subject"].astype(str)
meta_df["subject"] = meta_df["subject"].astype(str)

# Merge on subject
df = pd.merge(mask_df, meta_df, on="subject")
print(f"✅ Merged {len(df)} subjects.")
print("📋 Columns in merged df:", df.columns.tolist())

# Auto-detect feature columns
def find_column(df, candidates, name):
    for col in candidates:
        if col in df.columns:
            print(f"✅ Using '{col}' as {name}")
            return col
    raise ValueError(f"❌ None of {candidates} found for {name} in columns.")

sex_col = find_column(df, ["sex", "Sex", "gender"], "sex")
age_col = find_column(df, ["age", "Age"], "age")
volume_col = find_column(df, ["volume", "Volume", "normalized_volume"], "volume")

# Select features
features = df[[sex_col, age_col, volume_col]].copy()

# Drop participants with NaNs in features
before_drop = len(features)
features = features.dropna()
df = df.loc[features.index]
print(f"🧹 Dropped {before_drop - len(features)} participants due to NaNs in features.")

# Reset index so iloc works correctly
df = df.reset_index(drop=True)
features = features.reset_index(drop=True)

# Normalize age (volume is assumed already normalized)
scaler = StandardScaler()
features[age_col] = scaler.fit_transform(features[[age_col]])

# KMeans clustering for stratified sampling
kmeans = KMeans(n_clusters=10, random_state=42)
df["cluster"] = kmeans.fit_predict(features)

# Stratified K-Fold split
skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
splits = list(skf.split(df, df["cluster"]))

# ✅ Correct disjoint split: use validation indices from folds
folds = [split[1] for split in splits]
train_idx = np.concatenate([folds[i] for i in range(17)])
val_idx = folds[17]
test_idx = np.concatenate([folds[18], folds[19]])

# Create dataframes for each split
train_df = df.iloc[train_idx].copy()
val_df = df.iloc[val_idx].copy()
test_df = df.iloc[test_idx].copy()

# Save CSV files
train_df.to_csv(f"{output_dir}/Rskeleton_subject_train.csv", index=False)
val_df.to_csv(f"{output_dir}/Rskeleton_subject_val.csv", index=False)
test_df.to_csv(f"{output_dir}/Rskeleton_subject_test.csv", index=False)

print("✅ CSVs saved.")
print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# Plot and save age + sex distribution
def save_distributions_plot(train_df, val_df, test_df, save_path, age_col, sex_col):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Age distribution (KDE)
    sns.kdeplot(train_df[age_col], label="Train", fill=True, ax=axes[0])
    sns.kdeplot(val_df[age_col], label="Val", fill=True, ax=axes[0])
    sns.kdeplot(test_df[age_col], label="Test", fill=True, ax=axes[0])
    axes[0].set_title("Age Distribution")
    axes[0].legend()

    # Sex distribution (bar plot)
    sex_counts = pd.DataFrame({
        "Train": train_df[sex_col].value_counts(normalize=True),
        "Val": val_df[sex_col].value_counts(normalize=True),
        "Test": test_df[sex_col].value_counts(normalize=True)
    }).T
    sex_counts.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Sex Distribution")
    axes[1].set_ylabel("Proportion")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, "split_distributions.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"✅ Age + Sex distribution plot saved to {fig_path}")

# Plot and save volume distribution
def plot_volume_distribution(train_df, val_df, test_df, volume_col, save_path):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(train_df[volume_col], label="Train", fill=True)
    sns.kdeplot(val_df[volume_col], label="Val", fill=True)
    sns.kdeplot(test_df[volume_col], label="Test", fill=True)
    plt.title("Volume Distribution")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(save_path, "volume_distribution.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Volume distribution plot saved to {fig_path}")

# Print mean and std of volume for each split
def print_volume_stats(train_df, val_df, test_df, volume_col):
    print("\nVolume Statistics:")
    for name, split in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        mean = split[volume_col].mean()
        std = split[volume_col].std()
        print(f"{name:<5} → Mean: {mean:.4f} | Std: {std:.4f}")

# Run plots and stats
save_distributions_plot(train_df, val_df, test_df, output_dir, age_col, sex_col)
plot_volume_distribution(train_df, val_df, test_df, volume_col, output_dir)
print_volume_stats(train_df, val_df, test_df, volume_col)
