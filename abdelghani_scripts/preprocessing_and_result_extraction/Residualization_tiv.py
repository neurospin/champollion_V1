import os
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# === Paths ===
tiv_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_tiv_volumes_normalized.csv"
dest_dirs = "/home/cb283697/Bureau/SC_right_ukb_embeddings/dim_32_tiv",



# === Load TIV data ===
df_tiv = pd.read_csv(tiv_path)
df_tiv.columns = df_tiv.columns.str.strip().str.lower()
df_tiv = df_tiv.drop_duplicates(subset="participant_id").dropna(subset=["volume"])
df_tiv.set_index("participant_id", inplace=True)

# === Function to residualize one embedding file ===
def residualize_embeddings(embedding_path, tiv_df, output_path):
    embeddings = pd.read_csv(embedding_path, index_col=0)

    # Match TIV rows
    embeddings = embeddings.loc[embeddings.index.intersection(tiv_df.index)]
    tiv = tiv_df.loc[embeddings.index][["volume"]]

    # Standardize both
    scaler = StandardScaler()
    embeddings_std = pd.DataFrame(
        scaler.fit_transform(embeddings),
        index=embeddings.index,
        columns=embeddings.columns
    )
    tiv_std = scaler.fit_transform(tiv)

    # Add constant for regression
    tiv_std = sm.add_constant(tiv_std)

    residualized = pd.DataFrame(index=embeddings_std.index, columns=embeddings_std.columns)

    for col in embeddings_std.columns:
        model = sm.OLS(embeddings_std[col], tiv_std).fit()
        residualized[col] = model.resid

    residualized.to_csv(output_path)
    print(f"✅ Residualized and saved: {output_path}")

# === Process both dimensions ===
for dim, folder in dest_dirs.items():
    print(f"\n Processing dimension {dim} in folder: {folder}")
    output_folder = folder + "_residualized"
    os.makedirs(output_folder, exist_ok=True)

    for file in sorted(os.listdir(folder)):
        if file.endswith(".csv"):
            embed_path = os.path.join(folder, file)
            output_path = os.path.join(output_folder, file)
            residualize_embeddings(embed_path, df_tiv, output_path)

print("\n All embeddings residualized by TIV.")
