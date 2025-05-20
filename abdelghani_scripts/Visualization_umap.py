import pandas as pd
import umap
import matplotlib.pyplot as plt

# Path to your embeddings CSV
csv_path = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/Large_cingulate_right/sigma_0.01_batch_size_32/ACC_custom_embeddings/custom_cross_val_embeddings.csv"

# Load data
df = pd.read_csv(csv_path)
embedding_cols = [col for col in df.columns if col.startswith("dim")]
X = df[embedding_cols].values

# UMAP to 2D
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# Plot and save
plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=10, cmap="Spectral")
plt.title("UMAP Projection of 32D Embeddings")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.grid(True)
plt.tight_layout()

# Save to file
output_path = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/umap_embedding_plot.png"
plt.savefig(output_path)
print(f"UMAP plot saved to: {output_path}")
