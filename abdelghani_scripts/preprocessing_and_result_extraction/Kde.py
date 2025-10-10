import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr, gaussian_kde

# Load CSV
csv_path = "/neurospin/dico/data/deep_folding/current/datasets/hcp/hcp_subjects_gender_age_volume.csv"
df = pd.read_csv(csv_path)

# Drop NaNs for brain volume
df = df.dropna(subset=["both.brain_volume"])
volumes = df['both.brain_volume']

# Z-score normalization
mean_volume = volumes.mean()
std_volume = volumes.std()
df['volume'] = (df['both.brain_volume'] - mean_volume) / std_volume
z_scores = df['volume'].values

# Compute stats
n = len(z_scores)
sigma = np.std(z_scores)
iqr_val = iqr(z_scores)
A = min(sigma, iqr_val / 1.34)
h = 0.9 * A * n**(-1/5)

# KDE estimation
kde = gaussian_kde(z_scores, bw_method=h / sigma)  # scipy's bw_method expects h/σ
x_vals = np.linspace(-3, 3, 300)
kde_vals = kde(x_vals)

# Define histogram bins
bin_edges = np.arange(-3, 4, 0.2)
bin_labels = [f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f})" for i in range(len(bin_edges) - 1)]
counts, _ = np.histogram(z_scores, bins=bin_edges)

# Plot
plt.figure(figsize=(10, 5))
plt.hist(z_scores, bins=bin_edges, color='orchid', edgecolor='black', alpha=0.6, label='Histogram')
plt.plot(x_vals, kde_vals, color='black', linewidth=2, label=f'KDE (h={h:.3f})')

# Annotate values
text = (f"n = {n}\n"
        f"$\\hat\\sigma$ = {sigma:.3f}\n"
        f"IQR = {iqr_val:.3f}\n"
        f"A = {A:.3f}\n"
        f"h = {h:.3f}")
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.title("KDE over Z-scored Brain Volumes")
plt.xlabel("Z-scored Volume")
plt.ylabel("Count / Density")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Save the plot
plot_path = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/abdelghani_scripts/kde_hcp_zrange.png"
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print(f"KDE plot saved to: {plot_path}")
