import pandas as pd
import numpy as np
from kernels import KernelMetric  # make sure this is correct

# Load your dataset
csv_path = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_sex_age_volumes.csv"
df = pd.read_csv(csv_path)

# Drop NaNs and normalize
df = df.dropna(subset=["volume"])
X = df['volume'].values.reshape(-1, 1)

# Manually compute std and Scott's factor
n, d = X.shape
std = np.std(X, ddof=0)  # population std (bias=False in cov also uses this)
scott_factor = n ** (-1. / (d + 4))
sigma = std * scott_factor

# Fit KernelMetric with Scott's rule
kernel_metric = KernelMetric(kernel="gaussian", bw_method="scott")
kernel_metric.fit(X)

# Retrieve bandwidth matrix
H = kernel_metric.sqr_bandwidth_
sigma_kernel = H[0, 0]

# Print results
print(f"Number of samples (n): {n}")
print(f"Dimension (d): {d}")
print(f"Standard deviation of X: {std:.6f}")
print(f"Scott's factor: {scott_factor:.6f}")
print(f"σ = std × scott_factor = {sigma:.6f}")
print(f"Bandwidth from KernelMetric (σ): {sigma_kernel:.6f}")
