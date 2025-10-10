import os
import json
import re
import pandas as pd

root_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/SC_right"
results = []

# Loop over config folders like batch_size128_sigma_0.01
for config_name in sorted(os.listdir(root_dir)):
    config_path = os.path.join(root_dir, config_name)
    if not os.path.isdir(config_path):
        continue

    embeddings_root = os.path.join(config_path, "ukb40_random_embeddings")

    r2_dict = {"config": config_name}

    # Extract Batch and Sigma from config name
    batch_match = re.search(r'batch_size(\d+)', config_name)
    sigma_match = re.search(r'sigma_([0-9.]+)', config_name)

    r2_dict["Batch"] = int(batch_match.group(1)) if batch_match else None
    r2_dict["Sigma"] = float(sigma_match.group(1)) if sigma_match else None

    for dim in range(1, 7):
        json_path = os.path.join(embeddings_root, f"Isomap_central_right_dim{dim}", "full_values.json")
        key = f"dim{dim}_r2"

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    r2_dict[key] = data.get("full_r2", None)
            except Exception as e:
                print(f"[WARN] Could not read {json_path}: {e}")
                r2_dict[key] = None
        else:
            r2_dict[key] = None

    # Average full R² across dims
    r2_values = [r2_dict[f"dim{d}_r2"] for d in range(1, 7) if r2_dict[f"dim{d}_r2"] is not None]
    r2_dict["avg_r2"] = sum(r2_values) / len(r2_values) if r2_values else None

    results.append(r2_dict)

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values(by=["Batch", "Sigma"]).reset_index(drop=True)

# Display
with pd.option_context('display.float_format', '{:,.6f}'.format):
    print(df[["Batch", "Sigma", "config", "dim1_r2", "dim2_r2", "dim3_r2", "dim4_r2", "dim5_r2", "dim6_r2", "avg_r2"]].to_string(index=False))
