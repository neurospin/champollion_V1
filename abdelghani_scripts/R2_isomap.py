import os
import json

root_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/SC_left"

print(f"\nChecking folder: {root_dir}")
print(f"{'='*30} Avg Test R2 Results (Isomap dims 1–6) {'='*30}")
print(f"{'Config':<40} {'Avg Test R2':<10}")
print("-" * 70)

for config_name in sorted(os.listdir(root_dir)):
    config_path = os.path.join(root_dir, config_name)
    if os.path.isdir(config_path):
        isomap_dir = os.path.join(config_path, "hcp_isomap_custom_embeddings")
        r2_values = []

        for dim in range(1, 7):
            json_path = os.path.join(isomap_dir, f"Isomap_central_left_dim{dim}", "test_values.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        r2 = data.get("test_r2", None)
                        if r2 is not None:
                            r2_values.append(r2)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")

        if r2_values:
            avg_r2 = sum(r2_values) / len(r2_values)
            print(f"{config_name:<40} {avg_r2:<10.4f}")
        else:
            print(f"{config_name:<40} No valid R2 values found")

print("=" * 70)
