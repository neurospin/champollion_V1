import os
import json

root_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/80_epoch_FIP_right"

print(f"\nChecking folder: {root_dir}")
print(f"{'='*30} CV Score Summary {'='*30}")
print(f"{'Config':<45} {'CV Score (± Std)':<20}")
print("-" * 70)

for config_name in sorted(os.listdir(root_dir)):
    config_path = os.path.join(root_dir, config_name)

    if os.path.isdir(config_path):
        json_path = os.path.join(
            config_path,
            "FIP_right_custom_embeddings",
            "Right_FIP",
            "test_values.json"
        )

        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    cv = data.get("cv_score", None)
                    std = data.get("cv_std", None)

                    if cv is not None and std is not None:
                        print(f"{config_name:<45} {cv:.4f} ± {std:.4f}")
                    else:
                        print(f"{config_name:<45} Missing score")
            except Exception as e:
                print(f"{config_name:<45} Error reading file: {e}")
        else:
            print(f"{config_name:<45} test_values.json not found")

print("=" * 70)
