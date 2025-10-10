import os
import shutil


root_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Output/yaware_volume/FCLp-subsc-FCLa-INSULA_left/"

dest_dir_32 = "/home/cb283697/Bureau/FCLp-subsc-FCLa-INSULA_left/dim_32"
dest_dir_256 = "/home/cb283697/Bureau/FCLp-subsc-FCLa-INSULA_left/dim_256"


os.makedirs(dest_dir_32, exist_ok=True)
os.makedirs(dest_dir_256, exist_ok=True)

for config_name in sorted(os.listdir(root_dir)):
    config_path = os.path.join(root_dir, config_name)

    if os.path.isdir(config_path):
        embedding_file = os.path.join(
            config_path,
            "full_ukb_all_sub_random_embeddings",
            "full_embeddings.csv"
        )

        if os.path.exists(embedding_file):
            if config_name.startswith("dim_32"):
                dest_file = os.path.join(dest_dir_32, f"full_embeddings_{config_name}.csv")
            elif config_name.startswith("dim_256"):
                dest_file = os.path.join(dest_dir_256, f"full_embeddings_{config_name}.csv")
            else:
                print(f"Skipped unknown config: {config_name}")
                continue
            try:
                shutil.copy(embedding_file, dest_file)
                print(f" Copied: {config_name} → {dest_file}")
            except Exception as e:
                print(f" Error copying {config_name}: {e}")
        else:
            print(f" Missing full_embeddings.csv in {config_name}")

print("\n Done copying all full_embeddings.csv files.")
