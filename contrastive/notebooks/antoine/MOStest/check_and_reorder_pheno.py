import sys
import os
import pandas as pd
from glob import glob

def check_and_reorder_pheno(pheno_file, fam_folder):
    """
    Checks that all .fam files have the same order and reorders the phenotype file accordingly.
    Missing rows in the phenotype file are filled with 'NaN' values.

    Parameters:
    -----------
    pheno_file : str
        Path to the phenotype file with the IID column.
    fam_folder : str
        Path to the folder containing .fam files to check and reorder against.

    Returns:
    --------
    None
    """
    # Step 1: Read the phenotype file
    pheno_df = pd.read_csv(pheno_file, sep='\t')
    if 'IID' not in pheno_df.columns:
        print(f"Error: 'IID' column not found in {pheno_file}")
        return
    iid_index = pheno_df.columns.get_loc('IID')
    pheno_iids = set(pheno_df['IID'].tolist())

    # Step 2: Find all .fam files in the specified folder
    fam_files = glob(os.path.join(fam_folder, "imputed_chr*_decim_maf-*.fam"))
    #fam_files = glob(os.path.join(fam_folder, "*.fam"))
    
    # Print the number of .fam files found
    print(f"Number of .fam files found: {len(fam_files)}")
    print("Files found:")
    for file in fam_files:
        print(file)

    # Step 3: Read and check the order of .fam files
    fam_orders = []
    for fam_file in fam_files:
        fam_df = pd.read_csv(fam_file, delim_whitespace=True, header=None)
        fam_df.columns = ['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype']
        fam_orders.append(fam_df['IID'].tolist())

    # Check if all .fam files have the same order
    if not all(order == fam_orders[0] for order in fam_orders):
        print("Error: Not all .fam files have the same order")
        return

    fam_order = fam_orders[0]

    # Step 4: Reorder the phenotype file according to the .fam file order
    reordered_pheno_df = pd.DataFrame(columns=pheno_df.columns)
    for iid in fam_order:
        if iid in pheno_iids:
            row = pheno_df[pheno_df['IID'] == iid]
        else:
            row = pd.DataFrame([['NaN'] * pheno_df.shape[1]], columns=pheno_df.columns)
            row.iloc[0, iid_index] = iid
        reordered_pheno_df = pd.concat([reordered_pheno_df, row], ignore_index=True)

    # Save the reordered phenotype file
    if "#FID" in reordered_pheno_df.columns:
        reordered_pheno_df = reordered_pheno_df.drop(["#FID"], axis=1)
    if "FID" in reordered_pheno_df.columns:
        reordered_pheno_df = reordered_pheno_df.drop(["FID"], axis=1)
    if "IID" in reordered_pheno_df.columns:
        reordered_pheno_df = reordered_pheno_df.drop(["IID"], axis=1)

    output_pheno_file = pheno_file.replace('.txt', '_reordered.txt')
    reordered_pheno_df.to_csv(output_pheno_file, sep='\t', index=False)
    print(f"Reordered phenotype file saved as {output_pheno_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 check_and_reorder_pheno.py <pheno_file> <fam_folder>")
        sys.exit(1)

    pheno_file = sys.argv[1]
    fam_folder = sys.argv[2]

    check_and_reorder_pheno(pheno_file, fam_folder)