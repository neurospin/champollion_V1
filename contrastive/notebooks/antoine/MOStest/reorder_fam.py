"""
Reorder .fam File Based on Phenotype File

This script reorders the rows of a list of `.fam` file to match the order of IIDs (individual IDs) given in a phenotype file. It ensures that the second column (IID) of the `.fam` file is reordered according to the IID column in the phenotype file and saves the reordered `.fam` file with `_reordered.fam` appended to the original name.

Additionally, it creates a new phenotype file without the `IID` and `#FID` (or `FID`) columns.

Usage:
    python3 reorder_fam.py pre_residualized_phenotype.txt /path/to/fam_file.fam

Arguments:
    pre_residualized_phenotype.txt: Path to the phenotype file containing the IID column. This file is used to determine the desired order of IIDs.
    /path/to/fam_file.fam: Path to the `.fam` files to be reordered. This `.fam` files will be processed.

The script performs the following steps:
1. Reads the IID column from the phenotype file.
2. Reads the `.fam` files.
3. Filters out rows where the IID is not present in the phenotype file.
4. Reorders the rows to match the IID order specified in the phenotype file.
5. Saves the reordered file with `_reordered.fam` appended to the original filename.
6. Saves a new phenotype file without the IID and `#FID` (or `FID`) columns.

Author: Antoine Dufournet
Date: 2024-07-26
"""

import sys
import pandas as pd
import os

def reorder_fam_files(pheno_file, fam_files):
    """
    Reorders .fam files based on the IID order from the phenotype file.
    Saves the reordered files with '_reordered.fam' appended to the original file name.

    Parameters:
    -----------
    pheno_file : str
        Path to the phenotype file with the IID column.
    fam_files : list of str
        List of paths to the .fam files to be reordered.

    Returns:
    --------
    None
    """
    # Step 1: Read the IID column from the phenotype file
    pheno_df = pd.read_csv(pheno_file, sep='\t')
    if 'IID' not in pheno_df.columns:
        print(f"Error: 'IID' column not found in {pheno_file}")
        return
    iid_order = pheno_df['IID'].tolist()
    iid_set = set(iid_order)

    # Step 2: Process each .fam file
    for fam_file in fam_files:
        # Read the .fam file
        fam_df = pd.read_csv(fam_file, delim_whitespace=True, header=None)
        
        # The second column (index 1) is the IID column in the .fam file
        fam_df.columns = ['FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype']
        
        # Filter out rows where IID is not in the phenotype IID list
        fam_df_filtered = fam_df[fam_df['IID'].isin(iid_set)]
        
        # Create a dictionary from the filtered .fam file rows keyed by IID
        fam_dict = fam_df_filtered.set_index('IID', drop=False).to_dict('index')

        # Reorder the rows based on the IID order from the phenotype file
        reordered_rows = [fam_dict[iid] for iid in iid_order if iid in fam_dict]

        # Convert the reordered rows back to a DataFrame
        reordered_fam_df = pd.DataFrame(reordered_rows)

        # Define the output file name
        output_file = fam_file.replace('.fam', '_reordered.fam')
        
        # Save the reordered .fam file
        reordered_fam_df.to_csv(output_file, sep=' ', header=False, index=False)
        print(f"Reordered {fam_file} and saved as {output_file} successfully")

    # Save the new pre_residualized_pheno_for_mostest.txt without IID and FID columns
    new_pheno_file = pheno_file.replace('.txt', '_for_mostest.txt')
    pheno_df.drop(columns=['#FID', 'IID'], errors='ignore').to_csv(new_pheno_file, sep='\t', index=False)
    print(f"Saved new phenotype file without IID and FID columns as {new_pheno_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 reorder_fam.py pre_residualized_phenotype.txt fam_file1.fam [fam_file2.fam ...]")
        sys.exit(1)
    
    pheno_file = sys.argv[1]
    fam_files = sys.argv[2:]
    
    reorder_fam_files(pheno_file, fam_files)